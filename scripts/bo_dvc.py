#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    print("Missing dependency: PyYAML (yaml). Please install it first.", file=sys.stderr)
    sys.exit(1)

try:
    import optuna
except Exception:
    print("Missing dependency: optuna. Please install it first.", file=sys.stderr)
    sys.exit(1)


def load_space(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "bo" not in data:
        raise ValueError("BO config must have top-level 'bo' mapping")
    space = data["bo"]
    if not isinstance(space, dict) or not space:
        raise ValueError("'bo' mapping is empty or invalid")
    return space


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("_-") or "bo"


def format_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value)


def sample_params(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(f"Invalid spec for {name}")
        stype = str(spec["type"]).lower()
        if stype == "categorical":
            values = spec.get("values", [])
            if not isinstance(values, list) or not values:
                raise ValueError(f"Categorical values missing for {name}")
            if any(isinstance(v, (list, tuple)) for v in values):
                encoded = [json.dumps(v, separators=(",", ":")) for v in values]
                choice = trial.suggest_categorical(name, encoded)
                params[name] = json.loads(choice)
            else:
                params[name] = trial.suggest_categorical(name, values)
        elif stype == "uniform":
            low = float(spec["low"])
            high = float(spec["high"])
            step = spec.get("step")
            if step is not None:
                step = float(step)
                if step <= 0:
                    raise ValueError(f"Step must be > 0 for {name}")
                params[name] = trial.suggest_float(name, low, high, step=step)
            else:
                params[name] = trial.suggest_float(name, low, high)
        elif stype == "loguniform":
            if "step" in spec:
                raise ValueError(f"Step is not supported for loguniform ({name})")
            low = float(spec["low"])
            high = float(spec["high"])
            params[name] = trial.suggest_float(name, low, high, log=True)
        else:
            raise ValueError(f"Unknown type '{stype}' for {name}")
    return params


def build_overrides(params: dict, run_name: str) -> dict:
    overrides = {"run_name": run_name}
    for name, value in params.items():
        if name == "input_resolution":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("input_resolution must be a 2-item list")
            overrides["input_resolution_h"] = int(value[0])
            overrides["input_resolution_w"] = int(value[1])
        else:
            overrides[name] = value
    return overrides


def build_param_map(params_file: Path) -> dict:
    if not params_file.exists():
        return {}
    with params_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    mapping: dict[str, list[str]] = {}

    def walk(prefix, node):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(prefix + [key], value)
        else:
            if not prefix:
                return
            leaf = prefix[-1]
            path = ".".join(prefix)
            mapping.setdefault(leaf, []).append(path)

    for key, value in data.items():
        walk([key], value)
    return mapping


def resolve_param_key(key: str, param_map: dict) -> str:
    if "." in key:
        return key
    paths = param_map.get(key)
    if not paths:
        return key
    if len(paths) == 1:
        return paths[0]
    raise ValueError(f"Ambiguous param key '{key}', use dotted path: {paths}")


def resolve_overrides(overrides: dict, param_map: dict) -> dict:
    resolved = {}
    for key, value in overrides.items():
        resolved_key = resolve_param_key(key, param_map)
        resolved[resolved_key] = value
    return resolved


def read_metric(metrics_path: Path, metric_key: str) -> float:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if metric_key not in data:
        raise KeyError(f"Metric '{metric_key}' not found in {metrics_path}")
    return float(data[metric_key])


def build_dvc_command(
    dvc_bin: str,
    exp_name: str,
    overrides: dict,
    use_exp_name: bool,
) -> list[str]:
    cmd = [dvc_bin, "exp", "run"]
    if use_exp_name:
        cmd.extend(["--name", exp_name])
    for key, value in overrides.items():
        cmd.extend(["-S", f"{key}={format_value(value)}"])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Bayesian optimization via DVC experiments")
    parser.add_argument("--space", default="config/BOparamas.yaml", help="Path to BO search space YAML")
    parser.add_argument("--metric", default="best_precision", help="Metric key in metrics.json")
    parser.add_argument("--direction", choices=["maximize", "minimize"], default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--study-name", default="bo")
    parser.add_argument("--storage", default=None, help="Optuna storage, e.g. sqlite:///bo_study.db")
    parser.add_argument("--resume", action="store_true", help="Resume an existing study (requires --storage)")
    parser.add_argument("--run-name-prefix", default="bo", help="Prefix for train.run_name")
    parser.add_argument("--metrics-path", default="metrics/metrics.json")
    parser.add_argument("--params-file", default="params.yaml", help="Params file for key resolution")
    parser.add_argument("--no-param-map", action="store_true", help="Use raw keys without params.yaml mapping")
    parser.add_argument("--dvc-bin", default="dvc")
    parser.add_argument("--name-experiments", action="store_true", help="Pass --name to dvc exp run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.resume and not args.storage:
        print("--resume requires --storage", file=sys.stderr)
        return 2

    space = load_space(Path(args.space))

    direction = args.direction
    if direction is None:
        if args.metric == "best_loss":
            direction = "minimize"
        else:
            direction = "maximize"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction=direction,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.resume,
        sampler=sampler,
    )

    metrics_path = Path(args.metrics_path)
    param_map = {} if args.no_param_map else build_param_map(Path(args.params_file))
    if not args.no_param_map and not param_map:
        print(
            f"Warning: params file not found or empty: {args.params_file}. Using raw keys.",
            file=sys.stderr,
        )
    run_prefix = sanitize_name(args.run_name_prefix)
    study_tag = sanitize_name(args.study_name)

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, space)
        run_name = f"{run_prefix}_{study_tag}_t{trial.number:04d}"
        exp_name = run_name
        overrides = build_overrides(params, run_name)
        dvc_overrides = resolve_overrides(overrides, param_map)

        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("exp_name", exp_name)

        cmd = build_dvc_command(args.dvc_bin, exp_name, dvc_overrides, args.name_experiments)
        print("Running:", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            trial.set_user_attr("dvc_failed", True)
            raise

        value = read_metric(metrics_path, args.metric)
        return value

    study.optimize(objective, n_trials=args.trials, catch=(subprocess.CalledProcessError,))

    best = study.best_trial
    best_overrides = build_overrides(
        sample_params(optuna.trial.FixedTrial(best.params), space),
        best.user_attrs.get("run_name", "best"),
    )
    best_dvc_overrides = resolve_overrides(best_overrides, param_map)

    print("Best metric:", best.value)
    print("Best trial number:", best.number)
    if "run_name" in best.user_attrs:
        print("Best run_name:", best.user_attrs["run_name"])
    print("Best overrides:")
    for key in sorted(best_dvc_overrides.keys()):
        print(f"  {key}={format_value(best_dvc_overrides[key])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
