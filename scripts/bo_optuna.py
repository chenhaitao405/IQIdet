#!/usr/bin/env python3
"""Bayesian optimization runner (Optuna + DVC experiments).

Example:
  python scripts/bo_optuna.py \
    --bo_config config/BOparamas.yaml \
    --params params.yaml \
    --study_name bo_iqi \
    --n_trials 20
"""

import argparse
import json
import os
import subprocess
import sys
import time

try:
    import yaml  # type: ignore
    _USE_PYYAML = True
except Exception:
    _USE_PYYAML = False
    try:
        from ruamel.yaml import YAML  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("YAML parser not found. Install PyYAML or ruamel.yaml.") from exc

try:
    import optuna  # type: ignore
except Exception:
    print("Optuna is required. Install with: pip install optuna", file=sys.stderr)
    sys.exit(1)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        if _USE_PYYAML:
            return yaml.safe_load(f)
        yaml_loader = YAML(typ="safe")
        return yaml_loader.load(f)


def ensure_pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a pair like [H, W], got: {value}")
    return int(value[0]), int(value[1])


def _as_float(value, name, field):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{name}.{field} must be numeric, got '{value}'") from exc
    raise ValueError(f"{name}.{field} must be numeric, got {type(value)}")


def _normalize_categorical(values):
    normalized = []
    for v in values:
        if isinstance(v, list):
            normalized.append(tuple(v))
        else:
            normalized.append(v)
    return normalized


def suggest_param(trial, name, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"Spec for {name} must be a dict, got: {spec}")
    ptype = spec.get("type")
    if ptype == "categorical":
        values = _normalize_categorical(spec["values"])
        return trial.suggest_categorical(name, values)
    if ptype == "uniform":
        low = _as_float(spec["low"], name, "low")
        high = _as_float(spec["high"], name, "high")
        return trial.suggest_float(name, low, high)
    if ptype == "loguniform":
        low = _as_float(spec["low"], name, "low")
        high = _as_float(spec["high"], name, "high")
        return trial.suggest_float(name, low, high, log=True)
    if ptype == "int":
        low = int(_as_float(spec["low"], name, "low"))
        high = int(_as_float(spec["high"], name, "high"))
        return trial.suggest_int(name, low, high)
    raise ValueError(f"Unknown type for {name}: {ptype}")


def build_dvc_cmd(params, run_name, processed_dir_template=None):
    cmd = ["dvc", "exp", "run"]

    def _set(key, val):
        cmd.extend(["-S", f"{key}={val}"])

    _set("train.run_name", run_name)

    if "heatmap_resolution" in params:
        _set("preprocess.heatmap_resolution", params["heatmap_resolution"])

    if "input_resolution" in params:
        h, w = ensure_pair(params["input_resolution"], "input_resolution")
        _set("preprocess.input_resolution_h", h)
        _set("preprocess.input_resolution_w", w)

    if "lr" in params:
        _set("train.lr", params["lr"])

    for key in (
        "lcmap_weight",
        "lcoff_weight",
        "lleng_weight",
        "angle_weight",
        "count_weight",
        "count_sigma",
    ):
        if key in params:
            _set(f"loss.{key}", params[key])

    if processed_dir_template:
        h, w = ensure_pair(params["input_resolution"], "input_resolution")
        processed_dir = processed_dir_template.format(
            heatmap_resolution=params["heatmap_resolution"],
            input_resolution=f"{h}x{w}",
            input_resolution_h=h,
            input_resolution_w=w,
        )
        _set("data.processed_dir", processed_dir)

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Bayesian optimization for F-Clip via DVC exp run.")
    parser.add_argument("--bo_config", default="config/BOparamas.yaml", help="BO search space YAML.")
    parser.add_argument("--params", default="params.yaml", help="Base params.yaml.")
    parser.add_argument("--study_name", default="bo", help="Optuna study name.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--out_dir", default="logs/bo", help="Output directory for BO logs.")
    parser.add_argument(
        "--storage",
        default="",
        help="Optuna storage URL (e.g. sqlite:///logs/bo/bo_iqi.db). "
             "If empty, a sqlite file under out_dir will be used.",
    )
    parser.add_argument("--processed_dir_template", default="", help=(
        "Optional template for data.processed_dir, e.g. "
        "IQIdata/processed_1D_{heatmap_resolution}_{input_resolution}"
    ))
    args = parser.parse_args()

    bo_cfg = load_yaml(args.bo_config) or {}
    space = bo_cfg.get("bo", {})
    if not space:
        raise ValueError(f"No 'bo' section found in {args.bo_config}")

    params_cfg = load_yaml(args.params) or {}
    logdir = params_cfg.get("train", {}).get("logdir", "logs")

    os.makedirs(args.out_dir, exist_ok=True)
    result_path = os.path.join(args.out_dir, f"{args.study_name}.jsonl")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    storage = args.storage.strip()
    if not storage:
        db_path = os.path.join(args.out_dir, f"{args.study_name}.db")
        storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial):
        sampled = {}
        for name, spec in space.items():
            sampled[name] = suggest_param(trial, name, spec)

        run_name = f"{args.study_name}_t{trial.number:04d}"
        cmd = build_dvc_cmd(
            sampled,
            run_name=run_name,
            processed_dir_template=args.processed_dir_template or None,
        )

        start = time.time()
        ret = subprocess.run(cmd, check=False)
        duration = time.time() - start

        precision = None
        metrics_path = os.path.join(logdir, run_name, "metrics.json")
        if ret.returncode == 0 and os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            precision = metrics.get("best_precision")
        else:
            metrics = {}

        record = {
            "trial": trial.number,
            "run_name": run_name,
            "params": sampled,
            "returncode": ret.returncode,
            "duration_sec": round(duration, 3),
            "metrics": metrics,
        }
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if precision is None:
            return float("-inf")
        return float(precision)

    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print(f"best_trial={best.number} value={best.value:.6f}")
    print(json.dumps(best.params, indent=2))


if __name__ == "__main__":
    main()
