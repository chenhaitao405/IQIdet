#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dump IQI/FClip intermediate trace for cross-machine comparison."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_iqi_grade_infer import parse_args  # noqa: E402
from gauge.iqi_inferencer import IQIInferencer, collect_input_images  # noqa: E402
from gauge.pipeline_utils import crop_rotated_polygon, enhance_windowing_gray, ensure_dir, order_points, rotate_if_wide  # noqa: E402
from FClip.infer_utils import (  # noqa: E402
    infer_heatmaps,
    parse_lines_1d,
    preprocess_gray_image,
    scale_lines,
)
from FClip.line_parsing import OneStageLineParsing  # noqa: E402


CRITICAL_CODE_FILES = [
    "run_iqi_grade_infer.py",
    "src/gauge/iqi_inferencer.py",
    "src/gauge/roi_stage.py",
    "src/gauge/pipeline_utils.py",
    "src/gauge/fclip_stage.py",
    "src/FClip/infer_utils.py",
    "src/FClip/line_parsing.py",
    "src/FClip/models.py",
    "scripts/debug/dump_iqi_fclip_trace.py",
]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (Path.cwd() / file_path).resolve()
    if not file_path.is_file():
        return None
    return _file_sha256(file_path)


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(str(tuple(arr.shape)).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _array_stats(array: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if array is None:
        return None
    arr = np.asarray(array)
    finite = arr.astype(np.float64, copy=False) if np.issubdtype(arr.dtype, np.number) else None
    stats: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": _array_sha256(arr),
        "contiguous": bool(arr.flags["C_CONTIGUOUS"]),
    }
    if finite is not None and arr.size:
        is_finite = np.isfinite(finite)
        stats.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "sum": float(np.sum(finite)),
                "nonzero": int(np.count_nonzero(arr)),
                "nan_count": int(np.count_nonzero(np.isnan(finite))),
                "inf_count": int(np.count_nonzero(np.isinf(finite))),
                "finite_count": int(np.count_nonzero(is_finite)),
            }
        )
    return stats


def _array_diff_stats(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    if a is None or b is None:
        return None
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if arr_a.shape != arr_b.shape:
        return {"shape_a": list(arr_a.shape), "shape_b": list(arr_b.shape), "same_shape": False}
    diff = arr_a.astype(np.float64) - arr_b.astype(np.float64)
    abs_diff = np.abs(diff)
    return {
        "same_shape": True,
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "sum_abs": float(np.sum(abs_diff)) if abs_diff.size else 0.0,
        "nonzero": int(np.count_nonzero(abs_diff)),
    }


def _safe_distribution_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_module_version(module_name: str, dist_name: Optional[str] = None) -> Optional[str]:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        if version is not None:
            return str(version)
    except Exception:
        pass
    return _safe_distribution_version(dist_name or module_name)


def _module_origin(module_name: str) -> Optional[str]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        return None
    if spec is None:
        return None
    return spec.origin


def _command_output(cmd: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=8).strip()
    except Exception:
        return None


def _pip_freeze_relevant() -> list[str]:
    output = _command_output([sys.executable, "-m", "pip", "freeze"])
    if not output:
        return []
    keywords = (
        "torch",
        "cuda",
        "cudnn",
        "nvidia",
        "opencv",
        "numpy",
        "scipy",
        "scikit",
        "skimage",
        "ultralytics",
        "paddle",
        "paddlex",
        "pillow",
        "yaml",
        "pyyaml",
        "onnx",
    )
    return [line for line in output.splitlines() if any(key in line.lower() for key in keywords)]


def _rng_state_fingerprints() -> Dict[str, Any]:
    py_state = repr(random.getstate()).encode("utf-8", errors="replace")
    np_state = repr(np.random.get_state()).encode("utf-8", errors="replace")
    payload: Dict[str, Any] = {
        "python_random_state_sha256": hashlib.sha256(py_state).hexdigest(),
        "numpy_random_state_sha256": hashlib.sha256(np_state).hexdigest(),
        "torch_initial_seed": int(torch.initial_seed()),
        "torch_cpu_rng_state_sha256": _array_sha256(torch.get_rng_state().cpu().numpy()),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_initial_seed"] = int(torch.cuda.initial_seed())
        payload["torch_cuda_rng_state_sha256"] = [
            _array_sha256(state.cpu().numpy()) for state in torch.cuda.get_rng_state_all()
        ]
    return payload


def _opencv_build_info_summary() -> Dict[str, Any]:
    try:
        info = cv2.getBuildInformation()
    except Exception as exc:
        return {"error": str(exc)}
    wanted_prefixes = (
        "General configuration for OpenCV",
        "  Version control:",
        "  Platform:",
        "  CPU/HW features:",
        "  C/C++:",
        "  OpenCV modules:",
        "  GUI:",
        "  Media I/O:",
        "  Video I/O:",
        "  Parallel framework:",
        "  Trace:",
        "  Other third-party libraries:",
        "  NVIDIA CUDA:",
        "  cuDNN:",
        "  Python 3:",
    )
    lines = [
        line.rstrip()
        for line in info.splitlines()
        if line.startswith(wanted_prefixes)
        or "NVIDIA CUDA" in line
        or "cuDNN" in line
        or "Parallel framework" in line
    ]
    return {
        "sha256": hashlib.sha256(info.encode("utf-8", errors="replace")).hexdigest(),
        "summary_lines": lines[:80],
    }


def _runtime_environment() -> Dict[str, Any]:
    package_names = [
        "torch",
        "torchvision",
        "torchaudio",
        "ultralytics",
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "numpy",
        "scipy",
        "scikit-image",
        "pandas",
        "paddlepaddle",
        "paddlepaddle-gpu",
        "paddleocr",
        "paddlex",
        "pillow",
        "pyyaml",
    ]
    env_keys = [
        "CONDA_DEFAULT_ENV",
        "CONDA_PREFIX",
        "PYTHONPATH",
        "PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_LAUNCH_BLOCKING",
        "CUDNN_DETERMINISTIC",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
    ]
    cuda_info: Dict[str, Any] = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "torch_cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
        "torch_tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cuda_info["devices"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                    "total_memory": int(props.total_memory),
                    "multi_processor_count": int(props.multi_processor_count),
                }
            )

    return {
        "platform": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "uname": list(platform.uname()),
        },
        "packages": {name: _safe_distribution_version(name) for name in package_names},
        "module_versions": {
            "cv2": cv2.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ultralytics": _safe_distribution_version("ultralytics"),
            "paddle": _safe_distribution_version("paddlepaddle-gpu") or _safe_distribution_version("paddlepaddle"),
            "paddleocr": _safe_distribution_version("paddleocr"),
            "skimage": _safe_distribution_version("scikit-image"),
        },
        "module_origins": {
            name: _module_origin(name)
            for name in [
                "gauge",
                "gauge.iqi_inferencer",
                "gauge.pipeline_utils",
                "gauge.roi_stage",
                "gauge.fclip_stage",
                "FClip",
                "FClip.infer_utils",
                "FClip.line_parsing",
                "ultralytics",
                "cv2",
                "numpy",
                "torch",
            ]
        },
        "cuda": cuda_info,
        "opencv_runtime": {
            "num_threads": int(cv2.getNumThreads()),
            "opencl_available": bool(cv2.ocl.haveOpenCL()),
            "opencl_enabled": bool(cv2.ocl.useOpenCL()),
        },
        "commands": {
            "nvidia_smi_query": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,cuda_version,pci.bus_id,uuid",
                    "--format=csv,noheader",
                ]
            ),
            "nvcc_version": _command_output(["nvcc", "--version"]),
        },
        "env": {key: os.environ.get(key) for key in env_keys},
        "pip_freeze_relevant": _pip_freeze_relevant(),
        "rng_state_fingerprints_after_run": _rng_state_fingerprints(),
        "opencv_build": _opencv_build_info_summary(),
    }


def _asset_hashes(args: Any, inferencer: IQIInferencer) -> Dict[str, Any]:
    paths = {
        "gauge_weights": args.gauge_weights,
        "fclip_ckpt": args.fclip_ckpt,
        "fclip_config": args.fclip_config,
        "fclip_params": args.fclip_params,
        "ocr_orientation_model": args.ocr_orientation_model if args.enable_ocr_orientation else None,
        "correction_model": args.correction_model if args.enable_correction else None,
        "ocr_rec_model_dir": args.ocr_rec_model_dir,
        "ocr_det_model_dir": args.ocr_det_model_dir,
        "runtime_gauge_weights": inferencer.gauge_weights,
        "runtime_fclip_ckpt": inferencer.fclip_ckpt,
        "runtime_fclip_model_config": inferencer.fclip_model_config,
        "runtime_fclip_params": inferencer.fclip_params,
    }
    payload: Dict[str, Any] = {}
    for key, raw_path in paths.items():
        if not raw_path:
            payload[key] = None
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        item: Dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
        }
        if path.is_file():
            item["size"] = int(path.stat().st_size)
            item["sha256"] = _file_sha256(path)
        elif path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
            item["num_files"] = len(files)
            item["files"] = [
                {
                    "path": str(p.relative_to(path)),
                    "size": int(p.stat().st_size),
                    "sha256": _file_sha256(p),
                }
                for p in files[:200]
            ]
            digest = hashlib.sha256()
            for row in item["files"]:
                digest.update(row["path"].encode("utf-8", errors="replace"))
                digest.update(str(row["size"]).encode("ascii"))
                digest.update(str(row["sha256"]).encode("ascii"))
            item["tree_sha256_first_200_files"] = digest.hexdigest()
        payload[key] = item
    return payload


def _code_hashes() -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for rel_path in CRITICAL_CODE_FILES:
        path = REPO_ROOT / rel_path
        item: Dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
        }
        if path.is_file():
            item["size"] = int(path.stat().st_size)
            item["sha256"] = _file_sha256(path)
        payload[rel_path] = item
    return payload


def _args_snapshot(args: Any) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
    }


def _model_summaries(inferencer: IQIInferencer) -> Dict[str, Any]:
    gauge_model = getattr(inferencer, "gauge_model", None)
    fclip = getattr(inferencer, "fclip_inferencer", None)
    fclip_model = getattr(fclip, "model", None) if fclip is not None else None
    payload: Dict[str, Any] = {
        "gauge_model_class": None if gauge_model is None else f"{gauge_model.__class__.__module__}.{gauge_model.__class__.__name__}",
        "gauge_model_task": getattr(gauge_model, "task", None) if gauge_model is not None else None,
        "gauge_model_names": getattr(gauge_model, "names", None) if gauge_model is not None else None,
        "fclip_inferencer_class": None if fclip is None else f"{fclip.__class__.__module__}.{fclip.__class__.__name__}",
        "fclip_model_class": None if fclip_model is None else f"{fclip_model.__class__.__module__}.{fclip_model.__class__.__name__}",
    }
    if fclip_model is not None:
        try:
            params = list(fclip_model.parameters())
            payload["fclip_num_parameters"] = int(sum(p.numel() for p in params))
            payload["fclip_trainable_parameters"] = int(sum(p.numel() for p in params if p.requires_grad))
            payload["fclip_first_param"] = _array_stats(params[0].detach().cpu().numpy()) if params else None
            payload["fclip_last_param"] = _array_stats(params[-1].detach().cpu().numpy()) if params else None
        except Exception as exc:
            payload["fclip_param_summary_error"] = str(exc)
    return payload


def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_status_short() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line for line in output.splitlines() if line.strip()]


def _bbox_from_polygon(poly: np.ndarray) -> list[float]:
    return [
        float(np.min(poly[:, 0])),
        float(np.min(poly[:, 1])),
        float(np.max(poly[:, 0])),
        float(np.max(poly[:, 1])),
    ]


def _round_boundary_info(value: float) -> Dict[str, Any]:
    rounded = int(round(float(value)))
    return {
        "value": float(value),
        "round": rounded,
        "floor": int(np.floor(value)),
        "ceil": int(np.ceil(value)),
        "distance_to_round_value": float(value - rounded),
        "distance_to_lower_half_boundary": float(value - (rounded - 0.5)),
        "distance_to_upper_half_boundary": float((rounded + 0.5) - value),
    }


def _roi_crop_diagnostics(roi_info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not roi_info or not roi_info.get("polygon"):
        return None
    polygon = np.asarray(roi_info["polygon"], dtype=np.float32).reshape(-1, 2)
    if polygon.shape[0] != 4:
        return {"error": f"expected 4 polygon points, got {int(polygon.shape[0])}"}

    box = order_points(polygon)
    w1 = float(np.linalg.norm(box[0] - box[1]))
    w2 = float(np.linalg.norm(box[2] - box[3]))
    h1 = float(np.linalg.norm(box[0] - box[3]))
    h2 = float(np.linalg.norm(box[1] - box[2]))
    width_float = max(w1, w2)
    height_float = max(h1, h2)
    width = int(round(width_float))
    height = int(round(height_float))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(box, dst)
    return {
        "ordered_polygon": box.tolist(),
        "edge_lengths": {
            "w1_top": w1,
            "w2_bottom": w2,
            "h1_left": h1,
            "h2_right": h2,
        },
        "width": _round_boundary_info(width_float),
        "height": _round_boundary_info(height_float),
        "crop_size": [width, height],
        "crop_matrix": matrix.tolist(),
        "crop_matrix_sha256": _array_sha256(matrix),
    }


def _replay_roi_preprocess(
    image: Optional[np.ndarray],
    roi_info: Optional[Dict[str, Any]],
    enable_rotation: bool,
    actual_roi_cropped: Optional[np.ndarray],
    actual_roi_image: Optional[np.ndarray],
    actual_roi_gray: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    if image is None or not roi_info or not roi_info.get("polygon"):
        return None
    polygon = np.asarray(roi_info["polygon"], dtype=np.float32).reshape(-1, 2)
    replay_crop, replay_matrix = crop_rotated_polygon(image, polygon)
    if replay_crop is None or replay_matrix is None:
        return {"status": "failed"}
    replay_image, rotated, rotation = rotate_if_wide(replay_crop, enable=enable_rotation)
    replay_gray = enhance_windowing_gray(replay_image)
    return {
        "status": "ok",
        "crop_matrix": replay_matrix.tolist(),
        "crop_matrix_sha256": _array_sha256(replay_matrix),
        "roi_cropped": _array_stats(replay_crop),
        "roi_image": _array_stats(replay_image),
        "roi_gray": _array_stats(replay_gray),
        "rotated": bool(rotated),
        "rotation": int(rotation),
        "diff_vs_actual_roi_cropped": _array_diff_stats(replay_crop, actual_roi_cropped),
        "diff_vs_actual_roi_image": _array_diff_stats(replay_image, actual_roi_image),
        "diff_vs_actual_roi_gray": _array_diff_stats(replay_gray, actual_roi_gray),
    }


def _extract_gauge_candidates(result: Any) -> list[Dict[str, Any]]:
    obb = getattr(result, "obb", None)
    if obb is None:
        return []
    polys = getattr(obb, "xyxyxyxy", None)
    if polys is None:
        return []
    polys_np = polys.detach().cpu().numpy()
    if polys_np.ndim == 2 and polys_np.shape[1] == 8:
        polys_np = polys_np.reshape(-1, 4, 2)

    confs = getattr(obb, "conf", None)
    confs_np = confs.detach().cpu().numpy() if confs is not None else None
    classes = getattr(obb, "cls", None)
    classes_np = classes.detach().cpu().numpy().astype(int) if classes is not None else None

    candidates: list[Dict[str, Any]] = []
    for idx, poly in enumerate(polys_np):
        poly = poly.astype(np.float32)
        candidates.append(
            {
                "index": int(idx),
                "polygon": poly.astype(float).tolist(),
                "bbox": _bbox_from_polygon(poly),
                "area": float(cv2.contourArea(poly)),
                "conf": None if confs_np is None else float(confs_np[idx]),
                "class_id": None if classes_np is None else int(classes_np[idx]),
            }
        )
    return candidates


def _select_gauge_candidate(
    candidates: list[Dict[str, Any]],
    select: str,
    class_filter: Optional[int],
) -> Optional[Dict[str, Any]]:
    filtered = [
        item
        for item in candidates
        if class_filter is None or item.get("class_id") == class_filter
    ]
    if not filtered:
        return None
    if select == "conf" and any(item.get("conf") is not None for item in filtered):
        return max(filtered, key=lambda item: float(item.get("conf") or float("-inf")))
    return max(filtered, key=lambda item: float(item.get("area") or 0.0))


def _dump_yolo_result(
    result: Any,
    resize_scale: float,
    note: str,
    select: Optional[str] = None,
    class_filter: Optional[int] = None,
) -> Dict[str, Any]:
    candidates = _extract_gauge_candidates(result) if result is not None else []
    selected_by_conf: Optional[Dict[str, Any]] = None
    selected_by_area: Optional[Dict[str, Any]] = None
    selected: Optional[Dict[str, Any]] = None
    if candidates:
        if any(item.get("conf") is not None for item in candidates):
            selected_by_conf = max(candidates, key=lambda item: float(item.get("conf") or float("-inf")))
        selected_by_area = max(candidates, key=lambda item: float(item.get("area") or 0.0))
        if select is not None:
            selected = _select_gauge_candidate(candidates, select=select, class_filter=class_filter)

    obb = getattr(result, "obb", None) if result is not None else None
    boxes = getattr(result, "boxes", None) if result is not None else None
    payload: Dict[str, Any] = {
        "note": note,
        "num_candidates": len(candidates),
        "candidates_resized": candidates,
        "candidates_original": [_scale_candidate_to_original(item, resize_scale) for item in candidates],
        "selected_by_conf_resized": selected_by_conf,
        "selected_by_conf_original": None
        if selected_by_conf is None
        else _scale_candidate_to_original(selected_by_conf, resize_scale),
        "selected_by_area_resized": selected_by_area,
        "selected_by_area_original": None
        if selected_by_area is None
        else _scale_candidate_to_original(selected_by_area, resize_scale),
        "select": select,
        "class_filter": class_filter,
        "selected_resized": selected,
        "selected_original": None
        if selected is None
        else _scale_candidate_to_original(selected, resize_scale),
    }
    if obb is not None:
        for attr in ["xyxyxyxy", "conf", "cls", "xywhr", "xyxy"]:
            value = getattr(obb, attr, None)
            if value is not None:
                payload[f"obb_{attr}_stats"] = _array_stats(value.detach().cpu().numpy())
    if boxes is not None:
        for attr in ["xyxy", "conf", "cls", "xywh"]:
            value = getattr(boxes, attr, None)
            if value is not None:
                payload[f"boxes_{attr}_stats"] = _array_stats(value.detach().cpu().numpy())
    return payload


def _scale_candidate_to_original(candidate: Dict[str, Any], resize_scale: float) -> Dict[str, Any]:
    if not resize_scale or resize_scale == 1.0:
        return dict(candidate)
    scaled = dict(candidate)
    poly = np.asarray(candidate["polygon"], dtype=np.float32).reshape(-1, 2) / float(resize_scale)
    scaled["polygon"] = poly.astype(float).tolist()
    scaled["bbox"] = _bbox_from_polygon(poly)
    scaled["area"] = float(cv2.contourArea(poly.astype(np.float32)))
    return scaled


def _dump_gauge_second_pass(
    inferencer: IQIInferencer,
    sampled_image: Optional[np.ndarray],
    resize_scale: float,
) -> Optional[Dict[str, Any]]:
    if sampled_image is None:
        return None
    yolo_result = inferencer.gauge_model.predict(
        source=sampled_image,
        conf=inferencer.gauge_conf,
        iou=inferencer.gauge_iou,
        imgsz=inferencer.gauge_imgsz,
        device=inferencer.gauge_device,
        verbose=False,
    )
    if not yolo_result:
        return {"note": "second gauge pass from trace script", "num_candidates": 0, "candidates_resized": []}

    candidates = _extract_gauge_candidates(yolo_result[0])
    selected = _select_gauge_candidate(
        candidates,
        select=inferencer.gauge_select,
        class_filter=inferencer.gauge_class,
    )

    payload = _dump_yolo_result(
        yolo_result[0],
        resize_scale,
        note="second gauge pass from trace script; compare with gauge_first_pass to check gauge repeatability",
        select=inferencer.gauge_select,
        class_filter=inferencer.gauge_class,
    )
    payload.update(
        {
            "sampled_image_shape": list(sampled_image.shape),
            "resize_scale": float(resize_scale),
            "selected_resized": selected,
            "selected_original": None
            if selected is None
            else _scale_candidate_to_original(selected, float(resize_scale)),
        }
    )
    return payload


def _topk_scores(scores: torch.Tensor, limit: int) -> list[dict[str, Any]]:
    k = min(int(limit), int(scores.numel()))
    if k <= 0:
        return []
    vals, idx = torch.topk(scores, k=k)
    return [
        {"rank": int(i), "index": int(index.item()), "score": float(value.item())}
        for i, (value, index) in enumerate(zip(vals, idx))
    ]


def _dump_fclip_trace(fclip: Any, roi_gray: np.ndarray) -> Dict[str, Any]:
    image_tensor = preprocess_gray_image(
        roi_gray,
        input_resolution=fclip.input_resolution,
        mean=fclip.mean,
        std=fclip.std,
        device=fclip.device,
    )
    heatmaps = infer_heatmaps(fclip.model, image_tensor)
    count_logits_t = heatmaps["count"][0].detach().cpu().float()
    count_softmax_t = torch.softmax(count_logits_t, dim=0)
    count_pred = int(torch.argmax(count_logits_t).item())

    lcmap = heatmaps["lcmap"][0]
    lcoff = heatmaps["lcoff"][0]
    angle = heatmaps["angle"][0]
    nms_scores = OneStageLineParsing._nms_1d(
        lcmap.reshape(-1),
        delta=fclip.threshold,
        kernel=3,
    ).detach()

    lines_t, scores_t = parse_lines_1d(
        lcmap=lcmap,
        lcoff=lcoff,
        angle=angle,
        threshold=fclip.threshold,
        nlines=fclip.nlines,
        resolution=fclip.resolution,
        ang_type=fclip.ang_type,
        count_pred=count_pred,
    )
    lines_scaled = scale_lines(lines_t.clone(), fclip.resolution, roi_gray.shape)

    return {
        "device": str(fclip.device),
        "threshold": float(fclip.threshold),
        "nlines": int(fclip.nlines),
        "resolution": int(fclip.resolution),
        "input_resolution": list(fclip.input_resolution),
        "mean": float(fclip.mean),
        "std": float(fclip.std),
        "input_tensor": _array_stats(image_tensor.detach().cpu().numpy()),
        "input_tensor_shape": list(image_tensor.shape),
        "input_tensor_sha256": _array_sha256(image_tensor.detach().cpu().numpy()),
        "count_logits": count_logits_t.tolist(),
        "count_softmax": count_softmax_t.tolist(),
        "count_argmax": count_pred,
        "count_logits_stats": _array_stats(count_logits_t.numpy()),
        "count_margin_top1_top2": float(
            torch.topk(count_logits_t, k=min(2, int(count_logits_t.numel()))).values[0]
            - torch.topk(count_logits_t, k=min(2, int(count_logits_t.numel()))).values[-1]
        )
        if int(count_logits_t.numel()) >= 2
        else None,
        "lcmap_shape": list(lcmap.shape),
        "lcmap_sha256": _array_sha256(lcmap.detach().cpu().numpy()),
        "lcoff_sha256": _array_sha256(lcoff.detach().cpu().numpy()),
        "angle_sha256": _array_sha256(angle.detach().cpu().numpy()),
        "lcmap": _array_stats(lcmap.detach().cpu().numpy()),
        "lcoff": _array_stats(lcoff.detach().cpu().numpy()),
        "angle": _array_stats(angle.detach().cpu().numpy()),
        "nms_scores": _array_stats(nms_scores.detach().cpu().numpy()),
        "nms_top_scores": _topk_scores(nms_scores, fclip.nlines),
        "parsed_scores": scores_t.detach().cpu().tolist(),
        "parsed_lines_roi_yx": lines_t.detach().cpu().tolist(),
        "parsed_lines_scaled_yx": lines_scaled.detach().cpu().tolist(),
    }


def main() -> None:
    args = parse_args()
    image_paths = collect_input_images(
        image_path=args.image_path,
        image_dir=args.image_dir,
        image_list=args.image_list,
        max_images=args.max_images,
    )
    if len(image_paths) != 1:
        raise SystemExit(f"dump_iqi_fclip_trace expects exactly one image, got {len(image_paths)}")

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = (Path.cwd() / output_json).resolve()
    ensure_dir(output_json.parent)

    image_path = image_paths[0]
    inferencer = IQIInferencer(
        gauge_weights=args.gauge_weights,
        fclip_ckpt=args.fclip_ckpt,
        gauge_conf=args.gauge_conf,
        gauge_iou=args.gauge_iou,
        gauge_imgsz=args.gauge_imgsz,
        gauge_device=args.gauge_device,
        gauge_select=args.gauge_select,
        gauge_class=args.gauge_class,
        enhance_mode=args.enhance_mode,
        rotate_roi=not args.no_rotate,
        enable_correction=args.enable_correction,
        correction_model=args.correction_model,
        correction_device=args.correction_device,
        correction_verbose=args.correction_verbose,
        ocr_device=args.ocr_device,
        ocr_det_model_name=args.ocr_det_model_name,
        ocr_det_model_dir=args.ocr_det_model_dir,
        ocr_rec_model_name=args.ocr_rec_model_name,
        ocr_rec_model_dir=args.ocr_rec_model_dir,
        ocr_det_limit_side_len=args.ocr_det_limit_side_len,
        ocr_det_limit_type=args.ocr_det_limit_type,
        ocr_min_score=args.ocr_min_score,
        ocr_number_range=args.ocr_number_range,
        enable_ocr_orientation=args.enable_ocr_orientation,
        ocr_orientation_model=args.ocr_orientation_model,
        ocr_orientation_device=args.ocr_orientation_device,
        fclip_device=args.fclip_device,
        fclip_model_config=args.fclip_config,
        fclip_params=args.fclip_params,
        fclip_threshold=args.fclip_threshold,
    )

    try:
        record, artifacts = inferencer.infer_image_path(
            image_path,
            return_debug_artifacts=True,
            debug_timer=True,
        )
        roi_gray = None if artifacts is None else artifacts.get("roi_gray")
        sampled_image = None if artifacts is None else artifacts.get("sampled_image")
        original_image = None if artifacts is None else artifacts.get("image")
        full_ocr_input = None if artifacts is None else artifacts.get("full_ocr_input")
        roi_cropped = None if artifacts is None else artifacts.get("roi_cropped")
        roi_image = None if artifacts is None else artifacts.get("roi_image")
        roi_crop_matrix = None if artifacts is None else artifacts.get("roi_crop_matrix")
        gauge_yolo_result = None if artifacts is None else artifacts.get("gauge_yolo_result")
        fclip_trace = None
        if roi_gray is not None and inferencer.fclip_inferencer is not None:
            fclip_trace = _dump_fclip_trace(inferencer.fclip_inferencer, roi_gray)
        resize_scale = float((record.get("full_image_preprocess") or {}).get("resize_scale") or 1.0)
        gauge_first_pass = _dump_yolo_result(
            gauge_yolo_result,
            resize_scale,
            note="first gauge pass used by IQIInferencer",
            select=inferencer.gauge_select,
            class_filter=inferencer.gauge_class,
        )
        gauge_second_pass = _dump_gauge_second_pass(inferencer, sampled_image, resize_scale)
        git_status = _git_status_short()
        roi_replay = _replay_roi_preprocess(
            original_image,
            record.get("roi"),
            enable_rotation=not args.no_rotate,
            actual_roi_cropped=roi_cropped,
            actual_roi_image=roi_image,
            actual_roi_gray=roi_gray,
        )

        payload = {
            "schema": "iqi_fclip_trace_v1",
            "schema_revision": 2,
            "git_head": _git_head(),
            "git_dirty": bool(git_status),
            "git_status_short": git_status,
            "image_path": str(image_path),
            "image_sha256": _file_sha256(image_path),
            "argv": sys.argv,
            "args": _args_snapshot(args),
            "code_hashes": _code_hashes(),
            "asset_hashes": _asset_hashes(args, inferencer),
            "runtime_environment": _runtime_environment(),
            "model_summaries": _model_summaries(inferencer),
            "versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cv2": cv2.__version__,
                "numpy": np.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            "record_core": {
                "ok": record.get("ok"),
                "result_code": record.get("result_code"),
                "result_name": record.get("result_name"),
                "grade": record.get("grade"),
                "plate_code": record.get("plate_code"),
                "plate_number": record.get("plate_number"),
                "wire_count": record.get("wire_count"),
                "warnings": record.get("warnings"),
                "errors": record.get("errors"),
            },
            "runtime_meta": inferencer.get_runtime_meta(),
            "artifact_stats": {
                "image": _array_stats(original_image),
                "sampled_image": _array_stats(sampled_image),
                "full_ocr_input": _array_stats(full_ocr_input),
                "roi_cropped": _array_stats(roi_cropped),
                "roi_crop_matrix": _array_stats(roi_crop_matrix),
                "roi_image": _array_stats(roi_image),
                "roi_gray": _array_stats(roi_gray),
            },
            "full_image_preprocess": record.get("full_image_preprocess"),
            "full_image_ocr_summary": {
                "status": (record.get("full_image_ocr") or {}).get("status"),
                "det_box_count": (record.get("full_image_ocr") or {}).get("det_box_count"),
                "rec_item_count": (record.get("full_image_ocr") or {}).get("rec_item_count"),
                "texts": (record.get("full_image_ocr") or {}).get("texts"),
                "scores": (record.get("full_image_ocr") or {}).get("scores"),
                "items_original": (record.get("full_image_ocr") or {}).get("items_original"),
                "timings_ms": (record.get("full_image_ocr") or {}).get("timings_ms"),
            },
            "full_image_plate": record.get("full_image_plate"),
            "roi": record.get("roi"),
            "roi_crop_diagnostics": _roi_crop_diagnostics(record.get("roi")),
            "roi_replay": roi_replay,
            "gauge_first_pass": gauge_first_pass,
            "gauge_second_pass": gauge_second_pass,
            "preprocess": record.get("preprocess"),
            "roi_ocr_summary": {
                "status": (record.get("roi_ocr") or {}).get("status"),
                "det_box_count": (record.get("roi_ocr") or {}).get("det_box_count"),
                "rec_item_count": (record.get("roi_ocr") or {}).get("rec_item_count"),
                "texts": (record.get("roi_ocr") or {}).get("texts"),
                "scores": (record.get("roi_ocr") or {}).get("scores"),
                "items": (record.get("roi_ocr") or {}).get("items"),
                "items_image": (record.get("roi_ocr") or {}).get("items_image"),
                "timings_ms": (record.get("roi_ocr") or {}).get("timings_ms"),
            },
            "roi_plate": record.get("roi_plate"),
            "selected_plate": {
                "plate_source": record.get("plate_source"),
                "iqi_type": record.get("iqi_type"),
                "plate_code": record.get("plate_code"),
                "plate_number": record.get("plate_number"),
            },
            "roi_gray": None
            if roi_gray is None
            else {
                "shape": list(roi_gray.shape),
                "dtype": str(roi_gray.dtype),
                "sha256": _array_sha256(roi_gray),
            },
            "wire": record.get("wire"),
            "visualization_wire_lines": (record.get("visualization") or {}).get("wire_lines"),
            "fclip_trace": fclip_trace,
        }
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)
        print(f"trace json: {output_json}")
    finally:
        inferencer.close()


if __name__ == "__main__":
    main()