#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Gauge pipeline stage 1: inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from gauge.pipeline_utils import (
    SUPPORTED_IMAGE_EXTS,
    collect_images,
    crop_rotated_polygon,
    enhance_windowing_gray,
    ensure_dir,
    format_polygon,
    load_image,
    rotate_if_wide,
    safe_list,
    to_gray,
)

from FClip.infer_utils import (
    build_infer_model,
    draw_count,
    draw_lines,
    get_count_pred,
    infer_heatmaps,
    load_config_from_yaml,
    parse_lines_1d,
    preprocess_gray_image,
    scale_lines,
)

try:  # pragma: no cover
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(items, **_kwargs):
        return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gauge inference pipeline (stage 1).")
    parser.add_argument("--image-dir", help="Batch image directory to process.")
    parser.add_argument("--image-list", help="Optional text file with image paths (one per line).")
    parser.add_argument("--output-dir", default="outputs/gauge_infer", help="Output root directory.")
    parser.add_argument("--results-json", default="inference_results.json", help="JSON filename.")
    parser.add_argument("--max-images", type=int, help="Max images to process.")
    parser.add_argument("--vis", action="store_true", help="Save visualizations to output_dir/vis.")

    parser.add_argument("--gauge-weights", required=True, help="OBB gauge detector weights.")
    parser.add_argument("--gauge-conf", type=float, default=0.25, help="OBB confidence threshold.")
    parser.add_argument("--gauge-iou", type=float, default=0.45, help="OBB IoU threshold.")
    parser.add_argument("--gauge-imgsz", type=int, default=640, help="OBB inference image size.")
    parser.add_argument("--gauge-device", default=None, help="OBB device, e.g. 0/cuda:0/cpu.")
    parser.add_argument("--gauge-select", choices=["conf", "area"], default="conf",
                        help="How to pick the ROI when multiple detections exist.")
    parser.add_argument("--gauge-class", type=int, help="Optional class id filter.")

    parser.add_argument("--fclip-config", default="config/model.yaml", help="F-Clip model config.")
    parser.add_argument("--fclip-ckpt", required=True, help="F-Clip checkpoint path.")
    parser.add_argument("--fclip-params", default="params.yaml", help="F-Clip params yaml.")
    parser.add_argument("--fclip-device", default=None, help="F-Clip device, e.g. 0/cuda:0/cpu.")
    parser.add_argument("--threshold", type=float, default=0.4, help="Line parsing threshold.")

    parser.add_argument("--enhance-mode", choices=["original", "windowing"], default="windowing",
                        help="Image enhancement mode before F-Clip.")
    parser.add_argument("--no-rotate", action="store_true",
                        help="Disable width>height -> CCW90 rotation.")
    return parser.parse_args()


def resolve_torch_device(device_str: Optional[str]) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")
    return torch.device(device_str)


def resolve_yolo_device(device_str: Optional[str]) -> Optional[str]:
    if device_str is None:
        return None
    return device_str


def extract_best_obb(result, select: str, class_filter: Optional[int]) -> Optional[Dict[str, Any]]:
    obb = getattr(result, "obb", None)
    if obb is None:
        return None

    polys = getattr(obb, "xyxyxyxy", None)
    if polys is None:
        return None
    polys = polys.detach().cpu().numpy()
    if polys.ndim == 2 and polys.shape[1] == 8:
        polys = polys.reshape(-1, 4, 2)

    confs = getattr(obb, "conf", None)
    confs_np = confs.detach().cpu().numpy() if confs is not None else None
    classes = getattr(obb, "cls", None)
    classes_np = classes.detach().cpu().numpy().astype(int) if classes is not None else None

    candidates: List[Tuple[int, float]] = []
    for idx, poly in enumerate(polys):
        if class_filter is not None and classes_np is not None:
            if classes_np[idx] != class_filter:
                continue
        area = float(cv2.contourArea(poly.astype(np.float32)))
        candidates.append((idx, area))

    if not candidates:
        return None

    if select == "conf" and confs_np is not None:
        idx = int(max(candidates, key=lambda item: confs_np[item[0]])[0])
    else:
        idx = int(max(candidates, key=lambda item: item[1])[0])

    poly = polys[idx].astype(float)
    conf = float(confs_np[idx]) if confs_np is not None else None
    cls_id = int(classes_np[idx]) if classes_np is not None else None
    x_min = float(np.min(poly[:, 0]))
    y_min = float(np.min(poly[:, 1]))
    x_max = float(np.max(poly[:, 0]))
    y_max = float(np.max(poly[:, 1]))
    return {
        "polygon": format_polygon(poly),
        "bbox": [x_min, y_min, x_max, y_max],
        "conf": conf,
        "class_id": cls_id,
    }


def prepare_fclip_model(config_path: str, params_path: str, ckpt_path: str, device: torch.device):
    load_config_from_yaml(config_path, params_yaml=params_path, ckpt=ckpt_path)
    model = build_infer_model(device)
    return model


def infer_fclip_lines(
    model,
    image_gray: np.ndarray,
    device: torch.device,
    threshold: float,
) -> Tuple[int, List[List[List[float]]], List[float]]:
    from FClip.config import M

    input_h = getattr(M, "input_resolution_h", M.resolution * 4)
    input_w = getattr(M, "input_resolution_w", M.resolution * 4)
    mean = float(M.image.mean[0])
    std = float(M.image.stddev[0])

    tensor = preprocess_gray_image(image_gray, (input_w, input_h), mean, std, device)
    heatmaps = infer_heatmaps(model, tensor)
    count_pred = get_count_pred(heatmaps)
    if count_pred is None:
        raise ValueError("F-Clip outputs missing count head.")
    count_value = int(count_pred[0].item())

    lcmap = heatmaps["lcmap"][0]
    lcoff = heatmaps["lcoff"][0]
    angle = heatmaps["angle"][0]
    lines, scores = parse_lines_1d(
        lcmap=lcmap,
        lcoff=lcoff,
        angle=angle,
        threshold=threshold,
        nlines=M.nlines,
        resolution=M.resolution,
        ang_type=M.ang_type,
        count_pred=count_value,
    )
    lines = scale_lines(lines, M.resolution, image_gray.shape)
    lines_np = lines.detach().cpu().numpy() if hasattr(lines, "detach") else lines
    scores_np = scores.detach().cpu().numpy() if hasattr(scores, "detach") else scores

    lines_xy: List[List[List[float]]] = []
    for line in lines_np:
        y1, x1 = float(line[0][0]), float(line[0][1])
        y2, x2 = float(line[1][0]), float(line[1][1])
        lines_xy.append([[x1, y1], [x2, y2]])

    score_list = [float(s) for s in scores_np] if scores_np is not None else []
    return count_value, lines_xy, score_list


def build_vis_image(image_gray: np.ndarray, lines_rc: np.ndarray, count_value: int) -> np.ndarray:
    if image_gray.ndim == 2:
        vis = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        vis = image_gray.copy()
    draw_lines(vis, lines_rc)
    draw_count(vis, count_value)
    return vis


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    vis_dir = output_dir / "vis"
    if args.vis:
        ensure_dir(vis_dir)

    image_dir = Path(args.image_dir) if args.image_dir else None
    image_list = Path(args.image_list) if args.image_list else None
    image_paths = collect_images(image_dir, image_list, args.max_images)

    gauge_model = YOLO(args.gauge_weights)
    yolo_device = resolve_yolo_device(args.gauge_device)
    fclip_device = resolve_torch_device(args.fclip_device)
    fclip_model = prepare_fclip_model(args.fclip_config, args.fclip_params, args.fclip_ckpt, fclip_device)

    results: List[Dict[str, Any]] = []
    image_root = image_dir.resolve() if image_dir is not None else None

    for image_path in tqdm(image_paths, desc="Gauge inference"):
        record: Dict[str, Any] = {
            "image_path": str(image_path),
            "status": "ok",
        }
        try:
            image = load_image(image_path)
            h, w = image.shape[:2]
            record["width"] = w
            record["height"] = h

            yolo_result = gauge_model.predict(
                source=image,
                conf=args.gauge_conf,
                iou=args.gauge_iou,
                imgsz=args.gauge_imgsz,
                device=yolo_device,
                verbose=False,
            )
            if not yolo_result:
                record["status"] = "no_roi"
                results.append(record)
                continue

            roi_info = extract_best_obb(
                yolo_result[0],
                select=args.gauge_select,
                class_filter=args.gauge_class,
            )
            if roi_info is None:
                record["status"] = "no_roi"
                results.append(record)
                continue

            polygon = np.array(roi_info["polygon"], dtype=np.float32)
            roi_image, _ = crop_rotated_polygon(image, polygon)
            if roi_image is None:
                record["status"] = "roi_invalid"
                record["gauge"] = roi_info
                results.append(record)
                continue

            roi_image, rotated, rotation = rotate_if_wide(roi_image, enable=not args.no_rotate)
            if args.enhance_mode == "windowing":
                roi_gray = enhance_windowing_gray(roi_image)
            else:
                roi_gray = to_gray(roi_image)

            count_value, lines_xy, scores = infer_fclip_lines(
                fclip_model, roi_gray, fclip_device, args.threshold
            )

            record["gauge"] = roi_info
            record["preprocess"] = {
                "rotation": rotation,
                "rotated": rotated,
                "enhance_mode": args.enhance_mode,
                "roi_size": [int(roi_image.shape[1]), int(roi_image.shape[0])],
            }
            record["pred"] = {
                "count": count_value,
                "lines": lines_xy,
                "scores": scores,
                "line_format": "xy",
            }

            if args.vis:
                from FClip.config import M

                lines_rc = []
                for line in lines_xy:
                    (x1, y1), (x2, y2) = line
                    lines_rc.append([[y1, x1], [y2, x2]])
                lines_rc = np.array(lines_rc, dtype=np.float32)
                vis_img = build_vis_image(roi_gray, lines_rc, count_value)
                if image_root is None:
                    rel_name = image_path.name
                else:
                    try:
                        rel_name = str(image_path.relative_to(image_root))
                    except ValueError:
                        rel_name = image_path.name
                vis_path = (vis_dir / rel_name).with_suffix(".png")
                ensure_dir(vis_path.parent)
                cv2.imwrite(str(vis_path), vis_img)
                record["vis_path"] = str(vis_path.relative_to(output_dir))

        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        results.append(record)

    results_path = Path(args.results_json)
    if not results_path.is_absolute():
        results_path = output_dir / results_path
    ensure_dir(results_path.parent)

    meta = {
        "schema": "gauge_inference_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_root": str(image_root) if image_root is not None else None,
        "supported_exts": list(SUPPORTED_IMAGE_EXTS),
        "gauge_weights": args.gauge_weights,
        "gauge_conf": args.gauge_conf,
        "gauge_iou": args.gauge_iou,
        "gauge_imgsz": args.gauge_imgsz,
        "gauge_select": args.gauge_select,
        "gauge_class": args.gauge_class,
        "fclip_config": args.fclip_config,
        "fclip_ckpt": args.fclip_ckpt,
        "fclip_params": args.fclip_params,
        "threshold": args.threshold,
        "enhance_mode": args.enhance_mode,
        "rotation_rule": "ccw90_if_width_gt_height" if not args.no_rotate else "disabled",
    }
    payload = {"meta": meta, "results": results}
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n推理完成，共处理 {len(results)} 张图像。")
    print(f"结果JSON: {results_path}")


if __name__ == "__main__":
    main()
