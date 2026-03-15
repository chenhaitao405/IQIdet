#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Gauge pipeline stage 1 (OCR debug stage)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from gauge.ocr_stage import (
    build_ocr_statistics,
    create_paddle_ocr,
    draw_ocr_on_roi,
    infer_roi_ocr,
)
from gauge.roi_stage import build_roi_vis_image, extract_best_obb
from gauge.pipeline_utils import (
    SUPPORTED_IMAGE_EXTS,
    collect_images,
    crop_rotated_polygon,
    enhance_windowing_gray,
    ensure_dir,
    load_image,
    rotate_if_wide,
    to_gray,
)

try:  # pragma: no cover
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(items, **_kwargs):
        return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gauge inference pipeline (OCR-only stage).")
    parser.add_argument("--image-dir", help="Batch image directory to process.")
    parser.add_argument("--image-list", help="Optional text file with image paths (one per line).")
    parser.add_argument("--output-dir", default="outputs/gauge_ocr", help="Output root directory.")
    parser.add_argument("--results-json", default="ocr_results.json", help="Per-image JSON filename.")
    parser.add_argument("--ocr-stats-json", default="ocr_stats.json", help="OCR stats JSON filename.")
    parser.add_argument("--max-images", type=int, help="Max images to process.")
    parser.add_argument("--vis", action="store_true", help="Save visualizations to output_dir/vis.")

    parser.add_argument("--gauge-weights", required=True, help="OBB gauge detector weights.")
    parser.add_argument(
        "--fclip-ckpt",
        help="Optional FClip checkpoint. Accepted for compatibility but ignored in OCR-only stage.",
    )
    parser.add_argument(
        "--fclip-device",
        default=None,
        help="Optional FClip device. Accepted for compatibility but ignored in OCR-only stage.",
    )
    parser.add_argument("--gauge-conf", type=float, default=0.25, help="OBB confidence threshold.")
    parser.add_argument("--gauge-iou", type=float, default=0.45, help="OBB IoU threshold.")
    parser.add_argument("--gauge-imgsz", type=int, default=640, help="OBB inference image size.")
    parser.add_argument("--gauge-device", default=None, help="OBB device, e.g. 0/cuda:0/cpu.")
    parser.add_argument(
        "--gauge-select",
        choices=["conf", "area"],
        default="conf",
        help="How to pick the ROI when multiple detections exist.",
    )
    parser.add_argument("--gauge-class", type=int, help="Optional class id filter.")

    parser.add_argument(
        "--enhance-mode",
        choices=["original", "windowing"],
        default="windowing",
        help="Image enhancement mode before OCR.",
    )
    parser.add_argument("--no-rotate", action="store_true", help="Disable width>height -> CCW90 rotation.")
    parser.add_argument("--enable-correction", action="store_true", help="Enable weld orientation correction before ROI detection.")
    parser.add_argument("--correction-model", help="Orientation correction model weights (.pth). Required when --enable-correction is set.")
    parser.add_argument("--correction-device", help="Orientation correction device, e.g. cuda:0/cpu.")
    parser.add_argument("--correction-verbose", action="store_true", help="Print per-image orientation correction diagnostics.")

    parser.add_argument("--ocr-device", choices=["cpu", "gpu"], default="gpu", help="PaddleOCR device.")
    parser.add_argument("--ocr-lang", default="en", help="PaddleOCR language.")
    parser.add_argument(
        "--ocr-rec-model-name",
        default="en_PP-OCRv5_mobile_rec",
        help="PaddleOCR text recognition model name.",
    )
    parser.add_argument(
        "--ocr-rec-model-dir",
        help="Optional local PaddleOCR text recognition model directory.",
    )
    parser.add_argument("--ocr-min-score", type=float, default=0.0, help="Min OCR confidence score.")
    parser.add_argument("--ocr-topk", type=int, default=200, help="Top-K terms kept in OCR statistics.")
    return parser.parse_args()


def resolve_yolo_device(device_str: Optional[str]) -> Optional[str]:
    return device_str


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


def _get_rel_path(image_path: Path, image_root: Optional[Path]) -> Path:
    if image_root is None:
        return Path(image_path.name)
    try:
        return image_path.relative_to(image_root)
    except ValueError:
        return Path(image_path.name)


def _normalize_ocr_status(status: Optional[str]) -> str:
    value = str(status or "unknown")
    if value in {"ok", "no_jb", "skipped_no_roi"}:
        return value
    return "other"


def _build_status_sample_dir(
    vis_dir: Path,
    image_path: Path,
    image_root: Optional[Path],
    ocr_status: Optional[str],
) -> Path:
    rel_path = _get_rel_path(image_path, image_root)
    status_name = _normalize_ocr_status(ocr_status)
    return vis_dir / "ocr_status" / status_name / rel_path.with_suffix("")


def _save_no_roi_vis_image(
    output_dir: Path,
    vis_dir: Path,
    image_path: Path,
    image_root: Optional[Path],
    image: np.ndarray,
    ocr_status: str = "skipped_no_roi",
) -> Dict[str, str]:
    sample_dir = _build_status_sample_dir(vis_dir, image_path, image_root, ocr_status)
    ensure_dir(sample_dir)
    src_vis_path = sample_dir / "input.png"
    cv2.imwrite(str(src_vis_path), image)
    return {
        "vis_path": str(src_vis_path.relative_to(output_dir)),
        "status_vis_path": str(src_vis_path.relative_to(output_dir)),
    }


def _save_vis_images(
    output_dir: Path,
    vis_dir: Path,
    image_path: Path,
    image_root: Optional[Path],
    image: np.ndarray,
    roi_image: np.ndarray,
    roi_gray: np.ndarray,
    roi_info: Dict[str, Any],
    ocr_result: Dict[str, Any],
) -> Dict[str, str]:
    sample_dir = _build_status_sample_dir(
        vis_dir=vis_dir,
        image_path=image_path,
        image_root=image_root,
        ocr_status=ocr_result.get("status"),
    )
    ensure_dir(sample_dir)

    roi_vis = build_roi_vis_image(image, roi_info)
    roi_vis_path = sample_dir / "ROI.png"
    cv2.imwrite(str(roi_vis_path), roi_vis)

    ocr_input_path = sample_dir / "ocr_input.png"
    cv2.imwrite(str(ocr_input_path), roi_gray)

    ocr_vis = draw_ocr_on_roi(roi_image, ocr_result)
    ocr_vis_path = sample_dir / "ocr_result.png"
    cv2.imwrite(str(ocr_vis_path), ocr_vis)

    return {
        "roi_vis_path": str(roi_vis_path.relative_to(output_dir)),
        "ocr_input_path": str(ocr_input_path.relative_to(output_dir)),
        "ocr_vis_path": str(ocr_vis_path.relative_to(output_dir)),
        "status_vis_dir": str(sample_dir.relative_to(output_dir)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if args.fclip_ckpt:
        print(
            "[Info] 当前脚本为 OCR-only 阶段，已接收但忽略 --fclip-ckpt；"
            "FClip 丝识别尚未在此脚本中启用。"
        )

    vis_dir = output_dir / "vis"
    if args.vis:
        ensure_dir(vis_dir)

    image_dir = Path(args.image_dir) if args.image_dir else None
    image_list = Path(args.image_list) if args.image_list else None
    image_paths = collect_images(image_dir, image_list, args.max_images)

    ocr_engine = create_paddle_ocr(
        lang=args.ocr_lang,
        device=args.ocr_device,
        rec_model_name=args.ocr_rec_model_name,
        rec_model_dir=args.ocr_rec_model_dir,
    )

    corrector = None
    if args.enable_correction:
        if not args.correction_model:
            raise ValueError("--correction-model is required when --enable-correction is set.")
        from gauge.weld_correction import WeldOrientationCorrector

        corrector = WeldOrientationCorrector(
            model_path=args.correction_model,
            model_type="resnet50",
            device=args.correction_device,
        )

    from ultralytics import YOLO

    gauge_model = YOLO(args.gauge_weights)
    yolo_device = resolve_yolo_device(args.gauge_device)

    results: List[Dict[str, Any]] = []
    image_root = image_dir.resolve() if image_dir is not None else None

    for image_path in tqdm(image_paths, desc="Gauge OCR"):
        record: Dict[str, Any] = {
            "image_path": str(image_path),
            "status": "ok",
        }
        try:
            image = load_image(image_path)
            h, w = image.shape[:2]
            record["width"] = w
            record["height"] = h
            correction_info = {
                "label": 0,
                "confidence": None,
                "status": "disabled",
                "corrected": False,
                "actions": None,
            }
            if corrector is not None:
                image, correction_info = corrector.correct_image(image, verbose=args.correction_verbose)
                corrected_h, corrected_w = image.shape[:2]
                record["width"] = corrected_w
                record["height"] = corrected_h
            record["correction"] = correction_info

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
                record["ocr"] = {"status": "skipped_no_roi", "texts": [], "scores": [], "items": [], "num_items": 0}
                if args.vis:
                    record.update(
                        _save_no_roi_vis_image(
                            output_dir=output_dir,
                            vis_dir=vis_dir,
                            image_path=image_path,
                            image_root=image_root,
                            image=image,
                            ocr_status="skipped_no_roi",
                        )
                    )
                results.append(record)
                continue

            roi_info = extract_best_obb(yolo_result[0], select=args.gauge_select, class_filter=args.gauge_class)
            if roi_info is None:
                record["status"] = "no_roi"
                record["ocr"] = {"status": "skipped_no_roi", "texts": [], "scores": [], "items": [], "num_items": 0}
                if args.vis:
                    record.update(
                        _save_no_roi_vis_image(
                            output_dir=output_dir,
                            vis_dir=vis_dir,
                            image_path=image_path,
                            image_root=image_root,
                            image=image,
                            ocr_status="skipped_no_roi",
                        )
                    )
                results.append(record)
                continue

            polygon = np.array(roi_info["polygon"], dtype=np.float32)
            roi_image, _ = crop_rotated_polygon(image, polygon)
            if roi_image is None:
                record["status"] = "roi_invalid"
                record["gauge"] = roi_info
                record["ocr"] = {
                    "status": "skipped_roi_invalid",
                    "texts": [],
                    "scores": [],
                    "items": [],
                    "num_items": 0,
                }
                results.append(record)
                continue

            roi_image, rotated, rotation = rotate_if_wide(roi_image, enable=not args.no_rotate)
            if args.enhance_mode == "windowing":
                roi_gray = enhance_windowing_gray(roi_image)
            else:
                roi_gray = to_gray(roi_image)

            ocr_result = infer_roi_ocr(ocr_engine, roi_gray, min_score=args.ocr_min_score)

            record["gauge"] = roi_info
            record["preprocess"] = {
                "rotation": rotation,
                "rotated": rotated,
                "enhance_mode": args.enhance_mode,
                "roi_size": [int(roi_image.shape[1]), int(roi_image.shape[0])],
            }
            record["ocr"] = ocr_result

            if args.vis:
                vis_paths = _save_vis_images(
                    output_dir=output_dir,
                    vis_dir=vis_dir,
                    image_path=image_path,
                    image_root=image_root,
                    image=image,
                    roi_image=roi_image,
                    roi_gray=roi_gray,
                    roi_info=roi_info,
                    ocr_result=ocr_result,
                )
                record.update(vis_paths)

        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            record.setdefault("ocr", {"status": "error", "error": str(exc), "texts": [], "scores": [], "items": []})

        results.append(record)

    ocr_stats = build_ocr_statistics(results, topk=args.ocr_topk)

    results_path = Path(args.results_json)
    if not results_path.is_absolute():
        results_path = output_dir / results_path
    ensure_dir(results_path.parent)

    stats_path = Path(args.ocr_stats_json)
    if not stats_path.is_absolute():
        stats_path = output_dir / stats_path
    ensure_dir(stats_path.parent)

    meta = {
        "schema": "gauge_ocr_stage_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_root": str(image_root) if image_root is not None else None,
        "supported_exts": list(SUPPORTED_IMAGE_EXTS),
        "gauge_weights": args.gauge_weights,
        "fclip_ckpt": args.fclip_ckpt,
        "fclip_device": args.fclip_device,
        "gauge_conf": args.gauge_conf,
        "gauge_iou": args.gauge_iou,
        "gauge_imgsz": args.gauge_imgsz,
        "gauge_select": args.gauge_select,
        "gauge_class": args.gauge_class,
        "enhance_mode": args.enhance_mode,
        "rotation_rule": "ccw90_if_width_gt_height" if not args.no_rotate else "disabled",
        "enable_correction": args.enable_correction,
        "correction_model": args.correction_model,
        "correction_device": args.correction_device,
        "ocr_device": args.ocr_device,
        "ocr_lang": args.ocr_lang,
        "ocr_rec_model_name": args.ocr_rec_model_name,
        "ocr_rec_model_dir": args.ocr_rec_model_dir,
        "ocr_min_score": args.ocr_min_score,
        "note": "OCR debug stage only; FClip stage is intentionally skipped.",
    }

    payload = {
        "meta": meta,
        "ocr_stats": ocr_stats,
        "results": results,
    }
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(ocr_stats), f, indent=2, ensure_ascii=False)

    print(f"\nOCR阶段完成，共处理 {len(results)} 张图像。")
    print(f"结果JSON: {results_path}")
    print(f"OCR统计JSON: {stats_path}")


if __name__ == "__main__":
    main()
