#!/usr/bin/env python3
"""Shared helpers for the gauge pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_images(
    image_dir: Optional[Path],
    image_list: Optional[Path],
    max_images: Optional[int] = None,
) -> List[Path]:
    paths: List[Path] = []
    if image_list is not None:
        base = image_list.parent
        lines = image_list.read_text(encoding="utf-8").splitlines()
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            item = Path(line)
            if not item.is_absolute():
                item = (base / item).resolve()
            if item.is_dir():
                paths.extend(sorted(p for p in item.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS))
            else:
                paths.append(item)
    elif image_dir is not None:
        paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in SUPPORTED_IMAGE_EXTS)
    else:
        raise ValueError("image_dir or image_list must be provided.")

    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        root = image_dir if image_dir is not None else image_list
        raise FileNotFoundError(f"No supported images found under {root}")
    return paths


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def resize_long_side(image: np.ndarray, target_long_side: Optional[int]) -> Tuple[np.ndarray, float]:
    if target_long_side is None:
        return image, 1.0
    target = int(target_long_side)
    if target <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= target:
        return image, 1.0
    scale = float(target) / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_rotated_polygon(image: np.ndarray, polygon: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    box = order_points(polygon.astype(np.float32))
    w1 = np.linalg.norm(box[0] - box[1])
    w2 = np.linalg.norm(box[2] - box[3])
    h1 = np.linalg.norm(box[0] - box[3])
    h2 = np.linalg.norm(box[1] - box[2])
    width = int(round(max(w1, w2)))
    height = int(round(max(h1, h2)))
    if width < 2 or height < 2:
        return None, None
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix


def rotate_if_wide(image: np.ndarray, enable: bool = True) -> Tuple[np.ndarray, bool, int]:
    if not enable:
        return image, False, 0
    h, w = image.shape[:2]
    if w > h:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rotated, True, 90
    return image, False, 0


def auto_window_level(image: np.ndarray) -> Tuple[int, int]:
    percentiles = np.percentile(image, [2, 98])
    img_min, img_max = percentiles[0], percentiles[1]
    img_mean = np.mean(image)
    img_std = np.std(image)
    window_level = int(img_mean)
    window_width = int(min(4 * img_std, img_max - img_min))
    window_width = max(1, window_width)
    return window_width, window_level


def apply_window_level(image: np.ndarray, window_width: int, window_level: int) -> np.ndarray:
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2
    if window_max <= window_min:
        window_max = window_min + 1
    output = np.clip((image - window_min) / window_width * 255.0, 0, 255).astype(np.uint8)
    return output


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def enhance_windowing_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    img_float = gray.astype(np.float32, copy=False)
    ww, wl = auto_window_level(img_float)
    enhanced = apply_window_level(img_float, ww, wl)
    enhanced = apply_clahe(enhanced)
    return enhanced


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def format_polygon(points: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in points]


def safe_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]


# ---------------------------------------------------------------------------
# Pipeline helper functions (extracted from IQIInferencer static/class methods)
# ---------------------------------------------------------------------------

def build_skipped_ocr(status: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Build a placeholder OCR result when OCR is skipped."""
    payload: Dict[str, Any] = {
        "status": str(status),
        "texts": [],
        "scores": [],
        "items": [],
        "all_items": [],
        "num_items": 0,
        "selected_variant": str(status),
        "all_texts_original": [],
        "all_texts_mirror": [],
        "det_box_count": 0,
        "rec_item_count": 0,
        "jb_items": [],
        "jb_texts": [],
        "jb_item_count": 0,
        "item_errors": [],
        "timings_ms": {
            "text_det_ms": 0.0,
            "text_orientation_ms": 0.0,
            "text_rec_ms": 0.0,
            "text_total_ms": 0.0,
        },
    }
    if error:
        payload["error"] = str(error)
    return payload


def build_skipped_wire(status: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Build a placeholder wire result when wire inference is skipped."""
    payload: Dict[str, Any] = {
        "status": str(status),
        "wire_count": None,
        "parsed_line_count": 0,
        "lines": [],
        "warnings": [],
    }
    if error:
        payload["error"] = str(error)
    return payload


def is_usable_ocr_item(item: Dict[str, Any]) -> bool:
    """Check whether an OCR item has usable text."""
    return bool(
        str(item.get("text", "")).strip()
        and item.get("status") != "error"
        and item.get("accepted_by_score", True)
    )


def scale_box_points(box: Any, scale: float) -> Any:
    """Scale box coordinates by the given factor."""
    if box is None or scale == 1.0:
        return box
    try:
        pts = np.array(box, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return box
    pts = pts / float(scale)
    return pts.tolist()


def scale_ocr_items_to_original(items: Sequence[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    """Scale OCR item boxes from resized coordinates back to original image."""
    if scale == 1.0:
        return [dict(item) for item in items]
    scaled_items: List[Dict[str, Any]] = []
    for item in items:
        scaled = dict(item)
        scaled["box"] = scale_box_points(item.get("box"), scale)
        scaled_items.append(scaled)
    return scaled_items


def scale_roi_info_to_original(roi_info: Dict[str, Any], scale: float) -> Dict[str, Any]:
    """Scale ROI polygon and bbox from resized coordinates back to original image."""
    if scale == 1.0:
        return dict(roi_info)
    mapped = dict(roi_info)
    polygon = scale_box_points(roi_info.get("polygon"), scale)
    mapped["polygon"] = polygon
    bbox = roi_info.get("bbox")
    if bbox is not None:
        mapped["bbox"] = [float(value) / float(scale) for value in bbox]
    return mapped


def box_points_to_bbox(box: Any) -> Optional[List[float]]:
    """Convert a quad polygon to an axis-aligned bounding box."""
    if box is None:
        return None
    try:
        pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if pts.size == 0:
        return None
    return [
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    ]


def build_plate_visualization_items(
    items: Sequence[Dict[str, Any]],
    source: str,
) -> List[Dict[str, Any]]:
    """Build visualization-ready plate items from OCR items."""
    from gauge.domain.iqi_rules import normalize_text

    vis_items: List[Dict[str, Any]] = []
    text_index = 0
    for item in items:
        if not is_usable_ocr_item(item):
            continue
        box_image = item.get("box_image")
        if box_image is None:
            box_image = item.get("box")
        vis_items.append(
            {
                "text_index": int(text_index),
                "crop_index": item.get("crop_index"),
                "source": str(source),
                "text": str(item.get("text", "")),
                "normalized_text": normalize_text(item.get("text", "")),
                "score": item.get("score"),
                "det_score": item.get("det_score"),
                "status": item.get("status"),
                "accepted_by_score": bool(item.get("accepted_by_score", True)),
                "box_image_xy": box_image,
                "bbox_image": box_points_to_bbox(box_image),
                "box_roi_xy": item.get("box") if source == "roi" else None,
                "bbox_roi": box_points_to_bbox(item.get("box")) if source == "roi" else None,
            }
        )
        text_index += 1
    return vis_items


def merge_prefixed_ocr_timings(
    step_timings: Dict[str, float],
    prefix: str,
    ocr_result: Optional[Dict[str, Any]],
) -> None:
    """Merge OCR timings into step_timings with a prefix."""
    if not ocr_result:
        return
    for key, value in (ocr_result.get("timings_ms") or {}).items():
        normalized_key = str(key)
        if normalized_key == "text_total_ms":
            step_timings[f"{prefix}_ocr_ms"] = float(value)
        else:
            step_timings[f"{prefix}_{normalized_key}"] = float(value)
