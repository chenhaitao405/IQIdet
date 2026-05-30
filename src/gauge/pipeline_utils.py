#!/usr/bin/env python3
"""Shared helpers for the gauge pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

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
