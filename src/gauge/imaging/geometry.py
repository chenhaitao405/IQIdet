#!/usr/bin/env python3
"""Pure geometry helpers for IQI ROI and OCR coordinate projection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def ensure_float32_matrix(matrix: Any) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    return arr.reshape(3, 3)


def invert_perspective_matrix(matrix: Any) -> np.ndarray:
    return np.linalg.inv(ensure_float32_matrix(matrix))


def undo_ccw90_points(points_xy: np.ndarray, pre_rotate_size: Sequence[int]) -> np.ndarray:
    width = float(pre_rotate_size[0])
    out = np.asarray(points_xy, dtype=np.float32).copy()
    x_rot = out[:, 0].copy()
    y_rot = out[:, 1].copy()
    out[:, 0] = width - 1.0 - y_rot
    out[:, 1] = x_rot
    return out


def perspective_transform_points(points_xy: np.ndarray, matrix: Any) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, ensure_float32_matrix(matrix))
    return transformed.reshape(-1, 2)


def project_roi_box_to_image(
    box: Any,
    crop_inverse_matrix: Optional[np.ndarray],
    pre_rotate_size: Optional[Sequence[int]],
    rotated: bool,
) -> Tuple[Any, Any]:
    if box is None:
        return None, None
    try:
        roi_points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None, None
    if roi_points.size == 0:
        return None, None

    roi_unrotated = roi_points
    if rotated:
        if pre_rotate_size is None:
            return roi_points.tolist(), None
        roi_unrotated = undo_ccw90_points(roi_points, pre_rotate_size=pre_rotate_size)

    if crop_inverse_matrix is None:
        return roi_points.tolist(), roi_unrotated.tolist()

    image_points = perspective_transform_points(roi_unrotated, crop_inverse_matrix)
    return image_points.tolist(), roi_unrotated.tolist()


def project_ocr_items_to_image(
    items: Sequence[Dict[str, Any]],
    crop_inverse_matrix: Optional[np.ndarray],
    pre_rotate_size: Optional[Sequence[int]],
    rotated: bool,
) -> List[Dict[str, Any]]:
    projected_items: List[Dict[str, Any]] = []
    for item in items:
        projected = dict(item)
        box_image, box_unrotated = project_roi_box_to_image(
            item.get("box"),
            crop_inverse_matrix=crop_inverse_matrix,
            pre_rotate_size=pre_rotate_size,
            rotated=rotated,
        )
        projected["box_image"] = box_image
        projected["box_roi_unrotated"] = box_unrotated
        projected_items.append(projected)
    return projected_items
