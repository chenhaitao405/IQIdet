#!/usr/bin/env python3
"""Pipeline stage: OCR on the IQI ROI and plate marker matching."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gauge.fclip_stage import (
    invert_perspective_matrix,
    perspective_transform_points,
    undo_ccw90_points,
)
from gauge.iqi_rules import infer_plate_from_ocr_items
from gauge.ocr_stage import infer_roi_ocr
from gauge.pipeline_utils import (
    build_plate_visualization_items,
    is_usable_ocr_item,
    merge_prefixed_ocr_timings,
    scale_box_points,
)
from gauge.stages.base import PipelineStage, StageContext


def _project_roi_box_to_image(
    box: Any,
    crop_inverse_matrix: Optional[np.ndarray],
    pre_rotate_size: Optional[Sequence[int]],
    rotated: bool,
) -> Tuple[Any, Any]:
    """Project a box from ROI coordinates back to original image coordinates."""
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
        roi_unrotated = undo_ccw90_points(
            roi_points, pre_rotate_size=pre_rotate_size
        )

    if crop_inverse_matrix is None:
        return roi_points.tolist(), roi_unrotated.tolist()

    image_points = perspective_transform_points(
        roi_unrotated, crop_inverse_matrix
    )
    return image_points.tolist(), roi_unrotated.tolist()


def _project_ocr_items_to_image(
    items: Sequence[Dict[str, Any]],
    crop_inverse_matrix: Optional[np.ndarray],
    pre_rotate_size: Optional[Sequence[int]],
    rotated: bool,
) -> List[Dict[str, Any]]:
    """Project OCR items from ROI coordinates back to original image."""
    projected_items: List[Dict[str, Any]] = []
    for item in items:
        projected = dict(item)
        box_image, box_unrotated = _project_roi_box_to_image(
            item.get("box"),
            crop_inverse_matrix=crop_inverse_matrix,
            pre_rotate_size=pre_rotate_size,
            rotated=rotated,
        )
        projected["box_image"] = box_image
        projected["box_roi_unrotated"] = box_unrotated
        projected_items.append(projected)
    return projected_items


class ROIOCRStage(PipelineStage):
    """Run OCR on the cropped and preprocessed ROI, then match plate markers."""

    name = "roi_ocr"

    def should_run(self, ctx) -> bool:
        return ctx.roi_gray is not None

    def run(self, ctx: StageContext) -> StageContext:
        config = self.config
        ocr_backend = self.services.get("ocr_backend")
        ocr_text_corrector = self.services.get("ocr_text_corrector")

        # Run ROI OCR
        roi_ocr_result = infer_roi_ocr(
            ocr_backend,
            ocr_backend,
            ctx.roi_gray,
            min_score=config.ocr.min_score,
            text_orientation_corrector=ocr_text_corrector,
            text_orientation_verbose=config.ocr.orientation_verbose,
        )
        merge_prefixed_ocr_timings(ctx.timings_ms, "roi", roi_ocr_result)

        # Match plate markers
        roi_plate_result = infer_plate_from_ocr_items(
            roi_ocr_result.get("all_items") or [],
            require_jb=True,
            allowed_numbers=parse_allowed_numbers(config.ocr.number_range),
        )

        # Project items back to image coordinates
        crop_inverse_matrix = (
            np.array(
                ctx.roi_info.get("crop_inverse_matrix"), dtype=np.float32
            )
            if ctx.roi_info and ctx.roi_info.get("crop_inverse_matrix")
            else None
        )
        roi_projected_items = _project_ocr_items_to_image(
            roi_ocr_result.get("all_items") or [],
            crop_inverse_matrix=crop_inverse_matrix,
            pre_rotate_size=ctx.pre_rotate_size,
            rotated=ctx.rotated,
        )
        roi_plate_vis_items = build_plate_visualization_items(
            roi_projected_items, source="roi"
        )

        roi_ocr_result = dict(roi_ocr_result)
        roi_ocr_result["all_items_image"] = roi_projected_items
        roi_ocr_result["items_image"] = [
            item
            for item in roi_projected_items
            if is_usable_ocr_item(item)
        ]

        if roi_plate_result is not None:
            roi_plate_result = dict(roi_plate_result)
            roi_plate_result["raw_text_items"] = roi_plate_vis_items

        warnings: List[str] = []
        if roi_ocr_result.get("item_errors"):
            warnings.append("ROI OCR存在部分文本框识别异常")
        if roi_plate_result and roi_plate_result.get("corrections"):
            warnings.append("ROI OCR 标识解析触发了规则纠错")
        ctx.warnings.extend(warnings)

        ctx.roi_ocr_result = roi_ocr_result
        ctx.roi_plate_result = roi_plate_result
        ctx.roi_plate_vis_items = roi_plate_vis_items

        return ctx


def parse_allowed_numbers(spec: Optional[str]) -> Any:
    from gauge.iqi_rules import parse_allowed_numbers_spec

    return parse_allowed_numbers_spec(spec)
