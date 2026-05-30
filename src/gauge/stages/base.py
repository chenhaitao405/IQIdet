#!/usr/bin/env python3
"""Base classes for IQI pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gauge.config import PipelineConfig
from gauge.iqi_rules import (
    build_result_status,
    choose_primary_result_code,
    infer_plate_from_texts,
    normalize_text,
    parse_allowed_numbers_spec,
)
from gauge.models.record import IQIRecord


class StageContext(BaseModel):
    """Mutable state flowing through pipeline stages.

    Each stage reads from and writes to this context,
    building up the complete result incrementally.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input
    image_path: str
    config: PipelineConfig

    # Image (set by ImageLoadStage)
    image: Optional[np.ndarray] = None
    height: Optional[int] = None
    width: Optional[int] = None

    # Correction (set by CorrectionStage)
    correction_info: Dict[str, Any] = {}

    # Full image OCR (set by FullImageOCRStage)
    sampled_image: Optional[np.ndarray] = None
    resize_scale: float = 1.0
    full_ocr_result: Optional[Dict[str, Any]] = None
    full_plate_result: Optional[Dict[str, Any]] = None
    full_plate_vis_items: List[Dict[str, Any]] = []
    general_fields_data: Dict[str, Any] = {}
    field_statistics: Dict[str, Any] = {}

    # ROI detection (set by ROIDetectStage)
    roi_info: Optional[Dict[str, Any]] = None
    roi_cropped: Optional[np.ndarray] = None
    roi_image: Optional[np.ndarray] = None
    roi_gray: Optional[np.ndarray] = None
    roi_crop_matrix: Optional[np.ndarray] = None
    pre_rotate_size: Optional[List[int]] = None
    rotated: bool = False

    # ROI OCR (set by ROIOCRStage)
    roi_ocr_result: Optional[Dict[str, Any]] = None
    roi_plate_result: Optional[Dict[str, Any]] = None
    roi_plate_vis_items: List[Dict[str, Any]] = []

    # Wire detection (set by WireDetectStage)
    wire_result: Optional[Dict[str, Any]] = None

    # Grade fusion (set by GradeFusionStage)
    grade_result: Optional[Dict[str, Any]] = None

    # Selected plate
    plate_result: Optional[Dict[str, Any]] = None
    plate_source: Optional[str] = None

    # Tracking
    record_errors: List[Dict[str, Any]] = []
    warnings: List[str] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)

    # Debug
    debug_artifacts: Optional[Dict[str, np.ndarray]] = None
    _debug_artifacts: Optional[Dict[str, np.ndarray]] = None

    def to_record(self) -> IQIRecord:
        """Build the final IQIRecord from current context state."""
        primary_code = choose_primary_result_code(
            [entry["result_code"] for entry in self.record_errors]
        )
        status = build_result_status(primary_code)

        plate = self.plate_result or {}
        wire = self.wire_result or {}
        selected_plate_source = self.plate_source

        record = IQIRecord(
            image_path=self.image_path,
            ok=primary_code == 0,
            status="ok" if primary_code == 0 else "error",
            result_code=primary_code,
            result_name=status["result_name"],
            result_message=status["result_message"],
            grade=int(self.grade_result["grade"])
            if primary_code == 0
            and self.grade_result is not None
            and self.grade_result.get("grade") is not None
            else None,
            iqi_type=plate.get("iqi_type"),
            plate_code=plate.get("plate_code"),
            plate_number=plate.get("number"),
            plate_source=selected_plate_source,
            wire_count=wire.get("wire_count"),
            width=self.width,
            height=self.height,
            general_fields_found=bool(
                self.field_statistics.get("general_fields_found", False)
            ),
            iqi_marker_found=bool(plate.get("ok", False)),
            correction=dict(self.correction_info),
            warnings=list(self.warnings),
            errors=list(self.record_errors),
            timings_ms=dict(self.timings_ms),
        )

        # Attach full record sections
        record.fields = (
            self.general_fields_data.get("fields")
            if isinstance(self.general_fields_data.get("fields"), dict)
            else {}
        )
        record.field_statistics = dict(self.field_statistics)
        record.full_image_preprocess = self._build_full_image_preprocess()
        record.preprocess = self._build_roi_preprocess()
        record.ocr = self.full_ocr_result
        record.full_image_ocr = self.full_ocr_result
        record.full_image_plate = self.full_plate_result
        record.roi = self.roi_info
        record.roi_ocr = self.roi_ocr_result
        record.roi_plate = self.roi_plate_result

        # Plate and wire
        rec_plate = dict(plate)
        if selected_plate_source == "roi":
            rec_plate["raw_text_items"] = self.roi_plate_vis_items
        elif selected_plate_source == "full_image":
            rec_plate["raw_text_items"] = self.full_plate_vis_items
        else:
            rec_plate.setdefault("raw_text_items", [])
        record.plate = rec_plate

        record.wire = wire
        record.grade_rule = self.grade_result
        record.visualization = self._build_visualization()

        # Debug artifacts
        if self.debug_artifacts is not None:
            record._debug_artifacts = self.debug_artifacts

        return record

    def _build_full_image_preprocess(self) -> Dict[str, Any]:
        if self.full_ocr_result is None:
            return {}
        config = self.config
        return {
            "resize_scale": float(self.resize_scale),
            "resize_long_side": int(config.ocr.det_limit_side_len),
            "sampled_size": (
                [int(self.sampled_image.shape[1]), int(self.sampled_image.shape[0])]
                if self.sampled_image is not None
                else None
            ),
            "ocr_enhance_mode": "windowing",
        }

    def _build_roi_preprocess(self) -> Dict[str, Any]:
        if self.roi_image is None:
            return {}
        return {
            "rotation": 90 if self.rotated else 0,
            "rotated": bool(self.rotated),
            "enhance_mode": self.config.enhance.mode,
            "roi_size": (
                [int(self.roi_image.shape[1]), int(self.roi_image.shape[0])]
                if self.roi_image is not None
                else None
            ),
        }

    def _build_visualization(self) -> Dict[str, Any]:
        plate = self.plate_result or {}
        wire = self.wire_result or {}
        roi = self.roi_info or {}

        if self.plate_source == "roi":
            plate_items = list(self.roi_plate_vis_items or [])
        elif self.plate_source == "full_image":
            plate_items = list(self.full_plate_vis_items or [])
        else:
            plate_items = []

        allowed_numbers = parse_allowed_numbers_spec(
            self.config.ocr.number_range
        )
        target_code = normalize_text(plate.get("plate_code"))
        plate_items_selected: List[Dict[str, Any]] = []
        if target_code:
            for item in plate_items:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                parsed = infer_plate_from_texts(
                    [text],
                    require_jb=True,
                    allowed_numbers=allowed_numbers,
                )
                candidates = [
                    normalize_text(code)
                    for code in (parsed.get("candidate_codes") or [])
                ]
                if target_code in candidates:
                    plate_items_selected.append(item)

        wire_lines = []
        for line in wire.get("lines") or []:
            image_xy = line.get("image_xy")
            if not image_xy:
                continue
            wire_lines.append(
                {
                    "index": line.get("index"),
                    "score": line.get("score"),
                    "image_xy": image_xy,
                }
            )

        return {
            "roi_polygon_xy": roi.get("polygon"),
            "roi_bbox": roi.get("bbox"),
            "plate_source": self.plate_source,
            "plate_code": plate.get("plate_code"),
            "candidate_codes": plate.get("candidate_codes") or [],
            "raw_texts": plate.get("raw_texts") or [],
            "plate_text_items": plate_items,
            "plate_text_items_selected": plate_items_selected,
            "wire_lines": wire_lines,
        }


class PipelineStage(ABC):
    """Base class for a single pipeline stage."""

    name: str = "unnamed"

    def __init__(self, config: PipelineConfig, services: Optional[Dict[str, Any]] = None):
        self.config = config
        self.services = services or {}

    def should_run(self, ctx: StageContext) -> bool:
        """Return False to skip this stage."""
        return True

    @abstractmethod
    def run(self, ctx: StageContext) -> StageContext:
        """Execute the stage, mutating and returning the context."""
        ...
