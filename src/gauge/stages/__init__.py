#!/usr/bin/env python3
"""Pipeline stage implementations."""
from gauge.stages.base import PipelineStage, StageContext
from gauge.stages.image_load import ImageLoadStage
from gauge.stages.correction import CorrectionStage
from gauge.stages.full_image_ocr import FullImageOCRStage
from gauge.stages.roi_detect import ROIDetectStage
from gauge.stages.roi_ocr import ROIOCRStage
from gauge.stages.wire_detect import WireDetectStage
from gauge.stages.grade_fusion import GradeFusionStage

__all__ = [
    "PipelineStage", "StageContext",
    "ImageLoadStage", "CorrectionStage", "FullImageOCRStage",
    "ROIDetectStage", "ROIOCRStage", "WireDetectStage", "GradeFusionStage",
]
