#!/usr/bin/env python3
"""Pipeline stage: optional weld orientation correction."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gauge.stages.base import PipelineStage, StageContext


class CorrectionStage(PipelineStage):
    """Apply optional weld-orientation correction to the image."""

    name = "correction"

    def __init__(self, config, services=None):
        super().__init__(config, services)
        self.corrector = (services or {}).get("corrector")

    def should_run(self, ctx: StageContext) -> bool:
        return self.corrector is not None

    def run(self, ctx: StageContext) -> StageContext:
        corrector = self.corrector
        correction_info: Dict[str, Any] = {
            "label": 0,
            "confidence": None,
            "status": "disabled",
            "corrected": False,
            "actions": None,
        }

        image, correction_info = corrector.correct_image(
            ctx.image, verbose=self.config.correction.verbose
        )
        height, width = image.shape[:2]

        ctx.image = image
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.correction_info = correction_info

        if ctx.debug_artifacts is not None:
            ctx.debug_artifacts["image"] = image

        return ctx
