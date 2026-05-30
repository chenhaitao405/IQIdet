#!/usr/bin/env python3
"""Pipeline stage: FClip wire count inference on the ROI."""

from __future__ import annotations

import numpy as np

from gauge.pipeline_utils import build_skipped_wire
from gauge.stages.base import PipelineStage, StageContext


class WireDetectStage(PipelineStage):
    """Run FClip wire-count inference on the preprocessed ROI."""

    name = "wire_detect"

    def should_run(self, ctx: StageContext) -> bool:
        return ctx.roi_gray is not None

    def run(self, ctx: StageContext) -> StageContext:
        fclip_inferencer = self.services.get("fclip_inferencer")

        if fclip_inferencer is None:
            ctx.wire_result = build_skipped_wire(
                "error",
                "FClip is disabled because no checkpoint was provided.",
            )
            return ctx

        crop_inverse_matrix = (
            np.array(
                ctx.roi_info.get("crop_inverse_matrix"), dtype=np.float32
            )
            if ctx.roi_info and ctx.roi_info.get("crop_inverse_matrix")
            else None
        )

        wire_result = fclip_inferencer.infer(
            ctx.roi_gray,
            crop_inverse_matrix=crop_inverse_matrix,
            pre_rotate_size=ctx.pre_rotate_size,
            rotated=ctx.rotated,
        )

        ctx.wire_result = wire_result
        ctx.warnings.extend(wire_result.get("warnings") or [])
        return ctx
