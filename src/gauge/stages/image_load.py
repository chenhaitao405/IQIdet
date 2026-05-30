#!/usr/bin/env python3
"""Pipeline stage: load image and record dimensions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from gauge.pipeline_utils import load_image
from gauge.stages.base import PipelineStage, StageContext


class ImageLoadStage(PipelineStage):
    """Load the input image and record its dimensions."""

    name = "image_load"

    def run(self, ctx: StageContext) -> StageContext:
        image = load_image(Path(ctx.image_path))
        height, width = image.shape[:2]

        ctx.image = image
        ctx.height = int(height)
        ctx.width = int(width)

        if ctx.debug_artifacts is not None:
            ctx.debug_artifacts["image"] = image

        return ctx
