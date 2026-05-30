#!/usr/bin/env python3
"""Pipeline stage base exports.

Concrete stages are imported directly by gauge.pipeline to avoid importing
heavy optional runtime dependencies during light module imports.
"""

from gauge.stages.base import PipelineStage, StageContext

__all__ = ["PipelineStage", "StageContext"]
