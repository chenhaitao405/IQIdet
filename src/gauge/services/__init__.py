#!/usr/bin/env python3
"""Shared service base classes."""
from gauge.services.base import BaseRegionService


def __getattr__(name):
    if name == "BaseOrientationCorrector":
        from gauge.services.correction import BaseOrientationCorrector as _boc
        return _boc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BaseRegionService", "BaseOrientationCorrector"]
