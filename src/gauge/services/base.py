#!/usr/bin/env python3
"""Base class for region services (OCR and SNR) with base64 decode + singleton lifecycle."""
from __future__ import annotations
import base64
import atexit
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generic, Optional, TypeVar
import cv2
import numpy as np

try:
    from fastapi import HTTPException
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = int(status_code)
            self.detail = str(detail)

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseRegionService(Generic[T]):
    """Base for region processing services with shared base64 decode + thread pool."""

    _service: Optional[T] = None
    _executor: ThreadPoolExecutor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="region-svc"
    )

    @staticmethod
    def decode_base64(image_base64: str) -> np.ndarray:
        """Decode base64 image (supports data URL prefix)."""
        b64_data = str(image_base64 or "")
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="无法解码图片")
        return img

    @classmethod
    def get_service(cls) -> T:
        if cls._service is None:
            cls._service = cls._create_service()
        return cls._service

    @classmethod
    def close_service(cls) -> None:
        if cls._service is not None:
            cls._service.close()
            cls._service = None

    @classmethod
    def _create_service(cls) -> T:
        raise NotImplementedError("Subclass must implement _create_service()")

    @staticmethod
    def _shutdown_executor() -> None:
        try:
            BaseRegionService._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            BaseRegionService._executor.shutdown(wait=False)

    def close(self) -> None:
        pass


atexit.register(BaseRegionService.close_service)
atexit.register(BaseRegionService._shutdown_executor)
