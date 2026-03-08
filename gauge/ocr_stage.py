#!/usr/bin/env python3
"""OCR stage helpers for gauge pipeline."""

from __future__ import annotations

from collections import Counter
import inspect
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def create_paddle_ocr(lang: str = "en", device: str = "cpu"):
    """Create PaddleOCR engine close to WeldOCR/OCR_main.py usage."""
    device = str(device).lower()
    if device not in {"cpu", "gpu"}:
        device = "cpu"
    if device == "cpu":
        # Force Paddle to avoid GPU path in mixed torch+paddle runtime.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Explicitly bind paddle runtime device to avoid torch+paddle GPU cudnn conflicts.
    try:
        import paddle

        paddle.set_device("gpu" if device == "gpu" else "cpu")
    except Exception:
        pass

    from paddleocr import PaddleOCR

    # Match WeldOCR/OCR_main.py first.
    base_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
    }

    try:
        sig = inspect.signature(PaddleOCR.__init__)
        accepted = {k for k in sig.parameters.keys() if k != "self"}
    except Exception:
        accepted = set()

    if "lang" in accepted:
        base_kwargs["lang"] = lang
    if "device" in accepted:
        base_kwargs["device"] = device

    # Keep lang optional as a best-effort extension.
    try:
        return PaddleOCR(**base_kwargs)
    except Exception:
        pass

    try:
        fallback_kwargs = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
        }
        if "device" in accepted:
            fallback_kwargs["device"] = device
        if "lang" in accepted:
            fallback_kwargs["lang"] = lang
        return PaddleOCR(**fallback_kwargs)
    except Exception:
        # Last fallback: minimal default constructor.
        return PaddleOCR()


def _run_ocr_predict(ocr_engine, image: Any) -> Any:
    try:
        return ocr_engine.predict(input=image)
    except TypeError:
        pass

    try:
        return ocr_engine.predict(image)
    except Exception:
        pass

    return ocr_engine.ocr(image, cls=False)


def _ensure_bgr(image: Any) -> Any:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            # cv2 image is usually BGR; WeldOCR path feeds RGB arrays from PIL.
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _parse_ocr_output(raw_output: Any, min_score: float = 0.0) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []

    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
        block = raw_output[0]
        rec_texts = block.get("rec_texts", []) or []
        rec_scores = block.get("rec_scores", []) or []
        rec_polys = block.get("rec_polys") or block.get("dt_polys") or []
        for idx, text in enumerate(rec_texts):
            score = float(rec_scores[idx]) if idx < len(rec_scores) and rec_scores[idx] is not None else None
            if score is not None and score < min_score:
                continue
            box = rec_polys[idx] if idx < len(rec_polys) else None
            items.append({"text": str(text), "score": score, "box": box})
    elif isinstance(raw_output, list) and raw_output:
        rows = raw_output[0] if len(raw_output) == 1 and isinstance(raw_output[0], list) else raw_output
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            rec = row[1]
            if not isinstance(rec, (list, tuple)) or not rec:
                continue
            text = str(rec[0])
            score = float(rec[1]) if len(rec) > 1 and rec[1] is not None else None
            if score is not None and score < min_score:
                continue
            items.append({"text": text, "score": score, "box": row[0]})

    texts = [item["text"] for item in items]
    scores = [item["score"] for item in items]
    return {
        "status": "ok",
        "texts": texts,
        "scores": scores,
        "items": items,
        "num_items": len(items),
    }


def infer_roi_ocr(ocr_engine, roi_image: Any, min_score: float = 0.0) -> Dict[str, Any]:
    """Run OCR on one ROI image and return normalized result dict."""
    try:
        ocr_input = _ensure_bgr(roi_image)
        raw_output = _run_ocr_predict(ocr_engine, ocr_input)
        return _parse_ocr_output(raw_output, min_score=min_score)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "texts": [],
            "scores": [],
            "items": [],
            "num_items": 0,
        }


def _normalize_text(text: str) -> str:
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def build_ocr_statistics(results: List[Dict[str, Any]], topk: int = 200) -> Dict[str, Any]:
    """Aggregate OCR-level stats across all image records."""
    status_counter = Counter()
    ocr_status_counter = Counter()
    raw_text_counter = Counter()
    norm_text_counter = Counter()

    for record in results:
        status_counter[str(record.get("status", "unknown"))] += 1
        ocr = record.get("ocr") or {}
        ocr_status_counter[str(ocr.get("status", "missing"))] += 1
        for text in ocr.get("texts", []) or []:
            raw = str(text)
            norm = _normalize_text(raw)
            raw_text_counter[raw] += 1
            if norm:
                norm_text_counter[norm] += 1

    return {
        "images_total": len(results),
        "pipeline_status": {k: int(v) for k, v in status_counter.items()},
        "ocr_status": {k: int(v) for k, v in ocr_status_counter.items()},
        "top_raw_text": {k: int(v) for k, v in raw_text_counter.most_common(topk)},
        "top_normalized_text": {k: int(v) for k, v in norm_text_counter.most_common(topk)},
    }


def draw_ocr_on_roi(roi_image: np.ndarray, ocr_result: Dict[str, Any]) -> np.ndarray:
    """Draw OCR polygons and texts on ROI image."""
    vis = roi_image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.ndim == 3 and vis.shape[2] == 1:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    items = ocr_result.get("items", []) or []
    for item in items:
        box = item.get("box")
        text = str(item.get("text", ""))
        score = item.get("score")
        if box is None:
            continue
        try:
            pts = np.array(box, dtype=np.float32).reshape(-1, 2)
        except Exception:
            continue
        if pts.shape[0] < 3:
            continue
        pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts_i], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        label = text
        if score is not None:
            label = f"{text} ({float(score):.2f})"
        x = int(np.min(pts[:, 0]))
        y = int(np.min(pts[:, 1])) - 6
        y = max(y, 18)
        cv2.putText(
            vis,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    return vis
