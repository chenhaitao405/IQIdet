#!/usr/bin/env python3
"""OCR stage helpers for gauge pipeline."""

from __future__ import annotations

from collections import Counter
import inspect
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def create_paddle_ocr(
    lang: str = "en",
    device: str = "gpu",
    rec_model_name: str = "en_PP-OCRv5_mobile_rec",
    rec_model_dir: Optional[str] = None,
):
    """Create PaddleOCR engine and pin the recognition model when possible."""
    device = str(device).lower()
    if device not in {"cpu", "gpu"}:
        device = "cpu"
    if device == "cpu":
        # Force Paddle to avoid GPU path in mixed torch+paddle runtime.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    # Explicitly bind paddle runtime device to avoid torch+paddle GPU cudnn conflicts.
    try:
        import paddle

        paddle.set_device("gpu" if device == "gpu" else "cpu")
    except Exception:
        pass

    from paddleocr import PaddleOCR

    common_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
    }

    try:
        sig = inspect.signature(PaddleOCR.__init__)
        accepted = {k for k in sig.parameters.keys() if k != "self"}
    except Exception:
        accepted = set()

    def accepts(name: str) -> bool:
        return not accepted or name in accepted

    if accepts("lang"):
        common_kwargs["lang"] = lang
    if accepts("device"):
        common_kwargs["device"] = device

    candidate_kwargs: List[Dict[str, Any]] = []

    primary_kwargs = dict(common_kwargs)
    if rec_model_name and accepts("text_recognition_model_name"):
        primary_kwargs["text_recognition_model_name"] = rec_model_name
    if rec_model_dir:
        if accepts("text_recognition_model_dir"):
            primary_kwargs["text_recognition_model_dir"] = rec_model_dir
        elif accepts("rec_model_dir"):
            primary_kwargs["rec_model_dir"] = rec_model_dir
    candidate_kwargs.append(primary_kwargs)

    if rec_model_dir and "text_recognition_model_name" in primary_kwargs:
        dir_only_kwargs = dict(primary_kwargs)
        dir_only_kwargs.pop("text_recognition_model_name", None)
        candidate_kwargs.append(dir_only_kwargs)

    candidate_kwargs.append(dict(common_kwargs))
    candidate_kwargs.append(
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
        }
    )
    candidate_kwargs.append({})

    seen = set()
    for kwargs in candidate_kwargs:
        key = tuple(sorted(kwargs.items()))
        if key in seen:
            continue
        seen.add(key)
        try:
            return PaddleOCR(**kwargs)
        except Exception:
            continue

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


def _normalize_text(text: str) -> str:
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def _contains_jb(text: str) -> bool:
    return "J" in _normalize_text(text)


def _filter_items_with_jb(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in items if _contains_jb(item.get("text", ""))]


def _mirror_items_back(items: List[Dict[str, Any]], width: int) -> List[Dict[str, Any]]:
    mapped: List[Dict[str, Any]] = []
    for item in items:
        out = dict(item)
        box = item.get("box")
        if box is None:
            mapped.append(out)
            continue
        try:
            pts = np.array(box, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = (width - 1) - pts[:, 0]
            out["box"] = pts.tolist()
        except Exception:
            pass
        mapped.append(out)
    return mapped


def infer_roi_ocr(ocr_engine, roi_image: Any, min_score: float = 0.0) -> Dict[str, Any]:
    """Run OCR on original + mirrored ROI, output only J-containing results."""
    try:
        ocr_input = _ensure_bgr(roi_image)
        raw_output = _run_ocr_predict(ocr_engine, ocr_input)
        original = _parse_ocr_output(raw_output, min_score=min_score)
        original_jb = _filter_items_with_jb(original.get("items", []))

        mirrored_img = cv2.flip(ocr_input, 1)
        raw_output_m = _run_ocr_predict(ocr_engine, mirrored_img)
        mirrored = _parse_ocr_output(raw_output_m, min_score=min_score)
        mirrored_jb = _filter_items_with_jb(mirrored.get("items", []))

        selected_variant = "none"
        selected_items: List[Dict[str, Any]] = []
        if original_jb:
            selected_variant = "original"
            selected_items = original_jb
        elif mirrored_jb:
            selected_variant = "mirror"
            width = int(ocr_input.shape[1]) if hasattr(ocr_input, "shape") else 0
            if width > 0:
                selected_items = _mirror_items_back(mirrored_jb, width)
            else:
                selected_items = mirrored_jb

        texts = [str(item.get("text", "")) for item in selected_items]
        scores = [item.get("score") for item in selected_items]
        status = "ok" if selected_items else "no_jb"
        return {
            "status": status,
            "texts": texts,
            "scores": scores,
            "items": selected_items,
            "num_items": len(selected_items),
            "selected_variant": selected_variant,
            "all_texts_original": original.get("texts", []),
            "all_texts_mirror": mirrored.get("texts", []),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "texts": [],
            "scores": [],
            "items": [],
            "num_items": 0,
        }


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
