#!/usr/bin/env python3
"""One-click regression check for refactor branches.

Runs batch inference on the 8-image test set, then compares the output
against the golden baseline.

Usage:
    python scripts/compare_results.py

Exit code 0 = all match, 1 = mismatch or error.

Lifecycle: this script lives only on refactor/pipeline-stages* branches.
It should be deleted in the final cleanup commit before merging to main.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "outputs" / "iqi_grade_infer_szj" / "iqi_grade_results2.json"
TMP_OUTPUT = ROOT / "outputs" / "iqi_grade_infer_szj" / "iqi_grade_results_tmp.json"
IMAGE_DIR = Path("/home/cht/datasets/iqiTEST")
DEFAULT_INFER_PYTHON = Path("/home/cht/miniconda3/envs/weld-gpu/bin/python")


def _infer_python() -> str:
    override = os.environ.get("IQIDET_COMPARE_PYTHON")
    if override:
        return override
    if DEFAULT_INFER_PYTHON.exists():
        return str(DEFAULT_INFER_PYTHON)
    return sys.executable


def _cuda_available() -> bool:
    try:
        result = subprocess.run(
            [
                _infer_python(),
                "-c",
                "import torch; print('1' if torch.cuda.is_available() else '0')",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return False
    return result.returncode == 0 and result.stdout.strip().endswith("1")


CUDA_AVAILABLE = _cuda_available()
OCR_DEVICE = "gpu" if CUDA_AVAILABLE else "cpu"
OCR_ORIENTATION_DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"

# ---------------------------------------------------------------------------
# Inference args - kept in sync with .vscode/launch.json
# ---------------------------------------------------------------------------
INFER_ARGS = [
    _infer_python(),
    str(ROOT / "run_iqi_grade_infer.py"),
    "--image-dir", str(IMAGE_DIR),
    "--gauge-weights", "models/guagerotation.pt",
    "--fclip-ckpt", "models/fclip67.pth.tar",
    "--fclip-config", "models/fclip_config.yaml",
    "--output-json", str(TMP_OUTPUT),
    "--ocr-device", OCR_DEVICE,
    "--ocr-det-model-name", "PP-OCRv5_server_det",
    "--ocr-det-limit-side-len", "960",
    "--ocr-det-limit-type", "max",
    "--ocr-rec-model-dir", "models/OCR_rec_inference_best_accuracy0325",
    "--enable-ocr-orientation",
    "--ocr-orientation-model", "models/ocr_orientation_model.pth",
    "--ocr-orientation-device", OCR_ORIENTATION_DEVICE,
    "--ocr-number-range", "1-19",
]

# ---------------------------------------------------------------------------
# Deterministic comparison keys
# ---------------------------------------------------------------------------
RECORD_KEYS = [
    "ok", "result_code", "result_name", "result_message",
    "grade", "iqi_type", "plate_code", "plate_number",
    "plate_source", "wire_count",
    "general_fields_found", "iqi_marker_found",
]

FIELD_STATS_KEYS = [
    "component_code_count", "weld_film_pair_count",
    "weld_number_count", "film_number_count", "pipe_spec_count",
    "general_fields_found", "full_image_marker_found",
    "roi_marker_found", "iqi_marker_found",
]

SUMMARY_KEYS = [
    "images_total", "success_total", "failure_total",
    "result_code_hist", "result_code_hist_named",
    "iqi_type_hist", "grade_hist", "field_totals",
    "images_with_general_fields", "images_with_iqi_marker",
]


def _diff_record(new: dict, ref: dict, label: str) -> list[str]:
    diffs: list[str] = []
    for key in RECORD_KEYS:
        if new.get(key) != ref.get(key):
            diffs.append(f"  {label}.{key}: {ref.get(key)} -> {new.get(key)}")

    fs_new = new.get("field_statistics") or {}
    fs_ref = ref.get("field_statistics") or {}
    for key in FIELD_STATS_KEYS:
        if fs_new.get(key) != fs_ref.get(key):
            diffs.append(f"  {label}.field_statistics.{key}: {fs_ref.get(key)} -> {fs_new.get(key)}")

    errs_new = new.get("errors") or []
    errs_ref = ref.get("errors") or []
    if len(errs_new) != len(errs_ref):
        diffs.append(f"  {label}.errors count: {len(errs_ref)} -> {len(errs_new)}")
    else:
        for i, (en, er) in enumerate(zip(errs_new, errs_ref)):
            for ek in ("stage", "result_code", "result_name", "result_message"):
                if en.get(ek) != er.get(ek):
                    diffs.append(f"  {label}.errors[{i}].{ek}: {er.get(ek)} -> {en.get(ek)}")

    w_new = len(new.get("warnings") or [])
    w_ref = len(ref.get("warnings") or [])
    if w_new != w_ref:
        diffs.append(f"  {label}.warnings count: {w_ref} -> {w_new}")

    fields_new = new.get("fields") or {}
    fields_ref = ref.get("fields") or {}
    for fk in ("component_codes", "weld_film_pairs", "weld_numbers", "film_numbers", "pipe_specs"):
        cn = len(fields_new.get(fk) or [])
        cr = len(fields_ref.get(fk) or [])
        if cn != cr:
            diffs.append(f"  {label}.fields.{fk} count: {cr} -> {cn}")

    return diffs


def _diff_summary(new: dict, ref: dict) -> list[str]:
    diffs: list[str] = []
    for key in SUMMARY_KEYS:
        if new.get(key) != ref.get(key):
            diffs.append(f"  summary.{key}: {ref.get(key)} -> {new.get(key)}")
    return diffs


def main() -> int:
    missing = []
    if not BASELINE.exists():
        missing.append(str(BASELINE))
    if not IMAGE_DIR.exists():
        missing.append(str(IMAGE_DIR))
    if missing:
        print(f"SKIP: missing prerequisites: {', '.join(missing)}")
        return 0

    print(f"Running inference on {IMAGE_DIR} ...")
    result = subprocess.run(
        INFER_ARGS,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"FAIL: inference exited with code {result.returncode}")
        if result.stderr:
            print("STDERR:", result.stderr[-2000:])
        return 1
    if not TMP_OUTPUT.exists():
        print(f"FAIL: output JSON not found at {TMP_OUTPUT}")
        return 1

    with open(BASELINE, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(TMP_OUTPUT, encoding="utf-8") as f:
        current = json.load(f)

    all_diffs: list[str] = []
    all_diffs.extend(_diff_summary(current.get("summary") or {}, baseline.get("summary") or {}))

    ref_by_name = {r["image_path"].split("/")[-1]: r for r in baseline.get("results", [])}
    cur_by_name = {r["image_path"].split("/")[-1]: r for r in current.get("results", [])}

    all_names = sorted(set(ref_by_name) | set(cur_by_name))
    for name in all_names:
        ref_r = ref_by_name.get(name)
        cur_r = cur_by_name.get(name)
        if ref_r is None:
            all_diffs.append(f"  {name}: NEW IMAGE (not in baseline)")
        elif cur_r is None:
            all_diffs.append(f"  {name}: MISSING (in baseline but not in output)")
        else:
            all_diffs.extend(_diff_record(cur_r, ref_r, name))

    print()
    if all_diffs:
        print("MISMATCH - differences found:")
        for diff in all_diffs:
            print(diff)
        print(f"\n{len(all_diffs)} difference(s) total.")
        return 1

    batch_count = len(current.get("results", []))
    print(f"ALL MATCH — {batch_count} image(s), 0 differences.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
