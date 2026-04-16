# IQI Wire On ROI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make IQI inference run wire recognition whenever ROI is valid, while keeping marker-missing cases as overall failures that still return `wire_count` and `visualization.wire_lines`.

**Architecture:** Keep the pipeline structure intact in `gauge/iqi_inferencer.py`, but decouple FClip execution from marker success. ROI failure continues to short-circuit the pipeline, while marker failure becomes a final-status concern after wire inference has already populated `record["wire"]` and visualization payloads.

**Tech Stack:** Python, `unittest`, existing IQI inferencer pipeline, existing delivery JSON builders

---

### Task 1: Add Regression Test For ROI-Valid Marker-Missing Flow

**Files:**
- Create: `tests/test_iqi_inferencer.py`
- Modify: none
- Test: `tests/test_iqi_inferencer.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from unittest import mock

import numpy as np

from gauge.iqi_inferencer import IQIInferencer


class IQIInferencerMarkerFailureWireTest(unittest.TestCase):
    def test_infer_image_path_runs_wire_when_roi_exists_even_if_marker_fails(self) -> None:
        inferencer = object.__new__(IQIInferencer)
        inferencer.ocr_det_limit_side_len = 960
        inferencer.ocr_min_score = 0.0
        inferencer.ocr_text_corrector = None
        inferencer.ocr_orientation_verbose = False
        inferencer.enhance_mode = "windowing"
        inferencer.rotate_roi = True
        inferencer.gauge_conf = 0.25
        inferencer.gauge_iou = 0.45
        inferencer.gauge_imgsz = 640
        inferencer.gauge_device = None
        inferencer.gauge_select = "conf"
        inferencer.gauge_class = None
        inferencer.ocr_allowed_numbers = frozenset({6, 10, 11, 12, 13, 14, 15})
        inferencer.gauge_model = mock.Mock()
        inferencer.gauge_model.predict.return_value = ["fake-result"]
        inferencer.ocr_backend = mock.Mock()
        inferencer.fclip_inferencer = mock.Mock()
        inferencer.fclip_inferencer.infer.return_value = {
            "status": "ok",
            "wire_count": 4,
            "parsed_line_count": 1,
            "warnings": [],
            "lines": [
                {
                    "index": 0,
                    "score": 0.95,
                    "image_xy": [[10.0, 20.0], [30.0, 40.0]],
                    "roi_xy": [[1.0, 2.0], [3.0, 4.0]],
                }
            ],
        }

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        roi_gray = np.zeros((10, 10), dtype=np.uint8)

        with mock.patch("gauge.iqi_inferencer.load_image", return_value=image), \
            mock.patch("gauge.iqi_inferencer.resize_long_side", return_value=(image, 1.0)), \
            mock.patch("gauge.iqi_inferencer.enhance_windowing_gray", side_effect=[image, roi_gray]), \
            mock.patch(
                "gauge.iqi_inferencer.infer_roi_ocr",
                side_effect=[
                    {"all_items": [{"text": "BAD", "status": "ok", "accepted_by_score": True}], "item_errors": [], "timings_ms": {}},
                    {"all_items": [{"text": "BAD", "status": "ok", "accepted_by_score": True}], "item_errors": [], "timings_ms": {}},
                ],
            ), \
            mock.patch(
                "gauge.iqi_inferencer.extract_general_fields_from_ocr_items",
                return_value={
                    "fields": {
                        "component_codes": [],
                        "weld_film_pairs": [],
                        "weld_numbers": [],
                        "film_numbers": [],
                        "pipe_specs": [],
                    },
                    "field_statistics": {
                        "component_code_count": 0,
                        "weld_film_pair_count": 0,
                        "weld_number_count": 0,
                        "film_number_count": 0,
                        "pipe_spec_count": 0,
                        "general_fields_found": False,
                    },
                },
            ), \
            mock.patch(
                "gauge.iqi_inferencer.infer_plate_from_ocr_items",
                side_effect=[
                    {
                        "ok": False,
                        "result_code": 2002,
                        "result_name": "marker_missing_jb",
                        "result_message": "像质计标识识别失败：未识别到 J / JB",
                        "iqi_type": None,
                        "number": None,
                        "plate_code": None,
                        "raw_texts": ["BAD"],
                        "candidate_codes": [],
                        "corrections": [],
                    },
                    {
                        "ok": False,
                        "result_code": 2002,
                        "result_name": "marker_missing_jb",
                        "result_message": "像质计标识识别失败：未识别到 J / JB",
                        "iqi_type": None,
                        "number": None,
                        "plate_code": None,
                        "raw_texts": ["BAD"],
                        "candidate_codes": [],
                        "corrections": [],
                    },
                ],
            ), \
            mock.patch("gauge.iqi_inferencer.extract_best_obb", return_value={"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "bbox": [0, 0, 10, 10]}), \
            mock.patch("gauge.iqi_inferencer.crop_rotated_polygon", return_value=(np.zeros((10, 10, 3), dtype=np.uint8), np.eye(3, dtype=np.float32))), \
            mock.patch("gauge.iqi_inferencer.invert_perspective_matrix", return_value=np.eye(3, dtype=np.float32)), \
            mock.patch("gauge.iqi_inferencer.rotate_if_wide", return_value=(np.zeros((10, 10, 3), dtype=np.uint8), False, 0)), \
            mock.patch("gauge.iqi_inferencer.compute_iqi_grade") as compute_grade:
            record, _ = inferencer.infer_image_path("fake.png")

        self.assertEqual(record["result_code"], 2002)
        self.assertFalse(record["ok"])
        self.assertEqual(record["wire"]["status"], "ok")
        self.assertEqual(record["wire_count"], 4)
        self.assertEqual(record["visualization"]["wire_lines"], [{"index": 0, "score": 0.95, "image_xy": [[10.0, 20.0], [30.0, 40.0]]}])
        self.assertIsNone(record["grade"])
        self.assertNotEqual(record["wire"]["status"], "skipped_marker_not_found")
        compute_grade.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/cht/code/IQIdet python -m unittest tests.test_iqi_inferencer.IQIInferencerMarkerFailureWireTest.test_infer_image_path_runs_wire_when_roi_exists_even_if_marker_fails`

Expected: FAIL because current implementation returns before FClip when marker parsing fails.

- [ ] **Step 3: Write minimal implementation**

```python
if roi_error_code is not None:
    record["wire"] = self._build_skipped_wire(
        "skipped_no_roi" if roi_error_code == 1101 else "skipped_roi_invalid",
        roi_error_message,
    )
    error_entries = [{"stage": "roi", **build_result_status(roi_error_code, roi_error_message)}]
    self._attach_visualization_payload(
        record,
        roi_plate_vis_items=roi_plate_vis_items,
        full_plate_vis_items=full_plate_vis_items,
    )
    final_record = self._finalize_record(record, error_entries, warnings)
    return final_record, debug_artifacts

if self.fclip_inferencer is not None:
    wire_result = self.fclip_inferencer.infer(
        roi_gray,
        crop_inverse_matrix=np.array(record["roi"].get("crop_inverse_matrix"), dtype=np.float32),
        pre_rotate_size=record["roi"].get("crop_size_before_rotate"),
        rotated=bool(record.get("preprocess", {}).get("rotated", False)),
    )
else:
    wire_result = {
        "status": "error",
        "error": "FClip is disabled because no checkpoint was provided.",
        "wire_count": None,
        "parsed_line_count": 0,
        "lines": [],
        "warnings": [],
    }

record["wire"] = wire_result

if not selected_plate.get("ok"):
    error_entries = [{"stage": "marker", **build_result_status(int(selected_plate.get("result_code", 9001)))}]
    final_record = self._finalize_record(record, error_entries, warnings)
    return final_record, debug_artifacts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/cht/code/IQIdet python -m unittest tests.test_iqi_inferencer.IQIInferencerMarkerFailureWireTest.test_infer_image_path_runs_wire_when_roi_exists_even_if_marker_fails`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_iqi_inferencer.py gauge/iqi_inferencer.py
git commit -m "Run IQI wire inference whenever ROI is valid"
```

### Task 2: Preserve Existing ROI Failure Behavior

**Files:**
- Modify: `tests/test_iqi_inferencer.py`
- Modify: `gauge/iqi_inferencer.py`
- Test: `tests/test_iqi_inferencer.py`

- [ ] **Step 1: Write the failing test**

```python
def test_infer_image_path_skips_wire_when_roi_is_missing(self) -> None:
    inferencer = object.__new__(IQIInferencer)
    inferencer.ocr_det_limit_side_len = 960
    inferencer.ocr_min_score = 0.0
    inferencer.ocr_text_corrector = None
    inferencer.ocr_orientation_verbose = False
    inferencer.enhance_mode = "windowing"
    inferencer.rotate_roi = True
    inferencer.gauge_conf = 0.25
    inferencer.gauge_iou = 0.45
    inferencer.gauge_imgsz = 640
    inferencer.gauge_device = None
    inferencer.gauge_select = "conf"
    inferencer.gauge_class = None
    inferencer.ocr_allowed_numbers = frozenset({6, 10, 11, 12, 13, 14, 15})
    inferencer.gauge_model = mock.Mock()
    inferencer.gauge_model.predict.return_value = ["fake-result"]
    inferencer.ocr_backend = mock.Mock()
    inferencer.fclip_inferencer = mock.Mock()

    with mock.patch("gauge.iqi_inferencer.load_image", return_value=np.zeros((32, 32, 3), dtype=np.uint8)), \
        mock.patch("gauge.iqi_inferencer.resize_long_side", return_value=(np.zeros((32, 32, 3), dtype=np.uint8), 1.0)), \
        mock.patch("gauge.iqi_inferencer.enhance_windowing_gray", return_value=np.zeros((32, 32, 3), dtype=np.uint8)), \
        mock.patch(
            "gauge.iqi_inferencer.infer_roi_ocr",
            return_value={"all_items": [{"text": "FE11JB", "status": "ok", "accepted_by_score": True}], "item_errors": [], "timings_ms": {}},
        ), \
        mock.patch(
            "gauge.iqi_inferencer.extract_general_fields_from_ocr_items",
            return_value={
                "fields": {
                    "component_codes": [],
                    "weld_film_pairs": [],
                    "weld_numbers": [],
                    "film_numbers": [],
                    "pipe_specs": [],
                },
                "field_statistics": {
                    "component_code_count": 0,
                    "weld_film_pair_count": 0,
                    "weld_number_count": 0,
                    "film_number_count": 0,
                    "pipe_spec_count": 0,
                    "general_fields_found": False,
                },
            },
        ), \
        mock.patch(
            "gauge.iqi_inferencer.infer_plate_from_ocr_items",
            return_value={
                "ok": True,
                "result_code": 0,
                "result_name": "success",
                "result_message": "识别成功",
                "iqi_type": "uniform",
                "number": 11,
                "plate_code": "FE11JB",
                "raw_texts": ["FE11JB"],
                "candidate_codes": ["FE11JB"],
                "corrections": [],
            },
        ), \
        mock.patch("gauge.iqi_inferencer.extract_best_obb", return_value=None):
        record, _ = inferencer.infer_image_path("fake.png")

    self.assertEqual(record["result_code"], 1101)
    self.assertEqual(record["wire"]["status"], "skipped_no_roi")
    inferencer.fclip_inferencer.infer.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/cht/code/IQIdet python -m unittest tests.test_iqi_inferencer.IQIInferencerMarkerFailureWireTest.test_infer_image_path_skips_wire_when_roi_is_missing`

Expected: If the refactor accidentally moved wire inference too early, this test catches it by failing.

- [ ] **Step 3: Write minimal implementation**

```python
if roi_error_code is not None:
    record["wire"] = self._build_skipped_wire(
        "skipped_no_roi" if roi_error_code == 1101 else "skipped_roi_invalid",
        roi_error_message,
    )
    error_entries = [{"stage": "roi", **build_result_status(roi_error_code, roi_error_message)}]
    return final_record, debug_artifacts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/cht/code/IQIdet python -m unittest tests.test_iqi_inferencer.IQIInferencerMarkerFailureWireTest.test_infer_image_path_skips_wire_when_roi_is_missing`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_iqi_inferencer.py gauge/iqi_inferencer.py
git commit -m "Keep ROI failure short-circuit for IQI wire inference"
```

### Task 3: Run Focused Regression Verification

**Files:**
- Modify: none
- Test: `tests/test_iqi_inferencer.py`, `tests/test_iqi_delivery_record.py`

- [ ] **Step 1: Run the focused inferencer tests**

Run: `PYTHONPATH=/home/cht/code/IQIdet python -m unittest tests.test_iqi_inferencer -v`

Expected: PASS with all tests green.

- [ ] **Step 2: Run delivery record regression tests**

Run: `PYTHONPATH=/home/cht/code/IQIdet python -m unittest tests.test_iqi_delivery_record -v`

Expected: PASS to confirm JSON delivery formatting still preserves `visualization.wire_lines`.

- [ ] **Step 3: Review requirements against outputs**

Checklist:
- ROI valid + marker fail still returns marker failure
- `wire_count` is non-null when FClip succeeds
- `visualization.wire_lines` is present when FClip returns lines
- ROI invalid still skips wire inference
- `grade` remains null when marker fails

- [ ] **Step 4: Commit final code**

```bash
git add tests/test_iqi_inferencer.py gauge/iqi_inferencer.py
git commit -m "Preserve wire results for ROI-only IQI failures"
```
