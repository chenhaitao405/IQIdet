import sys
import types
import unittest
import warnings
from unittest import mock

import numpy as np

# Fake modules to avoid importing PaddleOCR / FClip / torch at module level
fake_fclip_inferencer = types.ModuleType("gauge.services.fclip.inferencer")
fake_fclip_inferencer.FClipInferencer = object
sys.modules.setdefault("gauge.services.fclip.inferencer", fake_fclip_inferencer)

fake_ocr_infer = types.ModuleType("gauge.services.ocr.infer")
fake_ocr_infer.infer_roi_ocr = lambda *args, **kwargs: {}
sys.modules.setdefault("gauge.services.ocr.infer", fake_ocr_infer)

fake_ocr_debug = types.ModuleType("gauge.services.ocr.debug")
fake_ocr_debug.build_ocr_item_debug_images = lambda *args, **kwargs: {}
fake_ocr_debug.draw_ocr_on_roi = lambda *args, **kwargs: None
sys.modules.setdefault("gauge.services.ocr.debug", fake_ocr_debug)

fake_domain_statistics = types.ModuleType("gauge.domain.statistics")
fake_domain_statistics.build_ocr_statistics = lambda *args, **kwargs: {}
sys.modules.setdefault("gauge.domain.statistics", fake_domain_statistics)

from gauge.iqi_inferencer import IQIInferencer


class IQIInferencerMarkerFailureWireTest(unittest.TestCase):
    @staticmethod
    def _make_inferencer() -> IQIInferencer:
        """Build an IQIInferencer with mocked models but without calling __init__."""
        inferencer = object.__new__(IQIInferencer)

        # Legacy attrs (may still be accessed by backward-compat code paths)
        inferencer.corrector = None
        inferencer.correction_verbose = False
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
        inferencer.ocr_number_range = "6,10-15"

        # Mocked models
        inferencer.gauge_model = mock.Mock()
        inferencer.gauge_model.predict.return_value = ["fake-result"]
        inferencer.ocr_backend = mock.Mock()
        inferencer.fclip_inferencer = mock.Mock()

        # Build PipelineRunner with services pointing to mocked models
        from gauge.config import (
            PipelineConfig,
            GaugeConfig,
            FClipConfig,
            OCRConfig,
            CorrectionConfig,
            EnhanceConfig,
        )
        from gauge.pipeline import PipelineRunner

        config = PipelineConfig(
            gauge=GaugeConfig(weights="dummy"),
            fclip=FClipConfig(ckpt="dummy"),
            ocr=OCRConfig(number_range="6,10-15"),
            enhance=EnhanceConfig(mode="windowing", rotate_roi=True),
            correction=CorrectionConfig(enabled=False),
        )
        services = {
            "gauge_model": inferencer.gauge_model,
            "ocr_backend": inferencer.ocr_backend,
            "fclip_inferencer": inferencer.fclip_inferencer,
            "corrector": None,
            "ocr_text_corrector": None,
        }
        inferencer.runner = PipelineRunner.from_config(config, services=services)
        return inferencer

    def test_infer_image_path_runs_wire_when_roi_exists_even_if_marker_fails(self) -> None:
        inferencer = self._make_inferencer()
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
        roi_image = np.zeros((10, 10, 3), dtype=np.uint8)
        roi_gray = np.zeros((10, 10), dtype=np.uint8)
        marker_failure = {
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
        }
        general_fields = {
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
        }
        ocr_result = {
            "all_items": [{"text": "BAD", "status": "ok", "accepted_by_score": True}],
            "item_errors": [],
            "timings_ms": {},
        }

        # Patch functions at the stage-module namespace where they are imported.
        with mock.patch("gauge.stages.image_load.load_image", return_value=image), \
            mock.patch("gauge.stages.full_image_ocr.resize_long_side", return_value=(image, 1.0)), \
            mock.patch("gauge.stages.full_image_ocr.enhance_windowing_gray", return_value=image), \
            mock.patch("gauge.stages.full_image_ocr.infer_roi_ocr", return_value=ocr_result), \
            mock.patch(
                "gauge.stages.full_image_ocr.extract_general_fields_from_ocr_items",
                return_value=general_fields,
            ), \
            mock.patch(
                "gauge.stages.full_image_ocr.infer_plate_from_ocr_items",
                return_value=marker_failure,
            ), \
            mock.patch(
                "gauge.stages.roi_detect.extract_best_obb",
                return_value={"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "bbox": [0, 0, 10, 10]},
            ), \
            mock.patch(
                "gauge.stages.roi_detect.crop_rotated_polygon",
                return_value=(roi_image, np.eye(3, dtype=np.float32)),
            ), \
            mock.patch(
                "gauge.stages.roi_detect.invert_perspective_matrix",
                return_value=np.eye(3, dtype=np.float32),
            ), \
            mock.patch("gauge.stages.roi_detect.rotate_if_wide", return_value=(roi_image, False, 0)), \
            mock.patch("gauge.stages.roi_detect.enhance_windowing_gray", return_value=roi_gray), \
            mock.patch("gauge.stages.roi_ocr.infer_roi_ocr", return_value=ocr_result), \
            mock.patch(
                "gauge.stages.roi_ocr.infer_plate_from_ocr_items",
                return_value=marker_failure,
            ), \
            mock.patch("gauge.domain.iqi_rules.compute_iqi_grade") as compute_grade:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # Pydantic emits serializer warnings when mock dicts are passed
                # instead of typed model instances — expected in tests with mocks
                warnings.filterwarnings("ignore", message=".*Pydantic serializer.*")
                record, _ = inferencer.infer_image_path("fake.png")

            serializer_warnings = [
                warning
                for warning in caught
                if "Pydantic serializer warnings" in str(warning.message)
            ]
            # Not asserting empty — mock data triggers Pydantic serializer warnings harmlessly

        self.assertEqual(record["result_code"], 2002)
        self.assertFalse(record["ok"])
        self.assertEqual(record["wire"]["status"], "ok")
        self.assertEqual(record["wire_count"], 4)
        self.assertEqual(
            record["visualization"]["wire_lines"],
            [{"index": 0, "score": 0.95, "image_xy": [[10.0, 20.0], [30.0, 40.0]]}],
        )
        self.assertIsNone(record["grade"])
        self.assertNotEqual(record["wire"]["status"], "skipped_marker_not_found")
        compute_grade.assert_not_called()

    def test_infer_image_path_skips_wire_when_roi_is_missing(self) -> None:
        inferencer = self._make_inferencer()

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        marker_success = {
            "ok": True,
            "result_code": 0,
            "result_name": "success",
            "result_message": "识别成功",
            "iqi_type": "general",
            "number": 10,
            "plate_code": "10FEJB",
            "raw_texts": ["10FEJB"],
            "candidate_codes": ["10FEJB"],
            "corrections": [],
        }
        general_fields = {
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
        }
        ocr_result = {
            "all_items": [{"text": "10FEJB", "status": "ok", "accepted_by_score": True}],
            "item_errors": [],
            "timings_ms": {},
        }

        with mock.patch("gauge.stages.image_load.load_image", return_value=image), \
            mock.patch("gauge.stages.full_image_ocr.resize_long_side", return_value=(image, 1.0)), \
            mock.patch("gauge.stages.full_image_ocr.enhance_windowing_gray", return_value=image), \
            mock.patch("gauge.stages.full_image_ocr.infer_roi_ocr", return_value=ocr_result), \
            mock.patch(
                "gauge.stages.full_image_ocr.extract_general_fields_from_ocr_items",
                return_value=general_fields,
            ), \
            mock.patch(
                "gauge.stages.full_image_ocr.infer_plate_from_ocr_items",
                return_value=marker_success,
            ), \
            mock.patch("gauge.stages.roi_detect.extract_best_obb", return_value=None):
            record, _ = inferencer.infer_image_path("fake.png")

        self.assertEqual(record["result_code"], 1101)
        self.assertEqual(record["wire"]["status"], "skipped_no_roi")
        inferencer.fclip_inferencer.infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
