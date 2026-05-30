import unittest
import warnings

from gauge.config import PipelineConfig
from gauge.domain.record_builders import build_iqi_record
from gauge.pipeline.context import StageContext


class RecordBuilderTest(unittest.TestCase):
    def test_build_iqi_record_validates_nested_sections_without_serializer_warning(self) -> None:
        ctx = StageContext(image_path="demo.png", config=PipelineConfig())
        ctx.width = 64
        ctx.height = 32
        ctx.general_fields_data = {
            "fields": {
                "component_codes": [{"text": "4S9", "match_text": "4S9", "value": "4S9"}],
                "weld_film_pairs": [],
                "weld_numbers": [],
                "film_numbers": [],
                "pipe_specs": [],
            }
        }
        ctx.field_statistics = {
            "component_code_count": 1,
            "weld_film_pair_count": 0,
            "weld_number_count": 0,
            "film_number_count": 0,
            "pipe_spec_count": 0,
            "general_fields_found": True,
            "full_image_marker_found": True,
            "roi_marker_found": True,
            "iqi_marker_found": True,
        }
        ctx.full_ocr_result = {"status": "ok", "texts": ["10FEJB"], "all_items": [], "items": []}
        ctx.roi_ocr_result = {"status": "ok", "texts": ["10FEJB"], "all_items": [], "items": []}
        ctx.full_plate_result = {
            "ok": True,
            "result_code": 0,
            "result_name": "success",
            "result_message": "识别成功",
            "iqi_type": "general",
            "number": 10,
            "plate_code": "10FEJB",
        }
        ctx.roi_plate_result = dict(ctx.full_plate_result)
        ctx.plate_result = dict(ctx.full_plate_result)
        ctx.plate_source = "roi"
        ctx.wire_result = {"status": "ok", "wire_count": 2, "parsed_line_count": 0, "lines": [], "warnings": []}
        ctx.grade_result = {
            "ok": True,
            "result_code": 0,
            "result_name": "success",
            "result_message": "识别成功",
            "grade": 11,
            "wire_count": 2,
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            record = build_iqi_record(ctx)
            dumped = record.model_dump()

        serializer_warnings = [w for w in caught if "Pydantic serializer warnings" in str(w.message)]
        self.assertEqual(serializer_warnings, [])
        self.assertTrue(dumped["ok"])
        self.assertEqual(dumped["grade"], 11)
        self.assertEqual(dumped["fields"]["component_codes"][0]["value"], "4S9")
        self.assertEqual(dumped["plate"]["plate_code"], "10FEJB")

    def test_build_iqi_record_treats_empty_roi_as_absent(self) -> None:
        ctx = StageContext(image_path="demo.png", config=PipelineConfig())
        ctx.roi_info = {}
        ctx.roi_ocr_result = {"status": "skipped_no_roi", "texts": [], "all_items": [], "items": []}
        ctx.wire_result = {"status": "skipped_no_roi", "wire_count": None}
        ctx.record_errors = [
            {
                "stage": "roi_detect",
                "result_code": 1101,
                "result_name": "roi_not_found",
                "result_message": "未检测到像质计 ROI",
            }
        ]

        record = build_iqi_record(ctx)
        dumped = record.model_dump()

        self.assertEqual(dumped["result_code"], 1101)
        self.assertIsNone(dumped["roi"])


if __name__ == "__main__":
    unittest.main()
