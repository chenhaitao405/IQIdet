#!/usr/bin/env python3

import importlib
import sys
import types
import unittest
from unittest.mock import patch


class DummyPaddleOCR:
    last_kwargs = None

    def __init__(
        self,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        lang=None,
        device=None,
        text_recognition_model_name=None,
        text_recognition_model_dir=None,
    ):
        DummyPaddleOCR.last_kwargs = {
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
            "lang": lang,
            "device": device,
            "text_recognition_model_name": text_recognition_model_name,
            "text_recognition_model_dir": text_recognition_model_dir,
        }


class TestCreatePaddleOCR(unittest.TestCase):
    def test_prefers_english_recognition_model_name_and_gpu(self):
        fake_paddle = types.SimpleNamespace(set_device=lambda *_args, **_kwargs: None)
        fake_paddleocr = types.ModuleType("paddleocr")
        fake_paddleocr.PaddleOCR = DummyPaddleOCR
        fake_cv2 = types.ModuleType("cv2")
        fake_numpy = types.ModuleType("numpy")

        with patch.dict(
            sys.modules,
            {
                "paddle": fake_paddle,
                "paddleocr": fake_paddleocr,
                "cv2": fake_cv2,
                "numpy": fake_numpy,
            },
        ):
            sys.modules.pop("gauge.ocr_stage", None)
            ocr_stage = importlib.import_module("gauge.ocr_stage")
            engine = ocr_stage.create_paddle_ocr(
                lang="en",
                device="gpu",
                rec_model_name="en_PP-OCRv5_mobile_rec",
            )

        self.assertIsInstance(engine, DummyPaddleOCR)
        self.assertEqual(
            DummyPaddleOCR.last_kwargs["text_recognition_model_name"],
            "en_PP-OCRv5_mobile_rec",
        )
        self.assertEqual(DummyPaddleOCR.last_kwargs["lang"], "en")
        self.assertEqual(DummyPaddleOCR.last_kwargs["device"], "gpu")


if __name__ == "__main__":
    unittest.main()
