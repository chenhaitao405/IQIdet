#!/usr/bin/env python3

import unittest

from gauge.iqi_ocr import compute_iqi_grade, infer_plate_from_texts, run_paddle_ocr


class TestIQIOcrRules(unittest.TestCase):
    def test_run_paddle_ocr_promotes_gray_to_bgr(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy is not available in current test runtime")

        class DummyOcr:
            def predict(self, input=None):
                return [{"shape": tuple(input.shape)}]

        gray = np.zeros((20, 10), dtype=np.uint8)
        out = run_paddle_ocr(DummyOcr(), gray)
        self.assertEqual(out[0]["shape"], (20, 10, 3))

    def test_infer_uniform_from_strict_code(self):
        result = infer_plate_from_texts(["FE12"])
        self.assertEqual(result["iqi_type"], "uniform")
        self.assertEqual(result["number"], 12)

    def test_infer_uniform_with_ee_correction(self):
        result = infer_plate_from_texts(["EE12"])
        self.assertEqual(result["iqi_type"], "uniform")
        self.assertEqual(result["number"], 12)
        self.assertIn("EE->FE", result["corrections"])

    def test_infer_gradient_with_n1_correction(self):
        result = infer_plate_from_texts(["N108"])
        self.assertEqual(result["iqi_type"], "gradient")
        self.assertEqual(result["number"], 8)

    def test_infer_from_split_tokens(self):
        result = infer_plate_from_texts(["N1", "08"])
        self.assertEqual(result["iqi_type"], "gradient")
        self.assertEqual(result["number"], 8)

    def test_uniform_grade_rules(self):
        self.assertEqual(compute_iqi_grade("uniform", 12, 3)["grade"], 12)
        self.assertEqual(compute_iqi_grade("uniform", 12, 2)["grade"], 0)

    def test_gradient_grade_rules(self):
        self.assertEqual(compute_iqi_grade("gradient", 8, 4)["grade"], 11)
        self.assertEqual(compute_iqi_grade("gradient", 8, 0)["grade"], 0)

    def test_unknown_or_missing(self):
        self.assertEqual(compute_iqi_grade(None, 8, 4)["grade"], 0)
        self.assertEqual(compute_iqi_grade("uniform", None, 4)["grade"], 0)


if __name__ == "__main__":
    unittest.main()
