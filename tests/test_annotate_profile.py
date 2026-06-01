import unittest

import numpy as np
from docopt import docopt

from gauge.imaging.profile import build_groundtruth_payload
from scripts.double_wire._dwlib.io_utils import default_groundtruth_path


class AnnotateProfilePathTest(unittest.TestCase):
    def test_default_groundtruth_path_is_next_to_profile_json(self) -> None:
        path = default_groundtruth_path(
            "outputs/double_wire_demo_3/sample_profile.json"
        )

        self.assertEqual(
            str(path),
            "outputs/double_wire_demo_3/sample_groundtruth.json",
        )

    def test_docopt_does_not_treat_help_text_as_output_default(self) -> None:
        from scripts.double_wire import annotate
        args = docopt(
            annotate.__doc__,
            argv=["outputs/double_wire_demo_3/sample.jpg"],
        )

        self.assertEqual(args["--output-dir"], "outputs/double_wire_demo")

    def test_build_groundtruth_payload_positive_uses_peak_valley_peak(self) -> None:
        profile = np.full(80, 100.0, dtype=np.float64)
        profile[[10, 30]] = 180.0
        profile[20] = 80.0
        markers = [
            {"type": "peak", "idx": 10},
            {"type": "valley", "idx": 20},
            {"type": "peak", "idx": 30},
        ]

        payload = build_groundtruth_payload(
            profile,
            markers,
            source_profile="sample_profile.json",
            band_width=21,
            film_type="positive",
        )

        self.assertEqual(payload["film_type"], "positive")
        self.assertEqual(payload["num_wire_pairs"], 1)
        self.assertEqual(
            payload["wire_pairs"][0],
            {
                "group": 1,
                "wire_a_idx": 10,
                "gap_idx": 20,
                "wire_b_idx": 30,
                "wire_a_gray": 180.0,
                "gap_gray": 80.0,
                "wire_b_gray": 180.0,
            },
        )

    def test_build_groundtruth_payload_negative_uses_valley_peak_valley(self) -> None:
        profile = np.full(80, 100.0, dtype=np.float64)
        profile[[10, 30]] = 80.0
        profile[20] = 180.0
        markers = [
            {"type": "valley", "idx": 10},
            {"type": "peak", "idx": 20},
            {"type": "valley", "idx": 30},
        ]

        payload = build_groundtruth_payload(
            profile,
            markers,
            source_profile="sample_profile.json",
            band_width=21,
            film_type="negative",
        )

        self.assertEqual(payload["film_type"], "negative")
        self.assertEqual(payload["num_wire_pairs"], 1)
        self.assertEqual(payload["wire_pairs"][0]["wire_a_idx"], 10)
        self.assertEqual(payload["wire_pairs"][0]["gap_idx"], 20)
        self.assertEqual(payload["wire_pairs"][0]["wire_b_idx"], 30)


if __name__ == "__main__":
    unittest.main()
