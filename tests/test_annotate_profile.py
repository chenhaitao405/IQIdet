import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from docopt import docopt

from gauge.imaging.profile import build_groundtruth_payload
from scripts.double_wire._dwlib.io_utils import default_groundtruth_path
from scripts.double_wire._dwlib.annotation import Annotator


class AnnotateProfilePathTest(unittest.TestCase):
    def test_default_groundtruth_path_is_next_to_profile_json(self) -> None:
        path = default_groundtruth_path(
            "outputs/double_wire_demo_3/sample_profile.json"
        )

        self.assertEqual(
            str(path),
            "outputs/double_wire_demo_3/sample_groundtruth.json",
        )

    def test_double_wire_artifact_paths_group_by_image_and_variant(self) -> None:
        from scripts.double_wire._dwlib.io_utils import double_wire_artifact_paths

        original = double_wire_artifact_paths(
            "outputs/double_wire_demo",
            "/data/img1.png",
            inverted=False,
        )
        inverted = double_wire_artifact_paths(
            "outputs/double_wire_demo",
            "/data/img1.png",
            inverted=True,
        )

        self.assertEqual(original.image_dir, Path("outputs/double_wire_demo/img1"))
        self.assertEqual(original.variant_dir, Path("outputs/double_wire_demo/img1/ori"))
        self.assertEqual(original.profile, Path("outputs/double_wire_demo/img1/ori/img1_profile.json"))
        self.assertEqual(original.groundtruth, Path("outputs/double_wire_demo/img1/ori/img1_groundtruth.json"))
        self.assertEqual(original.overlay, Path("outputs/double_wire_demo/img1/img1_overlay.png"))
        self.assertEqual(inverted.variant_dir, Path("outputs/double_wire_demo/img1/inver"))
        self.assertEqual(inverted.profile, Path("outputs/double_wire_demo/img1/inver/img1_inverted_profile.json"))
        self.assertEqual(inverted.groundtruth, Path("outputs/double_wire_demo/img1/inver/img1_inverted_groundtruth.json"))

    def test_find_profile_groundtruth_pairs_recurses_nested_outputs(self) -> None:
        from scripts.double_wire._dwlib.io_utils import find_profile_groundtruth_pairs

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [
                "img1/ori/img1_profile.json",
                "img1/ori/img1_groundtruth.json",
                "img1/inver/img1_inverted_profile.json",
                "img1/inver/img1_inverted_groundtruth.json",
                "legacy_profile.json",
                "legacy_groundtruth.json",
                "orphan/ori/orphan_profile.json",
            ]
            for rel in files:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")

            pairs = find_profile_groundtruth_pairs(root)

        rel_pairs = [
            (profile.relative_to(root).as_posix(), gt.relative_to(root).as_posix())
            for profile, gt in pairs
        ]
        self.assertEqual(
            rel_pairs,
            [
                ("img1/inver/img1_inverted_profile.json", "img1/inver/img1_inverted_groundtruth.json"),
                ("img1/ori/img1_profile.json", "img1/ori/img1_groundtruth.json"),
                ("legacy_profile.json", "legacy_groundtruth.json"),
            ],
        )

    def test_docopt_does_not_treat_help_text_as_output_default(self) -> None:
        from scripts.double_wire import annotate
        args = docopt(
            annotate.__doc__,
            argv=["outputs/double_wire_demo_3/sample.jpg"],
        )

        self.assertEqual(args["--output-dir"], "outputs/double_wire_demo")

    def test_bam_annotator_uses_default_output_dir_when_omitted(self) -> None:
        from scripts.double_wire.annotate import BAMAnnotator

        annotator = BAMAnnotator("sample.jpg")

        self.assertEqual(str(annotator.output_dir), "outputs/double_wire_demo")

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


class AnnotatorKeyboardRoutingTest(unittest.TestCase):
    def test_cv2_key_translation_supports_uppercase_and_escape(self) -> None:
        from scripts.double_wire.annotate import _cv2_key_to_annotation_key

        self.assertEqual(_cv2_key_to_annotation_key(ord("P")), "p")
        self.assertEqual(_cv2_key_to_annotation_key(ord("v")), "v")
        self.assertEqual(_cv2_key_to_annotation_key(27), "escape")
        self.assertIsNone(_cv2_key_to_annotation_key(-1))
        self.assertIsNone(_cv2_key_to_annotation_key(ord("n")))

    def test_cv2_key_route_drives_annotation_without_matplotlib_focus(self) -> None:
        save_calls = []
        toggle_calls = []
        next_calls = []
        quit_calls = []
        annotator = Annotator(
            np.arange(20, dtype=np.float64),
            band_width=21,
            on_save=lambda: save_calls.append("save"),
            on_toggle=lambda: toggle_calls.append("toggle"),
            on_next=lambda: next_calls.append("next"),
            on_quit=lambda: quit_calls.append("quit"),
        )
        annotator.add_marker(3)

        self.assertTrue(annotator.handle_key("p"))
        self.assertEqual(annotator.mode, Annotator.MODE_PEAK)
        self.assertTrue(annotator.handle_key("v"))
        self.assertEqual(annotator.mode, Annotator.MODE_VALLEY)
        self.assertTrue(annotator.handle_key("u"))
        self.assertEqual(annotator.markers, [])
        self.assertTrue(annotator.handle_key("s"))
        self.assertEqual(save_calls, ["save"])
        self.assertTrue(annotator.handle_key("escape"))
        self.assertEqual(toggle_calls, ["toggle"])
        self.assertTrue(annotator.handle_key("n"))
        self.assertEqual(next_calls, ["next"])
        self.assertTrue(annotator.handle_key("q"))
        self.assertEqual(quit_calls, ["quit"])

if __name__ == "__main__":
    unittest.main()
