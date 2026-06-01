import unittest

from docopt import docopt

from scripts.debug import annotate_profile


class AnnotateProfilePathTest(unittest.TestCase):
    def test_default_groundtruth_path_is_next_to_profile_json(self) -> None:
        path = annotate_profile.default_groundtruth_path(
            "outputs/double_wire_demo_3/sample_profile.json"
        )

        self.assertEqual(
            str(path),
            "outputs/double_wire_demo_3/sample_groundtruth.json",
        )

    def test_docopt_does_not_treat_help_text_as_output_default(self) -> None:
        args = docopt(
            annotate_profile.__doc__,
            argv=["outputs/double_wire_demo_3/sample_profile.json"],
        )

        self.assertIsNone(args["--output"])


if __name__ == "__main__":
    unittest.main()
