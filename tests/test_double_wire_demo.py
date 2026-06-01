import unittest

import numpy as np

try:
    from gauge.imaging.profile import bam_pair_marker_indices, normalize_profile_obb
except ImportError as exc:
    bam_pair_marker_indices = None
    normalize_profile_obb = None


@unittest.skipIf(bam_pair_marker_indices is None, "gauge.imaging.profile not importable")
class TestDoubleWireDemoMarkers(unittest.TestCase):
    def test_bam_pair_marker_indices_include_both_wire_points(self):
        pairs = [[10, 16, 22], [86, 90, 94], [243, 244, 245]]

        wires, gaps = bam_pair_marker_indices(pairs)

        self.assertEqual(wires.tolist(), [10, 22, 86, 94, 243, 245])
        self.assertEqual(gaps.tolist(), [16, 90, 244])


@unittest.skipIf(normalize_profile_obb is None, "gauge.imaging.profile not importable")
class TestDoubleWireDemoOBB(unittest.TestCase):
    def test_normalize_profile_obb_keeps_long_first_edge_as_profile_width(self):
        corners = np.array(
            [
                [0.0, 0.0],
                [240.0, 0.0],
                [240.0, 60.0],
                [0.0, 60.0],
            ],
            dtype=np.float32,
        )

        normalized, (start, end) = normalize_profile_obb(corners)

        width = np.linalg.norm(normalized[1] - normalized[0])
        height = np.linalg.norm(normalized[3] - normalized[0])
        self.assertGreater(width, height)
        self.assertTrue(np.allclose(normalized, corners))
        self.assertTrue(np.allclose(start, [0.0, 30.0]))
        self.assertTrue(np.allclose(end, [240.0, 30.0]))

    def test_normalize_profile_obb_rotates_short_first_edge_to_long_profile_width(self):
        corners = np.array(
            [
                [240.0, 0.0],
                [240.0, 60.0],
                [0.0, 60.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )

        normalized, (start, end) = normalize_profile_obb(corners)

        self.assertTrue(np.allclose(normalized, corners[[1, 2, 3, 0]]))
        self.assertTrue(np.allclose(start, [240.0, 30.0]))
        self.assertTrue(np.allclose(end, [0.0, 30.0]))

    def test_normalize_profile_obb_saved_sample_uses_long_edge(self):
        corners = np.array(
            [
                [1014.361572265625, 1211.9432373046875],
                [1003.52001953125, 1331.2000732421875],
                [590.5067138671875, 1293.6534423828125],
                [601.3482666015625, 1174.3966064453125],
            ],
            dtype=np.float32,
        )

        normalized, (start, end) = normalize_profile_obb(corners)

        width = np.linalg.norm(normalized[1] - normalized[0])
        height = np.linalg.norm(normalized[3] - normalized[0])
        self.assertGreater(width, height)
        self.assertGreater(width, 400.0)
        self.assertLess(height, 130.0)
        self.assertTrue(np.allclose(normalized, corners[[1, 2, 3, 0]]))
        self.assertTrue(np.allclose(start, [1008.9408, 1271.5717], atol=0.1))
        self.assertTrue(np.allclose(end, [595.9275, 1234.025], atol=0.1))


if __name__ == "__main__":
    unittest.main()
