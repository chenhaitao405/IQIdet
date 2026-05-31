"""Unit tests for gauge.imaging.profile — band-averaged profile extraction."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gauge.imaging.profile import (
    extract_profile_band,
    fit_obb_and_midline,
    unwarp_obb_region,
    detect_peaks_valleys,
)


class TestExtractProfileBand(unittest.TestCase):
    """Tests for extract_profile_band()."""

    def setUp(self):
        """Create a synthetic 16-bit test image with a known gradient pattern."""
        h, w = 200, 300
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        # Horizontal ramp: left=0, right=65535
        self.ramp_image = (x_coords.astype(np.float64) / (w - 1) * 65535).astype(np.uint16)
        # Vertical ramp: top=0, bottom=65535
        self.vramp_image = (y_coords.astype(np.float64) / (h - 1) * 65535).astype(np.uint16)

    def test_horizontal_profile_band_center(self):
        """Profile along horizontal midline of a vertical ramp should be uniform ~32767."""
        h = self.vramp_image.shape[0]
        start = (10.0, float(h) / 2.0)
        end = (290.0, float(h) / 2.0)
        profile = extract_profile_band(self.vramp_image, start, end, band_width=1)
        self.assertEqual(profile.ndim, 1)
        self.assertGreater(len(profile), 0)
        expected = self.vramp_image[h // 2, 10:291].astype(np.float64).mean()
        self.assertAlmostEqual(float(profile[len(profile) // 2]), expected, delta=2.0)

    def test_vertical_profile_band_center(self):
        """Profile along vertical midline of a horizontal ramp should be uniform ~32767."""
        w = self.ramp_image.shape[1]
        start = (float(w) / 2.0, 10.0)
        end = (float(w) / 2.0, 190.0)
        profile = extract_profile_band(self.ramp_image, start, end, band_width=1)
        self.assertEqual(profile.ndim, 1)
        self.assertGreater(len(profile), 0)
        expected = self.ramp_image[10:191, w // 2].astype(np.float64).mean()
        self.assertAlmostEqual(float(profile[len(profile) // 2]), expected, delta=2.0)

    def test_horizontal_ramp_profile(self):
        """Profile along the horizontal edge of a horizontal ramp should increase linearly."""
        start = (10.0, 100.0)
        end = (290.0, 100.0)
        profile = extract_profile_band(self.ramp_image, start, end, band_width=1)
        self.assertEqual(profile.ndim, 1)
        self.assertGreater(len(profile), 0)
        # Values should be monotonically increasing (horizontal ramp)
        self.assertGreater(profile[-1], profile[0])

    def test_band_width_multiple_lines(self):
        """band_width=21 should average multiple lines and reduce noise."""
        np.random.seed(42)
        noisy = self.vramp_image.astype(np.float64).copy()
        noise = np.random.normal(0, 500, noisy.shape)
        noisy = np.clip(noisy + noise, 0, 65535).astype(np.uint16)

        h = noisy.shape[0]
        start = (10.0, float(h) / 2.0)
        end = (290.0, float(h) / 2.0)

        profile_1 = extract_profile_band(noisy, start, end, band_width=1)
        profile_21 = extract_profile_band(noisy, start, end, band_width=21)

        var_1 = np.var(np.diff(profile_1))
        var_21 = np.var(np.diff(profile_21))
        self.assertLess(var_21, var_1,
                        "band_width=21 should be smoother (var_1={:.1f}, var_21={:.1f})".format(var_1, var_21))

    def test_band_width_odd_even(self):
        """Both odd and even band_width values should work."""
        h = self.vramp_image.shape[0]
        start = (10.0, float(h) / 2.0)
        end = (290.0, float(h) / 2.0)

        for bw in [1, 3, 10, 21, 30]:
            profile = extract_profile_band(self.vramp_image, start, end, band_width=bw)
            self.assertEqual(profile.ndim, 1)
            self.assertGreater(len(profile), 0)

    def test_diagonal_profile(self):
        """Profile along a diagonal line should work."""
        start = (50.0, 50.0)
        end = (250.0, 150.0)
        profile = extract_profile_band(self.ramp_image, start, end, band_width=1)
        self.assertEqual(profile.ndim, 1)
        self.assertGreater(len(profile), 0)

    def test_returns_float_array(self):
        """Output should be float64 regardless of input dtype."""
        img8 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        profile = extract_profile_band(img8, (10, 50), (90, 50), band_width=1)
        self.assertEqual(profile.dtype, np.float64)

        img16 = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
        profile = extract_profile_band(img16, (10, 50), (90, 50), band_width=1)
        self.assertEqual(profile.dtype, np.float64)

        imgf = np.random.rand(100, 100).astype(np.float32)
        profile = extract_profile_band(imgf, (10, 50), (90, 50), band_width=1)
        self.assertEqual(profile.dtype, np.float64)

    def test_num_samples_parameter(self):
        """num_samples should control output length."""
        h = self.vramp_image.shape[0]
        start = (10.0, float(h) / 2.0)
        end = (290.0, float(h) / 2.0)

        profile_default = extract_profile_band(self.vramp_image, start, end, band_width=1)
        profile_100 = extract_profile_band(self.vramp_image, start, end, band_width=1, num_samples=100)
        profile_500 = extract_profile_band(self.vramp_image, start, end, band_width=1, num_samples=500)

        self.assertEqual(len(profile_100), 100)
        self.assertEqual(len(profile_500), 500)
        self.assertGreater(len(profile_default), 200)


class TestFitOBBAndMidline(unittest.TestCase):
    """Tests for fit_obb_and_midline()."""

    @staticmethod
    def _rotated_rect_corners(width=240.0, height=60.0, angle_deg=35.0, center=(500.0, 400.0)):
        """Return p0=TL, p1=TR, p2=BR, p3=BL for a rotated rectangle."""
        angle = math.radians(angle_deg)
        c, s = math.cos(angle), math.sin(angle)
        w, h = float(width), float(height)
        corners = np.array([
            [-w / 2 * c + h / 2 * s, -w / 2 * s - h / 2 * c],
            [ w / 2 * c + h / 2 * s,  w / 2 * s - h / 2 * c],
            [ w / 2 * c - h / 2 * s,  w / 2 * s + h / 2 * c],
            [-w / 2 * c - h / 2 * s, -w / 2 * s + h / 2 * c],
        ], dtype=np.float32)
        return corners + np.array(center, dtype=np.float32)

    def test_axis_aligned_rectangle(self):
        """Axis-aligned rectangle: OBB should match input, midline along user's top edge."""
        pts = np.array([
            [0, 0],     # TL
            [300, 0],   # TR
            [300, 100], # BR
            [0, 100],   # BL
        ], dtype=np.float32)

        corners, (start, end) = fit_obb_and_midline(pts)

        # Should return 4 corners in TL-TR-BR-BL order
        self.assertEqual(corners.shape, (4, 2))
        # Midline runs along user's top edge (p0→p1, horizontal)
        # Left edge midpoint (0,50) → right edge midpoint (300,50)
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        self.assertGreater(dx, dy, "Midline should run along user's top-edge direction (horizontal)")
        self.assertAlmostEqual(dx, 300.0, delta=5.0)

    def test_irregular_quadrilateral(self):
        """Non-rectangular 4 points: should fit a true rectangle."""
        pts = np.array([
            [10, 5],
            [290, 0],
            [305, 95],
            [5, 105],
        ], dtype=np.float32)

        corners, (start, end) = fit_obb_and_midline(pts)

        self.assertEqual(corners.shape, (4, 2))
        self.assertIsInstance(start, tuple)
        self.assertIsInstance(end, tuple)
        # Fitted corners should form a rectangle: all angles ~90deg
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            c = corners[(i + 2) % 4]
            v1 = b - a
            v2 = c - b
            dot = np.dot(v1, v2)
            self.assertAlmostEqual(dot, 0.0, delta=50.0,
                msg=f"Corner {i} not ~90 degrees")

    def test_rotated_rectangle(self):
        """45-degree rotated rectangle."""
        import math
        angle = math.radians(45)
        c, s = math.cos(angle), math.sin(angle)
        w, h = 200.0, 50.0

        corners_raw = np.array([
            [-w/2 * c + h/2 * s,  -w/2 * s - h/2 * c],
            [ w/2 * c + h/2 * s,   w/2 * s - h/2 * c],
            [ w/2 * c - h/2 * s,   w/2 * s + h/2 * c],
            [-w/2 * c - h/2 * s,  -w/2 * s + h/2 * c],
        ], dtype=np.float32)
        pts = corners_raw + np.array([500, 400], dtype=np.float32)

        corners, (start, end) = fit_obb_and_midline(pts)

        self.assertEqual(corners.shape, (4, 2))
        # Midline runs along user's top edge: length ≈ w (200)
        mid_dist = math.hypot(end[0] - start[0], end[1] - start[1])
        self.assertAlmostEqual(mid_dist, w, delta=10.0)

    def test_preserves_user_clicked_corner_order_for_rotated_rectangle(self):
        """Fitted OBB corners should keep p0,p1,p2,p3 as TL,TR,BR,BL."""
        pts = self._rotated_rect_corners(angle_deg=35.0)

        corners, (start, end) = fit_obb_and_midline(pts)

        nearest_clicked = [
            int(np.argmin(np.linalg.norm(pts - corner, axis=1)))
            for corner in corners
        ]
        self.assertEqual(nearest_clicked, [0, 1, 2, 3])

        expected_start = (pts[3] + pts[0]) / 2.0
        expected_end = (pts[1] + pts[2]) / 2.0
        self.assertTrue(np.allclose(start, expected_start, atol=1.0))
        self.assertTrue(np.allclose(end, expected_end, atol=1.0))

    def test_profile_direction_matches_user_ordered_unwarped_columns(self):
        """Profile x-axis should run from clicked left edge to clicked right edge."""
        pts = self._rotated_rect_corners(
            width=180.0, height=50.0, angle_deg=35.0, center=(180.0, 160.0),
        )
        long_axis = pts[1] - pts[0]
        long_axis = long_axis / np.linalg.norm(long_axis)

        y_coords, x_coords = np.mgrid[0:320, 0:360]
        projection = (
            (x_coords.astype(np.float64) - float(pts[0][0])) * float(long_axis[0])
            + (y_coords.astype(np.float64) - float(pts[0][1])) * float(long_axis[1])
        )
        image = np.clip(projection + 32.0, 0, 255).astype(np.uint8)

        corners, (start, end) = fit_obb_and_midline(pts)
        unwarped, (uw, uh) = unwarp_obb_region(image, corners)
        profile = extract_profile_band(image, start, end, band_width=1, num_samples=uw)

        self.assertLess(unwarped[uh // 2, 0], unwarped[uh // 2, -1])
        self.assertLess(profile[0], profile[-1])
        self.assertAlmostEqual(float(profile[0]), float(unwarped[uh // 2, 0]), delta=5.0)
        self.assertAlmostEqual(float(profile[-1]), float(unwarped[uh // 2, -1]), delta=5.0)

    def test_short_top_edge_preserves_user_corner_order(self):
        """Short top edge: corners should NOT be rotated — preserve user ordering."""
        pts = self._rotated_rect_corners(
            width=60.0, height=240.0, angle_deg=35.0, center=(260.0, 260.0),
        )

        corners, (start, end) = fit_obb_and_midline(pts)

        # Corners should preserve user's order: 0→p0, 1→p1, 2→p2, 3→p3
        nearest_clicked = [
            int(np.argmin(np.linalg.norm(pts - corner, axis=1)))
            for corner in corners
        ]
        self.assertEqual(nearest_clicked, [0, 1, 2, 3])

        # Midline connects LEFT edge (p3→p0) midpoint to RIGHT edge (p1→p2) midpoint
        expected_start = (pts[3] + pts[0]) / 2.0
        expected_end = (pts[1] + pts[2]) / 2.0
        self.assertTrue(np.allclose(start, expected_start, atol=1.0))
        self.assertTrue(np.allclose(end, expected_end, atol=1.0))

    def test_short_top_edge_profile_matches_unwarped_columns(self):
        """Profile and unwarped image should share the same left→right direction."""
        pts = self._rotated_rect_corners(
            width=60.0, height=240.0, angle_deg=35.0, center=(260.0, 260.0),
        )
        # Profile runs LEFT to RIGHT: midline connects left edge midpoint
        # (p3,p0) to right edge midpoint (p1,p2).
        profile_start = (pts[3] + pts[0]) / 2.0
        profile_end = (pts[1] + pts[2]) / 2.0
        axis = profile_end - profile_start
        axis = axis / np.linalg.norm(axis)

        y_coords, x_coords = np.mgrid[0:520, 0:520]
        projection = (
            (x_coords.astype(np.float64) - float(profile_start[0])) * float(axis[0])
            + (y_coords.astype(np.float64) - float(profile_start[1])) * float(axis[1])
        )
        image = np.clip(projection + 32.0, 0, 255).astype(np.uint8)

        corners, (start, end) = fit_obb_and_midline(pts)
        unwarped, (uw, uh) = unwarp_obb_region(image, corners)
        profile = extract_profile_band(image, start, end, band_width=1, num_samples=uw)

        # After unwarp, profile runs left → right along the ramp
        self.assertLess(unwarped[uh // 2, 0], unwarped[uh // 2, -1])
        self.assertLess(profile[0], profile[-1])
        self.assertAlmostEqual(float(profile[0]), float(unwarped[uh // 2, 0]), delta=5.0)
        self.assertAlmostEqual(float(profile[-1]), float(unwarped[uh // 2, -1]), delta=5.0)

    def test_square(self):
        """Square: midline should still work."""
        pts = np.array([
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100],
        ], dtype=np.float32)

        corners, (start, end) = fit_obb_and_midline(pts)
        self.assertIsInstance(start, tuple)
        self.assertIsInstance(end, tuple)
        self.assertNotEqual(start, end)


class TestUnwarpOBBRegion(unittest.TestCase):
    """Tests for unwarp_obb_region()."""

    def setUp(self):
        h, w = 100, 300
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        self.img = (x_coords.astype(np.float64) / (w - 1) * 255).astype(np.uint8)

    def test_simple_rectangle(self):
        """Unwarp a simple axis-aligned rectangle."""
        corners = np.array([
            [50, 0],
            [250, 0],
            [250, 100],
            [50, 100],
        ], dtype=np.float32)

        unwarped, (uw, uh) = unwarp_obb_region(self.img, corners)

        self.assertEqual(unwarped.ndim, 2)
        self.assertEqual(uw, 200)  # width = 250-50
        self.assertEqual(uh, 100)  # height = 100-0
        # Left side should be dark, right side light (horizontal ramp)
        self.assertLess(unwarped[50, 0], unwarped[50, -1])

    def test_dimensions_wider_than_tall(self):
        """Width should be the longer dimension."""
        corners = np.array([
            [0, 0],
            [400, 0],
            [400, 50],
            [0, 50],
        ], dtype=np.float32)

        unwarped, (uw, uh) = unwarp_obb_region(self.img, corners)
        self.assertGreaterEqual(uw, uh)


class TestDetectPeaksValleys(unittest.TestCase):
    """Tests for detect_peaks_valleys()."""

    def test_simple_peak_valley_pattern(self):
        """Synthetic positive-film profile: peak - valley - peak pattern."""
        np.random.seed(0)
        x = np.linspace(0, 4 * np.pi, 300)
        profile = np.sin(x) + 0.1 * np.random.randn(300)
        profile = profile.astype(np.float64)

        peaks, valleys = detect_peaks_valleys(profile, min_distance=20, prominence=0.3)

        self.assertGreater(len(peaks), 0, "Should find at least one peak")
        self.assertGreater(len(valleys), 0, "Should find at least one valley")
        if len(peaks) > 0 and len(valleys) > 0:
            self.assertGreater(profile[peaks[0]], profile[valleys[0]])

    def test_flat_signal(self):
        """Flat signal should return empty arrays."""
        profile = np.ones(200, dtype=np.float64)
        peaks, valleys = detect_peaks_valleys(profile, min_distance=10, prominence=0.1)
        self.assertEqual(len(peaks), 0)
        self.assertEqual(len(valleys), 0)

    def test_inverted_signal(self):
        """Inverted signal (negative film): valleys detected on -profile."""
        np.random.seed(0)
        x = np.linspace(0, 4 * np.pi, 300)
        profile = -np.sin(x) + 0.1 * np.random.randn(300)
        profile = profile.astype(np.float64)

        peaks, valleys = detect_peaks_valleys(profile, min_distance=20, prominence=0.3)

        self.assertGreater(len(peaks), 0)
        self.assertGreater(len(valleys), 0)

    def test_min_distance_parameter(self):
        """Larger min_distance should yield fewer peaks."""
        np.random.seed(0)
        x = np.linspace(0, 4 * np.pi, 300)
        profile = np.sin(x).astype(np.float64)

        peaks_near, _ = detect_peaks_valleys(profile, min_distance=5, prominence=0.3)
        peaks_far, _ = detect_peaks_valleys(profile, min_distance=50, prominence=0.3)

        self.assertGreaterEqual(len(peaks_near), len(peaks_far))

    def test_single_peak(self):
        """Single Gaussian peak."""
        x = np.linspace(-5, 5, 200)
        profile = np.exp(-x**2).astype(np.float64)
        peaks, valleys = detect_peaks_valleys(profile, min_distance=10, prominence=0.2)
        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0], 100, delta=5)


if __name__ == "__main__":
    unittest.main()
