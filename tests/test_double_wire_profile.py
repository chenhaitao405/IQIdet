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


class TestBAMHelpers(unittest.TestCase):
    """Unit tests for BAM internal helpers."""

    def test_detect_film_type_positive(self):
        """正片: valley(丝·暗) < peak(间隙·亮) → film_type='positive'."""
        from gauge.imaging.profile import _detect_film_type
        valleys = np.array([10, 50])
        peaks = np.array([30, 70])
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[valleys] = 50.0   # 暗丝(dark wires)
        profile[peaks] = 200.0    # 亮间隙(bright gaps)
        result = _detect_film_type(profile, valleys, peaks)
        self.assertEqual(result, "positive")

    def test_detect_film_type_negative(self):
        """负片: peaks=wire(bright), valleys=gap(dark) → film_type='negative'."""
        from gauge.imaging.profile import _detect_film_type
        valleys = np.array([30, 70])   # gaps (dark minima)
        peaks = np.array([10, 50])     # wires (bright maxima)
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[peaks] = 200.0   # bright wires
        profile[valleys] = 50.0  # dark gaps
        result = _detect_film_type(profile, valleys, peaks)
        self.assertEqual(result, "negative")

    def test_detect_film_type_insufficient_data(self):
        """峰谷不足时默认返回 'positive'."""
        from gauge.imaging.profile import _detect_film_type
        profile = np.ones(100, dtype=np.float64)
        result = _detect_film_type(profile, np.array([], dtype=int), np.array([], dtype=int))
        self.assertEqual(result, "positive")

    def test_fit_quadratic_background_positive(self):
        """正片: 遮罩 valley 区后在 gap 区拟合二次背景."""
        from gauge.imaging.profile import _fit_quadratic_background
        x = np.arange(200, dtype=np.float64)
        sigma = 2.0
        true_bg = np.full_like(x, 100.0)
        profile = true_bg.copy()
        wire_idx = np.array([30, 50, 100, 120, 170], dtype=int)
        for w in wire_idx:
            profile -= 30.0 * np.exp(-0.5 * ((x - w) / sigma) ** 2)

        bg = _fit_quadratic_background(profile, wire_idx, inverted=True)

        self.assertEqual(bg.shape, profile.shape)
        self.assertEqual(bg.dtype, np.float64)
        gap_mask = np.ones(200, dtype=bool)
        for w in wire_idx:
            gap_mask[w - 10:w + 11] = False
        rmse = np.sqrt(np.mean((bg[gap_mask] - true_bg[gap_mask]) ** 2))
        self.assertLess(rmse, 5.0, f"Background RMSE={rmse:.1f} too high")

    def test_fit_quadratic_background_negative(self):
        """负片: 遮罩 peak 区，inverted=False."""
        from gauge.imaging.profile import _fit_quadratic_background
        x = np.arange(200, dtype=np.float64)
        sigma = 2.0
        true_bg = np.full_like(x, 120.0)
        profile = true_bg.copy()
        wire_idx = np.array([30, 50, 100, 120, 170], dtype=int)
        for w in wire_idx:
            profile += 30.0 * np.exp(-0.5 * ((x - w) / sigma) ** 2)

        bg = _fit_quadratic_background(profile, wire_idx, inverted=False)

        self.assertEqual(bg.shape, profile.shape)
        gap_mask = np.ones(200, dtype=bool)
        for w in wire_idx:
            gap_mask[w - 10:w + 11] = False
        rmse = np.sqrt(np.mean((bg[gap_mask] - true_bg[gap_mask]) ** 2))
        self.assertLess(rmse, 5.0, f"Background RMSE={rmse:.1f} too high")

    def test_fit_quadratic_background_short_profile(self):
        """极短剖面不应崩溃."""
        from gauge.imaging.profile import _fit_quadratic_background
        profile = np.array([10.0, 12.0, 10.0], dtype=np.float64)
        bg = _fit_quadratic_background(profile, np.array([1], dtype=int), inverted=True)
        self.assertEqual(bg.shape, (3,))


    def test_compute_dip_basic(self):
        """已知 profile 和 background 值 → 验证 dip 计算."""
        from gauge.imaging.profile import _compute_dip
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[20] = 90.0   # wire_a (dark)
        profile[30] = 105.0  # gap (bright)
        profile[40] = 90.0   # wire_b (dark)
        background = np.ones(100, dtype=np.float64) * 100.0

        dip = _compute_dip(profile, 20, 30, 40, background, half_w=0)
        # A=|100-90|=10, B=|100-90|=10, C=|100-105|=5
        # dip=100*(10+10-2*5)/(10+10)=100*10/20=50.0
        self.assertAlmostEqual(dip, 50.0, delta=0.01)

    def test_compute_dip_with_window(self):
        """half_w > 0 时使用邻域均值."""
        from gauge.imaging.profile import _compute_dip
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[18:23] = 90.0
        profile[28:33] = 105.0
        profile[38:43] = 90.0
        background = np.ones(100, dtype=np.float64) * 100.0

        dip = _compute_dip(profile, 20, 30, 40, background, half_w=2)
        self.assertAlmostEqual(dip, 50.0, delta=0.5)

    def test_compute_dip_fully_merged(self):
        """完全融合（denom≈0）→ 返回 0."""
        from gauge.imaging.profile import _compute_dip
        profile = np.ones(100, dtype=np.float64) * 100.0
        background = np.ones(100, dtype=np.float64) * 100.0

        dip = _compute_dip(profile, 20, 30, 40, background, half_w=0)
        self.assertEqual(dip, 0.0)

    def test_compute_dip_gap_exceeds_wires(self):
        """间隙偏差大于丝偏差 → 钳位到 0."""
        from gauge.imaging.profile import _compute_dip
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[20] = 110.0  # wire_a
        profile[30] = 80.0   # gap (deep gap — profile farther from bg than wires)
        profile[40] = 110.0  # wire_b
        background = np.ones(100, dtype=np.float64) * 100.0
        # A=10, B=10, C=20 → dip=100*(20-40)/20 = -100 → should clamp to 0
        dip = _compute_dip(profile, 20, 30, 40, background, half_w=0)
        self.assertAlmostEqual(dip, 0.0)

    def test_pair_wires_positive(self):
        """正片: wires=valleys, gaps=peaks, 相邻 valley 间距≤1.05*首对间距."""
        from gauge.imaging.profile import _pair_wires_and_compute_dips
        x = np.linspace(0, 6 * np.pi, 300)
        profile = (-np.sin(x) * 30.0 + 100.0).astype(np.float64)
        background = np.ones(300, dtype=np.float64) * 100.0
        peaks, valleys = detect_peaks_valleys(profile, min_distance=30, prominence=0.05)

        dips, pairs = _pair_wires_and_compute_dips(
            profile, valleys, peaks, background, half_w=1,
            dist_factor=1.05, film_type="positive",
        )
        self.assertGreater(len(dips), 0)
        self.assertEqual(len(dips), len(pairs))
        for w1, g, w2 in pairs:
            self.assertLess(w1, g)
            self.assertLess(g, w2)

    def test_pair_wires_negative(self):
        """负片: wires=peaks, gaps=valleys."""
        from gauge.imaging.profile import _pair_wires_and_compute_dips
        x = np.linspace(0, 6 * np.pi, 300)
        profile = (np.sin(x) * 30.0 + 100.0).astype(np.float64)
        background = np.ones(300, dtype=np.float64) * 100.0
        peaks, valleys = detect_peaks_valleys(profile, min_distance=30, prominence=0.05)

        dips, pairs = _pair_wires_and_compute_dips(
            profile, peaks, valleys, background, half_w=1,
            dist_factor=1.05, film_type="negative",
        )
        self.assertGreater(len(dips), 0)
        for w1, g, w2 in pairs:
            self.assertLess(w1, g)
            self.assertLess(g, w2)

    def test_pair_wires_filters_wide_gaps(self):
        """间距 > 1.05*dist[0] 的假丝被跳过."""
        from gauge.imaging.profile import _pair_wires_and_compute_dips
        profile = np.ones(200, dtype=np.float64) * 100.0
        wire_pos = np.array([20, 40, 60, 150], dtype=int)
        for w in wire_pos:
            profile[w] = 80.0
        gap_pos = np.array([30, 50, 105], dtype=int)
        for g in gap_pos:
            profile[g] = 120.0
        background = np.ones(200, dtype=np.float64) * 100.0

        dips, pairs = _pair_wires_and_compute_dips(
            profile, wire_pos, gap_pos, background, half_w=0,
            dist_factor=1.05, film_type="positive",
        )
        self.assertEqual(len(pairs), 2)

    def test_pair_wires_no_gap_between(self):
        """两丝之间无 gap → 跳过该对."""
        from gauge.imaging.profile import _pair_wires_and_compute_dips
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[[20, 40, 60]] = 80.0
        background = np.ones(100, dtype=np.float64) * 100.0

        dips, pairs = _pair_wires_and_compute_dips(
            profile, np.array([20, 40, 60]), np.array([10, 90]), background,
            half_w=0, dist_factor=1.05, film_type="positive",
        )
        self.assertEqual(len(pairs), 0)


class TestComputeContrast(unittest.TestCase):
    """Tests for compute_contrast()."""

    def setUp(self):
        """Create synthetic negative-film profile with 4 wire pairs (decreasing dip)."""
        np.random.seed(42)
        x = np.arange(400, dtype=np.float64)
        bg = 0.0003 * x**2 + 150.0
        self.profile = bg.copy()
        # D1 (deep): wires at ~50,90, gap at ~70
        self.profile[45:55] += 40.0   # wire height
        self.profile[85:95] += 40.0
        self.profile[65:75] -= 22.0   # gap dip (~55% of wire height)
        # D2: wires at ~140,180, gap at ~160
        self.profile[135:145] += 34.0
        self.profile[175:185] += 34.0
        self.profile[155:165] -= 20.0   # gap dip (~59% of wire height)
        # D3: wires at ~230,270, gap at ~250
        self.profile[225:235] += 24.0
        self.profile[265:275] += 24.0
        self.profile[245:255] -= 16.0   # gap dip (~67% of wire height)
        # D4 (nearly merged): wires at ~320,360, gap at ~340
        self.profile[315:325] += 12.0
        self.profile[355:365] += 12.0
        self.profile[335:345] -= 10.0   # gap dip (~83% of wire height, nearly merged)
        self.profile += np.random.normal(0, 1.5, 400).astype(np.float64)

    def test_auto_film_type(self):
        """film_type='auto' 应检测为 negative."""
        from gauge.imaging.profile import compute_contrast
        result = compute_contrast(self.profile, film_type="auto", min_distance=30)
        self.assertEqual(result.film_type, "negative")

    def test_explicit_film_type(self):
        """显式指定 film_type='positive' 应保留."""
        from gauge.imaging.profile import compute_contrast
        result = compute_contrast(self.profile, film_type="positive")
        self.assertEqual(result.film_type, "positive")

    def test_produces_dips_and_pairs(self):
        """应产出 dips 和 pairs."""
        from gauge.imaging.profile import compute_contrast
        result = compute_contrast(self.profile, film_type="negative", min_distance=30)
        self.assertGreater(len(result.dips), 0)
        self.assertEqual(len(result.dips), len(result.pairs))
        self.assertGreater(result.dips[0], result.dips[-1],
                          f"D1 dip ({result.dips[0]:.1f}) should exceed D4 dip ({result.dips[-1]:.1f})")

    def test_background_length(self):
        """background 与 profile 等长."""
        from gauge.imaging.profile import compute_contrast
        result = compute_contrast(self.profile)
        self.assertEqual(len(result.background), len(self.profile))
        self.assertEqual(result.background.dtype, np.float64)

    def test_short_profile_edge_case(self):
        """极短 profile 不应崩溃."""
        from gauge.imaging.profile import compute_contrast
        result = compute_contrast(np.array([10.0, 12.0], dtype=np.float64))
        self.assertEqual(len(result.dips), 0)
        self.assertEqual(result.film_type, "positive")

    def test_no_peaks_or_valleys(self):
        """平坦剖面 → 空结果."""
        from gauge.imaging.profile import compute_contrast
        flat = np.ones(200, dtype=np.float64) * 100.0
        result = compute_contrast(flat)
        self.assertEqual(len(result.dips), 0)


if __name__ == "__main__":
    unittest.main()
