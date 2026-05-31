# Double Wire Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive visualization tool for double-wire IQI profile band extraction — OBB selection + band-averaged gray profile + peak/valley detection.

**Architecture:** Two new files: `src/gauge/imaging/profile.py` (reusable pure functions, no UI) and `scripts/debug/double_wire_demo.py` (OpenCV HighGUI + matplotlib interactive demo). Follows existing project patterns: `docopt` CLI, `REPO_ROOT` sys.path setup, TDD with `unittest`.

**Tech Stack:** Python 3, numpy, scipy (ndimage.map_coordinates, signal.find_peaks), OpenCV (highgui, image I/O), matplotlib (profile curve plot), docopt (CLI)

---

## File Structure

| File | Responsibility | New/Modify |
|------|---------------|------------|
| `tests/test_double_wire_profile.py` | Unit tests for profile.py pure functions | Create |
| `src/gauge/imaging/profile.py` | `extract_profile_band`, `get_obb_long_edge_midline`, `detect_peaks_valleys` | Create |
| `scripts/debug/double_wire_demo.py` | `DoubleWireDemo` class + `main()` entry | Create |

---

### Task 1: Write unit test file for profile.py (TDD — tests first)

**Files:**
- Create: `tests/test_double_wire_profile.py`

- [ ] **Step 1: Create test file with all test cases**

```python
#!/usr/bin/env python3
"""Unit tests for gauge.imaging.profile — band-averaged profile extraction."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from gauge.imaging.profile import (
    extract_profile_band,
    get_obb_long_edge_midline,
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
        # Should be roughly uniform at the middle column's value
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

        # band_width=21 should have lower variance than band_width=1
        var_1 = np.var(np.diff(profile_1))
        var_21 = np.var(np.diff(profile_21))
        self.assertLess(var_21, var_1,
                        f"band_width=21 should be smoother (var_1={var_1:.1f}, var_21={var_21:.1f})")

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
        # uint8 input
        img8 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        profile = extract_profile_band(img8, (10, 50), (90, 50), band_width=1)
        self.assertEqual(profile.dtype, np.float64)

        # uint16 input
        img16 = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
        profile = extract_profile_band(img16, (10, 50), (90, 50), band_width=1)
        self.assertEqual(profile.dtype, np.float64)

        # float32 input
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
        # Default should be roughly the pixel distance
        self.assertGreater(len(profile_default), 200)


class TestGetOBBLongEdgeMidline(unittest.TestCase):
    """Tests for get_obb_long_edge_midline()."""

    def test_axis_aligned_rectangle(self):
        """Axis-aligned rectangle with known long edge."""
        # Rectangle: width=300, height=100 (long edge is horizontal)
        obb = np.array([
            [0, 0],     # top-left
            [300, 0],   # top-right
            [300, 100], # bottom-right
            [0, 100],   # bottom-left
        ], dtype=np.float32)

        (start, end) = get_obb_long_edge_midline(obb)

        # Midpoints of long edges (top and bottom edges)
        # top edge midpoint: (150, 0), bottom edge midpoint: (150, 100)
        self.assertAlmostEqual(start[0], 150.0, delta=1.0)
        self.assertAlmostEqual(start[1], 0.0, delta=1.0)
        self.assertAlmostEqual(end[0], 150.0, delta=1.0)
        self.assertAlmostEqual(end[1], 100.0, delta=1.0)

    def test_rotated_rectangle(self):
        """45-degree rotated rectangle."""
        # A diamond shape: width >> height in rotated frame
        import math
        angle = math.radians(45)
        c, s = math.cos(angle), math.sin(angle)
        w, h = 200.0, 50.0  # long edge = width direction

        corners = np.array([
            [-w/2 * c + h/2 * s,  -w/2 * s - h/2 * c],
            [ w/2 * c + h/2 * s,   w/2 * s - h/2 * c],
            [ w/2 * c - h/2 * s,   w/2 * s + h/2 * c],
            [-w/2 * c - h/2 * s,  -w/2 * s + h/2 * c],
        ], dtype=np.float32)
        corners += np.array([500, 400], dtype=np.float32)  # translate away from origin

        (start, end) = get_obb_long_edge_midline(corners)

        # The midline should be along the long edge direction (roughly 45 degrees)
        # Both start and end should be valid points
        self.assertIsInstance(start, tuple)
        self.assertIsInstance(end, tuple)
        self.assertEqual(len(start), 2)
        self.assertEqual(len(end), 2)

        # Midline length should be approximately h (50) — the distance between
        # the two long edge midpoints
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        mid_dist = math.hypot(dx, dy)
        self.assertAlmostEqual(mid_dist, h, delta=5.0)

    def test_square(self):
        """Square should not crash (all edges equal, picks first longest pair)."""
        obb = np.array([
            [0, 0],
            [100, 0],
            [100, 100],
            [0, 100],
        ], dtype=np.float32)

        (start, end) = get_obb_long_edge_midline(obd)
        self.assertIsInstance(start, tuple)
        self.assertIsInstance(end, tuple)


class TestDetectPeaksValleys(unittest.TestCase):
    """Tests for detect_peaks_valleys()."""

    def test_simple_peak_valley_pattern(self):
        """Synthetic positive-film profile: peak - valley - peak pattern."""
        np.random.seed(0)
        x = np.linspace(0, 4 * np.pi, 300)
        # Create a "peak-valley-peak" repeating pattern
        profile = np.sin(x) + 0.1 * np.random.randn(300)
        profile = profile.astype(np.float64)

        peaks, valleys = detect_peaks_valleys(profile, min_distance=20, prominence=0.3)

        self.assertGreater(len(peaks), 0, "Should find at least one peak")
        self.assertGreater(len(valleys), 0, "Should find at least one valley")
        # Peaks should have higher values than valleys
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
        # Inverted: valley-peak-valley pattern
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
```

- [ ] **Step 2: Run tests to verify they fail (module not yet created)**

```bash
cd /home/cht/code/IQIdet && conda run -n weld-gpu PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m pytest tests/test_double_wire_profile.py -v 2>&1 | tail -20
```

Expected: ImportError — `gauge.imaging.profile` module not found.

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_double_wire_profile.py
git commit -m "test: add failing tests for double-wire profile band extraction"
```

---

### Task 2: Implement `src/gauge/imaging/profile.py`

**Files:**
- Create: `src/gauge/imaging/profile.py`

- [ ] **Step 1: Create the profile module**

```python
#!/usr/bin/env python3
"""Profile band extraction for double-wire IQI analysis.

Pure functions with no OpenCV HighGUI or matplotlib dependency.
Suitable for integration into the automated pipeline (Phase 2).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


def extract_profile_band(
    image: np.ndarray,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    band_width: int = 21,
    num_samples: Optional[int] = None,
) -> np.ndarray:
    """Extract a band-averaged gray profile along a line, with sub-pixel interpolation.

    Samples ``band_width`` parallel lines centered on the profile midline, then
    averages them column-wise to produce a single 1D profile. This satisfies the
    JBT 7902-2025 requirement of ≥21 rows/columns averaged.

    Args:
        image: Input image (supports uint8, uint16, float32, float64).
        start_point: Midline start coordinate (x, y).
        end_point: Midline end coordinate (x, y).
        band_width: Number of parallel lines to sample and average (default 21).
        num_samples: Number of samples along the midline direction.
            If None, auto-calculated as ``ceil(line_length)``.

    Returns:
        1D float64 array of band-averaged gray values.
    """
    x0, y0 = float(start_point[0]), float(start_point[1])
    x1, y1 = float(end_point[0]), float(end_point[1])

    dx = x1 - x0
    dy = y1 - y0
    line_length = np.hypot(dx, dy)

    if line_length < 1e-6:
        return np.array([], dtype=np.float64)

    if num_samples is None:
        num_samples = max(1, int(np.ceil(line_length)))

    # Unit vectors
    ux = dx / line_length   # along profile direction
    uy = dy / line_length

    # Perpendicular unit vector (rotated 90° CCW)
    px = -uy
    py = ux

    # Generate sampling coordinates for the band
    # shape: (band_width, num_samples, 2) — [row_index, col_index, (y, x)]
    t_vals = np.linspace(0, 1, num_samples)  # along midline
    offsets = np.arange(band_width) - (band_width - 1) / 2.0  # perpendicular offsets

    # Midline coordinates at each t: [num_samples, 2]
    mid_x = x0 + t_vals * dx
    mid_y = y0 + t_vals * dy

    # Full coordinate grid: [band_width, num_samples]
    # y_coords[i, j] = midline_y[j] + offset[i] * py
    # x_coords[i, j] = midline_x[j] + offset[i] * px
    y_coords = mid_y[np.newaxis, :] + offsets[:, np.newaxis] * py
    x_coords = mid_x[np.newaxis, :] + offsets[:, np.newaxis] * px

    # map_coordinates expects (row=y, col=x) order
    coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=0)

    # Sample using spline interpolation (order=1 for bilinear, good speed/quality)
    sampled = map_coordinates(image.astype(np.float64), coords, order=1, mode="nearest")
    sampled = sampled.reshape(band_width, num_samples)

    # Average across band lines
    profile = sampled.mean(axis=0)
    return profile


def get_obb_long_edge_midline(
    obb_points: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute the midline connecting the midpoints of an OBB's two longest edges.

    This midline runs perpendicular to the metal wires in a double-wire IQI and
    serves as the center line for profile band extraction.

    Args:
        obb_points: OBB vertices, shape (4, 2), in counterclockwise order.

    Returns:
        (start_point, end_point): Midline endpoints (x, y) tuples.
    """
    pts = np.asarray(obb_points, dtype=np.float64).reshape(4, 2)

    # Four edges: (0→1), (1→2), (2→3), (3→0)
    edges = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        length = np.linalg.norm(b - a)
        midpoint = (a + b) / 2.0
        edges.append({"length": length, "midpoint": midpoint, "idx": i})

    # Sort by length descending
    edges.sort(key=lambda e: e["length"], reverse=True)

    # The two longest edges are the long edges of the OBB
    # Their midpoints define the midline
    long_edge_a = edges[0]
    long_edge_b = edges[1]

    start = (float(long_edge_a["midpoint"][0]), float(long_edge_a["midpoint"][1]))
    end = (float(long_edge_b["midpoint"][0]), float(long_edge_b["midpoint"][1]))

    return start, end


def detect_peaks_valleys(
    profile: np.ndarray,
    min_distance: int = 10,
    prominence: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peak and valley positions in a 1D gray profile.

    Peaks are detected on the profile directly; valleys are detected by
    inverting the profile and applying the same peak-finding algorithm.

    Args:
        profile: 1D gray profile array.
        min_distance: Minimum pixel distance between adjacent peaks.
        prominence: Minimum peak prominence relative to local baseline.
            Scaled to the profile's dynamic range (max - min).

    Returns:
        (peak_indices, valley_indices): Integer index arrays of peak and valley
        positions, sorted ascending.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if len(profile) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    data_range = float(np.max(profile) - np.min(profile))
    if data_range < 1e-10:
        return np.array([], dtype=int), np.array([], dtype=int)

    abs_prominence = prominence * data_range

    peaks, _ = find_peaks(profile, distance=min_distance, prominence=abs_prominence)
    valleys, _ = find_peaks(-profile, distance=min_distance, prominence=abs_prominence)

    return peaks.astype(int), valleys.astype(int)


# ---------------------------------------------------------------------------
# Phase 2 interface stubs (not implemented in this phase)
# ---------------------------------------------------------------------------


def compute_contrast(
    profile: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
) -> float:
    """Compute Contrast = (a + b - 2c) / (a + b) using the peak-two-valley method.

    Args:
        profile: 1D gray profile.
        peak_indices: Peak position indices.
        valley_indices: Valley position indices.

    Returns:
        Contrast value in [0, 1].

    Note:
        Phase 2 implementation. Must handle:
        - Positive/negative film determination
        - Neighborhood averaging of a, b values
        - Mismatched peak/valley counts
    """
    raise NotImplementedError("Phase 2 implementation")


def find_first_unresolved_group(
    profiles: list,
    threshold: float = 0.2,
) -> Optional[int]:
    """Find the first wire pair group with Contrast < threshold.

    Args:
        profiles: List of 1D gray profiles, one per wire pair group (D1 → Dn).
        threshold: Contrast threshold (default 0.2 = 20%).

    Returns:
        1-indexed group number of the first unresolved group, or None if all
        groups are resolved.

    Note:
        Phase 2 implementation.
    """
    raise NotImplementedError("Phase 2 implementation")
```

- [ ] **Step 2: Run tests**

```bash
cd /home/cht/code/IQIdet && conda run -n weld-gpu PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m pytest tests/test_double_wire_profile.py -v 2>&1
```

Expected: All tests PASS (or most — fix any issues).

- [ ] **Step 3: Fix test typo and re-run**

Note: The test in Task 1 has a typo — `get_obb_long_edge_midline(obd)` in `test_square` should be `get_obb_long_edge_midline(obb)`. Fix this after tests reveal the NameError.

```bash
cd /home/cht/code/IQIdet && sed -i 's/get_obb_long_edge_midline(obd)/get_obb_long_edge_midline(obb)/' tests/test_double_wire_profile.py && conda run -n weld-gpu PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m pytest tests/test_double_wire_profile.py -v 2>&1 | tail -30
```

Expected: All 16 tests PASS.

- [ ] **Step 4: Syntax check**

```bash
cd /home/cht/code/IQIdet && conda run -n weld-gpu python -m py_compile src/gauge/imaging/profile.py
```

Expected: No output (success).

- [ ] **Step 5: Commit**

```bash
git add src/gauge/imaging/profile.py tests/test_double_wire_profile.py
git commit -m "feat: add profile band extraction module for double-wire IQI

- extract_profile_band(): band-averaged profile with sub-pixel interpolation
- get_obb_long_edge_midline(): OBB long-edge midline computation
- detect_peaks_valleys(): peak/valley detection via scipy.signal.find_peaks
- Phase 2 stubs: compute_contrast(), find_first_unresolved_group()
- 16 unit tests covering band extraction, geometry, and peak detection

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Implement interactive demo script

**Files:**
- Create: `scripts/debug/double_wire_demo.py`

- [ ] **Step 1: Create the demo script**

```python
#!/usr/bin/env python3
"""Interactive double-wire IQI profile band visualization tool.

Usage:
    double_wire_demo.py <image_path> [options]
    double_wire_demo.py (-h | --help)

Arguments:
    <image_path>              双丝像质计图像路径（支持 16-bit TIFF / 8-bit JPEG）

Options:
    -h --help                 显示帮助信息
    --output-dir <dir>        输出目录 [default: outputs/double_wire_demo]
    --window-size <size>      显示窗口最大尺寸 [default: 1200]
    --band-width <N>          剖面带平行线数量（默认 21，符合 JBT 7902-2025） [default: 21]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("TkAgg")  # Non-blocking interactive backend
import matplotlib.pyplot as plt
from docopt import docopt
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from gauge.imaging.profile import (
    extract_profile_band,
    get_obb_long_edge_midline,
    detect_peaks_valleys,
)

# ── Color constants (BGR for OpenCV) ──
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_SEMI_RED = (0, 0, 255, 80)  # BGRA for overlay

# ── Profile curve color (hex for matplotlib) ──
PROFILE_COLOR = "#4C78A8"
PEAK_COLOR = "red"
VALLEY_COLOR = "blue"


class DoubleWireDemo:
    """Interactive double-wire IQI profile band visualization tool.

    State machine: IDLE → COLLECTING → LOCKED → (R key) → IDLE
    """

    STATE_IDLE = "idle"
    STATE_COLLECTING = "collecting"
    STATE_LOCKED = "locked"

    def __init__(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        window_size: int = 1200,
        band_width: int = 21,
    ):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir) if output_dir else None
        self.window_size = int(window_size)
        self.band_width = int(band_width)
        if self.band_width < 1:
            raise ValueError(f"band_width must be >= 1, got {self.band_width}")

        # Image data
        self.image_raw: Optional[np.ndarray] = None
        self.image_display: Optional[np.ndarray] = None

        # Interaction state
        self.state: str = self.STATE_IDLE
        self.obb_points: list = []  # list of (x, y) tuples

        # Profile parameters
        self.profile_offset_pct: int = 50  # 0–100
        self.film_type: str = "positive"  # "positive" | "negative"
        self.show_help: bool = False

        # Profile data (computed in LOCKED state)
        self.profile_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.profile: Optional[np.ndarray] = None
        self.peak_indices: Optional[np.ndarray] = None
        self.valley_indices: Optional[np.ndarray] = None

        # OpenCV / matplotlib handles
        self.window_name = "Double Wire Demo"
        self.trackbar_name = "offset%"
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    # ── Image Loading ────────────────────────────────────────────────

    def load_image(self) -> None:
        """Load image, handling 16-bit TIFF and 8-bit JPEG."""
        raw = cv2.imread(str(self.image_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Failed to read image: {self.image_path}")

        # Convert color to grayscale if needed
        if raw.ndim == 3:
            self.image_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            self.image_raw = raw

        # Build display image (8-bit normalized)
        if self.image_raw.dtype == np.uint16:
            self.image_display = cv2.normalize(
                self.image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
        elif self.image_raw.dtype == np.uint8:
            self.image_display = self.image_raw.copy()
        else:
            self.image_display = cv2.normalize(
                self.image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

        # Resize for display if needed
        h, w = self.image_display.shape[:2]
        long_side = max(h, w)
        if long_side > self.window_size:
            scale = self.window_size / long_side
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            self.image_display = cv2.resize(self.image_display, (new_w, new_h))

        # Convert to BGR for OpenCV colored overlay
        self.image_display = cv2.cvtColor(self.image_display, cv2.COLOR_GRAY2BGR)

    # ── Mouse Callback ───────────────────────────────────────────────

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events for OBB vertex collection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state in (self.STATE_IDLE, self.STATE_COLLECTING):
                self.obb_points.append((float(x), float(y)))
                if self.state == self.STATE_IDLE:
                    self.state = self.STATE_COLLECTING
                if len(self.obb_points) >= 4:
                    # Auto-lock on 4th point
                    self.state = self.STATE_LOCKED
                    self.update_profile()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.state == self.STATE_COLLECTING and self.obb_points:
                self.obb_points.pop()
                if not self.obb_points:
                    self.state = self.STATE_IDLE

    # ── Trackbar Callback ────────────────────────────────────────────

    def on_trackbar(self, value: int) -> None:
        """Trackbar callback: update offset and recompute profile."""
        self.profile_offset_pct = value
        if self.state == self.STATE_LOCKED and len(self.obb_points) >= 4:
            self.update_profile()

    # ── Profile Update ───────────────────────────────────────────────

    def update_profile(self) -> None:
        """Compute profile band from OBB and update matplotlib plot."""
        if len(self.obb_points) < 4:
            return

        obb = np.array(self.obb_points, dtype=np.float32)
        midline = get_obb_long_edge_midline(obb)

        # Apply offset along the short edge
        start, end = midline
        sx, sy = start
        ex, ey = end

        # Perpendicular unit vector
        dx = ex - sx
        dy = ey - sy
        length = np.hypot(dx, dy)
        if length < 1e-6:
            return
        px = -dy / length
        py = dx / length

        # Offset as fraction of OBB short edge
        # Estimate short edge length from the two short edges of OBB
        short_len = 0.0
        for i in range(4):
            a = obb[i]
            b = obb[(i + 1) % 4]
            edge_len = float(np.linalg.norm(b - a))
            if i == 0 or i == 2:
                pass  # could compute precisely, but midpoint method is simpler
        # Use a simpler approach: offset by pixel-proportional amount
        # The trackbar 0-100 maps to the full short-edge range
        # Approximate short edge as the edge between p0-p3 or p1-p2
        short_edge_0 = float(np.linalg.norm(obb[0] - obb[3]))
        short_edge_1 = float(np.linalg.norm(obb[1] - obb[2]))
        short_side = (short_edge_0 + short_edge_1) / 2.0

        # offset from center: -0.5*short_side to +0.5*short_side
        offset_frac = (self.profile_offset_pct - 50) / 50.0  # -1.0 to +1.0
        offset_amount = offset_frac * short_side * 0.45  # 0.45 margin from edges

        sx_off = sx + offset_amount * px
        sy_off = sy + offset_amount * py
        ex_off = ex + offset_amount * px
        ey_off = ey + offset_amount * py

        self.profile_line = ((sx_off, sy_off), (ex_off, ey_off))

        # Extract profile band
        self.profile = extract_profile_band(
            self.image_raw,
            self.profile_line[0],
            self.profile_line[1],
            band_width=self.band_width,
        )

        # Detect peaks/valleys
        self.peak_indices, self.valley_indices = detect_peaks_valleys(
            self.profile, min_distance=10, prominence=0.05
        )

        # Update matplotlib
        self.plot_profile()

    # ── Overlay Drawing ──────────────────────────────────────────────

    def draw_overlay(self) -> np.ndarray:
        """Render overlay annotations onto the display image."""
        vis = self.image_display.copy()

        # Draw OBB vertices and edges based on state
        if len(self.obb_points) >= 1:
            pts_int = [(int(x), int(y)) for x, y in self.obb_points]
            # Draw vertices
            for pt in pts_int:
                cv2.circle(vis, pt, 5, COLOR_YELLOW, -1, cv2.LINE_AA)
            # Draw edges between consecutive points
            for i in range(len(pts_int) - 1):
                cv2.line(vis, pts_int[i], pts_int[i + 1], COLOR_YELLOW, 1, cv2.LINE_AA)

        if self.state == self.STATE_LOCKED and len(self.obb_points) >= 4:
            pts_int = [(int(x), int(y)) for x, y in self.obb_points]
            # Close OBB
            cv2.polylines(vis, [np.array(pts_int)], isClosed=True, color=COLOR_GREEN, thickness=2, lineType=cv2.LINE_AA)

            # Draw profile band
            if self.profile_line is not None:
                (sx, sy), (ex, ey) = self.profile_line
                dx = ex - sx
                dy = ey - sy
                length = np.hypot(dx, dy)
                if length > 1e-6:
                    px = -dy / length
                    py = dx / length
                    half_band = (self.band_width - 1) / 2.0

                    # Band edges
                    s1 = (int(sx + half_band * px), int(sy + half_band * py))
                    e1 = (int(ex + half_band * px), int(ey + half_band * py))
                    s2 = (int(sx - half_band * px), int(sy - half_band * py))
                    e2 = (int(ex - half_band * px), int(ey - half_band * py))

                    # Semi-transparent band fill
                    overlay = vis.copy()
                    band_pts = np.array([s1, e1, e2, s2], dtype=np.int32)
                    cv2.fillPoly(overlay, [band_pts], (0, 0, 255))
                    vis = cv2.addWeighted(overlay, 0.2, vis, 0.8, 0)

                    # Band boundary dashed lines
                    cv2.line(vis, s1, e1, COLOR_RED, 1, cv2.LINE_AA)
                    cv2.line(vis, s2, e2, COLOR_RED, 1, cv2.LINE_AA)

                    # Midline solid
                    mid_s = (int(sx), int(sy))
                    mid_e = (int(ex), int(ey))
                    cv2.line(vis, mid_s, mid_e, COLOR_RED, 1, cv2.LINE_AA)

        # Status text (bottom-left)
        status_map = {
            self.STATE_IDLE: "Ready",
            self.STATE_COLLECTING: f"Points {len(self.obb_points)}/4",
            self.STATE_LOCKED: f"Locked | offset={self.profile_offset_pct}% | band={self.band_width} | {self.film_type}",
        }
        status = status_map.get(self.state, "")
        cv2.putText(vis, status, (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, COLOR_WHITE, 1, cv2.LINE_AA)

        # Help overlay
        if self.show_help:
            help_lines = [
                "L-click: add OBB vertex (4 to lock)",
                "R-click: undo last vertex",
                "R: reset OBB   S: save   Q/ESC: quit",
                "F: toggle pos/neg film   H: hide help",
                f"Trackbar: adjust profile offset | band_width={self.band_width}",
            ]
            overlay = vis.copy()
            h, w = overlay.shape[:2]
            # Semi-transparent panel
            panel_h = 20 * len(help_lines) + 20
            cv2.rectangle(overlay, (10, 30), (min(520, w - 10), 30 + panel_h),
                          COLOR_BLACK, -1)
            vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)
            for i, line in enumerate(help_lines):
                cv2.putText(vis, line, (20, 55 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, COLOR_WHITE, 1, cv2.LINE_AA)

        return vis

    # ── Matplotlib Plot ──────────────────────────────────────────────

    def plot_profile(self) -> None:
        """Clear and redraw the matplotlib profile curve with peak/valley markers."""
        if self.profile is None:
            return

        self.ax.clear()
        x = np.arange(len(self.profile))

        # Gray profile curve
        self.ax.plot(x, self.profile, color=PROFILE_COLOR, linewidth=1.2, label="Profile")

        # Normalize to [0, 1] for consistent prominence computation
        prof_range = self.profile.max() - self.profile.min()
        y_min = self.profile.min() - 0.05 * max(prof_range, 1.0)
        y_max = self.profile.max() + 0.05 * max(prof_range, 1.0)

        # Peak markers (red triangles)
        if self.peak_indices is not None and len(self.peak_indices) > 0:
            peaks = self.peak_indices
            self.ax.plot(peaks, self.profile[peaks], "r^", markersize=8,
                         label="Peaks", markeredgewidth=1, markeredgecolor="darkred")
            for p in peaks:
                val = self.profile[p]
                self.ax.annotate(f"{val:.2f}", (p, val), textcoords="offset points",
                                 xytext=(0, 8), fontsize=7, color="red", ha="center")

        # Valley markers (blue inverted triangles)
        if self.valley_indices is not None and len(self.valley_indices) > 0:
            valleys = self.valley_indices
            self.ax.plot(valleys, self.profile[valleys], "bv", markersize=8,
                         label="Valleys", markeredgewidth=1, markeredgecolor="darkblue")
            for v in valleys:
                val = self.profile[v]
                self.ax.annotate(f"{val:.2f}", (v, val), textcoords="offset points",
                                 xytext=(0, -12), fontsize=7, color="blue", ha="center")

        # Compute OBB stats for title
        if len(self.obb_points) >= 4:
            obb = np.array(self.obb_points, dtype=np.float32)
            e0 = float(np.linalg.norm(obb[0] - obb[1]))
            e1 = float(np.linalg.norm(obb[1] - obb[2]))
            w = max(e0, e1)
            h = min(e0, e1)
            # Angle of long edge relative to horizontal
            if e0 >= e1:
                angle = np.degrees(np.arctan2(obb[1][1] - obb[0][1], obb[1][0] - obb[0][0]))
            else:
                angle = np.degrees(np.arctan2(obb[2][1] - obb[1][1], obb[2][0] - obb[1][0]))
        else:
            w, h, angle = 0, 0, 0.0

        stem = self.image_path.stem
        title = (
            f"{stem} | OBB: {w:.0f}x{h:.0f} @ {angle:.1f}deg"
            f" | offset:{self.profile_offset_pct}%"
            f" | band:{self.band_width}"
            f" | {self.film_type}"
        )
        self.ax.set_title(title, fontsize=9)
        self.ax.set_xlabel("Pixel position along profile")
        self.ax.set_ylabel("Gray value (raw)")
        self.ax.set_ylim(y_min, y_max)
        self.ax.legend(fontsize=7, loc="upper right")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ── Save ─────────────────────────────────────────────────────────

    def save_results(self) -> None:
        """Save overlay image and profile JSON to output directory."""
        if self.output_dir is None:
            print("[save] No output directory configured. Skipping.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem

        # Save overlay image
        overlay_path = self.output_dir / f"{stem}_overlay.png"
        overlay = self.draw_overlay()
        cv2.imwrite(str(overlay_path), overlay)
        print(f"[save] Overlay: {overlay_path}")

        # Save profile JSON
        json_path = self.output_dir / f"{stem}_profile.json"
        payload = {
            "image_path": str(self.image_path),
            "obb_points": [[float(x), float(y)] for x, y in self.obb_points],
            "profile_midline": {
                "start": list(self.profile_line[0]) if self.profile_line else None,
                "end": list(self.profile_line[1]) if self.profile_line else None,
            },
            "band_width": self.band_width,
            "profile_offset_pct": self.profile_offset_pct,
            "film_type": self.film_type,
            "profile_values": self.profile.tolist() if self.profile is not None else [],
            "peak_indices": self.peak_indices.tolist() if self.peak_indices is not None else [],
            "valley_indices": self.valley_indices.tolist() if self.valley_indices is not None else [],
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[save] Profile data: {json_path}")

    # ── Main Loop ────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the interactive demo."""
        self.load_image()

        # Setup OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.createTrackbar(self.trackbar_name, self.window_name, 50, 100, self.on_trackbar)

        # Setup matplotlib figure (non-blocking interactive mode)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_xlabel("Pixel position along profile")
        self.ax.set_ylabel("Gray value (raw)")
        self.fig.tight_layout()

        print(f"[demo] Loaded: {self.image_path.name}")
        print(f"[demo] Shape: {self.image_raw.shape}, dtype: {self.image_raw.dtype}")
        print(f"[demo] band_width: {self.band_width}")
        print("[demo] Keys: L-click=add point, R-click=undo, R=reset, S=save, Q=quit, F=flip film, H=help")

        while True:
            overlay = self.draw_overlay()
            cv2.imshow(self.window_name, overlay)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("r"):
                self.state = self.STATE_IDLE
                self.obb_points.clear()
                self.profile_line = None
                self.profile = None
                self.peak_indices = None
                self.valley_indices = None
                cv2.setTrackbarPos(self.trackbar_name, self.window_name, 50)
                self.profile_offset_pct = 50
                if self.ax is not None:
                    self.ax.clear()
                    self.fig.canvas.draw()
                print("[demo] Reset")

            elif key == ord("s"):
                if self.state == self.STATE_LOCKED:
                    self.save_results()
                else:
                    print("[demo] Lock OBB first (complete 4 points) before saving")

            elif key == ord("q") or key == 27:  # ESC
                print("[demo] Quit")
                break

            elif key == ord("h"):
                self.show_help = not self.show_help

            elif key == ord("f"):
                if self.state == self.STATE_LOCKED:
                    self.film_type = "negative" if self.film_type == "positive" else "positive"
                    self.update_profile()
                    print(f"[demo] Film type: {self.film_type}")

        cv2.destroyAllWindows()
        plt.close("all")


# ── CLI Entry Point ──────────────────────────────────────────────────

def main() -> None:
    args = docopt(__doc__)

    image_path = args["<image_path>"]
    output_dir = args["--output-dir"]
    window_size = int(args["--window-size"])
    band_width = int(args["--band-width"])

    if not Path(image_path).is_file():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    demo = DoubleWireDemo(
        image_path=image_path,
        output_dir=output_dir,
        window_size=window_size,
        band_width=band_width,
    )
    demo.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

```bash
cd /home/cht/code/IQIdet && conda run -n weld-gpu python -m py_compile scripts/debug/double_wire_demo.py
```

Expected: No output (success).

- [ ] **Step 3: Smoke test — run with a sample image (non-headless)**

```bash
cd /home/cht/code/IQIdet && timeout 5 conda run -n weld-gpu python scripts/debug/double_wire_demo.py "outputs/候选双丝像质计/t2-3hj__3ZF1-118-3.jpg" --output-dir /tmp/dw_test --band-width 21 2>&1 || true
```

Expected: Program starts, loads image, prints `[demo] Loaded: ...`, then exits after 5s timeout without crashing.

- [ ] **Step 4: Commit**

```bash
git add scripts/debug/double_wire_demo.py
git commit -m "feat: add interactive double-wire profile band demo tool

Interactive visualization of double-wire IQI profile bands:
- 4-point OBB selection with right-click undo
- Band-averaged profile extraction (band_width default 21 per JBT 7902-2025)
- Trackbar slider for profile band offset (0%-100%)
- F-key positive/negative film toggle
- H-key help overlay
- S-key save: overlay PNG + profile JSON
- matplotlib profile curve with peak/valley annotation
- docopt CLI with --band-width and --output-dir options

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Final verification

- [ ] **Step 1: Run all tests one more time**

```bash
cd /home/cht/code/IQIdet && conda run -n weld-gpu PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m pytest tests/test_double_wire_profile.py -v 2>&1
```

Expected: All 16 tests pass.

- [ ] **Step 2: Verify CLI help text**

```bash
cd /home/cht/code/IQIdet && conda run -n weld-gpu python scripts/debug/double_wire_demo.py --help 2>&1
```

Expected: Prints usage documentation with `--band-width` option visible.

- [ ] **Step 3: Quick test of extract_profile_band against real image**

```bash
cd /home/cht/code/IQIdet && conda run -n weld-gpu python -c "
import sys
sys.path.insert(0, 'src')
from gauge.imaging.profile import extract_profile_band, detect_peaks_valleys
import cv2
img = cv2.imread('outputs/候选双丝像质计/t2-3hj__3ZF1-118-3.jpg', cv2.IMREAD_GRAYSCALE)
print(f'Image: {img.shape}, dtype: {img.dtype}')
# Extract a horizontal band at center
h, w = img.shape
profile = extract_profile_band(img, (0, h//2), (w-1, h//2), band_width=21)
peaks, valleys = detect_peaks_valleys(profile)
print(f'Profile length: {len(profile)}, peaks: {len(peaks)}, valleys: {len(valleys)}')
print(f'Profile min/max: {profile.min():.1f} / {profile.max():.1f}')
print('OK')
" 2>&1
```

Expected: Prints profile stats and "OK" without error.

- [ ] **Step 4: Final commit if any fixes needed, otherwise done**

```bash
git status
```
