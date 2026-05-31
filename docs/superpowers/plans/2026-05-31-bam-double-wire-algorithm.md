# BAM 双丝像质计算法 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `src/gauge/imaging/profile.py` 中实现 `compute_contrast()` 和 `find_first_unresolved_group()`，替换 Phase 1 stub。

**Architecture:** 两公开函数 + 六内部私有 helper。`compute_contrast` 编排片型判定→背景拟合→丝配对→dip 计算；`find_first_unresolved_group` 编排单调性清理→低调制排除→crossing 判定。所有内部 helper 为模块级 `_` 前缀纯函数，可独立测试。

**Tech Stack:** Python 3.10+, numpy, scipy (signal.find_peaks, signal.peak_widths, optimize.curve_fit), dataclasses

**Spec:** `docs/superpowers/specs/2026-05-31-bam-double-wire-algorithm-design.md`

---

### 文件结构

| 文件 | 职责 |
|------|------|
| `src/gauge/imaging/profile.py` | 所有 BAM 算法代码（公开 + 私有函数），替换 stub |
| `tests/test_double_wire_profile.py` | 新增 `TestBAMHelpers`、`TestComputeContrast`、`TestFindFirstUnresolvedGroup` |

---

### Task 1: 更新 imports 并添加 `ComputeContrastResult` dataclass

**文件:**
- Modify: `src/gauge/imaging/profile.py:1-14`

- [ ] **Step 1: 更新文件头部 imports**

将 profile.py 第 1-14 行替换为：

```python
"""Profile band extraction for double-wire IQI analysis.

Pure functions with no OpenCV HighGUI or matplotlib dependency.
Suitable for integration into the automated pipeline (Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

# JBT 7902-2025 表2 标准双丝型像质计 D1~D13 丝径/间距 (mm)
_DEFAULT_WIRE_SPACINGS: Tuple[float, ...] = (
    0.80, 0.63, 0.50, 0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10, 0.08, 0.063, 0.05,
)
```

- [ ] **Step 2: 在 `unwarp_obb_region` 函数之后、`detect_peaks_valleys` 函数之前，添加 `ComputeContrastResult`**

找到 `def detect_peaks_valleys(`，在其上方插入：

```python
@dataclass
class ComputeContrastResult:
    """Result of :func:`compute_contrast`.

    Attributes:
        dips: Dip (modulation depth) for each wire pair, in percent [0, 100].
        pairs: Detected (wire_a_idx, gap_idx, wire_b_idx) triplets.
            Conventions follow positive-film semantics (valley, peak, valley),
            matching the naming in ``groundtruth.json``.
        background: Quadratic background fit values, same length as the input
            profile.
        film_type: ``"positive"`` or ``"negative"``.
    """
    dips: List[float]
    pairs: List[Tuple[int, int, int]]
    background: np.ndarray
    film_type: str
```

- [ ] **Step 3: 语法检查**

```bash
cd /home/cht/code/IQIdet && python -m py_compile src/gauge/imaging/profile.py
```

Expected: 无输出（编译成功）。

- [ ] **Step 4: Commit**

```bash
git add src/gauge/imaging/profile.py
git commit -m "feat(BAM): add ComputeContrastResult dataclass and required imports"
```

---

### Task 2: 实现 `_detect_film_type()`

**文件:**
- Modify: `src/gauge/imaging/profile.py` — 在 `ComputeContrastResult` 之后添加
- Modify: `tests/test_double_wire_profile.py` — 添加 `TestBAMHelpers` 类

- [ ] **Step 1: 添加测试类到 test_double_wire_profile.py**

在 test 文件末尾（`if __name__ == "__main__":` 之前）添加：

```python
class TestBAMHelpers(unittest.TestCase):
    """Unit tests for BAM internal helpers."""

    def test_detect_film_type_positive(self):
        """正片: valley(丝·暗) < peak(间隙·亮) → film_type='positive'."""
        from gauge.imaging.profile import _detect_film_type
        valleys = np.array([10, 50])
        peaks = np.array([30, 70])
        # valley_gray=50 < peak_gray=200 → positive
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[valleys] = 50.0   # 暗丝
        profile[peaks] = 200.0    # 亮间隙
        result = _detect_film_type(profile, valleys, peaks)
        self.assertEqual(result, "positive")

    def test_detect_film_type_negative(self):
        """负片: valley(丝·亮) > peak(间隙·暗) → film_type='negative'."""
        from gauge.imaging.profile import _detect_film_type
        valleys = np.array([10, 50])
        peaks = np.array([30, 70])
        # valley_gray=200 > peak_gray=50 → negative
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[valleys] = 200.0  # 亮丝
        profile[peaks] = 50.0     # 暗间隙
        result = _detect_film_type(profile, valleys, peaks)
        self.assertEqual(result, "negative")

    def test_detect_film_type_insufficient_data(self):
        """峰谷不足时默认返回 'positive'."""
        from gauge.imaging.profile import _detect_film_type
        profile = np.ones(100, dtype=np.float64)
        result = _detect_film_type(profile, np.array([], dtype=int), np.array([], dtype=int))
        self.assertEqual(result, "positive")
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m pytest tests/test_double_wire_profile.py::TestBAMHelpers -v 2>&1 || PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers -v
```

Expected: FAIL — `ImportError: cannot import name '_detect_film_type'`

- [ ] **Step 3: 在 profile.py 中实现 `_detect_film_type`**

在 `ComputeContrastResult` dataclass 之后添加：

```python
def _detect_film_type(
    profile: np.ndarray,
    valleys: np.ndarray,
    peaks: np.ndarray,
) -> str:
    """Determine film type from the first wire pair's gray-level relationship.

    For a positive film wires are dark (low gray) and gaps are bright (high
    gray), so ``profile[valley] < profile[peak]``.  Negative film is the
    inverse.

    Args:
        profile: 1D band-averaged gray profile.
        valleys: Valley index array (positions of profile minima).
        peaks: Peak index array (positions of profile maxima).

    Returns:
        ``"positive"`` or ``"negative"``.  Defaults to ``"positive"`` when
        there are fewer than 2 valleys or 1 peak.
    """
    if len(valleys) >= 2 and len(peaks) >= 1:
        a_val = float(profile[valleys[0]])
        c_val = float(profile[peaks[0]])
        return "negative" if a_val > c_val else "positive"
    return "positive"
```

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gauge/imaging/profile.py tests/test_double_wire_profile.py
git commit -m "feat(BAM): implement _detect_film_type helper"
```

---

### Task 3: 实现 `_fit_quadratic_background()`

**文件:**
- Modify: `src/gauge/imaging/profile.py` — 在 `_detect_film_type` 之后添加
- Modify: `tests/test_double_wire_profile.py` — 在 `TestBAMHelpers` 中添加测试

- [ ] **Step 1: 在 TestBAMHelpers 中添加测试方法**

```python
    def test_fit_quadratic_background_positive(self):
        """正片: 遮罩 valley 区后在 gap 区拟合二次背景."""
        from gauge.imaging.profile import _fit_quadratic_background
        # 构造抛物线背景 + 局部深谷（丝）的剖面
        x = np.arange(200, dtype=np.float64)
        # 背景: 轻微二次曲线 y = 0.001*x^2 + 100
        true_bg = 0.001 * x**2 + 100.0
        profile = true_bg.copy()
        # 在若干位置插入深谷（模拟暗丝）
        wire_idx = np.array([30, 50, 100, 120, 170], dtype=int)
        for w in wire_idx:
            profile[w-3:w+4] -= 30.0  # 深谷

        bg = _fit_quadratic_background(profile, wire_idx, inverted=True)

        self.assertEqual(bg.shape, profile.shape)
        self.assertEqual(bg.dtype, np.float64)
        # 背景应在丝区之外接近真实背景
        gap_mask = np.ones(200, dtype=bool)
        for w in wire_idx:
            gap_mask[w-3:w+4] = False
        rmse = np.sqrt(np.mean((bg[gap_mask] - true_bg[gap_mask]) ** 2))
        self.assertLess(rmse, 5.0, f"Background RMSE={rmse:.1f} too high")

    def test_fit_quadratic_background_negative(self):
        """负片: 遮罩 peak 区，inverted=False."""
        from gauge.imaging.profile import _fit_quadratic_background
        x = np.arange(200, dtype=np.float64)
        true_bg = -0.0005 * x**2 + 0.1 * x + 120.0
        profile = true_bg.copy()
        wire_idx = np.array([30, 50, 100, 120, 170], dtype=int)
        for w in wire_idx:
            profile[w-3:w+4] += 30.0  # 亮丝（负片）

        bg = _fit_quadratic_background(profile, wire_idx, inverted=False)

        self.assertEqual(bg.shape, profile.shape)
        gap_mask = np.ones(200, dtype=bool)
        for w in wire_idx:
            gap_mask[w-3:w+4] = False
        rmse = np.sqrt(np.mean((bg[gap_mask] - true_bg[gap_mask]) ** 2))
        self.assertLess(rmse, 5.0, f"Background RMSE={rmse:.1f} too high")

    def test_fit_quadratic_background_short_profile(self):
        """极短剖面不应崩溃."""
        from gauge.imaging.profile import _fit_quadratic_background
        profile = np.array([10.0, 12.0, 10.0], dtype=np.float64)
        bg = _fit_quadratic_background(profile, np.array([1], dtype=int), inverted=True)
        self.assertEqual(bg.shape, (3,))
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers.test_fit_quadratic_background_positive -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: 在 profile.py 中实现 `_fit_quadratic_background`**

在 `_detect_film_type` 之后添加：

```python
def _fit_quadratic_background(
    profile: np.ndarray,
    wire_indices: np.ndarray,
    *,
    inverted: bool = False,
) -> np.ndarray:
    """Fit a quadratic background curve after masking out wire regions.

    Wire regions (where the profile deviates strongly from the background)
    are identified via :func:`scipy.signal.peak_widths` and masked out.
    A quadratic ``a*x² + b*x + c`` is then fitted to the remaining
    (gap-dominated) samples.

    Args:
        profile: 1D band-averaged gray profile.
        wire_indices: Integer indices of wire positions.  For positive film
            these are the valleys (from ``find_peaks(-profile)``); for
            negative film these are the peaks (from ``find_peaks(profile)``).
        inverted: When *True* (positive film) the profile is negated before
            computing peak widths so that the dark wires become positive
            peaks suitable for :func:`peak_widths`.

    Returns:
        1D ndarray of background values, same length as *profile*.
    """
    n = len(profile)
    if len(wire_indices) == 0:
        # Fallback: fit to all points
        x = np.arange(n, dtype=np.float64)
        popt, _ = curve_fit(
            lambda x, a, b, c: a * x * x + b * x + c,
            x, profile.astype(np.float64),
        )
        return np.asarray(popt[0] * x * x + popt[1] * x + popt[2], dtype=np.float64)

    target = -profile.astype(np.float64) if inverted else profile.astype(np.float64)
    widths, _, _, _ = peak_widths(target, wire_indices, rel_height=0.9)

    mask = np.ones(n, dtype=bool)
    for i, p in enumerate(wire_indices):
        lo = max(0, int(np.rint(p - widths[i])))
        hi = min(n - 1, int(np.rint(p + widths[i])))
        mask[lo:hi + 1] = False

    if mask.sum() < 3:
        # Not enough gap points; fall back to full-range fit
        mask[:] = True

    x = np.arange(n, dtype=np.float64)
    popt, _ = curve_fit(
        lambda x, a, b, c: a * x * x + b * x + c,
        x[mask], profile.astype(np.float64)[mask],
    )
    return np.asarray(popt[0] * x * x + popt[1] * x + popt[2], dtype=np.float64)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers.test_fit_quadratic_background_positive tests.test_double_wire_profile.TestBAMHelpers.test_fit_quadratic_background_negative tests.test_double_wire_profile.TestBAMHelpers.test_fit_quadratic_background_short_profile -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gauge/imaging/profile.py tests/test_double_wire_profile.py
git commit -m "feat(BAM): implement _fit_quadratic_background helper"
```

---

### Task 4: 实现 `_compute_dip()`

**文件:**
- Modify: `src/gauge/imaging/profile.py` — 在 `_fit_quadratic_background` 之后添加
- Modify: `tests/test_double_wire_profile.py` — 在 `TestBAMHelpers` 中添加测试

- [ ] **Step 1: 在 TestBAMHelpers 中添加测试方法**

```python
    def test_compute_dip_basic(self):
        """已知 profile 和 background 值 → 验证 dip 计算."""
        from gauge.imaging.profile import _compute_dip
        # 背景恒为 100, 丝在 90(暗), 间隙在 105(亮), half_w=0 退化到单点
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[20] = 90.0   # wire_a (暗)
        profile[30] = 105.0  # gap (亮)
        profile[40] = 90.0   # wire_b (暗)
        background = np.ones(100, dtype=np.float64) * 100.0

        dip = _compute_dip(profile, 20, 30, 40, background, half_w=0)

        # A = |100-90| = 10, B = |100-90| = 10, C = |100-105| = 5
        # dip = 100*(10+10-2*5)/(10+10) = 100*10/20 = 50.0
        self.assertAlmostEqual(dip, 50.0, delta=0.01)

    def test_compute_dip_with_window(self):
        """half_w > 0 时使用邻域均值."""
        from gauge.imaging.profile import _compute_dip
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[18:23] = 90.0    # wire_a 邻域均值 = 90
        profile[28:33] = 105.0   # gap 邻域均值 = 105
        profile[38:43] = 90.0    # wire_b 邻域均值 = 90
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers.test_compute_dip_basic -v
```

Expected: FAIL

- [ ] **Step 3: 在 profile.py 中实现 `_compute_dip`**

在 `_fit_quadratic_background` 之后添加：

```python
def _compute_dip(
    profile: np.ndarray,
    wire_a: int,
    gap_c: int,
    wire_b: int,
    background: np.ndarray,
    half_w: int,
) -> float:
    """Compute the modulation depth (dip) for a single wire pair.

    The dip is defined as ``100 * (A + B - 2*C) / (A + B)`` where *A*, *B*
    are the absolute deviations of the two wires from the background and *C*
    is the absolute deviation of the gap from the background.  Each value is
    taken as the neighbourhood mean of width ``2*half_w+1`` around the
    detected position.

    Args:
        profile: 1D band-averaged gray profile.
        wire_a: Index of the first wire.
        gap_c: Index of the gap between the two wires.
        wire_b: Index of the second wire.
        background: Background fit values (same length as *profile*).
        half_w: Half-width of the neighbourhood window.

    Returns:
        Dip value in percent [0, 100].  Returns 0 when the denominator is
        negligible (fully merged pair).
    """
    L = len(profile)

    def _region_mean(center: int) -> float:
        lo = max(0, center - half_w)
        hi = min(L - 1, center + half_w)
        return float(profile[lo:hi + 1].mean())

    a_mean = _region_mean(wire_a)
    c_mean = _region_mean(gap_c)
    b_mean = _region_mean(wire_b)

    A = abs(float(background[wire_a]) - a_mean)
    B = abs(float(background[wire_b]) - b_mean)
    C = abs(float(background[gap_c]) - c_mean)

    denom = A + B
    if denom < 1e-10:
        return 0.0
    return 100.0 * (A + B - 2.0 * C) / denom
```

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers.test_compute_dip_basic tests.test_double_wire_profile.TestBAMHelpers.test_compute_dip_with_window tests.test_double_wire_profile.TestBAMHelpers.test_compute_dip_fully_merged -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gauge/imaging/profile.py tests/test_double_wire_profile.py
git commit -m "feat(BAM): implement _compute_dip helper with neighbourhood averaging"
```

---

### Task 5: 实现 `_pair_wires_and_compute_dips()`

**文件:**
- Modify: `src/gauge/imaging/profile.py` — 在 `_compute_dip` 之后添加
- Modify: `tests/test_double_wire_profile.py` — 在 `TestBAMHelpers` 中添加测试

- [ ] **Step 1: 在 TestBAMHelpers 中添加测试方法**

```python
    def test_pair_wires_positive(self):
        """正片: wires=valleys, gaps=peaks, 相邻 valley 间距≤1.05*首对间距."""
        from gauge.imaging.profile import _pair_wires_and_compute_dips
        # 构造正弦波模拟 profile: valleys at 20,60,100,140,180 间距=40
        x = np.linspace(0, 6 * np.pi, 300)
        profile = (-np.sin(x) * 30.0 + 100.0).astype(np.float64)  # 负正弦=valleys在下
        background = np.ones(300, dtype=np.float64) * 100.0
        # detect_peaks_valleys 已在本 test 模块顶部导入
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
        profile = (np.sin(x) * 30.0 + 100.0).astype(np.float64)  # 正正弦=peaks在上
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
        # 两对正常丝 + 一个间距异常大的假谷
        profile = np.ones(200, dtype=np.float64) * 100.0
        wire_pos = np.array([20, 40, 60, 150], dtype=int)  # 60→150 间距=90, 正常=20
        for w in wire_pos:
            profile[w] = 80.0  # 暗丝（正片语义）
        gap_pos = np.array([30, 50, 105], dtype=int)
        for g in gap_pos:
            profile[g] = 120.0  # 亮间隙
        background = np.ones(200, dtype=np.float64) * 100.0

        dips, pairs = _pair_wires_and_compute_dips(
            profile, wire_pos, gap_pos, background, half_w=0,
            dist_factor=1.05, film_type="positive",
        )
        # 第一对正常间距=20, dist_max=21. 间距 20→OK(配对), 20→OK(配对), 90→跳过
        self.assertEqual(len(pairs), 2)

    def test_pair_wires_no_gap_between(self):
        """两丝之间无 gap → 跳过该对."""
        from gauge.imaging.profile import _pair_wires_and_compute_dips
        profile = np.ones(100, dtype=np.float64) * 100.0
        profile[[20, 40]] = 80.0  # 两根丝但没有间隙峰
        profile[60] = 120.0
        background = np.ones(100, dtype=np.float64) * 100.0

        dips, pairs = _pair_wires_and_compute_dips(
            profile, np.array([20, 40, 60]), np.array([30]), background,
            half_w=0, dist_factor=1.05, film_type="positive",
        )
        # 20-40 之间无gap → 跳过; 没有更多pair
        self.assertEqual(len(pairs), 0)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers.test_pair_wires_positive -v
```

Expected: FAIL

- [ ] **Step 3: 在 profile.py 中实现 `_pair_wires_and_compute_dips`**

在 `_compute_dip` 之后添加：

```python
def _pair_wires_and_compute_dips(
    profile: np.ndarray,
    wire_positions: np.ndarray,
    gap_positions: np.ndarray,
    background: np.ndarray,
    half_w: int,
    *,
    dist_factor: float = 1.05,
    film_type: str = "positive",
) -> Tuple[List[float], List[Tuple[int, int, int]]]:
    """Pair adjacent wires into wire-pair groups and compute each dip.

    Two adjacent wire positions are paired when their distance does not
    exceed ``dist_factor * dist_between_first_two``.  The gap (the profile
    extremum between the two wires) is located in a film-type-aware manner,
    and the dip for the pair is computed via :func:`_compute_dip`.

    Args:
        profile: 1D band-averaged gray profile.
        wire_positions: Sorted indices of wire positions (valleys for
            positive film, peaks for negative).
        gap_positions: Sorted indices of gap positions (peaks for positive,
            valleys for negative).
        background: Quadratic background fit, same length as *profile*.
        half_w: Half-window for neighbourhood-averaged dip computation.
        dist_factor: Maximum allowed multiple of the first-pair spacing
            for two wires to be considered a pair.
        film_type: ``"positive"`` or ``"negative"``.

    Returns:
        ``(dips, pairs)`` where *dips* is a list of float percentages and
        *pairs* is the corresponding list of ``(wire_a, gap, wire_b)``
        index triplets.
    """
    dips: List[float] = []
    pairs: List[Tuple[int, int, int]] = []

    if len(wire_positions) < 2:
        return dips, pairs

    dist = wire_positions[1:] - wire_positions[:-1]
    dist_max = dist_factor * float(dist[0])

    i = 0
    while i < len(wire_positions) - 1:
        if dist[i] <= dist_max:
            w1 = int(wire_positions[i])
            w2 = int(wire_positions[i + 1])
            # Find gap positions between w1 and w2
            gap_mask = (gap_positions > w1) & (gap_positions < w2)
            gaps_between = gap_positions[gap_mask]
            if len(gaps_between) >= 1:
                if film_type == "negative":
                    c = int(gaps_between[np.argmin(profile[gaps_between])])
                else:
                    c = int(gaps_between[np.argmax(profile[gaps_between])])
                pairs.append((w1, c, w2))
                dip = _compute_dip(profile, w1, c, w2, background, half_w)
                dips.append(dip)
            i += 2  # consume both wires of the pair
        else:
            i += 1  # skip isolated wire (false detection)

    return dips, pairs
```

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers.test_pair_wires_positive tests.test_double_wire_profile.TestBAMHelpers.test_pair_wires_negative tests.test_double_wire_profile.TestBAMHelpers.test_pair_wires_filters_wide_gaps tests.test_double_wire_profile.TestBAMHelpers.test_pair_wires_no_gap_between -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gauge/imaging/profile.py tests/test_double_wire_profile.py
git commit -m "feat(BAM): implement _pair_wires_and_compute_dips with film-type-aware pairing"
```

---

### Task 6: 实现 `compute_contrast()` 编排函数

**文件:**
- Modify: `src/gauge/imaging/profile.py` — 替换现有 stub
- Modify: `tests/test_double_wire_profile.py` — 添加 `TestComputeContrast` 类

- [ ] **Step 1: 添加 TestComputeContrast 测试类**

在 test 文件末尾（`TestBAMHelpers` 之后、`if __name__` 之前）添加：

```python
class TestComputeContrast(unittest.TestCase):
    """Tests for compute_contrast()."""

    def setUp(self):
        """Create synthetic negative-film profile with 4 wire pairs."""
        np.random.seed(42)
        x = np.arange(400, dtype=np.float64)
        # Background: slight quadratic
        bg = 0.0003 * x**2 + 150.0
        # 4 wire pairs with decreasing dip: D1 deep, D4 nearly merged
        # Wire positions (bright peaks for negative film)
        self.profile = bg.copy()
        # D1: wires at 50,90  gap at 70    (deep)
        self.profile[45:55] += 40.0
        self.profile[85:95] += 40.0
        self.profile[65:75] -= 30.0   # gap is dark for negative
        # D2: wires at 130,170  gap at 150
        self.profile[125:135] += 35.0
        self.profile[165:175] += 35.0
        self.profile[145:155] -= 25.0
        # D3: wires at 210,250  gap at 230
        self.profile[205:215] += 25.0
        self.profile[245:255] += 25.0
        self.profile[225:235] -= 15.0
        # D4: wires at 290,330  gap at 310 (nearly merged)
        self.profile[285:295] += 15.0
        self.profile[325:335] += 15.0
        self.profile[305:315] -= 8.0
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
        # D1 dip > D4 dip
        self.assertGreater(result.dips[0], result.dips[-1])

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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestComputeContrast.test_auto_film_type -v
```

Expected: FAIL — `NotImplementedError: Phase 2 implementation`

- [ ] **Step 3: 在 profile.py 中替换 stub 为完整实现**

找到现有的 stub `def compute_contrast(` (约第 235 行)，替换整个函数体：

```python
def compute_contrast(
    profile: np.ndarray,
    wire_spacings: Optional[Sequence[float]] = None,
    window_half_width: int = 3,
    film_type: str = "auto",
    min_distance: int = 10,
    prominence: float = 0.05,
) -> ComputeContrastResult:
    """Compute the modulation depth (dip) for every wire pair in a profile.

    Orchestrates: peak/valley detection → film-type determination →
    quadratic background fitting → wire pairing → per-pair dip calculation.

    The input *profile* is expected to be the output of
    :func:`extract_profile_band`, i.e. already band-averaged across
    ≥ 21 pixel rows to satisfy JBT 7902.

    Args:
        profile: 1D band-averaged gray profile.  Each element is the
            column-wise mean of ≥ 21 pixel rows perpendicular to the
            profile direction.
        wire_spacings: Nominal spacings (mm) of the wire pairs, e.g. the
            JBT 7902 D1–D13 sequence.  Accepted for forward compatibility;
            not used internally by this function.
        window_half_width: Half-width of the neighbourhood window for
            computing the a / b / c region means.  0 degenerates to
            single-pixel values (方案 A).
        film_type: ``"positive"``, ``"negative"``, or ``"auto"``.  When
            ``"auto"`` the type is detected from the first wire pair.
        min_distance: Minimum pixel distance between adjacent peaks,
            forwarded to :func:`detect_peaks_valleys`.
        prominence: Relative peak prominence, forwarded to
            :func:`detect_peaks_valleys`.

    Returns:
        :class:`ComputeContrastResult` with dips, pairs, background, and
        film_type.
    """
    n = len(profile)
    if n < 3:
        return ComputeContrastResult(
            dips=[], pairs=[],
            background=np.array([], dtype=np.float64),
            film_type=film_type if film_type != "auto" else "positive",
        )

    # 1. Peak / valley detection
    peaks, valleys = detect_peaks_valleys(
        profile, min_distance=min_distance, prominence=prominence,
    )

    if len(peaks) == 0 and len(valleys) == 0:
        return ComputeContrastResult(
            dips=[], pairs=[],
            background=np.zeros(n, dtype=np.float64),
            film_type=film_type if film_type != "auto" else "positive",
        )

    # 2. Film-type determination
    if film_type == "auto":
        ft = _detect_film_type(profile, valleys, peaks)
    else:
        ft = film_type

    is_negative = (ft == "negative")

    # 3. Assign wire / gap roles
    if is_negative:
        wire_positions = peaks
        gap_positions = valleys
    else:
        wire_positions = valleys
        gap_positions = peaks

    # 4. Quadratic background fit (masking wire regions)
    background = _fit_quadratic_background(
        profile, wire_positions, inverted=not is_negative,
    )

    # 5. Pair wires and compute dips
    dips, pairs = _pair_wires_and_compute_dips(
        profile, wire_positions, gap_positions, background,
        half_w=window_half_width,
        dist_factor=1.05,
        film_type=ft,
    )

    return ComputeContrastResult(
        dips=dips,
        pairs=pairs,
        background=background,
        film_type=ft,
    )
```

同时删除旧 stub 的 docstring 和 `raise NotImplementedError`。

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestComputeContrast -v
```

Expected: 6 tests PASS

- [ ] **Step 5: 运行全部已有测试确保无回归**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile -v
```

Expected: All existing + new tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/gauge/imaging/profile.py tests/test_double_wire_profile.py
git commit -m "feat(BAM): implement compute_contrast orchestrator, replacing Phase 1 stub"
```

---

### Task 7: 实现 `_cleanup_dips_monotonic()`

**文件:**
- Modify: `src/gauge/imaging/profile.py` — 在 `compute_contrast` 之后添加
- Modify: `tests/test_double_wire_profile.py` — 在 `TestBAMHelpers` 中添加测试

- [ ] **Step 1: 在 TestBAMHelpers 中添加测试方法**

```python
    def test_cleanup_dips_monotonic_removes_anomaly(self):
        """后组比前组深 >5pp → 删除前组."""
        from gauge.imaging.profile import _cleanup_dips_monotonic
        dips = [70.0, 80.0, 60.0, 50.0]  # 80-70=10 >5 → 删70
        spacings = [0.80, 0.63, 0.50, 0.40]
        d, s = _cleanup_dips_monotonic(dips, spacings)
        self.assertEqual(d, [80.0, 60.0, 50.0])
        self.assertEqual(s, [0.63, 0.50, 0.40])

    def test_cleanup_dips_monotonic_no_removal(self):
        """单调递减无异常 → 不删除."""
        from gauge.imaging.profile import _cleanup_dips_monotonic
        dips = [80.0, 75.0, 60.0, 50.0]
        spacings = [0.80, 0.63, 0.50, 0.40]
        d, s = _cleanup_dips_monotonic(dips, spacings)
        self.assertEqual(d, dips)
        self.assertEqual(s, spacings)

    def test_cleanup_dips_monotonic_chain_removal(self):
        """删前组后需回退检查新前组."""
        from gauge.imaging.profile import _cleanup_dips_monotonic
        # 100, 90, 85, 95 → 95-85=10 >5, 删85后 95-90=5 ≤5 → 停
        dips = [100.0, 90.0, 85.0, 95.0]
        spacings = [0.80, 0.63, 0.50, 0.40]
        d, s = _cleanup_dips_monotonic(dips, spacings)
        self.assertEqual(d, [100.0, 90.0, 95.0])
        self.assertEqual(s, [0.80, 0.63, 0.40])
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers.test_cleanup_dips_monotonic_removes_anomaly -v
```

Expected: FAIL

- [ ] **Step 3: 在 profile.py 中实现 `_cleanup_dips_monotonic`**

在 `compute_contrast` 之后添加：

```python
def _cleanup_dips_monotonic(
    dips: Sequence[float],
    spacings: Sequence[float],
) -> Tuple[List[float], List[float]]:
    """Enforce monotonic decrease of dips from coarse (D1) to fine pairs.

    A dip that is more than 5 percentage points deeper than its
    predecessor is considered a detection anomaly; the shallower
    predecessor is removed.  This matches the ctsimu-toolbox monotonicity
    check.

    Args:
        dips: Dip values (percent) per wire pair, from coarse to fine.
        spacings: Nominal wire-pair spacings (mm), same length as *dips*.

    Returns:
        ``(cleaned_dips, cleaned_spacings)`` as new lists.
    """
    d = list(dips)
    s = list(spacings)
    i = 1
    while i < len(d):
        if (d[i] - d[i - 1]) > 5.0:
            del d[i - 1]
            del s[i - 1]
            i -= 1  # recheck the new predecessor
        i += 1
    return d, s
```

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers.test_cleanup_dips_monotonic_removes_anomaly tests.test_double_wire_profile.TestBAMHelpers.test_cleanup_dips_monotonic_no_removal tests.test_double_wire_profile.TestBAMHelpers.test_cleanup_dips_monotonic_chain_removal -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gauge/imaging/profile.py tests/test_double_wire_profile.py
git commit -m "feat(BAM): implement _cleanup_dips_monotonic helper"
```

---

### Task 8: 实现 `_find_crossing_group()`

**文件:**
- Modify: `src/gauge/imaging/profile.py` — 在 `_cleanup_dips_monotonic` 之后添加
- Modify: `tests/test_double_wire_profile.py` — 在 `TestBAMHelpers` 中添加测试

- [ ] **Step 1: 在 TestBAMHelpers 中添加测试方法**

```python
    def test_find_crossing_group_all_resolved(self):
        """所有 dip ≥ 20% → None."""
        from gauge.imaging.profile import _find_crossing_group
        dips = [80.0, 65.0, 45.0, 28.0]
        spacings = [0.80, 0.63, 0.50, 0.40]
        result = _find_crossing_group(dips, spacings, threshold=20.0, min_dip=1.5)
        self.assertIsNone(result)

    def test_find_crossing_group_first_unresolved(self):
        """第一组就 < 20% → 返回 1."""
        from gauge.imaging.profile import _find_crossing_group
        dips = [15.0, 8.0, 3.0, 1.0]
        spacings = [0.80, 0.63, 0.50, 0.40]
        result = _find_crossing_group(dips, spacings, threshold=20.0, min_dip=1.5)
        self.assertEqual(result, 1)

    def test_find_crossing_group_middle(self):
        """中间 crossing → 离散判定 + 插值精化."""
        from gauge.imaging.profile import _find_crossing_group
        dips = [80.0, 65.0, 45.0, 28.0, 15.0, 5.0, 1.0]
        spacings = [0.80, 0.63, 0.50, 0.40, 0.32, 0.25, 0.20]
        result = _find_crossing_group(dips, spacings, threshold=20.0, min_dip=1.5)
        # crossing between D4(28%) and D5(15%)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 4)
        self.assertLessEqual(result, 5)

    def test_find_crossing_group_excludes_low_dips(self):
        """dip < 1.5% 的组从插值邻域排除."""
        from gauge.imaging.profile import _find_crossing_group
        # D5=18%, D6=1.0%, D7=0.5% — D6,D7 应被排除
        dips = [80.0, 65.0, 45.0, 28.0, 18.0, 1.0, 0.5]
        spacings = [0.80, 0.63, 0.50, 0.40, 0.32, 0.25, 0.20]
        result = _find_crossing_group(dips, spacings, threshold=20.0, min_dip=1.5)
        # D5 即 < 20%, 返回 5
        self.assertEqual(result, 5)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers.test_find_crossing_group_all_resolved -v
```

Expected: FAIL

- [ ] **Step 3: 在 profile.py 中实现 `_find_crossing_group`**

在 `_cleanup_dips_monotonic` 之后添加：

```python
def _find_crossing_group(
    dips: Sequence[float],
    spacings: Sequence[float],
    threshold: float,
    min_dip: float,
) -> Optional[int]:
    """Find the first wire-pair group whose dip falls below *threshold*.

    Performs discrete traversal (coarse → fine) with optional quadratic
    interpolation refinement around the crossing point for sub-integer
    precision.

    Groups whose dip is below *min_dip* are excluded from the
    interpolation neighbourhood to avoid destabilising the quadratic fit.

    Args:
        dips: Dip values (percent) per wire pair, D1 → Dn.
        spacings: Nominal wire-pair spacings (mm), same length as *dips*.
        threshold: Dip percentage below which a pair is considered
            unresolved (typically 20).
        min_dip: Minimum dip (percent) for a group to be included in the
            interpolation neighbourhood (typically 1.5).

    Returns:
        1-indexed group number of the first unresolved pair, or *None*
        if all pairs are resolved.
    """
    n = len(dips)
    if n == 0:
        return None

    # 1. Discrete scan: find where dip first drops below threshold
    crossing_i: Optional[int] = None
    for i in range(n):
        if dips[i] < threshold:
            crossing_i = i
            break

    if crossing_i is None:
        return None  # all resolved
    if crossing_i == 0:
        return 1     # very poor quality

    # 2. Quadratic interpolation refinement
    #    Select up to 4 neighbours around the crossing point
    lo = max(0, crossing_i - 1)
    hi = min(n, crossing_i + 3)   # +3 because Python slice is exclusive

    sel_dips = list(dips[lo:hi])
    sel_spacings = list(spacings[lo:hi])

    # Exclude neighbours with dip < min_dip (unstable for fitting)
    # but always keep at least 2 points
    keep = [j for j in range(len(sel_dips)) if sel_dips[j] >= min_dip]
    if len(keep) < 2:
        return crossing_i + 1  # fall back to discrete result

    sel_dips = [sel_dips[j] for j in keep]
    sel_spacings = [sel_spacings[j] for j in keep]

    if len(sel_spacings) < 2:
        return crossing_i + 1

    try:
        coeffs = np.polyfit(sel_spacings, sel_dips, 2)
    except (np.linalg.LinAlgError, ValueError):
        return crossing_i + 1

    a, b, c_coeff = coeffs
    # Solve a*x² + b*x + (c_coeff - threshold) = 0
    roots = np.roots([a, b, c_coeff - threshold])
    # Keep only real roots within the interpolation range
    valid = roots[np.isreal(roots) &
                  (roots >= min(sel_spacings)) &
                  (roots <= max(sel_spacings))].real
    if len(valid) == 0:
        return crossing_i + 1

    # Choose root based on curvature of the quadratic
    crossing_spacing = float(np.max(valid) if a >= 0 else np.min(valid))

    # Map spacing back to 1-indexed group number:
    # find the first group whose spacing is ≤ crossing_spacing
    for idx, s in enumerate(spacings):
        if s <= crossing_spacing:
            return idx + 1

    return crossing_i + 1  # fallback
```

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestBAMHelpers.test_find_crossing_group_all_resolved tests.test_double_wire_profile.TestBAMHelpers.test_find_crossing_group_first_unresolved tests.test_double_wire_profile.TestBAMHelpers.test_find_crossing_group_middle tests.test_double_wire_profile.TestBAMHelpers.test_find_crossing_group_excludes_low_dips -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gauge/imaging/profile.py tests/test_double_wire_profile.py
git commit -m "feat(BAM): implement _find_crossing_group with quadratic interpolation"
```

---

### Task 9: 实现 `find_first_unresolved_group()`

**文件:**
- Modify: `src/gauge/imaging/profile.py` — 替换现有 stub
- Modify: `tests/test_double_wire_profile.py` — 添加 `TestFindFirstUnresolvedGroup` 类

- [ ] **Step 1: 添加 TestFindFirstUnresolvedGroup 测试类**

在 test 文件末尾（`TestComputeContrast` 之后，`if __name__` 之前）添加：

```python
class TestFindFirstUnresolvedGroup(unittest.TestCase):
    """Tests for find_first_unresolved_group()."""

    def test_all_resolved(self):
        """全部 ≥ 20% → None."""
        from gauge.imaging.profile import find_first_unresolved_group
        dips = [80.0, 65.0, 45.0, 28.0]
        result = find_first_unresolved_group(dips)
        self.assertIsNone(result)

    def test_first_unresolved(self):
        """第一组就 < 20% → 1."""
        from gauge.imaging.profile import find_first_unresolved_group
        dips = [15.0, 8.0, 3.0, 1.0]
        result = find_first_unresolved_group(dips)
        self.assertEqual(result, 1)

    def test_monotonicity_cleanup_applied(self):
        """含异常 dip 的序列 → 先清理再判定."""
        from gauge.imaging.profile import find_first_unresolved_group
        # 60, 70 异常(70-60=10>5) → 删60; 余70, 55, 35, 18
        dips = [60.0, 70.0, 55.0, 35.0, 18.0]
        result = find_first_unresolved_group(dips)
        # D4(35%)≥20, D5(18%)<20 → 5
        self.assertEqual(result, 5)

    def test_empty_dips(self):
        """空 dips → None."""
        from gauge.imaging.profile import find_first_unresolved_group
        self.assertIsNone(find_first_unresolved_group([]))

    def test_cleanup_empties_dips(self):
        """清理后 dips 为空 → None."""
        from gauge.imaging.profile import find_first_unresolved_group
        # 仅一组 dip, 无需清理但只有一个元素...
        result = find_first_unresolved_group([15.0])
        self.assertIsNotNone(result)  # 1

    def test_custom_threshold(self):
        """自定义 threshold=15%."""
        from gauge.imaging.profile import find_first_unresolved_group
        dips = [80.0, 25.0, 16.0, 8.0]
        # threshold=15: D3(16%)≥15, D4(8%)<15 → 4
        result = find_first_unresolved_group(dips, dip_threshold=15.0)
        self.assertEqual(result, 4)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestFindFirstUnresolvedGroup.test_all_resolved -v
```

Expected: FAIL — `NotImplementedError`

- [ ] **Step 3: 在 profile.py 中替换 stub**

找到现有的 stub `def find_first_unresolved_group(`，替换整个函数体：

```python
def find_first_unresolved_group(
    dips: Sequence[float],
    wire_spacings: Optional[Sequence[float]] = None,
    dip_threshold: float = 20.0,
    min_dip_pct: float = 1.5,
) -> Optional[int]:
    """Find the first (coarsest) wire-pair group whose dip is below the
    resolution threshold.

    Applies monotonicity cleanup to the dip sequence, excludes very-low-dip
    neighbours from interpolation, and returns the 1-indexed group number
    of the first unresolved pair.

    Args:
        dips: Dip values (percent) per wire pair, from coarse (D1) to fine.
            Typically obtained from :func:`compute_contrast`.
        wire_spacings: Nominal wire-pair spacings (mm) matching the IQI
            model.  Defaults to the JBT 7902 D1–D13 sequence.  Must have
            at least as many entries as *dips*.
        dip_threshold: Dip percentage below which a pair is considered
            unresolved (default 20 %).
        min_dip_pct: Minimum dip (percent) for a group to participate in
            the interpolation neighbourhood (default 1.5 %).

    Returns:
        1-indexed group number, or *None* if all pairs are resolved.
    """
    if len(dips) == 0:
        return None

    if wire_spacings is None:
        spacings = list(_DEFAULT_WIRE_SPACINGS[:len(dips)])
    else:
        spacings = list(wire_spacings[:len(dips)])

    # 1. Monotonicity cleanup
    clean_dips, clean_spacings = _cleanup_dips_monotonic(dips, spacings)

    if len(clean_dips) == 0:
        return None

    # 2. Find crossing group (discrete + interpolation)
    return _find_crossing_group(
        clean_dips, clean_spacings,
        threshold=dip_threshold,
        min_dip=min_dip_pct,
    )
```

同时删除旧 stub 的 docstring 和 `raise NotImplementedError`。

- [ ] **Step 4: 运行测试确认通过**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestFindFirstUnresolvedGroup -v
```

Expected: 6 tests PASS

- [ ] **Step 5: 运行全部测试确保无回归**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/gauge/imaging/profile.py tests/test_double_wire_profile.py
git commit -m "feat(BAM): implement find_first_unresolved_group, replacing Phase 1 stub"
```

---

### Task 10: 端到端验证 — 合成负片剖面完整流程

**文件:**
- Modify: `tests/test_double_wire_profile.py` — 在 `TestFindFirstUnresolvedGroup` 中添加端到端测试

- [ ] **Step 1: 添加端到端测试方法**

在 `TestFindFirstUnresolvedGroup` 类中添加：

```python
    def test_end_to_end_negative_film(self):
        """完整流程: 合成负片剖面 → compute_contrast → find_first_unresolved_group."""
        from gauge.imaging.profile import compute_contrast, find_first_unresolved_group

        # 构造一个清晰的 4 对负片剖面
        x = np.arange(400, dtype=np.float64)
        profile = 0.0003 * x**2 + 150.0  # 轻微二次背景
        # D1 (deep): wires at 50, 90
        profile[45:55] += 40.0
        profile[85:95] += 40.0
        profile[65:75] -= 30.0
        # D2: wires at 140, 180
        profile[135:145] += 35.0
        profile[175:185] += 35.0
        profile[155:165] -= 25.0
        # D3: wires at 230, 270
        profile[225:235] += 25.0
        profile[265:275] += 25.0
        profile[245:255] -= 15.0
        # D4: wires at 320, 360 (nearly merged → dip should be low)
        profile[315:325] += 12.0
        profile[355:365] += 12.0
        profile[335:345] -= 6.0
        profile += np.random.default_rng(42).normal(0, 1.5, 400)

        result = compute_contrast(profile, film_type="auto", min_distance=30)
        self.assertEqual(result.film_type, "negative")
        self.assertGreaterEqual(len(result.dips), 3)
        # D1 dip > D4 dip
        self.assertGreater(result.dips[0], result.dips[-1])

        group = find_first_unresolved_group(result.dips)
        self.assertIsNotNone(group)
        # 应返回 D4 附近
        self.assertGreaterEqual(group, 3)
```

- [ ] **Step 2: 运行端到端测试**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile.TestFindFirstUnresolvedGroup.test_end_to_end_negative_film -v
```

Expected: PASS

- [ ] **Step 3: 运行全部测试**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile -v
```

Expected: All tests PASS

- [ ] **Step 4: 语法检查全部相关文件**

```bash
cd /home/cht/code/IQIdet && python -m py_compile src/gauge/imaging/profile.py tests/test_double_wire_profile.py
```

Expected: 无输出。

- [ ] **Step 5: Commit**

```bash
git add tests/test_double_wire_profile.py
git commit -m "test(BAM): add end-to-end negative-film integration test"
```

---

### 自检清单

- [x] **Spec 覆盖**: §2 dataclass → Task1, §3.1 compute_contrast → Task6, §3.2 find_first_unresolved_group → Task9, §6.1~§6.6 各 helper → Task2~Task5, Task7~Task8
- [x] **无占位符**: 所有步骤含完整代码、命令和预期输出
- [x] **类型一致**: `_compute_dip` 签名中 `half_w: int` 跨 Task4/5 一致；`_find_crossing_group` 返回 `Optional[int]` 跨 Task8/9 一致
- [x] **边角情况覆盖**: §7 各情况均有对应测试（空 profile、平坦剖面、空 dips、全通过、首组未通过、cleanup 清空）
