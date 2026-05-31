# BAM 双丝像质计算法设计

**日期：** 2026-05-31
**状态：** 待实现
**分支：** `feature/BAM`
**参考：** [[方案调研]], `src/3rdparty/ctsimu-toolbox/ctsimu/image_analysis/isrb.py`

## 1. 概述

在 `src/gauge/imaging/profile.py` 中实现 `compute_contrast()` 和 `find_first_unresolved_group()`，遵循 BAM ctsimu-toolbox 的 dip 计算与 iSRb 插值算法。

Phase 1 已完成剖面提取（`extract_profile_band`）、OBB 拟合（`fit_obb_and_midline`）、OBB 展开（`unwarp_obb_region`）和峰谷检测（`detect_peaks_valleys`）。Phase 2 在此基础上实现完整的 BAM 双丝分析链。

**输入前置条件**：`profile` 参数应为 `extract_profile_band()` 的输出——已沿垂直于剖面方向对 ≥21 行像素做束带平均，满足 JBT 7902 "不少于 21 行或列像素叠加平均" 的要求。`compute_contrast` 本身不再做束带方向的平均。

## 2. 新增数据结构

```python
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

@dataclass
class ComputeContrastResult:
    """compute_contrast() 的返回结果。

    Attributes:
        dips: 各组 dip 值，百分比 [0, 100]。
        pairs: 配对的 (wire_a_idx, gap_idx, wire_b_idx) 三元组列表。
            对应对应正片语义的 (valley, peak, valley)，与 groundtruth.json 命名一致。
        background: 二次背景拟合值，ndarray 与 profile 等长。
        film_type: "positive" 或 "negative"。
    """
    dips: List[float]
    pairs: List[Tuple[int, int, int]]
    background: np.ndarray
    film_type: str
```

## 3. 公开函数签名

### 3.1 `compute_contrast`

```python
def compute_contrast(
    profile: np.ndarray,
    wire_spacings: Optional[Sequence[float]] = None,
    window_half_width: int = 3,
    film_type: str = "auto",
    min_distance: int = 10,
    prominence: float = 0.05,
) -> ComputeContrastResult:
    """计算双丝剖面各组的 dip（调制度）。

    完成：正/负片判定 → 二次背景拟合 → 谷配对 → 逐组 dip 计算。

    前置条件：profile 应为 extract_profile_band() 的输出，即已沿垂直于剖面
    方向对 ≥21 行像素做束带平均，满足 JBT 7902 要求。

    Args:
        profile: 1D 束带平均灰度剖面（每个元素 = 垂直于剖面方向的 21 行像素列向均值）。
        wire_spacings: 各组丝对的标准间距（mm）。默认使用 JBT 7902 D1~D13 序列。
        window_half_width: a/b/c 邻域均值窗口半宽（0=单点极值，退化到方案A）。
        film_type: "positive" / "negative" / "auto"。auto 时基于 D1 信号方向自动判定。
        min_distance: 峰谷最小间距，传给 detect_peaks_valleys。
        prominence: 峰谷显著性阈值，传给 detect_peaks_valleys。

    Returns:
        ComputeContrastResult，包含 dips, pairs, background, film_type。
    """
```

### 3.2 `find_first_unresolved_group`

```python
def find_first_unresolved_group(
    dips: Sequence[float],
    wire_spacings: Optional[Sequence[float]] = None,
    dip_threshold: float = 20.0,
    min_dip_pct: float = 1.5,
) -> Optional[int]:
    """查找首个不可分辨的丝对组号。

    完成：单调性清理 → 排除低调制组 → 遍历/插值判定。

    Args:
        dips: 各组 dip 值列表（来自 compute_contrast）。
        wire_spacings: 各组丝对的标准间距（mm）。默认使用 JBT 7902 D1~D13 序列。
        dip_threshold: dip 百分比阈值，默认 20%。
        min_dip_pct: 排除 dip 低于此值的组（%），避免低调制组破坏插值稳定性。

    Returns:
        1-indexed 首个不可分辨组号，None 表示全部可分辨。
    """
```

## 4. 内部私有函数

| 函数 | 职责 |
|------|------|
| `_detect_film_type(profile, valleys, peaks) -> str` | D1 组比较 valley/peak 灰度判定正/负片 |
| `_fit_quadratic_background(profile, wire_indices, *, inverted=False) -> np.ndarray` | curve_fit 二次背景，mask 掉丝区 |
| `_pair_wires_and_compute_dips(profile, wire_positions, gap_positions, background, half_w, *, dist_factor, film_type) -> Tuple[List[float], List[Tuple]]` | 片型感知的丝配对 + 逐对 dip |
| `_compute_dip(profile, wire_a, gap_c, wire_b, background, half_w) -> float` | 单组 dip 计算（邻域均值 + 背景减除） |
| `_cleanup_dips_monotonic(dips, spacings) -> Tuple[List, List]` | 单调性清理：dip[i] 比 dip[i-1] 深 >5pp → 删 dip[i-1] |
| `_find_crossing_group(dips, spacings, threshold, min_dip) -> Optional[int]` | 离散判定 + 二次插值精化 → 返回首个未分辨组号 |

注：`detect_peaks_valleys()` 仍使用 Phase 1 已实现的公开版本。

## 5. 核心数据流

```
profile (1D ndarray)
        │
        ▼
detect_peaks_valleys(profile)  ← Phase 1 已有
        │
        ├── peaks, valleys
        ▼
_detect_film_type(profile, valleys, peaks)  ← 仅 film_type="auto" 时
        │
        ├── 确定 wire_positions / gap_positions 角色
        ▼
_fit_quadratic_background(profile, wire_positions, inverted=is_negative)
  - 用 peak_widths 计算丝宽 → mask 掉丝区
  - curve_fit(quadratic, x[mask], profile[mask]) → background
        │
        ▼
_pair_wires_and_compute_dips(profile, wire_positions, gap_positions, background, half_w)
  - dist_max = 1.05 * dist[0] (以第一对丝间距为基准)
  - 相邻丝间距 ≤ dist_max → 配成一对
  - 每对: 片型感知找中间间隙 → _compute_dip()
  - A = |bg[a] - mean_a|, B = |bg[b] - mean_b|, C = |bg[c] - mean_c|
  - dip = 100 * (A + B - 2*C) / (A + B)
        │
        ▼
ComputeContrastResult(dips, pairs, background, film_type)
        │
        ▼
find_first_unresolved_group(dips, wire_spacings)
  - _cleanup_dips_monotonic(dips, spacings)  # 单调性清理
  - 排除 dip < 1.5% 的组（仅从插值邻域排除，不删除）
  - _find_crossing_group():
      - 全部 ≥ threshold → return None
      - 第一组 < threshold → return 1
      - 否则 → 二次插值 → 查表返回组号
        │
        ▼
Optional[int]  # 1-indexed 组号
```

## 6. 关键算法细节

### 6.1 二次背景拟合

遮罩掉丝区（wire regions），在剩余间隙区域上拟合二次曲线。丝区的判定取决于片型：
- 正片：丝=valley，用 `peak_widths(-profile, valleys)` 计算丝宽并遮罩
- 负片：丝=peak，用 `peak_widths(profile, peaks)` 计算丝宽并遮罩

```python
from scipy.signal import peak_widths
from scipy.optimize import curve_fit

def _fit_quadratic_background(profile, wire_indices, *, inverted=False):
    """在 profile 上拟合二次背景，遮罩掉丝区。

    Args:
        profile: 1D 束带平均灰度剖面。
        wire_indices: 丝位置的整数索引数组（正片=valleys, 负片=peaks）。
        inverted: True 表示丝本身是 profile 的峰值（负片），此时对 -profile 做 peak_widths。
    """
    target = -profile if inverted else profile
    widths, _, _, _ = peak_widths(target, wire_indices, rel_height=0.9)
    mask = np.ones(len(profile), dtype=bool)
    for i, p in enumerate(wire_indices):
        lo = max(0, int(np.rint(p - widths[i])))
        hi = min(len(profile) - 1, int(np.rint(p + widths[i])))
        mask[lo:hi+1] = False

    x = np.arange(len(profile))
    popt, _ = curve_fit(lambda x, a, b, c: a*x**2 + b*x + c, x[mask], profile[mask])
    return popt[0]*x**2 + popt[1]*x + popt[2]
```

### 6.2 谷/峰配对策略

片型决定丝和间隙的角色：

| 片型 | 丝（a, b 位置） | 间隙（c 位置） |
|------|----------------|---------------|
| 正片 | valleys（暗） | peaks（亮） |
| 负片 | peaks（亮） | valleys（暗） |

配对以丝位（wire positions）为锚点，相邻丝间距不超过 `1.05 * dist[0]` 时配成一对。
两丝之间的间隙位置通过片型感知的极值搜索确定。

```python
def _pair_wires_and_compute_dips(
    profile, wire_positions, gap_positions, background, half_w,
    dist_factor=1.05, film_type="positive",
):
    """以丝位为锚点配对并计算 dip。

    Args:
        wire_positions: 丝的位置索引（正片=valleys, 负片=peaks）。
        gap_positions: 间隙的位置索引（正片=peaks, 负片=valleys）。
        两组数组均需已排序。

    配对规则：
        dist = wire_positions[1:] - wire_positions[:-1]
        dist_max = dist_factor * dist[0]   # 以首对间距为基准
        相邻丝间距 ≤ dist_max → 配成一对

    每对内的间隙查找：
        正片（丝=暗, 间隙=亮）→ argmax(profile) 在两丝之间
        负片（丝=亮, 间隙=暗）→ argmin(profile) 在两丝之间
    """
```

具体实现：

```python
# 配对
dist = wire_positions[1:] - wire_positions[:-1]
dist_max = dist_factor * dist[0]

pairs = []
i = 0
while i < len(wire_positions) - 1:
    if dist[i] <= dist_max:
        w1, w2 = wire_positions[i], wire_positions[i+1]
        # 找 w1, w2 之间的间隙
        gaps_between = gap_positions[(gap_positions > w1) & (gap_positions < w2)]
        if len(gaps_between) >= 1:
            if film_type == "negative":
                c = gaps_between[np.argmin(profile[gaps_between])]
            else:
                c = gaps_between[np.argmax(profile[gaps_between])]
            pairs.append((w1, int(c), w2))
        i += 2   # 跳过已配对的丝
    else:
        i += 1   # 间距过大→假丝，跳过
```

### 6.3 dip 公式

A、B、C 均先取邻域均值（`window_half_width`），再与背景做差取绝对值。
正负片下公式自然统一——背景始终穿过间隙区域。

```python
def _compute_dip(profile, wire_a, gap_c, wire_b, background, half_w):
    """计算单组丝对的 dip 值。返回百分比 [0, 100]."""
    L = len(profile)

    def _region_mean(center):
        lo = max(0, center - half_w)
        hi = min(L - 1, center + half_w)
        return float(profile[lo:hi+1].mean())

    a_mean = _region_mean(wire_a)
    c_mean = _region_mean(gap_c)
    b_mean = _region_mean(wire_b)

    A = abs(background[wire_a] - a_mean)
    B = abs(background[wire_b] - b_mean)
    C = abs(background[gap_c] - c_mean)

    denom = A + B
    if denom < 1e-10:
        return 0.0
    return 100.0 * (A + B - 2.0 * C) / denom
```

### 6.4 正/负片自动判定

```python
def _detect_film_type(profile, valleys, peaks):
    if len(valleys) >= 2 and len(peaks) >= 1:
        # 正片：valley(丝·暗) < peak(间隙·亮)
        a_val = profile[valleys[0]]
        c_val = profile[peaks[0]]
        return "negative" if a_val > c_val else "positive"
    return "positive"
```

### 6.5 单调性清理

```python
def _cleanup_dips_monotonic(dips, spacings):
    """若 dip[i] - dip[i-1] > 5（后组比前组深超 5pp），删除前组。"""
    d = list(dips)
    s = list(spacings)
    i = 1
    while i < len(d):
        if (d[i] - d[i-1]) > 5.0:
            del d[i-1], s[i-1]
            i -= 1
        i += 1
    return d, s
```

### 6.6 iSRb 插值与 crossing 判定

```python
def _find_crossing_group(dips, spacings, threshold, min_dip):
    """通过离散判定+可选插值精化，返回首个 dip < threshold 的组号。

    步骤：
    1. 从粗到细遍历，找到 dip 首次 < threshold 的组 i。
       - 若全部 ≥ threshold → return None
       - 若第一组就 < threshold → return 1
    2. (可选精化) 若 crossing 出现在 i 和 i+1 之间:
       - 取 i-1, i, i+1, i+2 共 4 组（裁剪到有效范围）
       - 排除其中 dip < min_dip 的邻近组
       - 二次拟合 dip = f(spacing)
       - 解 f(d) = threshold → 得到连续 crossing spacing
       - 查表：crossing spacing 对应哪个组号 → 返回该组号(int)
    3. 返回组号（1-indexed int）。

    若精化步骤失败（如拟合数据不足），回退到离散判定结果。
    """

# 二次插值核心：
coeffs = np.polyfit(selected_spacings, selected_dips, 2)
a, b, c_coeff = coeffs
# 解 a*x^2 + b*x + (c_coeff - threshold) = 0
roots = np.roots([a, b, c_coeff - threshold])
# 依曲率选择有效根（凸函数取小根，凹函数取大根）
valid = roots[(roots >= min(selected_spacings)) & (roots <= max(selected_spacings))]
crossing_spacing = max(valid) if a >= 0 else min(valid)

# 查表：找到 crossing_spacing 对应或小于它的最小组号
for idx, s in enumerate(spacings):
    if s <= crossing_spacing:
        return idx + 1   # 1-indexed
```

### 6.7 wire_spacings 默认值

JBT 7902 标准 D1~D13 序列（mm）：
```
[0.80, 0.63, 0.50, 0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10, 0.08, 0.063, 0.05]
```

---

## 7. 边角情况处理

| 情况 | 处理 |
|------|------|
| profile 长度 < 3 | `ComputeContrastResult(dips=[], pairs=[], background=np.array([]), film_type="positive")` |
| 峰谷检测为空 | 返回空结果，不抛异常 |
| 所有 dip ≥ threshold | `find_first_unresolved_group` 返回 `None` |
| 第一组 dip < threshold | 返回 1（图像质量极差） |
| 谷间距过大（噪声假谷） | `dist > dist_max` 的间隔跳过，不强行配对 |
| valley 之间无 peak（已完全融合） | 该组 dip ≈ 0，仍记录，不影响其他组 |
| monotonicity 清理导致 dips 为空 | 返回 `None` |
| 负片映射 | 自动判定 + 背景绝对值统一处理 |

---

## 8. 验证计划 (Phase 3)

Phase 3 由独立 subagent 负责：
- 测试脚本使用 `groundtruth.json` 作为基准
- 对比：算法输出的 valley/peak 配对 vs 人工标注
- 评估指标：峰值/谷值检出率、配对准确率、dip 误差
- `find_first_unresolved_group` 输出组号验证

## 9. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/gauge/imaging/profile.py` | 修改 | 实现 `compute_contrast` 和 `find_first_unresolved_group`，替换 stub |
| `tests/test_double_wire_profile.py` | 修改 | 新增 `TestComputeContrast` 和 `TestFindFirstUnresolvedGroup` 测试类 |
| `scripts/debug/double_wire_demo.py` | 无需修改 | 现有接口不变 |
