# BAM 双丝像质计算法原理

**日期：** 2026-06-01
**来源：** BAM `ctsimu-toolbox` 源码分析 + [[../需求/方案调研|方案调研]] + `src/gauge/imaging/profile.py` 实现
**上游参考：** ISO 19232-5:2018、ASTM E2002-15、JBT 7902-2025

---

## 1. 背景

BAM（Bundesanstalt für Materialforschung und -prüfung，德国联邦材料研究与测试研究所）开源了 `ctsimu-toolbox`，是目前最成熟的 ASTM E2002 双丝 IQI 自动分析实现。核心代码位于 `src/3rdparty/ctsimu-toolbox/ctsimu/image_analysis/isrb.py`（iSRb = interpolated Spatial Resolution basic）。

本项目已将此算法本地化实现在 `src/gauge/imaging/profile.py` 中，作为 Phase 2 自动判定的核心引擎。

---

## 2. 核心公式：调制度 Dip

双丝分辨率的数学基础是 ISO 19232-5 / ASTM E2002 定义的**调制度（Modulation Depth / Dip）**：

$$R = \frac{a + b - 2c}{a + b} \times 100\%$$

三个符号来自灰度剖面上每对双丝的三个关键位置：

| 符号 | 正片含义 | 负片含义 | 剖面形态 |
|------|---------|---------|----------|
| **a** | 第一根丝的灰度值（丝→暗→局部极小值） | 第一根丝的灰度值（丝→亮→局部极大值） | 波谷1（正片） |
| **c** | 两丝间隙的灰度值（间隙→亮→局部极大值） | 两丝间隙的灰度值（间隙→暗→局部极小值） | 波峰（正片） |
| **b** | 第二根丝的灰度值（局部极小值） | 第二根丝的灰度值（局部极大值） | 波谷2（正片） |

**直观理解：** 双丝越可分辨 → 两谷越深、中间峰越高 → Dip 越大。双丝完全融合时 a ≈ b ≈ c → Dip ≈ 0。

**判定标准：** $R < 20\%$ 时该线对不可分辨。从最粗组 D1 向最细遍历，第一个不可分辨组对应的丝径即为基本空间分辨率 SRb，$U_T = 2d$ 为总不清晰度。

### 丝对参数表（JBT 7902-2025 表2）

| 组号 | d (mm) | UT (mm) | SRb (mm) | LP/mm |
|------|--------|---------|----------|-------|
| D1  | 0.800  | 1.600   | 0.800    | 0.63  |
| D2  | 0.630  | 1.260   | 0.630    | 0.79  |
| D3  | 0.500  | 1.000   | 0.500    | 1.00  |
| D4  | 0.400  | 0.800   | 0.400    | 1.25  |
| D5  | 0.320  | 0.640   | 0.320    | 1.56  |
| D6  | 0.250  | 0.500   | 0.250    | 2.00  |
| D7  | 0.200  | 0.400   | 0.200    | 2.50  |
| D8  | 0.160  | 0.320   | 0.160    | 3.13  |
| D9  | 0.130  | 0.260   | 0.130    | 3.85  |
| D10 | 0.100  | 0.200   | 0.100    | 5.00  |
| D11 | 0.080  | 0.160   | 0.080    | 6.25  |
| D12 | 0.063  | 0.126   | 0.063    | 7.94  |
| D13 | 0.050  | 0.100   | 0.050    | 10.00 |
| D14*| 0.040  | 0.080   | 0.040    | 12.50 |
| D15*| 0.032  | 0.064   | 0.032    | 15.60 |
| D16*| 0.025  | 0.050   | 0.025    | 20.00 |
| D17*| 0.020  | 0.040   | 0.020    | 25.00 |

> D1~D3 为钨丝，D4~D17 为铂丝。\* 标记为扩展组（新团标）。

---

## 3. 算法流水线

BAM `isrb.py` 的 `Interpolation` 类定义四步核心方法：`profile()` → `calc_dips()` → `interpolate()`。我们的实现扩展为正/负片自动判定 + 细粒度配对验证，共五步：

```
Step 1: 剖面提取        → 沿双丝方向提取灰度剖面线（束带平均）
Step 2: 峰谷检测        → 去趋势后用 scipy.signal.find_peaks 检测波峰/波谷
Step 3: 二次背景拟合    → 消除辐照不均匀（heel effect）
Step 4: 谷配对 + Dip    → 细粒度相邻 wire-gap-wire 配对 + 单点极值 dip
Step 5: 二次插值 iSRb   → 解 dip=20% 对应丝径（离散判定可跳过）
```

### 3.1 Step 1：提取灰度剖面线

**BAM 实现**（`isrb.py:158-251`）：
沿 RDI 方向用 `skimage.measure.profile_line` 提取灰度剖面。支持通过最小化"垂直方差的方差"来自动优化剖面线角度（Powell 法），适应像质计放置的微小角度偏差。

**我们的实现**（`profile.py:41-110`：`extract_profile_band()`）：
采样 `band_width`（默认 21）条平行线，用 `scipy.ndimage.map_coordinates` 做双线性子像素插值，沿列向平均。满足 JBT 7902-2025 "不少于 21 行或列像素叠加平均"的要求。

```python
# profile.py:105-109
sampled = map_coordinates(image.astype(np.float64), coords, order=1, mode="nearest")
sampled = sampled.reshape(band_width, num_samples)
profile = sampled.mean(axis=0)  # 沿束带方向平均
```

### 3.2 Step 2：峰谷检测

**BAM 实现**（`isrb.py:301`）：
只检测 valleys（向下峰），对取反后的剖面调用 `find_peaks`：
```python
peaks, prop = find_peaks(-self.measure, prominence=prominence, height=height, width=width)
```

**我们的基础检测函数**（`profile.py`：`detect_peaks_valleys()`）：
同时检测 peaks 和 valleys，使用**自适应 prominence 阈值**：
```python
data_range = float(np.max(profile) - np.min(profile))
abs_prominence = prominence * data_range  # 默认 0.05 * 动态范围
peaks, _ = find_peaks(profile, distance=min_distance, prominence=abs_prominence)
valleys, _ = find_peaks(-profile, distance=min_distance, prominence=abs_prominence)
```

关键参数：

| 参数 | 含义 | 默认值 | 依据 |
|------|------|--------|------|
| `min_distance` | 相邻峰/谷最小间距（像素） | 10 | 2024 灰度直方图论文：$D_{min}=2, D_{max}=20$ |
| `prominence` | 峰/谷显著性（相对动态范围） | 0.05 | 自适应优于固定值 |

**当前 BAM 主路径不是直接在原始 profile 上配对。** `compute_contrast()` 会先对剖面做二次趋势去除，再执行两级极值检测：

```python
# 粗检测：用于基础片型判断和负片候选
peaks, valleys = detect_peaks_valleys(
    detrended, min_distance=min_distance, prominence=prominence
)

# 细检测：用于正片细线对配对，允许 D6-D8 这类 1~3px 间距
fine_peaks, fine_valleys = detect_peaks_valleys(
    detrended, min_distance=1, prominence=max(0.005, prominence * 0.5)
)
```

原因：双丝越往细组，左右 wire 与中间 gap 的间距可缩小到 1~4 像素。若沿用显示层的 `min_distance=10`，会漏掉一侧 wire valley；若只调小全局 `min_distance`，又会引入尾部平台噪声。因此主路径使用细检测生成候选，再通过物理配对规则过滤。

### 3.3 Step 3：二次背景拟合（核心创新）

这是 BAM 算法最关键的步骤，消除 X 射线束不均匀照射（heel effect）导致的剖面基线偏移。

**BAM 实现**（`isrb.py:308-332`）：
```python
# 用 peak_widths 确定谷的宽度区域
widths, _, _, _ = peak_widths(-self.measure, peaks, rel_height=0.9, ...)
# 掩膜排除峰区
mask_bg[lim1:lim2+1] = False
# 在剩余数据点（间隙区）上做二次拟合
popt_bg, _ = curve_fit(bg_func, self.ind[mask_bg], self.measure[mask_bg])
bg_val = bg_func(self.ind, *popt_bg)   # y = a·x² + b·x + c
```

**我们的实现**（`profile.py:279-336`：`_fit_quadratic_background()`）：
```python
# 正片：丝是谷 → 取反后调用 peak_widths
target = -profile if inverted else profile
widths, _, _, _ = peak_widths(target, wire_indices, rel_height=0.9)
# 掩膜排除丝区
mask[lo:hi+1] = False
# 对间隙区做二次拟合
popt, _ = curve_fit(lambda x, a, b, c: a*x²+b*x+c, x[mask], profile[mask])
```

**为什么需要背景拟合？** 不减背景时，Dip 依赖绝对灰度值，不同曝光条件下同一像质计的 Dip 结果不一致。减去二次背景后，A、B、C 变为**丝/间隙相对于局部背景的偏离量**，使 Dip 在正负片上统一。

**适用条件：** 当双丝 ROI 较小（在图像中局部区域）时，背景变化有限，此步可跳过。若全图提取长剖面（跨越整个 IQI），背景拟合很有价值。

### 3.4 Step 4：谷配对 + Dip 计算

#### 配对策略

**BAM 实现**（`isrb.py:335-373`）：
```python
dist = peaks[1:] - peaks[:-1]              # 相邻谷间距
dist_max = 1.05 * dist[0]                  # 以第一对间距的 1.05 倍为上限

for i in range(len(dist)):
    if dist[i] <= dist_max:                # 间距在阈值内 → 视为线对
        A = abs(bg_val[lim1] - self.measure[lim1])
        B = abs(bg_val[lim2] - self.measure[lim2])
        C_pos = np.argmax(self.measure[lim1:lim2+1]) + lim1
        C = abs(bg_val[C_pos] - self.measure[C_pos])
        dips.append(100 * (A + B - 2*C) / (A + B))
```

**配对核心：** 以最粗丝对 D1 的谷间距为锚点，只有相邻谷间距不超过第一对间距的 1.05 倍时才认为是同一对。这自动过滤了噪声假谷，优于固定像素阈值。

**基础配对函数**（`_pair_wires_and_compute_dips()`）：
保留 BAM 的相邻 wire 配对思路，并增加片型感知的间隙极值选择：
```python
if film_type == "negative":
    c = int(gaps_between[np.argmin(profile[gaps_between])])  # 负片取最小
else:
    c = int(gaps_between[np.argmax(profile[gaps_between])])  # 正片取最大
```

**当前正片主路径**使用 `_pair_adjacent_wires_with_gaps()`，更严格地按物理三元组配对：

```text
正片: valley(wire_a) - peak(gap_c) - valley(wire_b)
负片: peak(wire_a)   - valley(gap_c) - peak(wire_b)
```

配对步骤：

1. 从 `fine_valleys / fine_peaks` 生成相邻 `wire-gap-wire` 候选。
2. 检查灰度方向：正片要求 `gap > wire_a` 且 `gap > wire_b`；负片相反。
3. 用第一组 wire 间距作为锚点，只保留 `dist <= 1.05 * first_dist` 的候选。
4. 如果后续候选中心距突然大幅跳变，截断尾部噪声，只保留物理连续前缀。

这一步解决了实际验证中出现的问题：D 标签位置正确但普通蓝色 valley 漏标。普通显示层峰谷检测不再作为 BAM 配对依据，BAM 只认最终 `bam_pairs`。

#### Dip 计算：当前采用单点极值法

`_compute_dip()` 仍保留 `half_w` 参数，支持邻域均值；但**当前 `compute_contrast()` 主路径固定使用 `half_w=0` 的单点极值法**：

```python
dip_half_w = 0
```

原因：实际样本中 D3 这类细线对的三元组可能是 `(84, 88, 92)`，半间距约 4px。若使用 `half_w=3` 的 7 像素窗口，窗口会把左右 valley 与中间 peak 相互混入，导致 D3 的 dip 从视觉上明显可分辨却被压到约 6%。改为单点法后，同一组 D3 约为 96%。

单点法退化形式：
```python
def _region_mean(center):
    lo = max(0, center - half_w)
    hi = min(L - 1, center + half_w)
    return float(profile[lo:hi+1].mean())

# 当前主路径 half_w=0，因此 _region_mean(center) == profile[center]
```

完整公式仍然相同：
```python
def _region_mean(center):
    lo = max(0, center - half_w)
    hi = min(L - 1, center + half_w)
    return float(profile[lo:hi+1].mean())

A = abs(float(background[wire_a]) - _region_mean(wire_a))
B = abs(float(background[wire_b]) - _region_mean(wire_b))
C = abs(float(background[gap_c])   - _region_mean(gap_c))
dip = 100.0 * (A + B - 2.0 * C) / (A + B)
return max(0.0, dip)  # clamp 到 [0, 100]
```

| 方案 | 方法 | 噪声鲁棒性 |
|------|------|-----------|
| **A（单点极值，当前主路径）** | 取检测位置的单一像素值 | 低，但不会抹平细丝 |
| B（邻域均值，保留能力） | 取检测位置周围窗口灰度均值 | 中，粗丝更稳但细丝会被抹平 |
| C（全段积分） | 用极值点间自然分段积分 | 高 |

**文献依据：** 2024 灰度直方图 20% 下凹法论文明确指出：
> "a、b：双丝中丝对对应各像素的**最小灰度的平均值**"
> "c：双丝对中像素**最高灰度值的平均值**"

实现上保留 `window_half_width` 参数作 API 兼容；当前 BAM 配对路径统一按 `half_w=0` 执行。后续若恢复邻域均值，应改为**按线对间距自适应窗口**，不能对所有 D 组固定使用 `half_w=3`。

### 3.5 Step 5：二次插值求 iSRb（可选）

标准离散判定精确到 ±1 组。BAM 用二次插值达到 ±0.5 组精度。

**BAM 实现**（`isrb.py:377-486`）：
```python
# 清理 Dip 序列：剔除不符合单调递减规律的异常值
while i < len(use_dips):
    if (use_dips[i] - use_dips[i-1]) > 5:  # 后续不得比前一个深 5% 以上
        use_dips = np.delete(use_dips, i-1)

# 找 20% crossing 点，取前后各 2 组参与插值
# 排除 dip < 1.5% 的低调制组（背景拟合误差被放大，破坏插值稳定性）

# 二次拟合 dip = f(wire_spacing)，解 f(d) = 20%
popt, _ = curve_fit(Interpolation.quadratic, dists, dips)
dips20 = Interpolation.inverted_quadratic(20, *popt)

# 根据曲率选择有效根
if self.a >= 0:
    self.dip20 = max(dips20)   # 开口向上 → 取大根
else:
    self.dip20 = min(dips20)   # 开口向下 → 取小根
```

**数据处理细节：**
- 排除 `dip < 1.5%` 的低调制组（"to prevent unpleasant fits"）
- 后续 Dip 不得比前一个深超过 5 个百分点，否则视为异常并丢弃
- 取 crossing 点前后各 2 组参与插值（共最多 5 组）

---

## 4. 正/负片统一性

公式 $R = \frac{a+b-2c}{a+b}$ 在正负片上**自然统一**：

| 片型 | a, b 对应 | c 对应 | 公式行为 |
|------|----------|--------|---------|
| 正片 | 两个波谷（低灰度） | 波峰（高灰度） | c > a,b → R 大 |
| 负片 | 两个波峰（高灰度） | 波谷（低灰度） | c < a,b → R 大 |

减背景后，A、B、C 都是 `|background - 实际值|`，方向性完全消除。

### 自动判定方法（`_detect_film_type`）

基础方法仍会找剖面中振幅最大的交替三元组（v-p-v 或 p-v-p）：

```python
# profile.py:220-276
# v-p-v with deep valleys → positive film  (wires are dark valleys)
# p-v-p with tall peaks   → negative film (wires are bright peaks)
if best_type == "v": return "positive"
if best_type == "p": return "negative"
```

但当前 `compute_contrast()` 不再只依赖这个结果。实际样本中，最大振幅三元组可能被背景平台或局部强响应误导，曾将正片误判为 `negative`。因此主路径额外构建 positive 候选序列并计算方向显著性：

```python
positive_scores = _pair_direction_scores(profile, positive_pairs, film_type="positive")
min_positive_score = 0.08 * profile_range
has_strong_positive_series = (
    len(positive_pairs) >= max(3, len(peaks) // 3)
    and median(positive_scores) >= min_positive_score
)
```

若存在足够强且连续的 positive 三元组序列，则判为 `positive`；否则回退到 `_detect_film_type()` 的基础三元组结果。这保证真实正片样本能判对，同时保留负片合成测试不被噪声 positive 候选误判。

---

## 5. 整体编排：compute_contrast

`compute_contrast()` 将上述步骤编排为完整的 orchestrator：

```
compute_contrast(profile, ...)
  │
  ├─ 1. 二次趋势去除
  │     └─ profile - polyfit(profile, deg=2)
  │
  ├─ 2. detect_peaks_valleys()          → peaks, valleys
  │     └─ 粗检测: min_distance/prominence
  │
  ├─ 3. detect_peaks_valleys()          → fine_peaks, fine_valleys
  │     └─ 细检测: min_distance=1, prominence=max(0.005, prominence*0.5)
  │
  ├─ 4. 片型判定
  │     ├─ _detect_film_type() 基础三元组判定
  │     └─ positive 候选序列 + 方向显著性校正
  │
  ├─ 5. 角色分配
  │     ├─ 正片: wire_positions = valleys, gap_positions = peaks
  │     └─ 负片: wire_positions = peaks,  gap_positions = valleys
  │
  ├─ 6. _fit_quadratic_background()     → background array
  │
  └─ 7. 配对 + dip
        ├─ 正片: _pair_adjacent_wires_with_gaps(fine_valleys, fine_peaks)
        ├─ 负片: _pair_wires_and_compute_dips(peaks, valleys)
        ├─ 1.05× 首组间距过滤 + 中心距前缀截断
        └─ 单点极值 _compute_dip(half_w=0)

返回 ComputeContrastResult(dips, pairs, background, film_type)
```

---

## 6. 与 BAM 原文的差异总结

| 维度 | BAM `isrb.py` | 本项目 `profile.py` |
|------|--------------|-------------------|
| 峰谷检测方向 | 只检测 valleys | 同时检测 peaks + valleys |
| 背景拟合方向 | 固定方向（假设已知片型） | 支持正/负片自动取反 |
| Dip 值来源 | 单像素值（`measure[idx]`） | 当前主路径为单点极值（`half_w=0`） |
| 配对策略 | `1.05 * dist[0]` | 正片用细粒度 `valley-peak-valley`，并增加中心距前缀截断 |
| 插值求 iSRb | ✓ 已实现 | Phase 2.5 可选（当前用离散判定） |
| 片型判定 | 不在 isrb 源码内（外部输入） | `_detect_film_type()` + positive 序列显著性校正 |
| 束带平均 | `profile_line` 单线 | 21 行并行采样 + `map_coordinates` 子像素插值 |
| 返回值 | `dip20`（插值丝径） | `ComputeContrastResult`（完整中间结果） |

---

## 7. 验证与调试脚本口径

### 7.1 `validate_bam_gt.py`

GT 验证脚本以最终 `compute_contrast()` 输出为准，验收指标为：

- `film_type` 与 GT 一致
- 算法 pair 数量等于 GT `num_wire_pairs`
- 无 extra pair / missing pair
- 每组 `(wire_a, gap, wire_b)` 最大点位误差 ≤ 5px
- 全部点位平均误差 ≤ 3px

当前样本 `outputs/double_wire_demo_3/...` 的验证结果：

```text
Validation result: PASS
Detected 8 wire pairs, GT has 8
Mean point error: 0.083px
Max triplet err: 1.000px
```

注意：脚本中的 raw extrema / parameter sweep 只是诊断信息，不参与 PASS/FAIL。真正验收只看 `bam_pairs` 对 GT 的三元组点位误差。

### 7.2 `double_wire_demo.py`

交互 demo 当前只展示真实 BAM 点：

- `BAM wires`：来自 `bam_pairs` 的 `wire_a / wire_b`
- `BAM gaps`：来自 `bam_pairs` 的 `gap`
- `D1:xx%` 等标签：来自 `bam_dips`

普通 `detect_peaks_valleys()` 的红色 peak 三角 / 蓝色 valley 三角已删除，原因是它们使用显示层粗参数，容易漏掉 D3-D8 细线对的一侧 valley，造成“D 标签正确但蓝点缺失”的误导。

---

## 8. 异常处理

| 异常情况 | 原因 | 处理 |
|---------|------|------|
| 峰谷数量不匹配 | 噪声假谷 | `prominence` 阈值过滤 |
| 谷间距过大 | 图像边缘或非丝区域 | `dist_max` 上限过滤（1.05×） |
| 谷间距过小 | 噪声 | `min_distance` 下限过滤 |
| valley 之间无 peak | 双丝已完全融合 | 该组 R ≈ 0，可直接判定为不可分辨 |
| 所有组 R ≥ 20% | 全部可分辨 | 返回全部可分辨，超出量程上限 |
| 最粗组即 R < 20% | 图像质量极差 | 返回 D1（最粗组不可分辨） |
| `dip < 1.5%` | 背景拟合误差放大 | 从插值邻域中排除 |
| 后续 dip 比前一个深 > 5% | 异常值 | 丢弃 |

---

## 9. 参考文献

1. **ISO 19232-5:2018** — Determination of the image unsharpness and basic spatial resolution value using duplex wire-type IQIs
2. **ASTM E2002-15** — Standard Practice for Determining Total Image Unsharpness and Basic Spatial Resolution
3. **BAM ctsimu-toolbox** (开源) — https://github.com/BAMresearch/ctsimu-toolbox，核心代码 `ctsimu/image_analysis/isrb.py`
4. **JBT 7902-2025** — 双丝型像质计
5. **Sun Chao-ming (2017)** — "Automatic Determination Method of the Modulation of Duplex Wire IQI", *Nondestructive Testing*, 39(2): 22-25
6. **2024 灰度直方图 20% 下凹法与内插值法** — 无损检测期刊, DOI: 10.11973/wsjc240371
7. **杨庆国等 (2025)** — "Determining performance of radiographic examination system with duplex-wire type IQI", *Optics and Precision Engineering*, 33(7): 1051-1064
8. **CN120253177A** — "数字射线成像系统高精度自动测量基本空间分辨率的方法"
