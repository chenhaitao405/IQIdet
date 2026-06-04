# BAM 双丝像质计算法原理

**日期：** 2026-06-04（根据最新代码重构后更新）
**来源：** BAM `ctsimu-toolbox` 源码分析 + `src/gauge/imaging/double_wire.py` 实现
**上游参考：** ISO 19232-5:2018、ASTM E2002-15、JBT 7902-2025

**代码结构（2026-06-04 重构后）：**

| 模块 | 职责 |
|------|------|
| `src/gauge/imaging/profile.py` | 剖面提取（`extract_profile_band`/`_strip`）、OBB 几何、`detect_peaks_valleys` |
| `src/gauge/imaging/double_wire.py` | BAM 分析核心：片型判定、背景拟合、丝配对、dip 计算、等级判定 |
| `scripts/double_wire/annotate.py` | 交互标注工具（OpenCV + matplotlib） |
| `scripts/double_wire/validate_bam_gt.py` | GT 验证脚本 |

---

## 1. 背景

BAM（Bundesanstalt für Materialforschung und -prüfung）开源了 `ctsimu-toolbox`，是目前最成熟的 ASTM E2002 双丝 IQI 自动分析实现。

本项目将核心算法本地化实现，分为两层：
- **double_wire.py**：纯算法层，无 GUI 依赖，可集成到自动推理 pipeline
- **annotate.py**：交互标注层，用于建立 ground truth

---

## 2. 核心公式：调制度 Dip

### 2.1 数学定义

ISO 19232-5 / ASTM E2002 定义的调制度（Modulation Depth / Dip）：

```
R = (A + B - 2·C) / (A + B) × 100%
```

其中 A、B、C 是相对于**局部背景**的偏离量，而非绝对灰度值：

```
A = |background[wire_a] - profile[wire_a]|
B = |background[wire_b] - profile[wire_b]|
C = |background[gap_c]   - profile[gap_c]|
```

**关键点**：减去背景后，A、B、C 变为丝/间隙相对于局部基线的**偏离幅度**，因此正负片上的 dip 计算公式完全统一。

### 2.2 物理含义

| 情况 | Dip 值 | 含义 |
|------|--------|------|
| 双丝清晰可辨 | 接近 100% | 两丝深、中间谷浅 → A+B 远大于 2C |
| 双丝勉强可辨 | ~20% | C 接近 (A+B)/2 |
| 双丝完全融合 | 接近 0% | A ≈ B ≈ C → 分子趋近 0 |

**判定标准：** R < 20% 时该线对不可分辨。从最粗组 D1 向最细遍历，第一个不可分辨组即为基本空间分辨率。

### 2.3 丝对参数表（JBT 7902-2025 表2）

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

> D1~D3 为钨丝，D4~D13 为铂丝。D14~D17 为扩展组（新团标）。

---

## 3. 算法流水线

`compute_contrast()` 是核心编排函数，共 7 步：

```
profile (1D, band-averaged)
    │
    ├─ 1. 去趋势（2次多项式拟合 → 相减）
    │      └─ 目的：消除 heel effect，为峰谷检测拉平基线
    │
    ├─ 2. 两级峰谷检测（均在去趋势后的剖面上）
    │      ├─ 粗检测：min_distance=10, prominence=5%（仅做 early-exit 检查）
    │      └─ 细检测：min_distance=1, prominence=2.5%（用于实际配对）
    │
    ├─ 3. 双向候选构建 + 片型自动判定
    │      ├─ 构建 positive 候选：peaks→wires, valleys→gaps
    │      ├─ 构建 negative 候选：valleys→wires, peaks→gaps
    │      ├─ 各自用 _pair_adjacent_wires_with_gaps() 配对
    │      └─ 比较 median contrast score → 选优胜方向为 pairing_ft
    │
    ├─ 4. 选择优胜方的 pairs + background
    │
    ├─ 5. _remove_overlapping_pairs()  → 删除共享 wire 的重复 pair
    │
    ├─ 6. _recover_tail_pair()         → 在尾部找回被 prominence 漏掉的最细丝对
    │
    └─ 7. 重新计算 dips（用清理后的 pair 列表）

返回 ComputeContrastResult(dips, pairs, background, film_type)
```

### 3.1 Step 1：去趋势（Detrend）

**这是峰谷检测前最关键的前处理。** X 光管阳极的 heel effect 使剖面基线呈缓慢弯曲，导致不同位置的峰/谷 prominence 不可比较。

```
detrended = profile - polyfit(profile, deg=2)
```

- 用 2 次多项式拟合全局背景弯曲（低频）
- 减去趋势后基线拉平，所有位置的峰谷按统一标准判定
- 2 次是"能表达曲率的最低阶多项式"——不会过拟合丝的调制信号（高频）
- **去趋势结果仅用于峰谷检测**，不参与 dip 计算

### 3.2 Step 2：两级峰谷检测

基于 `scipy.signal.find_peaks`，在**去趋势后的剖面**上检测：

| 检测 | min_distance | prominence | 用途 |
|------|-------------|------------|------|
| **粗检测** | 10 px | 5% × range | early-exit 检查（两方向都无峰谷则返回空） |
| **细检测** | 1 px | max(0.005, 2.5% × range) | 实际丝配对 |

需要细检测的原因：双丝越往细组，相邻 wire 与中间 gap 的间距可缩小到 1~4 像素。若用 `min_distance=10` 会漏掉细丝 wire；若全局调小 min_distance 又会引入尾部平台噪声。因此用细检测生成候选 → 通过物理配对规则过滤。

### 3.3 Step 3：背景拟合

使用 **Savitzky-Golay 低通滤波器**（`scipy.signal.savgol_filter`）估计背景曲线：

```python
window = min(n // 8 * 2 + 1, 201)  # 奇数，最大 201
background = savgol_filter(profile, window, 2, mode='mirror')
```

**为什么不用 2 次多项式拟合？**（RC-4 修复）

单次 2 次多项式在丝调制强烈的区域会 overshoot——背景曲线被迫跟随丝的峰谷走，导致 C（gap 偏离背景）被高估，可能出现 "C >> A, B" → dip 异常为负。SG 滤波器用宽窗口（约 profile 长度的 1/8）只跟踪低频趋势，自然忽略窄带丝特征。

**注意**：此"背景拟合"与 Step 1 的"去趋势"是两个独立步骤，服务于不同目的：

| 步骤 | 方法 | 用途 |
|------|------|------|
| Step 1 去趋势 | 2次多项式相减 | 峰谷检测的基线拉平 |
| Step 3 背景拟合 | SG 低通滤波 | dip 计算的局部背景 |

### 3.4 Step 4：丝配对（`_pair_adjacent_wires_with_gaps`）

**核心规则——BAM 首间距规则：**

```
ref_dist = 第一对 candidate 的丝间距
dist_max = max(2.0, ref_dist × 1.05)

相邻丝间距 ≤ dist_max → 配成一对
相邻丝间距 > dist_max → 不配对
```

配对流程：

1. 遍历相邻 wire 位置，找两者之间的所有 gap 极值点
2. **片型感知选择 gap**：
   - Positive（亮丝暗隙）：选灰度最低的 gap（最深暗谷）
   - Negative（暗丝亮隙）：选灰度最高的 gap（最亮峰）
3. **灰度方向验证**：gap 灰度必须在两丝之间（正片：`profile[gap] < profile[w1]` 且 `< profile[w2]`；负片相反）
4. **首间距过滤**：只保留 `dist ≤ dist_max` 的候选
5. **`_trim_pairs_to_stable_center_prefix`**：若某 pair 的中心间距比前面中位数跳变超过 1.8 倍或 +20px，截断后续

### 3.5 Step 5：片型自动判定

**不再依赖单一的 `_detect_film_type()` 三元组法。** 当前采用**双向假设 + 对比投票**：

```
同时构建 positive 假设 (peaks=wires, valleys=gaps) 和
          negative 假设 (valleys=wires, peaks=gaps)

对每种假设，计算 _pair_direction_scores() → 每对丝的对比度分数

判定逻辑（优先级递减）：
  1. 一方 pair 数 ≥3 而另一方 <3  → 选多的一方（RC-6）
  2. neg_med > pos_med              → negative
  3. pos_med > neg_med              → positive
  4. 平局                            → norm_mean ≥ 0.5 选 negative
```

**额外规则**：若判定 `pairing_ft = "positive"` 但 `norm_mean ≤ 0.38`（图像整体偏暗），
说明这是反相底片（negative film 经 photometric inversion），最终 `film_type` 仍标为 `"negative"`。

其中 `_pair_direction_scores()` 为每对丝计算方向对比度：
```python
# Positive: 两丝灰度都应高于间隙
score = min(profile[w1] - profile[gap], profile[w2] - profile[gap])

# Negative: 两丝灰度都应低于间隙
score = min(profile[gap] - profile[w1], profile[gap] - profile[w2])
```

### 3.6 Step 6：Dip 计算（`_compute_dip`）

**当前主路径使用单点极值法（half_w=0）**：

```python
def _compute_dip(profile, wire_a, gap_c, wire_b, background, half_w=0):
    # half_w=0 时 _region_mean(center) == profile[center]
    A = |background[wire_a] - profile[wire_a附近平均]|
    B = |background[wire_b] - profile[wire_b附近平均]|
    C = |background[gap_c]   - profile[gap_c附近平均]|

    dip = 100 × (A + B - 2C) / (A + B)
    return max(0.0, dip)
```

**为什么用 half_w=0？** D6~D8 这类细丝对的三元组可能只有 3~5px 间距。若用 `half_w=3`（7px 窗口），会把左右 wire 与中间 gap 的灰度相互混入，导致 dip 被严重压低。单点法避免了细丝的窗口抹平问题。

| 方案 | half_w | 优点 | 缺点 |
|------|--------|------|------|
| 单点极值（当前） | 0 | 细丝不被抹平 | 对单像素噪声敏感 |
| 邻域均值 | 3 | 粗丝更稳 | 细丝 dip 被压低 |
| 自适应窗口 | 按间距变化 | 兼顾粗细 | 实现复杂 |

### 3.7 Step 7：后处理清理

#### 7a. 去重叠 pair（`_remove_overlapping_pairs`）

当两个相邻候选 pair 共享同一根 wire（第一个的 wire_b == 第二个的 wire_a），说明其中一个是噪声假阳性。保留 gap-wire 对比度更强的那个。

#### 7b. 尾部恢复（`_recover_tail_pair`）

最细丝对（如 D13）的调制深度可能太低，连细检测的 prominence 阈值都达不到。此函数在尾部区域用极低阈值（1% prominence）重新扫描候选丝对，验证其位置和 dip 的合理性后追加。

#### 7c. 单调性清理（`_cleanup_dips_monotonic`）

粗丝→细丝，dip 应单调递减。若后续 dip 比前一个深超过 5 个百分点，视为检测异常，删除前一个较浅的 pair。

### 3.8 未分辨组判定（`find_first_unresolved_group`）

```
1. _cleanup_dips_monotonic() → 清理反常 dip
2. 遍历 dips[]: 第一个 dip < 20% 的位置即为 crossing 点
3. 在 crossing 点附近取 ±2 组数据做 2次多项式插值
4. 解 polyfit(spacings, dips, 2) = 20% → 细化 crossing 丝径
5. 返回 1-indexed 组号（D1=1, D2=2, ...），全部可辨返回 None
```

---

## 4. 整体编排流程图

```
                    extract_profile_band(image, start, end, band_width=21)
                                   │
                                   ▼
                          1D profile (float64)
                                   │
                     ┌─────────────┴─────────────┐
                     │   compute_contrast(profile) │
                     └─────────────┬─────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    ▼                    │
              │  1. detrended = profile - polyfit²      │
              │                    │                    │
              │  2. 粗检测       细检测                 │
              │     (d=10,p=5%)  (d=1,p=2.5%)          │
              │                    │                    │
              │  3. ┌──────────────┴──────────────┐     │
              │     │  双向候选构建 + 片型判定      │     │
              │     │  pos: peaks→wires, valleys→gaps │  │
              │     │  neg: valleys→wires, peaks→gaps │  │
              │     │  比较 median score → 选优胜方   │     │
              │     └──────────────┬──────────────┘     │
              │                    │                    │
              │  4. 选择优胜方的 pairs + background    │
              │                    │                    │
              │  5. _remove_overlapping_pairs()        │
              │                    │                    │
              │  6. _recover_tail_pair()               │
              │                    │                    │
              │  7. recompute dips → ComputeContrast   │
              └────────────────────┼────────────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │ find_first_unresolved_group │
                     │   → 1-indexed D group or None│
                     └─────────────────────────────┘
```

---

## 5. 与 BAM 原文的差异总结

| 维度 | BAM `isrb.py` | 本项目 `double_wire.py` |
|------|--------------|------------------------|
| 峰谷检测方向 | 只检测 valleys | 同时 peaks + valleys |
| 去趋势 | 无（直接对原始 profile 操作） | 2次多项式去趋势后再检测峰谷 |
| 背景拟合 | `curve_fit(quadratic)` 对掩膜后的间隙区 | `savgol_filter` 低通滤波（RC-4） |
| 片型判定 | 不在 isrb 内（外部输入） | 双向假设 + 对比投票 + 亮度归一化兜底 |
| 配对函数 | `_pair_wires_and_compute_dips` | `_pair_adjacent_wires_with_gaps`（增加灰度方向验证 + 中心距前缀截断） |
| Dip 计算 | 单像素值 | 当前 half_w=0（等效单点），保留窗口参数 |
| 参考间距 | `dist[0]`（第一对） | `median(dist[:5])`（前5对中位数，RC-3） |
| 后处理清理 | 无 | 去重叠 + 尾部恢复 + 单调性清理 |
| 剖面提取 | `profile_line` 单线 | 21 行双线性子像素插值 + 平均（JBT 7902） |
| 返回值 | `dip20`（插值丝径） | `ComputeContrastResult`（完整中间结果） |

---

## 6. 验证工具

### 6.1 交互标注工具（`annotate.py`）

```
conda activate weld-gpu
python scripts/double_wire/annotate.py <image_path> [--band-width 21] [--expand 60]
```

- OpenCV 窗口：2 点画剖面线 → 自动锁定
- matplotlib 窗口：显示 strip 图 + 剖面曲线 + BAM 检测结果
- 标注模式：左键点剖面标记 peak/valley → 建立 ground truth
- 保存：同时输出原图 + 反相(255-x) 双版本

### 6.2 GT 验证（`validate_bam_gt.py`）

以 `compute_contrast()` 输出为准，验收指标：
- `film_type` 与 GT 一致
- 算法 pair 数量等于 GT `num_wire_pairs`
- 无 extra / missing pair
- 每组 `(wire_a, gap, wire_b)` 最大点位误差 ≤ 5px
- 全部点位平均误差 ≤ 3px

---

## 7. 异常处理

| 异常情况 | 原因 | 处理 |
|---------|------|------|
| 剖面过短（< 3px） | 无效输入 | 返回空结果 |
| 两方向均无峰谷 | 剖面平坦无信号 | 返回空结果 |
| 片型判定平局 | 双向分数相等 | 用 `norm_mean`（全局亮度）裁决 |
| 一方候选过少 | 稀疏假阳性主导 | 直接选候选多的一方（RC-6） |
| 共享 wire 的重叠 pair | 噪声假阳性 | 保留 contrast score 更高者 |
| 尾部细丝漏检 | prominence 不足 | `_recover_tail_pair` 低阈值扫描 |
| dip 非单调递减 | 检测异常 | `_cleanup_dips_monotonic` 剔除 |
| 全部 dip ≥ 20% | 全部可分辨 | 返回 None（超出量程上限） |
| 最粗组 dip < 20% | 图像质量极差 | 返回 D1 |
| dip < 1.5% | 背景拟合误差被放大 | 从插值邻域排除 |

---

## 8. 参考文献

1. **ISO 19232-5:2018** — Determination of the image unsharpness and basic spatial resolution value using duplex wire-type IQIs
2. **ASTM E2002-15** — Standard Practice for Determining Total Image Unsharpness and Basic Spatial Resolution
3. **BAM ctsimu-toolbox** (开源) — https://github.com/BAMresearch/ctsimu-toolbox，核心代码 `ctsimu/image_analysis/isrb.py`
4. **JBT 7902-2025** — 双丝型像质计
5. **Sun Chao-ming (2017)** — "Automatic Determination Method of the Modulation of Duplex Wire IQI", *Nondestructive Testing*, 39(2): 22-25
6. **2024 灰度直方图 20% 下凹法与内插值法** — 无损检测期刊, DOI: 10.11973/wsjc240371
7. **杨庆国等 (2025)** — "Determining performance of radiographic examination system with duplex-wire type IQI", *Optics and Precision Engineering*, 33(7): 1051-1064
