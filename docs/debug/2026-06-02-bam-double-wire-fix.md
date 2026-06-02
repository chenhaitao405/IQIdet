# BAM 双丝算法修复记录

**日期：** 2026-06-02
**基线：** `python scripts/double_wire/validate_bam_gt.py "outputs/double_wire_demo"` — 0/14 PASS
**最终：** 14/14 PASS，新增样本 1/1 PASS

## 基线状态

| 指标 | 基线 (a28f6aa) | 修复后 (dcc6f79) |
|------|---------------|-----------------|
| 片型匹配 | 0/14 (全部误判为 positive) | 14/14 |
| 配对数量 | 多样本 extra/missing | 14/14 |
| 最大 MAE | 63.9px | 0.125px |
| 最大 triplet err | 100px | 1.0px |
| PASS | 0/14 | 14/14 |

---

## RC-1: 片型判定 — 正片候选修正过于激进

**根因：** `compute_contrast()` 中的 `has_strong_positive_series` 检查总是通过，导致 `_detect_film_type` 被永远覆盖，14 个样本全部误判为 `positive`。

**数据证据：** 所有 ori 样本 GT film_type 为 `negative`，但算法始终返回 `positive`。

**修复：** (commit `2bfba42`)
1. 同时构建正片/负片两个方向的细粒度候选（之前只构建正片）
2. 比较 median direction score，高分者胜出（替代永远选 positive 的阈值检查）
3. 新增 photometric inversion 检测：`norm_mean <= 0.38` 时说明 profile 被反相，报告 `film_type="negative"` 但使用正片配对方向
4. 平分情况用 `norm_mean >= 0.5` 打破平局

---

## RC-2: 配对数量 — 重叠假对 (overlapping spurious pairs)

**根因：** 细极值检测产生的噪声极值导致相邻的候选 pair 共享同一个 wire 位置（pair[i].wire_b == pair[i+1].wire_a），其中一个是假对。

**数据证据：** A06+005（7 GT → 8 algo）、A06+009-01（8 GT → 9 algo）均有额外 pair。

**修复：** (commit `1d75e77`)
- 新增 `_remove_overlapping_pairs()` 函数
- 检测共享 wire 位置的连续候选对，保留 gap-wire 对比度更强的真对
- 在 `compute_contrast()` 片型选定后执行，不影响片型评分的计算

---

## RC-3: 配对数量 — 尾部漏对 (missing finest pair)

**根因：** 最细线对（D8 或 D6）的 extrema 调制深度低于 fine prominence 阈值 `max(0.005, prominence * 0.5)`，细极值检测完全遗漏。

**数据证据：** A05+002（D8 缺失）、A20-001（D8 缺失）、A06+009-16（D6 缺失）。

**修复：** (commit `1d75e77`)
- 新增 `_recover_tail_pair()` 函数
- 在主配对完成后，在 profile 尾部用放松的门槛搜索遗漏对
- 三重门控：位置合理性（在间距趋势预测的 0.6× delta 范围内）+ dip 单调性（≤ 前一真对 dip 的 1.2×）+ 最小 dip（≥ 2%）

---

## RC-4: 负片路径用 coarse extrema 导致配对流偏移

**根因：** `compute_contrast()` 负片分支调用 `_pair_wires_and_compute_dips()` 使用 coarse extrema（`min_distance=5`），而细线对间距仅 1-4px，大量 wire 位置被漏掉。

**数据证据：** 强制 `film_type="negative"` 时 ori 样本仅检测到 5 对（GT=8）。

**修复：** (commit `2bfba42`)
- 负片分支改为 `_pair_adjacent_wires_with_gaps(profile, fine_valleys, fine_peaks, ...)`，与正片分支对称使用 fine extrema

---

## RC-5: 移除 coarse extrema 的角色分配

**关联修复：** 与 RC-4 同时处理。旧代码将 coarse extrema 分配为 wire/gap 角色后传给 `_pair_wires_and_compute_dips`，修复后直接使用 fine extrema 配对，coarse extrema 仅用于早期空值检查。

---

## RC-6: 稀疏假对主导分数对比 (额外样本)

**根因：** 新样本（A06+005-01 inver）的负方向仅找到 2 个假 pair，但它们在背景陡坡上 gap-wire 对比度极高（scores [62.2, 50.1]），median 反而超过了有 8 个正确 pair 的正方向（median 24.4）。

**修复：** (commit `dcc6f79`)
- 片型选择新增 pair 数量门槛：一方 <3 pairs 而另一方 ≥3 时，跳过分数对比，直接选多的那方

---

## 修改文件

仅修改 `src/gauge/imaging/profile.py`，共 3 个 commit：

| Commit | 内容 |
|--------|------|
| `2bfba42` | 双向片型检测 + 反相检测 + 负片路径 fine extrema |
| `1d75e77` | 重叠对去除 + 尾部细对恢复 |
| `dcc6f79` | 稀疏假对防护 (RC-6) |

## 新增函数

- `_remove_overlapping_pairs(pairs, profile, film_type)` — 删除共享 wire 位置的重叠假对
- `_recover_tail_pair(profile, pairs, background, half_w, film_type)` — 在 profile 尾部恢复遗漏的最细对

## 验证命令

```bash
conda activate weld-gpu
PYTHONPATH=$(pwd):$(pwd)/src python scripts/double_wire/validate_bam_gt.py "outputs/double_wire_demo"
```

---

## 下一步：OBB 扰动增强验证计划

### 动机

实际使用中发现 OBB 选点对算法结果影响很大——同一张图，OBB 角点差几个像素就可能导致识别失败。当前 14 个 profile 的验证集只覆盖了单一的 OBB 标注，不足以评估算法对 OBB 变化的鲁棒性。

### 目标

通过微扰动 OBB 角点生成 profile 变体，验证算法在 OBB 不确定性下的鲁棒性。**最终目标：原始 14 个 profile + 所有扰动变体 100% PASS。**

### 数据布局

```
outputs/double_wire_demo/<样本名>/
  ori/
    *_profile.json        ← 含 obb_corners_raw（4×2 角点）、profile_values（1D）
    *_groundtruth.json    ← 含 wire_pairs（每组 wire_a/gap/wire_b 的 profile 索引）
  inver/
    *_inverted_profile.json
    *_inverted_groundtruth.json

outputs/候选双丝像质计/<样本名>.jpg   ← 原始图像
```

### 执行步骤

#### Step 1: GT 反投（profile 索引 → 原图坐标）

- 从 `groundtruth.json` 读取每组 `wire_a_idx, gap_idx, wire_b_idx`（profile 坐标系索引）
- 利用 `obb_corners_raw` 构建透视变换矩阵，做正向映射：
  - profile 索引 i → unwarp 图像列坐标 x = i（profile 第 i 个采样点对应 unwarp 图像第 i 列）
  - unwarp 列坐标 x → 原图坐标：通过 `cv2.getPerspectiveTransform` 的逆矩阵
- 每个 GT 关键点（wire_a, gap, wire_b）映射为原图坐标 (u, v)
- 保存为 `gt_image_points.json`（与原 profile 同级目录）

#### Step 2: OBB 扰动生成

对每个样本的 `obb_corners_raw`（4 个角点，TL-TR-BR-BL）生成 N 个变体（建议 N=5~8）：

- 每个角点独立加高斯噪声（σ = 2~5 px），模拟人工点击误差
- 可选：整体平移（±5~10 px）、整体旋转（±1~2°）
- 要求扰动后的 OBB 仍构成合理四边形（不自交、面积不剧烈变化）

#### Step 3: Profile 重提取 + GT 重投影

对每个扰动 OBB 变体：
- 用 `extract_profile_band()` 从原图重新提取 profile（`band_width=21`）
- 用新 OBB 构建正向透视变换，将 Step 1 的 GT 原图坐标重新投影回 profile 索引
- 生成新的 `*_profile.json` 和 `*_groundtruth.json`
- 变体输出目录：`outputs/double_wire_obb_aug/<样本名>/<变体ID>/`

#### Step 4: 全量验证

```bash
PYTHONPATH=$(pwd):$(pwd)/src \
  python scripts/double_wire/validate_bam_gt.py "outputs/double_wire_obb_aug"
```

产出基线报告，标记所有 FAIL 的变体。

#### Step 5: 诊断修复

对每个 FAIL 变体，按分层优先级诊断：

| Layer | 检查项 | 阈值 |
|-------|--------|------|
| L1 | 配对数量 == GT num_wire_pairs | exact |
| L2 | film_type == GT film_type | exact |
| L3 | max triplet err ≤ 5px, MAE ≤ 3px | 5px / 3px |

优先修算法层（`profile.py`），避免为每个变体单独调整参数。

### 实施脚本

建议新建 `scripts/double_wire/augment_obb_variants.py`，包含：
- `back_project_gt(profile_json, gt_json) -> gt_image_points` — GT 反投
- `perturb_obb(corners, sigma, seed) -> perturbed_corners` — OBB 扰动
- `reproject_gt(gt_image_points, perturbed_corners, image) -> new_gt` — GT 重投影
- `generate_variants(sample_dir, n_variants) -> None` — 主流程

### 提交规则

| 阶段 | 提交内容 |
|------|---------|
| 脚本完成 + 基线报告 | OBB 扰动脚本 + `gt_image_points.json` + 基线验证结果 |
| 每修复一类算法问题 | 至少一个变体从 FAIL→PASS，提交信息写明根因和修复 |
| 最终 | 所有原始 + 变体全部 PASS
