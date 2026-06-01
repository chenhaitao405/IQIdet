# BAM 双丝像质计算法 — 开发与验证 SOP

**适用分支**: `feature/BAM-validation-fix`（或任意 feature 分支）
**前置环境**: `conda activate weld-gpu`
**工作目录**: 仓库根目录

---

## 环境变量

修改路径时只需改下面 `export` 的值，后续命令直接复制使用。

```bash
export IMG="outputs/候选双丝像质计/wqxDR__SHLNG-PED-A05+002-Z-NJ01__01.jpg"
export OUT_DIR="outputs/double_wire_demo_3"
export PROFILE="${OUT_DIR}/wqxDR__SHLNG-PED-A05+002-Z-NJ01__01_profile.json"
export GT="${OUT_DIR}/wqxDR__SHLNG-PED-A05+002-Z-NJ01__01_groundtruth.json"
export PYTHONPATH=".:./src"
```

---

## 流程概览

```
┌─────────────────────────────────────────┐
│ Step 1: double_wire_demo.py             │
│   交互选 OBB → 锁定 → S 保存剖面 JSON    │
├─────────────────────────────────────────┤
│ Step 2: annotate_profile.py             │
│   交互标注峰/谷 → S 保存 groundtruth     │
├─────────────────────────────────────────┤
│ Step 3: 修改 src/gauge/imaging/         │
│   profile.py 中的算法逻辑               │
├─────────────────────────────────────────┤
│ Step 4: validate_bam_gt.py              │
│   算法 vs groundtruth → 验证报告         │
└─────────────────────────────────────────┘
```

---

## Step 1: 采集剖面数据

```bash
python scripts/debug/double_wire_demo.py "${IMG}" --output-dir "${OUT_DIR}"
```

**交互操作：**
1. 左键依次点击 OBB 四角（TL→TR→BR→BL，沿丝对排列方向）
2. Enter 锁定 OBB
3. 拖动 trackbar 调节剖面偏移位置，使剖面线穿过丝对中心
4. 按 **S** 保存结果

**产出文件（`${OUT_DIR}/`）：**
- `*_profile.json` — 剖面数据 + 峰谷检测 + BAM dips
- `*_obb.png` — unwarp OBB 图像
- `*_overlay.png` — 标注叠加图

---

## Step 2: 人工标注 Ground Truth

```bash
python scripts/debug/annotate_profile.py "${PROFILE}"
```

**交互操作：**
1. **p 键** → peak 模式，左键在剖面曲线上标注丝峰（亮区）
2. **v 键** → valley 模式，左键标注间隙谷（暗区）
3. 右键删除误标，**u 键**撤销
4. 标注完所有 7 组丝对后，按 **S** 保存

**产出文件：**
- 默认生成在输入 `*_profile.json` 的同目录下，文件名为同 stem 的 `*_groundtruth.json`
  - 示例：`${GT}`
- 如需指定其他位置，使用 `--output <path>`

**GT 标注规则：**
- 负片：丝 = 亮区（peak），间隙 = 暗区（valley）
- 每组标注 3 个点：左丝—间隙—右丝
- 双丝组的两个相邻 valley 会自动配对

---

## Step 3: 修改算法

算法代码位于：

```
src/gauge/imaging/profile.py
```

**核心函数：**

| 函数 | 职责 |
|------|------|
| `compute_contrast()` | 编排：去趋势 → 峰谷检测 → 片型判定 → 背景拟合 → 丝配对 → dip |
| `find_first_unresolved_group()` | 单调性清理 → 插值 → 返回首个未分辨组号 |
| `_detect_film_type()` | 交替极值三元组判定正/负片 |
| `_fit_quadratic_background()` | SG 低通滤波背景估计 |
| `_pair_wires_and_compute_dips()` | 1.05× 间距因子配对 + 逐组 dip |
| `_compute_dip()` | 邻域均值 + 背景减除 dip 公式 |
| `_cleanup_dips_monotonic()` | 单调性异常清理 |
| `_find_crossing_group()` | 二次插值 crossing 判定 |

**修改后检查编译：**

```bash
python -m py_compile src/gauge/imaging/profile.py
```

**运行单元测试：**

```bash
python -m unittest tests.test_double_wire_profile -v
```

---

## Step 4: 验证

```bash
python scripts/debug/validate_bam_gt.py "${PROFILE}" "${GT}"
```

**附加 `--vis` 可输出可视化图表：**

```bash
python scripts/debug/validate_bam_gt.py "${PROFILE}" "${GT}" --vis
```

**产出：**
- 终端输出：配对对比、位置误差、dip 值、参数敏感度
- `${OUT_DIR}/validation_report.txt`（报告文本）
- `${OUT_DIR}/validation_vis.png`（可视化，仅 `--vis` 时）：
  - Panel 1 — 逐对 GT vs 算法对比（每对单独子图，显示 Δ 误差）
  - Panel 2 — 算法全貌（SG 背景 + 丝对阴影 + dip 标注）
  - Panel 3 — GT vs 算法全局叠加

---

## 快速迭代（steps 3→4 循环）

```bash
# 1. 改代码
vim src/gauge/imaging/profile.py

# 2. 编译检查
python -m py_compile src/gauge/imaging/profile.py

# 3. 单元测试
python -m unittest tests.test_double_wire_profile -v

# 4. GT 验证
python scripts/debug/validate_bam_gt.py "${PROFILE}" "${GT}"

# 5. 提交
git add -A && git commit -m "fix(BAM): ..."
```

---

## 关键调参项

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| `min_distance` | 5 | `compute_contrast` 调用处 | 峰/谷最小间距(px)，越小检出越多 |
| `prominence` | 0.03 | `compute_contrast` 调用处 | 峰显著性阈值(相对动态范围) |
| `window_half_width` | 3 | `compute_contrast` 参数 | dip 邻域均值窗口半宽 |
| SG `window` | `n // 8 * 2 + 1` | `_fit_quadratic_background` | 背景低通滤波窗口大小 |
| `dist_factor` | 1.05 | `_pair_wires_and_compute_dips` | 配对间距因子 |
| `dip_threshold` | 20.0 | `find_first_unresolved_group` | 不可分辨阈值(%) |
