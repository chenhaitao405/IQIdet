# BAM ctsimu-toolbox 双丝像质计算法实现

**创建日期：** 2026-05-31
**状态：** Phase 2 — 算法实现
**分支：** `feature/BAM`
**参考：** [[方案调研]], `src/3rdparty/ctsimu-toolbox`

## Ground Truth 数据

| 文件 | 路径 |
|------|------|
| 输入剖面 | `outputs/double_wire_demo/wqxDR__SHLNG-PED-A05+002-Z-NJ01__01_profile.json` |
| 人工标注 | `outputs/double_wire_demo/groundtruth.json` |

**GT 关键参数：**
- 片型：**负片**（negative）— 丝=亮区(peak)，间隙=暗区(valley)
- 丝对数量：7 组（D1~D7）
- 谷间距范围：29~38 px
- dip 趋势：D1 深 → D7 浅（几乎融合）
- OBB 尺寸：519×118 px

## 目标

在 `src/gauge/imaging/profile.py` 的 `compute_contrast()` 和 `find_first_unresolved_group()` stub 中实现 BAM ctsimu-toolbox 的 dip 计算与 iSRb 插值算法。

## 阶段

### Phase 1: 数据准备 ✅ 完成
- [x] Clone ctsimu-toolbox 到 `src/3rdparty/`
- [x] 添加 `src/3rdparty/` 到 `.gitignore`
- [x] `double_wire_demo.py` 保存 OBB 图像 + 剖面数据
- [x] 标注脚本 + ground truth 产出

### Phase 2: BAM 算法实现 ⏳ 当前
- [ ] `compute_contrast()`: 二次背景拟合 + 谷配对 + dip 计算
- [ ] `find_first_unresolved_group()`: dip 单调性清理 + iSRb 插值
- [ ] 正/负片自动判定

### Phase 3: 验证
- [ ] 测试脚本: BAM 输出 vs `groundtruth.json` 对比
- [ ] 峰谷检出率、配对准确率、dip 误差评估

## BAM 算法关键差异（对比现有实现）

| 步骤 | BAM | 现有 |
|------|-----|------|
| 背景拟合 | `curve_fit` 二次曲线 | 无（直接用原始灰度） |
| 谷配对 | `dist_max = 1.05 * dist[0]` | 伪代码 |
| Dip 清理 | 单调性检查 + dip<1.5% 排除 | 无 |
| iSRb 求解 | 二次插值 f(d)=20% | stub |

## 数据文件约定

```
outputs/double_wire_demo/
├── wqxDR__SHLNG-PED-A05+002-Z-NJ01__01_overlay.png       # 标注叠加图
├── wqxDR__SHLNG-PED-A05+002-Z-NJ01__01_profile.json      # 剖面数据
├── wqxDR__SHLNG-PED-A05+002-Z-NJ01__01_obb.png           # unwarp OBB 图像
└── groundtruth.json                                       # 人工标注 ground truth
```
