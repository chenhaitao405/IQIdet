# BAM ctsimu-toolbox 双丝像质计算法实现

**创建日期：** 2026-05-31
**状态：** 数据准备阶段
**参考：** [[方案调研]], `src/3rdparty/ctsimu-toolbox`

## 目标

在 `src/gauge/imaging/profile.py` 的 `compute_contrast()` 和 `find_first_unresolved_group()` stub 中实现 BAM ctsimu-toolbox 的 dip 计算与 iSRb 插值算法。

## 阶段

### Phase 1: 数据准备 ✅ 进行中
- [x] Clone ctsimu-toolbox 到 `src/3rdparty/`
- [x] 添加 `src/3rdparty/` 到 `.gitignore`
- [ ] `double_wire_demo.py` 保存 OBB 图像 + 剖面数据
- [ ] 标注脚本: 手动标记峰/谷作为 ground truth

### Phase 2: BAM 算法实现
- [ ] `compute_contrast()`: 二次背景拟合 + 谷配对 + dip 计算
- [ ] `find_first_unresolved_group()`: dip 单调性清理 + iSRb 插值

### Phase 3: 验证
- [ ] 测试脚本: BAM 输出 vs ground truth 对比
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
├── <stem>_overlay.png      # 标注叠加图（已有）
├── <stem>_profile.json     # 剖面数据 + 峰谷检测结果（已有）
├── <stem>_obb.png          # unwarp OBB 图像（新增）
└── <stem>_groundtruth.json # 人工标注 ground truth（新增）
```
