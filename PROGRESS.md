# 项目进度

> 最后更新：2026-05-30

## 当前阶段

**Phase 1：单丝像质计等级识别** ✅ 已完成

- 交付入口 `run_iqi_grade_infer.py` 可产出一致性 JSON
- FClip 丝数推理、YOLO-OBB ROI 检测、PaddleOCR 全图/ROI OCR 均已集成
- 区域 OCR 和区域 SNR 旁路 API 可用
- 详细交付说明见 `docs/README_IQI_GRADE_INFERENCE_DELIVERY.md`

## 下一阶段

**Phase 2：双丝像质计分辨率计算** 🔜 即将开始

需求文档：`docs/需求/双丝像质计分辨率.md`

核心任务：
- 检测双丝像质计各组丝的位置
- 按波峰两波谷法计算每组的 Contrast（$Contrast = \frac{a + b - 2c}{a + b}$）
- 从粗到细遍历，找到第一个 Contrast < 20% 的组，确定最小可识别分辨率
- 支持正片（两黑夹一白）和负片（两白夹一黑）两种图像类型

已知约束：
- 双丝图像以 `GB`/`ISO` 文件名前缀区分
- 团标即将发布，双丝组数可能从 13 组扩展到 15-18 组，需支持分段处理
- 非分段双丝：所有组 Contrast 均 > 20% 时返回最细组（超出量程上限）

## 已完成的前置工作

- `scripts/tool/filter_nocrack_no_roi.py`：从 nocrack 数据集中筛选出未检测到单丝 ROI 的图像（即候选双丝像质计）
- 候选图像已保存到 `outputs/候选双丝像质计/`（311 张）
- 这些图像可用于双丝像质计定位和分辨率计算的开发和验证

## 已知问题 / 待修复

- `TODO.md` 中待办事项待整理

## 文件定位速查

| 目的 | 路径 |
|------|------|
| 代码架构 | `ARCHITECTURE.md` |
| 交付说明 | `docs/README_IQI_GRADE_INFERENCE_DELIVERY.md` |
| 模型资产 | `docs/MODEL_ASSETS.md` |
| 单丝等级规则契约 | `docs/contract/IQI_SINGLE_WIRE_MARKER_GRADE_RULE.md` |
| 双丝需求 | `docs/需求/双丝像质计分辨率.md` |
| 待办事项 | `TODO.md` |
| Agent 指南 | `AGENTS.md` |
