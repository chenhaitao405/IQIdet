# src/gauge

`src/gauge` 是 IQIdet 的自有推理和业务规则核心目录。

主要边界：

- `iqi_inferencer.py`：完整 IQI 推理服务门面，负责资源生命周期和 public inference API。
- `pipeline.py`：Stage 编排器。
- `stages/`：图像读取、方向矫正、全图 OCR、ROI 检测、ROI OCR、丝数识别、等级融合。
- `services/`：OCR、ROI、FClip、区域 OCR/SNR 等运行时适配。
- `models/`：Pydantic 数据模型。
- `iqi_rules.py`：OCR 字段、像质计标识和等级判断纯规则层。

完整架构说明见仓库根目录 `ARCHITECTURE.md`。
