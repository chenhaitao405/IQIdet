# IQIdet Agent Guide

本文件给维护本仓库的 agent 使用。开始改代码前先读 `ARCHITECTURE.md`，模型资产更新先读 `docs/MODEL_ASSETS.md`。

## 入口边界

根目录只保留对外交付入口和导入门面：

- `run_iqi_grade_infer.py`：统一 IQI 批处理入口，默认输出交付 JSON；本地调试可用隐藏整图矫正参数。
- `region_ocr_api.py`：区域 OCR 对外导入门面。
- `region_SNR_api.py`：区域归一化信噪比对外导入门面。

本地调试脚本放在 `scripts/debug/`：

- `scripts/debug/fclip_valid.py`
- `scripts/debug/run_region_ocr_batch.py`

不要在根目录新增临时调试脚本。确实需要长期保留的调试入口应放到 `scripts/debug/` 并写清用途。

## 代码边界

- `src/` 是源码根目录；根目录入口会把 `src/` 加入 `sys.path`，Python 包名仍是 `gauge`、`FClip`、`dataset`。
- `src/gauge/` 是自有推理和业务规则核心。
- `src/gauge/iqi_inferencer.py` 是完整 IQI 推理编排层。
- `src/gauge/iqi_rules.py` 是 OCR 字段、像质计标识和等级判断的纯规则层。
- `src/FClip/` 是像质丝模型代码，业务入口应通过 `src/gauge/fclip_stage.py` 适配。
- `src/dataset/` 是 FClip 训练数据预处理代码。
- `OCRtrain/third_party/PaddleOCR` 是第三方子模块，优先不要直接改第三方源码。
- `models/` 是交付模型目录，模型大文件通过 Git LFS 管理。

## 修改约束

- 对外交付 CLI 参数不要随意扩张；只面向本地维护的参数应使用 `argparse.SUPPRESS` 隐藏。
- 修改交付 JSON 字段时，同步检查 `docs/README_IQI_GRADE_INFERENCE_DELIVERY.md`。
- 修改主流程、目录边界或训练链路时，同步更新 `ARCHITECTURE.md`。
- 更新模型文件或模型路径时，同步更新 `docs/MODEL_ASSETS.md`。
- 不要把 `outputs/`、`logs/`、`local/`、`OCRtrain/runs/` 当作源码接口。

## 常用验证

纯 Python 入口和轻量测试可运行：

```bash
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py tests/test_iqi_delivery_record.py tests/test_iqi_inferencer.py tests/test_region_snr_service.py tests/test_script_layout.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_delivery_record.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_inferencer.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_snr_service.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_script_layout.py
```

真实模型推理、OCR worker、PaddleOCR 和 FClip 的端到端验证依赖本机模型和 GPU 环境，按交付文档命令单独执行。
