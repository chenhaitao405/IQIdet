# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

IQIdet 用于焊缝底片像质计（IQI）等级识别：定位像质计 ROI → OCR 识别标识 → FClip 识别丝数 → 规则层计算等级。

## 开发环境

```bash
# 激活 conda 环境（运行任何命令前先执行）
conda activate weld-gpu

# 语法检查（纯 Python，无构建步骤）
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py tests/test_iqi_delivery_record.py tests/test_iqi_inferencer.py tests/test_region_snr_service.py tests/test_script_layout.py

# 运行测试（必须设置 PYTHONPATH）
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_rules.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_inferencer.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_delivery_record.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_snr_service.py
```

测试使用标准库 `unittest`，没有 pytest 依赖。`test_iqi_rules.py` 是纯规则层测试，不需要 mock 模型；`test_iqi_inferencer.py` 通过 `types.ModuleType` + `unittest.mock` 绕过 fclip/ocr stage 模块导入。

## 代码边界

进入代码前先读 `ARCHITECTURE.md` 了解完整架构，模型资产更新先读 `docs/MODEL_ASSETS.md`。

核心约束：

- `src/` 是源码根目录。入口脚本把 `src/` 加入 `sys.path`，Python 包名仍是 `gauge`、`FClip`、`dataset`。
- `src/gauge/iqi_inferencer.py` 是完整 IQI 推理编排层，`src/gauge/iqi_rules.py` 是纯规则层。
- `src/FClip/` 是像质丝模型代码，推理主流程应通过 `src/gauge/fclip_stage.py` 适配，不要直接散落调用 FClip 内部函数。
- `OCRtrain/third_party/PaddleOCR` 是第三方子模块，优先不要直接改第三方源码。
- `models/` 由 Git LFS 管理，首次拉取需 `git lfs install && git lfs pull`。
- **不要在根目录新增临时调试脚本**，放到 `scripts/debug/`。
- 不要把 `outputs/`、`logs/`、`local/` 当作源码接口。

## 推理主流程

完整推理由 `IQIInferencer.infer_image_path()` 编排：

1. 读取原图 → 可选整图方向矫正（隐藏参数）
2. 全图 PaddleOCR 文本检测 → 逐框透视裁剪 → 可选文本方向矫正 → 识别
3. `iqi_rules` 提取焊道号/片号/部件代号/管道规格 + 像质计标识匹配
4. YOLO-OBB 检测像质计 ROI → 透视展开 → 灰度增强
5. ROI OCR（标识匹配）+ FClip（丝数推理）
6. 标识选择：ROI OCR 优先，全图 OCR 兜底
7. `compute_iqi_grade()` 计算最终等级

OCR 运行在**独立子进程**中（`ocr_paddle_worker.py`），通过 stdin/stdout JSON 通信，避免 PaddleOCR 与 PyTorch/Ultralytics/FClip 的 GPU runtime 冲突。

## 像质计规则关键点

- 单丝型像质计：通用型按 `标记丝号 + 可见丝数 - 1` 计算等级，专用型在 ≥1 根丝时直接输出标记丝号。
- 标识按标记顺序判定：`数字+材料+JB` 为通用（general），`材料+数字+JB` 为专用（special），材料代号不参与类型判定。
- 详细契约见 `docs/contract/IQI_SINGLE_WIRE_MARKER_GRADE_RULE.md`。

## 训练与数据

三条训练链路由 `dvc.yaml` + `params.yaml` 驱动：

| Stage | 功能 | 关键脚本 |
|-------|------|---------|
| `gauge_preprocess` → `train_gauge` | YOLO-OBB ROI 检测 | `src/gauge/convert_to_yolo_obb.py` → `src/gauge/train.py` |
| `preprocess_fclip` → `train_fclip` | FClip 像质丝模型 | `src/dataset/weld.py` → `src/FClip/train.py` |
| (独立于 DVC) | PaddleOCR 识别 | `OCRtrain/scripts/` + `OCRtrain/tools/` |

## 修改约束

- 对外交付 CLI 参数不随意扩张；仅面向本地维护的参数使用 `argparse.SUPPRESS` 隐藏。
- 修改交付 JSON 字段时，同步检查 `docs/README_IQI_GRADE_INFERENCE_DELIVERY.md`。
- 修改主流程或代码边界时，同步更新 `ARCHITECTURE.md`。
- 更新模型文件或路径时，同步更新 `docs/MODEL_ASSETS.md`。
- 规则层与模型层保持分离：OCR 文本修正、标识判定、等级计算属于 `iqi_rules.py`；图像裁剪、模型加载、坐标变换不混入规则层。
- **严禁修改** `run_iqi_grade_infer.py`、`region_ocr_api.py`、`region_SNR_api.py` 的对外接口（CLI 参数、函数签名、输出 JSON 字段），除非用户明确要求。即使要求修改，也应先提示影响范围。
- **重构分支专属**：`refactor/pipeline-stages*` 分支上，重构完成后运行 `python scripts/compare_results.py` 一键执行推理并对比基线。此脚本和本条说明将在合并到 `main` 前的最后一个 commit 中删除。
