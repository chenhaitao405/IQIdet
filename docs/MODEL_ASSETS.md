# 模型资产与 Git LFS 维护说明

`models/` 是本仓库的交付模型目录。交付入口默认从这里读取模型；集成侧也应优先同步这个目录中的模型资产。

## 当前模型资产

| 路径 | 用途 | 主要入口参数 | 说明 |
| --- | --- | --- | --- |
| `models/guagerotation.pt` | 像质计 ROI 检测 | `--gauge-weights` | Ultralytics YOLO-OBB 权重 |
| `models/fclip67.pth.tar` | 像质丝识别 | `--fclip-ckpt` | FClip checkpoint |
| `models/fclip_config.yaml` | 像质丝识别配置 | `--fclip-config` | FClip 推理模型配置 |
| `models/OCR_rec_inference_best_accuracy/` | OCR 识别模型 | `--ocr-rec-model-dir` | PaddleOCR TextRecognition inference 模型目录；当前仓库也可能存在带日期后缀的导出目录，例如 `models/OCR_rec_inference_best_accuracy0325/` |
| `models/ocr_orientation_model.pth` | 文本 crop 方向矫正 | `--ocr-orientation-model` | 本地文件存在时可启用 OCR 文本方向矫正 |
| `models/weld_orientation_model.pth` | 整图方向矫正 | 隐藏参数 `--correction-model` | 仅本地调试使用，不作为对外公开 CLI 参数 |

如果文件名或默认推荐模型发生变化，需要同步更新本文件、`docs/README_IQI_GRADE_INFERENCE_DELIVERY.md` 和相关调用命令。

## Git LFS 约定

`.gitattributes` 当前已配置：

```text
models/** filter=lfs diff=lfs merge=lfs -text
```

因此 `models/` 下的新模型文件默认会由 Git LFS 管理。不要把大模型文件放在 `models/` 之外再直接提交到普通 Git 历史。

首次拉取或新机器准备模型：

```bash
git lfs install
git lfs pull
```

只拉取模型目录：

```bash
git lfs pull --include="models/**"
```

## 更新已有模型

1. 覆盖或新增 `models/` 下对应模型文件。
2. 确认 Git LFS 跟踪规则仍覆盖该文件：

```bash
git lfs track
git check-attr filter -- models/<model-file>
```

`git check-attr` 应显示 `filter: lfs`。

3. 检查文件状态：

```bash
git status --short
git lfs status
```

4. 提交模型和文档：

```bash
git add models/<model-file> docs/MODEL_ASSETS.md
git commit -m "chore: update delivery model assets"
```

5. 推送 Git 和 LFS 对象：

```bash
git push
git lfs push origin <branch>
```

## 新增模型文件类型

如果后续需要把模型放到 `models/` 之外，或者新增不在现有规则覆盖范围内的资产路径，先更新 LFS 规则。例如：

```bash
git lfs track "some/path/*.onnx"
git add .gitattributes
```

当前更推荐把交付运行时模型统一放在 `models/`，减少入口参数和文档分歧。

## 更新后的检查项

模型更新后至少检查：

- `run_iqi_grade_infer.py` 默认参数或推荐命令是否需要调整。
- `docs/README_IQI_GRADE_INFERENCE_DELIVERY.md` 的模型路径是否仍正确。
- `ARCHITECTURE.md` 的模型资产描述是否仍正确。
- 若是 OCR 模型，确认导出目录包含 PaddleOCR inference 所需文件。
- 若是 FClip 模型，确认 checkpoint 与 `models/fclip_config.yaml`、`params.yaml` 匹配。
- 若是 ROI 检测模型，确认权重与当前 YOLO-OBB 推理代码兼容。
