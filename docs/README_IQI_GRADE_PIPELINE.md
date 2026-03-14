# IQI 像质计等级识别流程说明

本文档给出 IQI（像质计）等级识别的完整流程设计，包含当前已实现部分与当前被屏蔽（暂不执行）的像质丝识别部分。

## 1. 目标

输入一张原始底片图像，输出：
- ROI 检测与裁剪信息
- OCR 文本识别结果
- 像质丝根数（当前流程中暂时屏蔽）
- 最终像质计等级（OCR + 丝识别融合）

## 2. 当前实现状态

当前 `run_inference_pipeline.py` 处于 **OCR 调试阶段**：
- 已执行：图像读取 -> OBB 检测 -> ROI 透视裁剪 -> 预处理 -> OCR（原图+镜像）-> 结果统计与可视化
- 暂未执行：FClip 热图解码、丝根数计算、等级融合输出

也就是说，当前输出重点是 OCR 结果与 OCR 统计，便于先把文字识别调通。

## 3. 完整流程（含被屏蔽部分）

### Step 1. 输入收集
- 支持 `--image-dir` 批量目录输入
- 支持 `--image-list` 文本列表输入
- 初始化 `output_dir` 与可选 `vis` 输出目录

### Step 2. YOLO-OBB 检测 ROI
- 对整图做 OBB 检测
- 获取候选四边形
- 通过 `--gauge-select` 选择最佳候选：
  - `conf`：按置信度选最大
  - `area`：按面积选最大
- 可选 `--gauge-class` 做类别过滤

### Step 3. ROI 透视裁剪
- 对检测四边形做透视变换，得到规则 ROI 图
- ROI 无效（过小/异常）时标记 `roi_invalid`

### Step 4. ROI 几何与灰度预处理
- 默认宽图旋转到竖向（`--no-rotate` 可关闭）
- 灰度增强策略：
  - `windowing`（默认）
  - `original`

### Step 5. OCR 识别（已实现）
对 ROI 执行两路 OCR：
1. 原图 OCR
2. 水平镜像 OCR

当前默认将 PaddleOCR 的文本识别模型固定为英文专属模型 `en_PP-OCRv5_mobile_rec`，用于降低英数字符串（如 `FE12`、`Ni08`、`JB`）的误识别率。

筛选规则：
- 仅保留文本中包含 `JB` 的识别项
- 若原图命中 `JB`，优先用原图结果
- 否则若镜像命中 `JB`，使用镜像结果并将检测框坐标映射回原 ROI
- 若两路都未命中，状态记为 `no_jb`

### Step 6. 像质丝识别（当前屏蔽）
目标流程中应在 OCR 后执行：
- 调用 FClip 模型推理热图
- 解码 `count` 头得到丝根数
- 解码线段并缩放回 ROI 尺寸
- 输出 `pred.count`、`pred.lines`、`pred.scores`

> 当前此步骤为了 OCR 调试已屏蔽，后续 OCR 稳定后再恢复。

### Step 7. 等级融合计算（目标流程）
从 OCR 提取 `字母 + 2位数字`，并按规则计算等级：

1) 类型判定
- 包含 `FE` -> 均匀像质计
- 包含 `Ni/NI` -> 渐变像质计

2) 等级计算
- 均匀像质计：
  - 丝数 > 2 -> 等级 = OCR 数字
  - 否则等级 = 0
- 渐变像质计：
  - 丝数 > 0 -> 等级 = OCR 数字 + 丝数 - 1
  - 丝数 = 0 -> 等级 = 0

3) OCR 无有效数字或类型未知 -> 等级 = 0

### Step 8. 结果落盘
每图记录至少包含：
- `status`
- `gauge`（ROI 检测信息）
- `preprocess`（旋转与增强信息）
- `ocr`（OCR 结果）

完整目标流程还应补充：
- `pred`（丝识别结果）
- `iqi`（类型、OCR数字、丝数、等级、规则）

全量输出：
- `ocr_results.json`（逐图详情）
- `ocr_stats.json`（汇总统计）

## 4. 可视化输出约定

当开启 `--vis`：
- 每张图可视化会按 OCR 状态分类保存到：
  - `vis/ocr_status/ok/...`
  - `vis/ocr_status/no_jb/...`
  - `vis/ocr_status/skipped_no_roi/...`
- 典型文件：
  - `ROI.png`（原图上 ROI 框）
  - `ocr_input.png`（OCR 输入 ROI）
  - `ocr_result.png`（OCR 框与文本叠加）

## 5. 调试建议

建议先走两阶段：
1. 阶段A：仅 OCR 调试
- 观察 `ocr_stats.json` 中 `top_raw_text`、`top_normalized_text`
- 重点修正 FE/Ni 的误识别模式

2. 阶段B：恢复丝识别并做等级融合
- 恢复 FClip 推理
- 增加 `iqi` 字段输出
- 对比人工标注做准确率评估

## 6. 典型命令

### 仅 OCR 调试（当前）

```bash
python run_inference_pipeline.py \
  --image-dir IQIdata/ori/img \
  --gauge-weights local/gauge.pt \
  --gauge-device 0 \
  --ocr-device gpu \
  --ocr-rec-model-name en_PP-OCRv5_mobile_rec \
  --vis \
  --output-dir outputs/gauge_ocr
```

### 完整目标流程（未来恢复丝识别后）

```bash
python run_inference_pipeline.py \
  --image-dir IQIdata/ori/img \
  --gauge-weights local/gauge.pt \
  --fclip-ckpt local/fclip67.pth.tar \
  --gauge-device 0 \
  --fclip-device 0 \
  --ocr-device gpu \
  --vis \
  --output-dir outputs/gauge_infer_full
```
