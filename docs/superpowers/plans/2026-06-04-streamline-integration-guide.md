# 精简集成引导.md 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `集成引导.md` 从 760 行压缩到 ~80 行极简上手页

**Architecture:** 单文件覆写，四段结构：前置条件 → 功能速查卡片 ×3 → 组合建议 → 注意事项。所有被裁详细内容均可通过文末链接获取。

**Tech Stack:** Markdown

---

### Task 1: 覆写集成引导.md

**Files:**
- Modify: `集成引导.md`（完全覆写）
- No tests needed（纯文档变更）

- [ ] **Step 1: 写入精简版内容**

```markdown
# IQIdet 集成速查

## 1. 前置条件

### 环境
```bash
conda activate weld-gpu
# Python: /home/cht/miniconda3/envs/weld-gpu/bin/python
pip install -r requirements.txt
```

### 模型文件
`models/` 目录下需存在：

| 文件 | 说明 |
|------|------|
| `guagerotation.pt` | YOLO-OBB ROI 检测权重 |
| `fclip67.pth.tar` | FClip 像质丝模型 (**Git LFS**) |
| `fclip_config.yaml` | FClip 模型配置 |
| `OCR_rec_inference_best_accuracy/` | PaddleOCR 识别模型 |
| `ocr_orientation_model.pth` | 文本方向矫正模型 |

```bash
git lfs install
git lfs pull --include="models/fclip67.pth.tar"
```

## 2. 功能速查

### 2.1 底片信息自动识别

- **入口**: `run_iqi_grade_infer.py`
- **最小命令**:
  ```bash
  python run_iqi_grade_infer.py \
    --image-dir IQIdata/ori/img \
    --output-json outputs/iqi_grade_results.json \
    --ocr-number-range 1-19
  ```
  （其余参数有默认值，指向 `models/` 下对应文件）
- **输出核心字段**: `ok`, `result_code`, `grade`, `iqi_type`, `wire_count`, `fields.weld_numbers`, `fields.film_numbers`
- **负责**: 整图 → 全流程 IQI 等级输出（OCR + ROI检测 + 丝数识别 + 等级计算）
- **不负责**: 前端交互、单区域实时处理
- **详文**: [docs/README_IQI_GRADE_INFERENCE_DELIVERY.md](docs/README_IQI_GRADE_INFERENCE_DELIVERY.md)

### 2.2 区域 OCR 识别

- **入口**: `region_ocr_api.py`
- **核心调用**:
  ```python
  from region_ocr_api import (
      init_region_ocr_api, recognize_region, close_region_ocr_api,
  )
  init_region_ocr_api()                # 服务启动时一次
  result = await recognize_region(req) # 每请求
  close_region_ocr_api()               # 服务退出时
  ```
- **输入**: 单个已裁剪文字区域的 base64 图像（支持 `data:image/png;base64,...`）
- **输出**: `status`, `text`, `normalized_text`, `score`, `orientation`
- **负责**: 图像增强 → 方向矫正 → 文字识别 → 文本标准化
- **不负责**: 文本检测、多区域切分、整图方向矫正

### 2.3 区域归一化信噪比

- **入口**: `region_SNR_api.py`
- **核心调用**:
  ```python
  from region_SNR_api import (
      init_region_snr_api, compute_region_snr, close_region_snr_api,
  )
  init_region_snr_api()                  # 服务启动时一次
  result = await compute_region_snr(req) # 每请求
  close_region_snr_api()                 # 服务退出时
  ```
- **输入**: 单个已裁剪区域的 base64 图像
- **输出**: `snr_m`, `snr_n`, `gray_mean`, `gray_std`, `result_code`
- **负责**: 灰度统计 → 测量信噪比 → 归一化信噪比
- **不负责**: 文本检测、OCR、多区域切分
- **注意**: 区域面积需 ≥ 20×55 像素，否则返回 `4001`

## 3. 建议组合方式

| 需求 | 用哪个 | 方式 |
|------|--------|------|
| 整图批量 IQI 等级 | 功能 1 | 离线脚本 |
| 前端框选实时 OCR | 功能 2 | 挂在线服务 |
| 前端框选信噪比 | 功能 3 | 挂在线服务 |
| 三者都需要 | 功能 1 离线 + 功能 2/3 在线 | 互不影响 |

## 4. 常见注意事项

1. **不要每次请求初始化模型** — 在服务启动时初始化一次
2. **区域 OCR 输入必须是裁好的单块文字区域** — 该接口不做文本检测
3. **功能 2/3 的导入入口在仓库根目录** — 用 `from region_ocr_api import ...`，不要直接从 `src/gauge/` 导入
4. **PaddleOCR 以子进程运行** — 服务退出时务必调 `close_*` 释放资源
5. **GPU 环境确认 Paddle / Torch / CUDA 版本匹配**，纯 CPU 可设 `ocr_device="cpu"`

## 5. 详细文档

- [交付字段与 JSON schema](docs/README_IQI_GRADE_INFERENCE_DELIVERY.md)
- [代码架构](ARCHITECTURE.md)
- [模型资产](docs/MODEL_ASSETS.md)
```

- [ ] **Step 2: 验证文件行数**

```bash
wc -l 集成引导.md
```
Expected: ~80 行

- [ ] **Step 3: Commit**

```bash
git add 集成引导.md
git commit -m "docs: 精简集成引导.md，从 760 行压至 ~80 行极简上手页

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
```

## Self-Review

1. **Spec coverage**: 前置条件 ✓, 三个功能卡片 ✓, 组合方式 ✓, 注意事项 ✓, 文档链接 ✓
2. **Placeholder scan**: 无 TBD/TODO/占位符，所有步骤含完整代码
3. **Type consistency**: 单文件变更，无类型冲突风险
