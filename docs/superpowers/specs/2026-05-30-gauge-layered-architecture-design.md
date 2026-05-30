# Gauge 职责分层重排设计规格

**日期：** 2026-05-30  
**状态：** 草案  
**前置背景：** 已完成 `docs/superpowers/specs/2026-05-30-gauge-architecture-dedup-design.md` 中的第一轮架构去重与导入边界整理。

## 目标

对 `src/gauge` 做第二轮结构整理，把根目录散落的不同职责文件迁入明确子包，使维护者能够从目录层级直接判断模块责任、依赖方向和可测试边界。

本次重排采用“职责分层 + 直接迁移”策略：

- 新代码使用新的分层路径。
- 包内旧路径不保留 compatibility shim；实施时同步更新所有内部 import、测试和文档。
- 根目录交付入口保留，但入口内部 import 改到新路径。
- 不改变交付 JSON 字段、CLI 参数含义、推理算法和模型文件。

## 非目标

- 不重写 IQI 算法、OCR 规则、FClip 解析逻辑或等级计算公式。
- 不改变根目录交付入口：
  - `run_iqi_grade_infer.py`
  - `region_ocr_api.py`
  - `region_SNR_api.py`
- 不移动 `models/`、`OCRtrain/`、`src/FClip/`、`src/dataset/`。
- 不把训练链路做大规模重构。
- 不保留包内旧 import 路径；迁移完成后旧模块文件应删除或改名为新路径实现文件。

## 当前问题

第一轮重构后，`src/gauge` 已经有 `stages/`、`services/`、`models/` 等边界，但根目录仍混合多种职责：

- 门面和编排：
  - `iqi_inferencer.py`
  - `pipeline.py`
- 纯业务规则：
  - `iqi_rules.py`
  - `record_builders.py`
- 图像与可视化工具：
  - `geometry.py`
  - `pipeline_utils.py`
  - `visualization.py`
- runtime / 生命周期：
  - `ocr_runtime.py`
  - `region_runtime.py`
- 基础设施：
  - `exceptions.py`
  - `logging_setup.py`

这会导致两个维护问题：

1. 新维护者难以仅通过路径判断模块是否安全导入、是否依赖模型 runtime、是否属于业务规则。
2. 后续开发容易继续往根目录加工具文件，使已经清理出的边界再次变散。

## 目标目录结构

```text
src/gauge/
  app/
    __init__.py
    iqi_inferencer.py
    inputs.py
    region_ocr_api.py
    region_snr_api.py

  pipeline/
    __init__.py
    runner.py
    context.py
    stages/
      __init__.py
      image_load.py
      correction.py
      full_image_ocr.py
      roi_detect.py
      roi_ocr.py
      wire_detect.py
      grade_fusion.py

  domain/
    __init__.py
    iqi_rules.py
    record_builders.py
    statistics.py

  imaging/
    __init__.py
    adaptive.py
    geometry.py
    preprocess.py
    visualization.py

  runtime/
    __init__.py
    ocr_runtime.py
    ocr_paddle_worker.py
    region_runtime.py

  services/
    __init__.py
    fclip/
      __init__.py
      inferencer.py
      line_records.py
    ocr/
      __init__.py
      factory.py
      infer.py
      normalize.py
      debug.py
    roi/
      __init__.py
      yolo_obb.py
    orientation/
      __init__.py
      base.py
      ocr_text.py
      weld.py
    region/
      __init__.py
      ocr_service.py
      snr_service.py

  models/
  config/
  training/
```

迁移完成后，`src/gauge` 根目录不再保留 `iqi_rules.py`、`pipeline.py`、`pipeline_utils.py`、`ocr_runtime.py` 等旧实现或 shim。根目录只保留包初始化文件和确有包级意义的说明文件。

## 分层职责

### `app/`

职责：

- 对外交付层和服务门面。
- 持有模型资源生命周期。
- 组合 `PipelineRunner`、runtime、services 和最终 delivery payload。

初始迁移：

- `gauge.iqi_inferencer.IQIInferencer` 实现迁到 `gauge.app.iqi_inferencer.IQIInferencer`。
- 根目录 `run_iqi_grade_infer.py` 改为直接从 `gauge.app.iqi_inferencer` 导入。
- 删除旧 `gauge.iqi_inferencer` 模块，不保留兼容导出。

依赖允许：

- 可依赖 `pipeline/`、`domain/`、`runtime/`、`services/`、`models/`、`config/`。
- 可以延迟导入模型 runtime。

### `pipeline/`

职责：

- 管道编排。
- StageContext。
- 单个 Stage 的输入输出状态转换。

初始迁移：

- `gauge.pipeline.PipelineRunner` 迁到 `gauge.pipeline.runner.PipelineRunner`。
- `gauge.stages.base.StageContext` 迁到 `gauge.pipeline.context.StageContext`。
- 当前 `gauge.stages/*` 迁到 `gauge.pipeline.stages/*`。

依赖允许：

- 可依赖 `domain/`、`imaging/`、`services/`、`models/`、`config/`。
- 不持有模型构造细节；模型实例由 `app/` 注入。

### `domain/`

职责：

- OCR 字段规则。
- 像质计标识解析。
- 等级计算。
- result code / status / statistics / final record 构建。

初始迁移：

- `iqi_rules.py` -> `domain/iqi_rules.py`
- `record_builders.py` -> `domain/record_builders.py`

依赖规则：

- 只能依赖标准库、`models/` 和必要的纯配置值。
- 不能依赖 OpenCV、Numpy、Torch、PaddleOCR、Ultralytics。
- 不读取文件、不启动进程、不调用模型。

### `imaging/`

职责：

- 图像读取之外的纯图像处理工具。
- 几何、投影、灰度增强、可视化。

初始迁移：

- `geometry.py` -> `imaging/geometry.py`
- `pipeline_utils.py` 中图像处理函数 -> `imaging/preprocess.py`
- `visualization.py` -> `imaging/visualization.py`

边界说明：

- `collect_images()` 这类路径收集工具不属于 imaging，应迁到 `app/inputs.py` 或 `app/io.py`。
- `build_skipped_ocr()`、`build_skipped_wire()` 如果属于业务状态构建，应迁到 `domain/record_builders.py` 或 `domain/status.py`。

依赖允许：

- 可依赖 OpenCV 和 Numpy。
- 不依赖 Torch、PaddleOCR、Ultralytics。

### `runtime/`

职责：

- 子进程、线程池、锁、超时、atexit shutdown。
- 不理解 IQI 业务规则。

初始迁移：

- `ocr_runtime.py` -> `runtime/ocr_runtime.py`
- `region_runtime.py` -> `runtime/region_runtime.py`

依赖允许：

- 可依赖标准库、OpenCV/Numpy（用于图像编码/解码）。
- 不依赖 `domain/`，不拼 result code。

### `services/`

职责：

- 外部模型和第三方库适配。
- 服务类封装。
- 模型输出归一化。
- 与模型强相关的 infer helper。

保留原则：

- 当前 `services/` 不再继续扩大为“所有工具目录”。
- 只放 runtime adapter 或服务实现。
- 通用图像工具、业务规则、record 构建不应放入 `services/`。

边界判定：

- `services/` 是能力适配层，不是 API/controller 层。它对 `app/` 或 `pipeline/` 暴露 Python service 类或 infer 函数，但不定义 FastAPI/Pydantic request-response、async endpoint、base64 传输包装或全局入口门面。
- 与某个 service 实现强绑定、且没有通用价值的私有 helper 可以保留在对应 service 子包内。例如模型输出归一化、模型专用 debug 绘制、模型结果对象到内部字段的转换。
- 一旦 helper 可被多个 service 或 pipeline 复用，应迁出 `services/`：图像处理进 `imaging/`，纯业务规则和状态构建进 `domain/`，进程/线程/超时/关闭逻辑进 `runtime/`，输入收集和 API wrapper 进 `app/`。
- `region_ocr_service.py`、`region_snr_service.py` 属于区域能力实现，可以进入 `services/region/`。`region_ocr_api.py`、`region_snr_api.py` 即使会初始化并调用这些 service，也属于 `app/`，不属于 `services/`。
- `RegionSNRService` 这种无模型但对外表现为可调用能力的类可以作为 service 壳保留；通用灰度转换和像素统计拆到 `imaging/metrics.py`，只接收 mean/std/area 等纯数值输入的 SNR 公式可放到 `domain/statistics.py`，service 只负责编排和响应 payload。

### `services/` 现状与优化

当前 `src/gauge/services` 下文件职责如下：

| 当前文件 | 当前职责 | 目标处理 |
| --- | --- | --- |
| `adaptive_image_processor.py` | 方向矫正前的窗宽窗位/负片自适应预处理 | 迁到 `imaging/adaptive.py`，由 orientation service 调用 |
| `base.py` | 旧的区域服务 base64/threadpool/lifecycle 基类 | 删除；功能已由 `runtime/region_runtime.py` 承担 |
| `correction.py` | 8 类方向矫正 Torch 基类 | 迁到 `services/orientation/base.py` |
| `ocr_orientation.py` | OCR 文本 crop 方向矫正模型 | 迁到 `services/orientation/ocr_text.py` |
| `weld_correction.py` | 焊缝整图方向矫正模型 | 迁到 `services/orientation/weld.py` |
| `fclip_stage.py` | FClip 模型加载、丝数推理、线段坐标记录 | 拆到 `services/fclip/inferencer.py` 和 `services/fclip/line_records.py` |
| `ocr_stage.py` | PaddleOCR 组件构造、OCR 推理、输出归一化、debug 绘制、统计 | 拆到 `services/ocr/factory.py`、`services/ocr/infer.py`、`services/ocr/normalize.py`、`services/ocr/debug.py`；统计迁到 `domain/statistics.py` |
| `ocr_paddle_worker.py` | OCR 子进程协议入口 | 迁到 `runtime/ocr_paddle_worker.py` |
| `roi_stage.py` | YOLO-OBB 输出解析、ROI 可视化、ROI crop helper | YOLO 解析迁到 `services/roi/yolo_obb.py`；可视化/crop 迁到 `imaging/` |
| `region_ocr_service.py` | 前端框选区域 OCR service | 迁到 `services/region/ocr_service.py` |
| `region_snr_service.py` | 前端框选区域归一化 SNR service | 服务壳迁到 `services/region/snr_service.py`；通用灰度/统计 helper 如需复用则拆到 `imaging/metrics.py` |
| `region_ocr_api.py` | FastAPI 风格 request/response + async wrapper | 迁到 `app/region_ocr_api.py`；不得作为 service 子模块保留 |
| `region_snr_api.py` | FastAPI 风格 request/response + async wrapper | 迁到 `app/region_snr_api.py`；不得作为 service 子模块保留 |
| `__init__.py` | 旧 service base re-export | 简化为空包初始化；不再 re-export 已删除 base |

优化目标：

- `services/ocr_stage.py` 不能继续作为 500+ 行混合模块存在。
- `services/` 只保留“模型/服务适配”职责，不再承载图像通用工具、runtime worker、API wrapper、输入传输包装或业务统计。
- `record_builders.py` 不应依赖 `services.ocr_stage.build_ocr_statistics`；统计逻辑应迁入 `domain/statistics.py`。
- `visualization.py` 不应从 `services.roi_stage` 取 ROI 可视化；ROI 可视化应归入 `imaging/visualization.py`。
- `region_ocr_api.py` 和 `region_snr_api.py` 是应用边界，不是 service 实现，迁到 `app/` 后根目录门面直接导入 `gauge.app.region_ocr_api` / `gauge.app.region_snr_api`。

## 直接迁移策略

旧路径不保留 shim。每迁移一个层时，同一个提交内必须完成三件事：

- 移动实现文件到新目录。
- 更新所有调用点和测试 import。
- 删除旧路径文件，避免后续代码继续引用旧边界。

实施期间允许分阶段提交，但每个提交完成后仓库必须保持可导入、可测试状态。不得提交“新旧路径同时长期存在”的中间形态。

包内旧路径删除范围包括：

- `gauge.iqi_rules`
- `gauge.record_builders`
- `gauge.geometry`
- `gauge.pipeline_utils`
- `gauge.visualization`
- `gauge.ocr_runtime`
- `gauge.region_runtime`
- `gauge.pipeline`
- `gauge.stages`
- `gauge.iqi_inferencer`
- `gauge.services.ocr_stage`
- `gauge.services.fclip_stage`
- `gauge.services.roi_stage`
- `gauge.services.correction`
- `gauge.services.ocr_orientation`
- `gauge.services.weld_correction`
- `gauge.services.region_ocr_api`
- `gauge.services.region_snr_api`
- `gauge.services.region_ocr_service`
- `gauge.services.region_snr_service`
- `gauge.services.base`

外部交付入口保留：

- `run_iqi_grade_infer.py`
- `region_ocr_api.py`
- `region_SNR_api.py`

这些入口内部 import 必须直接指向新路径。

## 迁移顺序

按低风险到高风险迁移：

1. 建立新目录和新路径导入测试。
2. 迁移 `domain/`：
   - `iqi_rules.py`
   - `record_builders.py`
3. 迁移 `imaging/`：
   - `geometry.py`
   - `visualization.py`
   - 从 `pipeline_utils.py` 拆出图像预处理函数。
4. 迁移 `runtime/`：
   - `ocr_runtime.py`
   - `region_runtime.py`
   - `ocr_paddle_worker.py`
5. 拆分 `services/`：
   - `services/ocr_stage.py`
   - `services/fclip_stage.py`
   - `services/roi_stage.py`
   - `services/correction.py`
   - `services/ocr_orientation.py`
   - `services/weld_correction.py`
   - `services/region_*`
   - 删除 `services/base.py`
6. 迁移 `pipeline/`：
   - `PipelineRunner`
   - `StageContext`
   - `stages/`
7. 迁移 `app/`：
   - `IQIInferencer`
   - 输入收集、delivery-facing helpers。
   - 区域 API wrapper。
8. 更新文档：
   - `ARCHITECTURE.md`
   - `src/gauge/README.md`
9. 运行输出回归：
   - `python scripts/compare_results.py`
   - 必须输出 `ALL MATCH — 8 image(s), 0 differences.`

## 测试要求

### 导入边界测试

新增或扩展 import boundary 测试：

- `import gauge.domain.iqi_rules` 不得导入 OpenCV、Torch、PaddleOCR、Ultralytics。
- `import gauge.imaging.geometry` 不得导入 Torch、PaddleOCR、Ultralytics。
- `import gauge.pipeline.runner` 不得构造模型，不得导入 Torch/FClip/PaddleOCR。
- 关键调用点必须使用新路径 import。
- 旧路径模块文件应不存在；测试不再验证旧路径可用。

### 行为回归测试

继续保留并运行：

- `tests/test_iqi_rules.py`
- `tests/test_geometry.py`
- `tests/test_record_builders.py`
- `tests/test_iqi_inferencer.py`
- `tests/test_iqi_delivery_record.py`
- `tests/test_ocr_runtime.py`
- `tests/test_region_runtime.py`
- `tests/test_script_layout.py`

### 输出回归

最终验收必须运行：

```bash
python scripts/compare_results.py
```

通过条件：

```text
ALL MATCH — 8 image(s), 0 differences.
```

`SKIP: missing prerequisites` 不计为通过。

## 文档更新要求

完成实施后同步更新：

- `ARCHITECTURE.md`
  - 新目录图。
  - 新依赖方向。
  - 直接迁移说明。
- `src/gauge/README.md`
  - 简短说明当前 `src/gauge` 分层。
- 如交付 JSON 字段未变化，则无需修改交付 JSON 文档。

## 风险与控制

### 风险：import churn 过大

控制：

- 分阶段提交。
- 每次只迁移一个层。
- 每个迁移提交内同时更新 import 并删除旧路径，避免双路径长期共存。

### 风险：循环导入

控制：

- `domain/` 不反向依赖 `pipeline/`、`app/`、`services/`。
- `runtime/` 不依赖 `app/` 或 `pipeline/`。
- `app/` 是高层，可依赖其他层。

### 风险：输出字段漂移

控制：

- `build_delivery_record()` 行为不改。
- 每阶段运行 `tests/test_iqi_delivery_record.py`。
- 最终运行 `compare_results.py`。

### 风险：旧脚本 import 断裂

控制：

- 根目录入口不改调用方式。
- 根目录入口内部 import 更新到新路径。
- 包内旧路径不保证兼容。
- `tests/test_script_layout.py` 增加 root entrypoint import 检查。

## 验收标准

- `src/gauge` 根目录不保留门面 shim 或旧路径 shim；主要实现迁入清晰子包。
- 新代码 import 使用新路径。
- 包内旧路径 import 已清理；仓库内不再引用旧路径。
- 轻量测试全部通过。
- `python scripts/compare_results.py` 输出：

```text
ALL MATCH — 8 image(s), 0 differences.
```

## 后续计划

用户确认本 SPEC 后，再编写实施计划：

```text
docs/superpowers/plans/2026-05-30-gauge-layered-architecture-plan.md
```

实施计划应继续采用小步提交，每一层迁移一个独立 task，并在每个 task 后运行对应测试。
