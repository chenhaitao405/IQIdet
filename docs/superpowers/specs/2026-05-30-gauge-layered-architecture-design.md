# Gauge 职责分层重排设计规格

**日期：** 2026-05-30  
**状态：** 草案  
**前置背景：** 已完成 `docs/superpowers/specs/2026-05-30-gauge-architecture-dedup-design.md` 中的第一轮架构去重与导入边界整理。

## 目标

对 `src/gauge` 做第二轮结构整理，把根目录散落的不同职责文件迁入明确子包，使维护者能够从目录层级直接判断模块责任、依赖方向和可测试边界。

本次重排采用“职责分层 + 兼容导入”策略：

- 新代码使用新的分层路径。
- 旧路径保留薄转发，避免交付入口、历史脚本和外部集成一次性破坏。
- 不改变交付 JSON 字段、CLI 参数含义、推理算法和模型文件。

## 非目标

- 不重写 IQI 算法、OCR 规则、FClip 解析逻辑或等级计算公式。
- 不改变根目录交付入口：
  - `run_iqi_grade_infer.py`
  - `region_ocr_api.py`
  - `region_SNR_api.py`
- 不移动 `models/`、`OCRtrain/`、`src/FClip/`、`src/dataset/`。
- 不把训练链路做大规模重构。
- 不删除旧 import 路径；旧路径只降级为 compatibility shim。

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

  imaging/
    __init__.py
    geometry.py
    preprocess.py
    visualization.py

  runtime/
    __init__.py
    ocr_runtime.py
    region_runtime.py

  services/
    __init__.py
    fclip_stage.py
    ocr_stage.py
    roi_stage.py
    correction.py
    ocr_orientation.py
    weld_correction.py
    adaptive_image_processor.py
    region_ocr_service.py
    region_snr_service.py
    region_ocr_api.py
    region_snr_api.py
    ocr_paddle_worker.py

  models/
  config/
  training/

  # compatibility shims
  iqi_inferencer.py
  pipeline.py
  iqi_rules.py
  record_builders.py
  geometry.py
  pipeline_utils.py
  visualization.py
  ocr_runtime.py
  region_runtime.py
  exceptions.py
  logging_setup.py
```

## 分层职责

### `app/`

职责：

- 对外交付层和服务门面。
- 持有模型资源生命周期。
- 组合 `PipelineRunner`、runtime、services 和最终 delivery payload。

初始迁移：

- `gauge.iqi_inferencer.IQIInferencer` 实现迁到 `gauge.app.iqi_inferencer.IQIInferencer`。
- 旧 `gauge.iqi_inferencer` 保留兼容导出。

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

- `collect_images()` 这类路径收集工具不属于 imaging，可迁到 `app/inputs.py` 或保留兼容 shim。
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
- API request/response wrappers。

保留原则：

- 当前 `services/` 不再继续扩大为“所有工具目录”。
- 只放 runtime adapter 或服务实现。
- 通用图像工具、业务规则、record 构建不应放入 `services/`。

## 兼容导入策略

旧路径必须保留一轮：

```python
# src/gauge/iqi_rules.py
from gauge.domain.iqi_rules import *  # noqa
```

兼容 shim 原则：

- shim 只做 import/export，不保留业务实现。
- shim 文件顶部注释写清新路径。
- 新代码和新测试一律 import 新路径。
- 旧路径 compatibility 测试覆盖关键入口：
  - `from gauge.iqi_rules import compute_iqi_grade`
  - `from gauge.pipeline import PipelineRunner`
  - `from gauge.iqi_inferencer import IQIInferencer`
  - `from gauge.ocr_runtime import PaddleOCRSubprocessClient`

## 迁移顺序

按低风险到高风险迁移：

1. 建立新目录和 compatibility shim 测试。
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
5. 迁移 `pipeline/`：
   - `PipelineRunner`
   - `StageContext`
   - `stages/`
6. 迁移 `app/`：
   - `IQIInferencer`
   - 输入收集、delivery-facing helpers。
7. 更新文档：
   - `ARCHITECTURE.md`
   - `src/gauge/README.md`
8. 运行输出回归：
   - `python scripts/compare_results.py`
   - 必须输出 `ALL MATCH — 8 image(s), 0 differences.`

## 测试要求

### 导入边界测试

新增或扩展 import boundary 测试：

- `import gauge.domain.iqi_rules` 不得导入 OpenCV、Torch、PaddleOCR、Ultralytics。
- `import gauge.imaging.geometry` 不得导入 Torch、PaddleOCR、Ultralytics。
- `import gauge.pipeline.runner` 不得构造模型，不得导入 Torch/FClip/PaddleOCR。
- 旧路径 shim 可导入，并返回同一对象或兼容对象。

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
  - compatibility shim 说明。
- `src/gauge/README.md`
  - 简短说明当前 `src/gauge` 分层。
- 如交付 JSON 字段未变化，则无需修改交付 JSON 文档。

## 风险与控制

### 风险：import churn 过大

控制：

- 分阶段提交。
- 每次只迁移一个层。
- 旧路径 shim 先建立，调用点逐步改新路径。

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
- 旧 `gauge.*` import 路径保留 shim。
- `tests/test_script_layout.py` 增加 root entrypoint import 检查。

## 验收标准

- `src/gauge` 根目录只保留门面 shim、基础设施 shim 和包说明文件；主要实现迁入清晰子包。
- 新代码 import 使用新路径。
- 旧路径 import 仍兼容。
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
