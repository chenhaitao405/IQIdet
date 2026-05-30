# Gauge 架构与重复逻辑收敛设计规格

**日期：** 2026-05-30
**状态：** 草案

## 目标

对 `src/gauge` 做第二轮、增量式架构整理，使核心代码更容易导入、测试和维护，并收敛当前残留的重复逻辑。外部交付接口保持稳定：

- `run_iqi_grade_infer.py`
- `region_ocr_api.py`
- `region_SNR_api.py`
- 交付 JSON 字段和既有集成调用方式

本规格只约束架构边界、模块职责和重复逻辑清理，不改变模型权重、推理算法和像质计等级计算规则。

## 背景

第一轮重构已经引入了 `PipelineRunner`、Stage 管道、Pydantic 模型、集中配置和 service 目录。这条方向是正确的，但当前代码仍有迁移后的残留问题：

- `IQIInferencer` 仍然过大，混合了资源生命周期、推理入口、可视化、统计、交付 record 和 legacy helper。
- 几何投影、OCR item 投影、delivery record、debug 可视化等逻辑在多个文件中重复存在。
- 仅导入 `gauge.pipeline` 就可能触发 Torch/FClip 等重模型依赖导入，导致轻量测试环境无法导入纯管道代码。
- Pydantic 模型已经存在，但最终 record 构建中仍有 post-assignment dict 赋值，绕过了一部分验证。
- `BaseRegionService` 提供了共享生命周期思路，但区域 OCR/SNR API 实际仍主要用模块级全局变量，复用不彻底。

## 非目标

- 不训练或替换模型。
- 不改变模型文件路径和 Git LFS 资产组织。
- 不改变像质计标识解析和等级公式。
- 不新增对外交付 CLI 参数。
- 不删除 debug 可视化能力。
- 不把训练链路重构扩大到本次核心任务之外。

## 设计原则

1. 纯逻辑模块必须可以在无 Torch、无 PaddleOCR、无 Ultralytics 的轻量环境中导入。
2. `IQIInferencer` 只做 public facade、资源构造、资源关闭和 `infer_image_path`。
3. Stage 只表达阶段数据流和阶段局部决策，不保存跨模块工具逻辑。
4. 几何和坐标投影只有一个实现。
5. 最终 record 在边界处统一验证，避免多个位置手工拼 dict。
6. 区域服务共享代码要真实复用；不能真实复用的抽象应降级为明确的小工具。
7. 每一步保持行为兼容，并用回归比较脚本验证输出一致。

## 目标模块边界

### 保留职责

- `src/gauge/pipeline.py`
  - 只负责 Stage 编排、异常策略和默认 Stage 序列构造。

- `src/gauge/stages/*`
  - 只负责单个阶段的输入、输出和阶段局部判断。

- `src/gauge/iqi_rules.py`
  - 继续作为纯业务规则层，负责字段提取、标识解析、等级计算和结果码。

- `src/gauge/services/*`
  - 继续作为模型或外部运行时适配层，包括 YOLO、FClip、OCR worker、区域 OCR/SNR 服务。

- `src/gauge/models/*`
  - 继续作为 Pydantic schema 层。

- `src/gauge/config/__init__.py`
  - 继续作为唯一配置入口。

### 新增模块

- `src/gauge/geometry.py`
  - 负责纯几何和投影逻辑：
    - `invert_perspective_matrix`
    - `perspective_transform_points`
    - `undo_ccw90_points`
    - `project_roi_box_to_image`
    - `project_ocr_items_to_image`
  - 只依赖 `numpy` 和 `cv2`。
  - 不依赖 FClip、Torch、Pydantic 模型或 Stage。

- `src/gauge/record_builders.py`
  - 负责最终 record 和交付 payload：
    - `build_iqi_record(ctx)`
    - `build_delivery_record(record)`
    - `build_iqi_statistics(results, topk)`
  - 依赖 `models/*` 和 `iqi_rules.py`。
  - 不依赖 Torch、PaddleOCR、Ultralytics 或模型文件。

- `src/gauge/visualization.py`
  - 负责 debug/结果可视化：
    - `build_wire_vis_image`
    - `build_final_result_vis_image`
    - `save_debug_visualizations`
  - 从 `iqi_inferencer.py` 中迁出可视化函数。

- `src/gauge/ocr_runtime.py`
  - 负责 OCR worker 子进程协议：
    - `PaddleOCRSubprocessClient`
    - 内部请求锁
    - 请求超时
    - worker 退出或超时时的关闭策略
  - PaddleOCR 真实导入仍放在 worker 或构造函数内部。

- `src/gauge/region_runtime.py`
  - 替代当前半复用的 `BaseRegionService` 生命周期职责：
    - `decode_base64`
    - 共享 `ThreadPoolExecutor`
    - `register_region_service_shutdown`
    - `shutdown_region_runtime`
  - 区域 OCR/SNR API 继续保留现有 public 函数名。

- `src/gauge/cli.py`
  - 如果保留 `pyproject.toml` 的 console script，则新增此文件。
  - `main()` 委托到根目录 `run_iqi_grade_infer.main()`。

### 修改模块

- `src/gauge/iqi_inferencer.py`
  - 删除重复的 static/class helper。
  - 删除内部可视化、统计和交付 record 拼装实现。
  - 保留兼容导出：
    - `build_delivery_record`
    - `build_iqi_statistics`
    - `collect_input_images`
    - `save_debug_visualizations`
  - 这些兼容导出应转调新模块，不再保留重复实现。

- `src/gauge/stages/__init__.py`
  - 改为最小导出或 lazy `__getattr__`。
  - 导入 `gauge.stages.base` 时不能导入所有 Stage。

- `src/gauge/stages/base.py`
  - 将 mutable default 全部改成 `Field(default_factory=...)`。
  - `StageContext.to_record()` 改为薄代理，调用 `record_builders.build_iqi_record(ctx)`。

- `src/gauge/stages/roi_detect.py`
  - 从 `gauge.geometry` 导入透视矩阵工具。
  - 不再从 `services.fclip_stage` 导入通用几何函数。

- `src/gauge/stages/roi_ocr.py`
  - 删除本地 `_project_roi_box_to_image` 和 `_project_ocr_items_to_image`。
  - 改用 `gauge.geometry`。

- `src/gauge/services/fclip_stage.py`
  - 只保留 FClip 模型构造、预处理、丝数推理和线段解析适配。
  - 移出通用几何工具。
  - 将 `torch` 和 `FClip.*` 导入移动到 `FClipInferencer.__init__` 或私有 loader 函数中。

- `src/gauge/services/ocr_stage.py`
  - 移出 `PaddleOCRSubprocessClient` 到 `ocr_runtime.py`。
  - 保留 OCR 输出归一化、检测识别组合和 debug OCR 绘制逻辑。

- `src/gauge/services/region_ocr_api.py`、`src/gauge/services/region_snr_api.py`
  - 使用 `region_runtime.py` 提供的 base64 解码、executor 和 atexit 清理。
  - 保持 `init_*`、`get_*`、`close_*`、异步 API 函数名不变。

- `pyproject.toml`
  - 修正 `iqi-grade-infer = "gauge.cli:main"`，确保目标真实存在。

- `src/gauge/README.md`
  - 改为指向 `ARCHITECTURE.md` 的简短说明，或更新为当前 `src/gauge` 边界说明。

## 目标架构

```text
run_iqi_grade_infer.py
  -> IQIInferencer
      -> 构造模型/worker/服务资源
      -> PipelineRunner
          -> StageContext
          -> ImageLoadStage
          -> CorrectionStage
          -> FullImageOCRStage
          -> ROIDetectStage
          -> ROIOCRStage
          -> WireDetectStage
          -> GradeFusionStage
      -> record_builders.build_iqi_record
      -> record_builders.build_delivery_record

纯模块：
  iqi_rules.py
  geometry.py
  pipeline_utils.py
  models/*
  config/*

运行时适配模块：
  services/roi_stage.py
  services/fclip_stage.py
  services/ocr_stage.py
  ocr_runtime.py
  services/*region*
```

核心边界要求：纯模块可在轻量环境中导入；模型运行时依赖只在实例化对应资源时才需要。

## 详细需求

### 1. 导入边界

`python -c "import gauge.pipeline"` 不得导入 Torch、FClip、Ultralytics 或 PaddleOCR。

实现要求：

- `stages/__init__.py` 改为最小导出或 lazy 导出。
- `roi_detect.py` 不再通过 `services.fclip_stage` 使用几何工具。
- `fclip_stage.py` 顶层不导入 `torch` 和 `FClip.*`。
- `iqi_inferencer.py` 只在需要构造 FClip 时导入 `FClipInferencer`。

验收命令：

```bash
PYTHONPATH=/home/cht/code/IQIdet/src python -c "import gauge.pipeline; import gauge.iqi_rules; print('ok')"
```

期望输出：

```text
ok
```

### 2. 配置覆盖验证

`PipelineConfig.apply_cli_overrides(args)` 必须保持嵌套 Pydantic 类型，不得把 `ocr`、`gauge`、`enhance` 等字段替换成普通 dict。

实现要求：

- 替换当前 raw nested `model_copy(update=overrides, deep=True)`。
- 使用显式子模型 `model_copy(update=...)`，或使用 `model_dump()` 深合并后再 `PipelineConfig.model_validate(...)`。
- 增加覆盖测试，至少包含：
  - `gauge.conf`
  - `fclip.threshold`
  - `ocr.min_score`
  - `correction.enabled`
  - `enhance.rotate_roi`

验收断言：

```python
cfg = PipelineConfig().apply_cli_overrides(args)
assert isinstance(cfg.ocr, OCRConfig)
assert cfg.ocr.min_score == 0.5
assert isinstance(cfg.enhance, EnhanceConfig)
```

### 3. 几何和投影去重

坐标投影只能有一个实现。

实现要求：

- 从 `roi_ocr.py` 和 `iqi_inferencer.py` 中移除重复投影函数。
- 将投影逻辑集中到 `geometry.py`。
- 所有调用点统一使用 `gauge.geometry`。
- 保持旋转 ROI 和未旋转 ROI 的输出行为不变。

验收测试：

- `project_roi_box_to_image` 覆盖以下场景：
  - 未旋转 ROI + identity inverse matrix。
  - CCW90 旋转 ROI + `pre_rotate_size`。
  - 缺少 matrix 时返回 ROI 空间 fallback，且不抛异常。

### 4. 最终 record 构建契约

最终结果构建必须在一个边界统一验证。

实现要求：

- `StageContext.to_record()` 只调用 `record_builders.build_iqi_record(ctx)`。
- `IQIRecord` 必须在构造函数中一次性接收完整字段。
- 不再先创建 `IQIRecord` 再 post-assignment 设置 `fields`、`ocr`、`plate`、`wire` 等 typed 字段。
- 如果 Stage 中间结果仍是 dict，则在 record 边界调用：
  - `OCRResult.model_validate(...)`
  - `PlateResult.model_validate(...)`
  - `WireResult.model_validate(...)`
  - `GradeResult.model_validate(...)`
- `.model_dump()` 输出必须保持向后兼容。

验收标准：

- 现有 delivery record 测试通过。
- 正常非 mock record 不产生 Pydantic serializer warning。
- 交付 JSON 字段和字段类型不变。

### 5. `IQIInferencer` 职责收敛

`iqi_inferencer.py` 应成为 public service facade，而不是综合工具模块。

实现要求：

- 删除重复的 legacy helper 实现。
- 可视化函数迁移到 `visualization.py`。
- delivery/statistics 函数迁移到 `record_builders.py`。
- `iqi_inferencer.py` 保留兼容转发导出，避免外部 import 断裂。
- `IQIInferencer.infer_image_path(...)` 签名保持不变。

兼容验收：

```python
from gauge.iqi_inferencer import IQIInferencer
from gauge.iqi_inferencer import build_delivery_record
from gauge.iqi_inferencer import build_iqi_statistics
from gauge.iqi_inferencer import collect_input_images
from gauge.iqi_inferencer import save_debug_visualizations
```

### 6. OCR worker 运行时稳健性

OCR 子进程通信必须对共享实例安全，并且不能无限期阻塞。

实现要求：

- `PaddleOCRSubprocessClient` 内部增加 `threading.Lock`。
- 写 stdin 和读 stdout 必须在同一把锁内完成。
- 增加 request timeout。
- worker 超时或退出时，关闭进程并返回清晰错误。
- 区域 OCR API 的响应 schema 不变。

验收标准：

- 多线程并发调用同一个 OCR client 不会交错写入 worker 协议。
- fake worker 超时测试能稳定触发 timeout 错误。
- 正常 close 行为幂等。

### 7. 区域服务共享运行时

区域 OCR 和区域 SNR API 必须使用同一种清晰的生命周期模式。

实现要求：

- 新增 `region_runtime.py`，集中管理：
  - base64 解码。
  - 共享 executor。
  - atexit 关闭。
- `region_ocr_api.py` 和 `region_snr_api.py` 的模块级 service 必须注册到 atexit 清理。
- `close_region_ocr_api()` 和 `close_region_snr_api()` 保持幂等。
- base64 格式错误应返回 HTTP 400，而不是 HTTP 500。

验收标准：

- `close_region_ocr_api()` 会关闭 OCR worker。
- `close_region_snr_api()` 可重复调用。
- 非法 base64 请求返回 400。

### 8. 包入口和陈旧脚本清理

包配置和本地调试脚本必须符合仓库边界。

实现要求：

- 修正 `pyproject.toml` 的 `gauge.cli:main` 入口，或移除该 console script。
- 如果保留入口，则新增 `src/gauge/cli.py`，并委托根目录交付入口。
- 更新 `src/gauge/README.md`，避免继续描述旧的未实现状态。
- `src/gauge/training/OBBtraintest.py` 不得在 import 时直接执行验证。
- `training/infer.py` 和 `training/valid.py` 这两个未实现模块必须补成真实入口或移除。

验收命令：

```bash
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_script_layout.py
```

## 迁移策略

### 阶段 1：修复导入边界

先移动几何函数，改懒导入，保证纯模块可轻量导入。此阶段不改变推理行为。

### 阶段 2：修复配置和 record 边界

修复 `apply_cli_overrides`，并把最终 record 构建收敛到 `record_builders.py`。此阶段重点降低运行时类型风险。

### 阶段 3：拆出可视化、统计和交付 payload

从 `iqi_inferencer.py` 迁出 visualization、statistics 和 delivery record helper，同时保留兼容转发。

### 阶段 4：整理 OCR runtime 和区域服务生命周期

迁移 OCR subprocess client，增加 lock/timeout，再统一区域 API 的 executor、base64 decode 和 atexit cleanup。

### 阶段 5：清理包入口和脚本文档

最后处理 `pyproject.toml`、`src/gauge/README.md` 和训练/调试脚本，降低对主推理链路的影响。

## 测试策略

新增以下聚焦测试：

- `tests/test_import_boundaries.py`
  - 验证 `gauge.pipeline` 在无重模型依赖的轻量环境中可导入。

- `tests/test_pipeline_config.py`
  - 验证 CLI 覆盖后嵌套配置仍是 Pydantic 子模型。

- `tests/test_geometry.py`
  - 验证 ROI 坐标投影和旋转回投。

- `tests/test_record_builders.py`
  - 验证最终 `IQIRecord` 构建、嵌套 section 验证和 JSON 形状兼容。

- `tests/test_ocr_runtime.py`
  - 用 fake process/client 验证锁、超时和 close。

保留并继续运行现有测试：

- `tests/test_iqi_rules.py`
- `tests/test_iqi_inferencer.py`
- `tests/test_iqi_delivery_record.py`
- `tests/test_region_snr_service.py`
- `tests/test_script_layout.py`

## 总体验收标准

所有阶段完成后，必须满足：

1. 轻量导入通过：

```bash
PYTHONPATH=/home/cht/code/IQIdet/src python -c "import gauge.pipeline; import gauge.iqi_rules; print('ok')"
```

2. 轻量测试通过：

```bash
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py tests/test_iqi_delivery_record.py tests/test_iqi_inferencer.py tests/test_region_snr_service.py tests/test_script_layout.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_delivery_record.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_inferencer.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_snr_service.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_script_layout.py
```

3. 新增聚焦测试通过：

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_import_boundaries.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_pipeline_config.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_geometry.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_record_builders.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_ocr_runtime.py
```

4. 回归输出一致：

```bash
python scripts/compare_results.py
```

正式验收必须看到类似输出：

```text
ALL MATCH — 8 image(s), 0 differences.
```

如果脚本输出 `SKIP: missing prerequisites`，说明本机缺少 baseline 或测试图片，只能说明脚本未执行回归比较，不能视为本次架构重构的正式验收通过。

## 兼容要求

以下 import 必须继续可用：

```python
from gauge.iqi_inferencer import IQIInferencer
from gauge.iqi_inferencer import build_delivery_record
from gauge.iqi_inferencer import build_iqi_statistics
from gauge.iqi_inferencer import collect_input_images
from gauge.iqi_inferencer import save_debug_visualizations
```

以下 public API 函数名和请求/响应模型保持不变：

```python
from region_ocr_api import init_region_ocr_api, recognize_region, close_region_ocr_api
from region_SNR_api import init_region_snr_api, compute_region_snr, close_region_snr_api
```

## 成功标准

- `src/gauge/iqi_inferencer.py` 主要承担 facade 和资源生命周期职责。
- 纯模块导入不需要 Torch、FClip、PaddleOCR、Ultralytics。
- 几何投影逻辑只有一个实现。
- delivery JSON 代表性回归输出一致。
- CLI 配置覆盖返回类型正确的嵌套配置对象。
- OCR subprocess client 具备内部串行化和超时边界。
- 区域 API 生命周期清理明确且有测试覆盖。
- 陈旧脚本和文档入口被更新或移除。

## 风险与缓解

- **风险：** 移动 import 后真实模型依赖错误暴露时机改变。
  - **缓解：** 保持实例化时错误明确，并新增 import boundary 测试。

- **风险：** typed record 验证改变 `.model_dump()` 输出形状。
  - **缓解：** 保留 delivery record 测试，并使用 `compare_results.py` 验证代表性输出一致。

- **风险：** helper 迁移后外部仍从 `iqi_inferencer.py` 导入旧函数。
  - **缓解：** 在 `iqi_inferencer.py` 保留兼容转发导出。

- **风险：** OCR timeout 影响首次慢启动。
  - **缓解：** startup timeout 和 request timeout 分开配置，默认值保守。

## 自查

- 本规格不包含算法行为变更。
- 每个新增模块都有明确职责和依赖方向。
- 每个阶段都可以独立验证。
- 已将输出一致性回归脚本纳入正式验收标准。
- 文档中没有未决空项。
