# Gauge 架构与重复逻辑收敛 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 `docs/superpowers/specs/2026-05-30-gauge-architecture-dedup-design.md` 收敛 `src/gauge` 架构边界和重复逻辑，并保持交付输出一致。

**Architecture:** 分 8 个任务推进：先建立导入边界和几何模块，再修复配置、record 构建、`IQIInferencer` 职责、OCR runtime、区域服务生命周期，最后清理包入口和脚本。每个任务先写失败测试，再做最小实现，最后运行局部验证并提交该任务涉及文件。

**Tech Stack:** Python 3.10+、Pydantic v2、stdlib `unittest`、OpenCV/Numpy、现有 PaddleOCR/Torch/Ultralytics 运行时适配。

**Spec:** `docs/superpowers/specs/2026-05-30-gauge-architecture-dedup-design.md`

---

## 文件结构计划

### 新增文件

- `src/gauge/geometry.py`：纯几何和坐标投影工具。
- `src/gauge/record_builders.py`：最终 `IQIRecord`、交付 payload、统计 payload 构建。
- `src/gauge/visualization.py`：debug 和最终结果可视化。
- `src/gauge/ocr_runtime.py`：OCR worker 子进程 client、锁、超时、关闭策略。
- `src/gauge/region_runtime.py`：区域 API 共享 base64 解码、executor、atexit cleanup。
- `tests/test_import_boundaries.py`：轻量导入边界测试。
- `tests/test_pipeline_config.py`：配置覆盖验证测试。
- `tests/test_geometry.py`：几何投影测试。
- `tests/test_record_builders.py`：record 构建和 serializer warning 测试。
- `tests/test_ocr_runtime.py`：OCR runtime 锁、超时、close 测试。

### 修改文件

- `src/gauge/config/__init__.py`
- `src/gauge/stages/__init__.py`
- `src/gauge/stages/base.py`
- `src/gauge/stages/roi_detect.py`
- `src/gauge/stages/roi_ocr.py`
- `src/gauge/services/fclip_stage.py`
- `src/gauge/services/ocr_stage.py`
- `src/gauge/services/region_ocr_api.py`
- `src/gauge/services/region_snr_api.py`
- `src/gauge/services/region_ocr_service.py`
- `src/gauge/iqi_inferencer.py`
- `run_iqi_grade_infer.py`
- `src/gauge/README.md`
- `src/gauge/training/OBBtraintest.py`
- `src/gauge/training/infer.py`
- `src/gauge/training/valid.py`
- `pyproject.toml`
- `tests/test_script_layout.py`

### 不应修改

- 模型文件、`models/` 资产、`IQIdata/`、`outputs/` baseline。
- 交付 JSON 字段名和字段含义。
- 像质计等级计算规则。
- 用户当前已有的无关工作区修改。

## Task 0: 执行前工作区保护

**Files:**
- No code changes.

- [ ] **Step 1: 查看当前工作区**

Run:

```bash
git status --short
```

Expected: 能看到本次任务之外的既有修改，例如 `.vscode/`、`graphify-out/` 或用户改动。执行计划时不要 stage 这些无关文件。

- [ ] **Step 2: 创建执行分支**

Run:

```bash
git switch -c refactor/gauge-architecture-dedup
```

Expected: 新分支创建成功。若分支已存在，使用：

```bash
git switch refactor/gauge-architecture-dedup
```

- [ ] **Step 3: 记录只允许提交的路径**

Allowed paths for this plan:

```text
src/gauge/
tests/
run_iqi_grade_infer.py
pyproject.toml
docs/superpowers/plans/2026-05-30-gauge-architecture-dedup-plan.md
docs/superpowers/specs/2026-05-30-gauge-architecture-dedup-design.md
```

Do not include `.vscode/`, `graphify-out/`, `outputs/`, `logs/`, model assets, or unrelated user edits in commits.

## Task 1: 修复轻量导入边界并抽出纯几何模块

**Files:**
- Create: `tests/test_import_boundaries.py`
- Create: `src/gauge/geometry.py`
- Modify: `src/gauge/stages/__init__.py`
- Modify: `src/gauge/stages/roi_detect.py`
- Modify: `src/gauge/stages/roi_ocr.py`
- Modify: `src/gauge/services/fclip_stage.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_import_boundaries.py`:

```python
import importlib
import importlib.abc
import sys
import unittest


class _BlockHeavyImports(importlib.abc.MetaPathFinder):
    blocked_roots = {"torch", "FClip", "ultralytics", "paddleocr"}

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in self.blocked_roots:
            raise AssertionError(f"Blocked heavy import during pure import: {fullname}")
        return None


class ImportBoundaryTest(unittest.TestCase):
    def test_pipeline_import_does_not_load_heavy_model_runtime(self) -> None:
        blocked = _BlockHeavyImports()
        removed = {}
        for name in list(sys.modules):
            if name.split(".", 1)[0] in blocked.blocked_roots or name.startswith("gauge"):
                removed[name] = sys.modules.pop(name)
        sys.meta_path.insert(0, blocked)
        try:
            module = importlib.import_module("gauge.pipeline")
            self.assertTrue(hasattr(module, "PipelineRunner"))
        finally:
            sys.meta_path.remove(blocked)
            for name in list(sys.modules):
                if name.startswith("gauge"):
                    sys.modules.pop(name)
            sys.modules.update(removed)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认当前失败**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_import_boundaries.py
```

Expected: FAIL，错误包含 `Blocked heavy import` 或导入 `torch/FClip` 相关信息。

- [ ] **Step 3: 新增纯几何模块**

Create `src/gauge/geometry.py`:

```python
#!/usr/bin/env python3
"""Pure geometry helpers for IQI ROI and OCR coordinate projection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def ensure_float32_matrix(matrix: Any) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    return arr.reshape(3, 3)


def invert_perspective_matrix(matrix: Any) -> np.ndarray:
    return np.linalg.inv(ensure_float32_matrix(matrix))


def undo_ccw90_points(points_xy: np.ndarray, pre_rotate_size: Sequence[int]) -> np.ndarray:
    width = float(pre_rotate_size[0])
    out = np.asarray(points_xy, dtype=np.float32).copy()
    x_rot = out[:, 0].copy()
    y_rot = out[:, 1].copy()
    out[:, 0] = width - 1.0 - y_rot
    out[:, 1] = x_rot
    return out


def perspective_transform_points(points_xy: np.ndarray, matrix: Any) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, ensure_float32_matrix(matrix))
    return transformed.reshape(-1, 2)


def project_roi_box_to_image(
    box: Any,
    crop_inverse_matrix: Optional[np.ndarray],
    pre_rotate_size: Optional[Sequence[int]],
    rotated: bool,
) -> Tuple[Any, Any]:
    if box is None:
        return None, None
    try:
        roi_points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None, None
    if roi_points.size == 0:
        return None, None

    roi_unrotated = roi_points
    if rotated:
        if pre_rotate_size is None:
            return roi_points.tolist(), None
        roi_unrotated = undo_ccw90_points(roi_points, pre_rotate_size=pre_rotate_size)

    if crop_inverse_matrix is None:
        return roi_points.tolist(), roi_unrotated.tolist()

    image_points = perspective_transform_points(roi_unrotated, crop_inverse_matrix)
    return image_points.tolist(), roi_unrotated.tolist()


def project_ocr_items_to_image(
    items: Sequence[Dict[str, Any]],
    crop_inverse_matrix: Optional[np.ndarray],
    pre_rotate_size: Optional[Sequence[int]],
    rotated: bool,
) -> List[Dict[str, Any]]:
    projected_items: List[Dict[str, Any]] = []
    for item in items:
        projected = dict(item)
        box_image, box_unrotated = project_roi_box_to_image(
            item.get("box"),
            crop_inverse_matrix=crop_inverse_matrix,
            pre_rotate_size=pre_rotate_size,
            rotated=rotated,
        )
        projected["box_image"] = box_image
        projected["box_roi_unrotated"] = box_unrotated
        projected_items.append(projected)
    return projected_items
```

- [ ] **Step 4: 改 `stages/__init__.py` 为轻量导出**

Replace `src/gauge/stages/__init__.py` with:

```python
#!/usr/bin/env python3
"""Pipeline stage base exports.

Concrete stages are imported directly by gauge.pipeline to avoid importing
heavy optional runtime dependencies during light module imports.
"""

from gauge.stages.base import PipelineStage, StageContext

__all__ = ["PipelineStage", "StageContext"]
```

- [ ] **Step 5: 更新 ROI detect 的几何导入**

In `src/gauge/stages/roi_detect.py`, replace:

```python
from gauge.services.fclip_stage import invert_perspective_matrix
```

with:

```python
from gauge.geometry import invert_perspective_matrix
```

- [ ] **Step 6: 更新 ROI OCR 投影调用**

In `src/gauge/stages/roi_ocr.py`:

Delete imports from `gauge.services.fclip_stage`:

```python
from gauge.services.fclip_stage import (
    invert_perspective_matrix,
    perspective_transform_points,
    undo_ccw90_points,
)
```

Add:

```python
from gauge.geometry import project_ocr_items_to_image
```

Delete the local functions `_project_roi_box_to_image` and `_project_ocr_items_to_image`.

Replace the call:

```python
roi_projected_items = _project_ocr_items_to_image(
```

with:

```python
roi_projected_items = project_ocr_items_to_image(
```

- [ ] **Step 7: 让 FClip 模块顶层不导入 Torch/FClip**

In `src/gauge/services/fclip_stage.py`, remove these top-level imports:

```python
import torch
from FClip.config import M
from FClip.infer_utils import (
    build_infer_model,
    get_count_pred,
    infer_heatmaps,
    load_config_from_yaml,
    parse_lines_1d,
    preprocess_gray_image,
    scale_lines,
)
```

Add:

```python
from gauge.geometry import (
    invert_perspective_matrix,
    perspective_transform_points,
    undo_ccw90_points,
)
```

Replace `resolve_torch_device` with:

```python
def resolve_torch_device(device: Optional[str]):
    import torch

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

In `FClipInferencer.__init__`, add runtime imports and store callables:

```python
        import torch
        from FClip.config import M
        from FClip.infer_utils import (
            build_infer_model,
            get_count_pred,
            infer_heatmaps,
            load_config_from_yaml,
            parse_lines_1d,
            preprocess_gray_image,
            scale_lines,
        )

        self._torch = torch
        self._get_count_pred = get_count_pred
        self._infer_heatmaps = infer_heatmaps
        self._parse_lines_1d = parse_lines_1d
        self._preprocess_gray_image = preprocess_gray_image
        self._scale_lines = scale_lines
```

Then replace calls in `infer()`:

```python
image_tensor = self._preprocess_gray_image(...)
heatmaps = self._infer_heatmaps(self.model, image_tensor)
count_pred = self._get_count_pred(heatmaps)
lines_t, scores_t = self._parse_lines_1d(...)
lines_scaled = self._scale_lines(lines_scaled, self.resolution, roi_gray.shape)
if isinstance(lines_t, self._torch.Tensor):
...
```

- [ ] **Step 8: 运行导入边界测试**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_import_boundaries.py
```

Expected: PASS.

- [ ] **Step 9: 编译相关文件**

Run:

```bash
python -m py_compile src/gauge/geometry.py src/gauge/stages/__init__.py src/gauge/stages/roi_detect.py src/gauge/stages/roi_ocr.py src/gauge/services/fclip_stage.py src/gauge/pipeline.py
```

Expected: no output.

- [ ] **Step 10: Commit**

Run:

```bash
git add tests/test_import_boundaries.py src/gauge/geometry.py src/gauge/stages/__init__.py src/gauge/stages/roi_detect.py src/gauge/stages/roi_ocr.py src/gauge/services/fclip_stage.py
git commit -m "refactor: isolate geometry helpers from FClip runtime"
```

## Task 2: 增加几何投影单元测试

**Files:**
- Create: `tests/test_geometry.py`
- Modify: `src/gauge/geometry.py`

- [ ] **Step 1: 写几何测试**

Create `tests/test_geometry.py`:

```python
import unittest

import numpy as np

from gauge.geometry import project_roi_box_to_image, project_ocr_items_to_image


class GeometryProjectionTest(unittest.TestCase):
    def test_project_box_identity_matrix_without_rotation(self) -> None:
        box = [[1, 2], [3, 2], [3, 4], [1, 4]]
        image_box, unrotated_box = project_roi_box_to_image(
            box,
            crop_inverse_matrix=np.eye(3, dtype=np.float32),
            pre_rotate_size=None,
            rotated=False,
        )

        self.assertEqual(image_box, [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])
        self.assertEqual(unrotated_box, [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])

    def test_project_box_undoes_ccw90_rotation(self) -> None:
        box = [[2, 1], [4, 1]]
        image_box, unrotated_box = project_roi_box_to_image(
            box,
            crop_inverse_matrix=np.eye(3, dtype=np.float32),
            pre_rotate_size=[10, 20],
            rotated=True,
        )

        self.assertEqual(unrotated_box, [[8.0, 2.0], [8.0, 4.0]])
        self.assertEqual(image_box, [[8.0, 2.0], [8.0, 4.0]])

    def test_missing_matrix_returns_roi_space_fallback(self) -> None:
        box = [[1, 2], [3, 4]]
        image_box, unrotated_box = project_roi_box_to_image(
            box,
            crop_inverse_matrix=None,
            pre_rotate_size=None,
            rotated=False,
        )

        self.assertEqual(image_box, [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(unrotated_box, [[1.0, 2.0], [3.0, 4.0]])

    def test_project_ocr_items_adds_image_and_unrotated_boxes(self) -> None:
        items = [{"text": "10FEJB", "box": [[1, 2], [3, 2], [3, 4], [1, 4]]}]
        projected = project_ocr_items_to_image(
            items,
            crop_inverse_matrix=np.eye(3, dtype=np.float32),
            pre_rotate_size=None,
            rotated=False,
        )

        self.assertEqual(projected[0]["box_image"], [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])
        self.assertEqual(projected[0]["box_roi_unrotated"], [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])
        self.assertEqual(items[0].get("box_image"), None)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_geometry.py
```

Expected: PASS. If float formatting differs only by `-0.0`, normalize output in `geometry.py` with `float(value)` after `np.where(np.isclose(out, 0.0), 0.0, out)`.

- [ ] **Step 3: Commit**

Run:

```bash
git add tests/test_geometry.py src/gauge/geometry.py
git commit -m "test: cover ROI geometry projection helpers"
```

## Task 3: 修复 PipelineConfig CLI 覆盖

**Files:**
- Create: `tests/test_pipeline_config.py`
- Modify: `src/gauge/config/__init__.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_pipeline_config.py`:

```python
import unittest
from types import SimpleNamespace

from gauge.config import (
    CorrectionConfig,
    EnhanceConfig,
    FClipConfig,
    GaugeConfig,
    OCRConfig,
    PipelineConfig,
)


def _args(**overrides):
    defaults = {
        "gauge_weights": None,
        "gauge_conf": None,
        "gauge_iou": None,
        "gauge_imgsz": None,
        "gauge_device": None,
        "gauge_select": None,
        "gauge_class": None,
        "fclip_ckpt": None,
        "fclip_device": None,
        "fclip_config": None,
        "fclip_params": None,
        "fclip_threshold": None,
        "ocr_device": None,
        "ocr_det_model_name": None,
        "ocr_det_model_dir": None,
        "ocr_rec_model_name": None,
        "ocr_rec_model_dir": None,
        "ocr_det_limit_side_len": None,
        "ocr_det_limit_type": None,
        "ocr_min_score": None,
        "ocr_number_range": None,
        "enable_ocr_orientation": None,
        "ocr_orientation_model": None,
        "ocr_orientation_device": None,
        "ocr_orientation_verbose": False,
        "enable_correction": None,
        "correction_model": None,
        "correction_device": None,
        "correction_verbose": None,
        "enhance_mode": None,
        "no_rotate": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class PipelineConfigOverrideTest(unittest.TestCase):
    def test_apply_cli_overrides_preserves_nested_model_types(self) -> None:
        cfg = PipelineConfig().apply_cli_overrides(
            _args(
                gauge_conf=0.5,
                fclip_threshold=0.7,
                ocr_min_score=0.2,
                enable_correction=True,
                no_rotate=True,
            )
        )

        self.assertIsInstance(cfg.gauge, GaugeConfig)
        self.assertIsInstance(cfg.fclip, FClipConfig)
        self.assertIsInstance(cfg.ocr, OCRConfig)
        self.assertIsInstance(cfg.correction, CorrectionConfig)
        self.assertIsInstance(cfg.enhance, EnhanceConfig)
        self.assertEqual(cfg.gauge.conf, 0.5)
        self.assertEqual(cfg.fclip.threshold, 0.7)
        self.assertEqual(cfg.ocr.min_score, 0.2)
        self.assertTrue(cfg.correction.enabled)
        self.assertFalse(cfg.enhance.rotate_roi)

    def test_apply_cli_overrides_keeps_original_when_no_override(self) -> None:
        cfg = PipelineConfig()
        updated = cfg.apply_cli_overrides(_args())

        self.assertIs(updated, cfg)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认当前失败**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_pipeline_config.py
```

Expected: FAIL，当前实现会把 `cfg.ocr` 变成 `dict`。

- [ ] **Step 3: 修复 `apply_cli_overrides`**

In `src/gauge/config/__init__.py`, replace the method body from `overrides: dict = {}` through `return self` with:

```python
        updates: dict = {}
        mapping = {
            "gauge_weights": ("gauge", "weights"),
            "gauge_conf": ("gauge", "conf"),
            "gauge_iou": ("gauge", "iou"),
            "gauge_imgsz": ("gauge", "imgsz"),
            "gauge_device": ("gauge", "device"),
            "gauge_select": ("gauge", "select"),
            "gauge_class": ("gauge", "class_filter"),
            "fclip_ckpt": ("fclip", "ckpt"),
            "fclip_device": ("fclip", "device"),
            "fclip_config": ("fclip", "fclip_model_config"),
            "fclip_params": ("fclip", "params"),
            "fclip_threshold": ("fclip", "threshold"),
            "ocr_device": ("ocr", "device"),
            "ocr_det_model_name": ("ocr", "det_model_name"),
            "ocr_det_model_dir": ("ocr", "det_model_dir"),
            "ocr_rec_model_name": ("ocr", "rec_model_name"),
            "ocr_rec_model_dir": ("ocr", "rec_model_dir"),
            "ocr_det_limit_side_len": ("ocr", "det_limit_side_len"),
            "ocr_det_limit_type": ("ocr", "det_limit_type"),
            "ocr_min_score": ("ocr", "min_score"),
            "ocr_number_range": ("ocr", "number_range"),
            "enable_ocr_orientation": ("ocr", "enable_orientation"),
            "ocr_orientation_model": ("ocr", "orientation_model"),
            "ocr_orientation_device": ("ocr", "orientation_device"),
            "enable_correction": ("correction", "enabled"),
            "correction_model": ("correction", "model"),
            "correction_device": ("correction", "device"),
            "correction_verbose": ("correction", "verbose"),
            "enhance_mode": ("enhance", "mode"),
            "no_rotate": ("enhance", "rotate_roi"),
        }
        for attr_name, config_path in mapping.items():
            value = getattr(args, attr_name, None)
            if value is None or value is False:
                continue
            section, key = config_path
            updates.setdefault(section, {})
            updates[section][key] = False if attr_name == "no_rotate" else value
        if getattr(args, "ocr_orientation_verbose", False):
            updates.setdefault("ocr", {})["orientation_verbose"] = True
        if not updates:
            return self

        data = self.model_dump()
        for section, section_updates in updates.items():
            data.setdefault(section, {})
            data[section].update(section_updates)
        return PipelineConfig.model_validate(data)
```

- [ ] **Step 4: 运行配置测试**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_pipeline_config.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/test_pipeline_config.py src/gauge/config/__init__.py
git commit -m "fix: preserve typed config models during CLI overrides"
```

## Task 4: 收敛 record 构建边界

**Files:**
- Create: `src/gauge/record_builders.py`
- Create: `tests/test_record_builders.py`
- Modify: `src/gauge/stages/base.py`
- Modify: `src/gauge/iqi_inferencer.py`

- [ ] **Step 1: 写 record builder 测试**

Create `tests/test_record_builders.py`:

```python
import unittest
import warnings

from gauge.config import PipelineConfig
from gauge.record_builders import build_iqi_record
from gauge.stages.base import StageContext


class RecordBuilderTest(unittest.TestCase):
    def test_build_iqi_record_validates_nested_sections_without_serializer_warning(self) -> None:
        ctx = StageContext(image_path="demo.png", config=PipelineConfig())
        ctx.width = 64
        ctx.height = 32
        ctx.general_fields_data = {
            "fields": {
                "component_codes": [{"text": "4S9", "match_text": "4S9", "value": "4S9"}],
                "weld_film_pairs": [],
                "weld_numbers": [],
                "film_numbers": [],
                "pipe_specs": [],
            }
        }
        ctx.field_statistics = {
            "component_code_count": 1,
            "weld_film_pair_count": 0,
            "weld_number_count": 0,
            "film_number_count": 0,
            "pipe_spec_count": 0,
            "general_fields_found": True,
            "full_image_marker_found": True,
            "roi_marker_found": True,
            "iqi_marker_found": True,
        }
        ctx.full_ocr_result = {"status": "ok", "texts": ["10FEJB"], "all_items": [], "items": []}
        ctx.roi_ocr_result = {"status": "ok", "texts": ["10FEJB"], "all_items": [], "items": []}
        ctx.full_plate_result = {"ok": True, "result_code": 0, "result_name": "success", "result_message": "识别成功", "iqi_type": "general", "number": 10, "plate_code": "10FEJB"}
        ctx.roi_plate_result = dict(ctx.full_plate_result)
        ctx.plate_result = dict(ctx.full_plate_result)
        ctx.plate_source = "roi"
        ctx.wire_result = {"status": "ok", "wire_count": 2, "parsed_line_count": 0, "lines": [], "warnings": []}
        ctx.grade_result = {"ok": True, "result_code": 0, "result_name": "success", "result_message": "识别成功", "grade": 11, "wire_count": 2}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            record = build_iqi_record(ctx)
            dumped = record.model_dump()

        serializer_warnings = [w for w in caught if "Pydantic serializer warnings" in str(w.message)]
        self.assertEqual(serializer_warnings, [])
        self.assertTrue(dumped["ok"])
        self.assertEqual(dumped["grade"], 11)
        self.assertEqual(dumped["fields"]["component_codes"][0]["value"], "4S9")
        self.assertEqual(dumped["plate"]["plate_code"], "10FEJB")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认当前失败**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_record_builders.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gauge.record_builders'`.

- [ ] **Step 3: 创建 `record_builders.py`**

Create `src/gauge/record_builders.py` by moving the existing `StageContext.to_record()` logic and helper methods from `src/gauge/stages/base.py`.

The module must expose:

```python
#!/usr/bin/env python3
"""Builders for IQI records, delivery payloads, and batch statistics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from gauge.iqi_rules import (
    build_result_status,
    choose_primary_result_code,
    infer_plate_from_texts,
    normalize_text,
    parse_allowed_numbers_spec,
    summarize_result_codes,
    summarize_result_codes_named,
)
from gauge.models.record import IQIRecord
from gauge.services.ocr_stage import build_ocr_statistics


def build_iqi_record(ctx: Any) -> IQIRecord:
    primary_code = choose_primary_result_code(
        [entry["result_code"] for entry in ctx.record_errors]
    )
    status = build_result_status(primary_code)
    plate = ctx.plate_result or {}
    wire = ctx.wire_result or {}
    selected_plate_source = ctx.plate_source

    rec_plate = dict(plate)
    if selected_plate_source == "roi":
        rec_plate["raw_text_items"] = list(ctx.roi_plate_vis_items or [])
    elif selected_plate_source == "full_image":
        rec_plate["raw_text_items"] = list(ctx.full_plate_vis_items or [])
    else:
        rec_plate.setdefault("raw_text_items", [])

    return IQIRecord(
        image_path=ctx.image_path,
        ok=primary_code == 0,
        status="ok" if primary_code == 0 else "error",
        result_code=primary_code,
        result_name=status["result_name"],
        result_message=status["result_message"],
        grade=int(ctx.grade_result["grade"])
        if primary_code == 0 and ctx.grade_result is not None and ctx.grade_result.get("grade") is not None
        else None,
        iqi_type=plate.get("iqi_type"),
        plate_code=plate.get("plate_code"),
        plate_number=plate.get("number"),
        plate_source=selected_plate_source,
        wire_count=wire.get("wire_count"),
        width=ctx.width,
        height=ctx.height,
        general_fields_found=bool(ctx.field_statistics.get("general_fields_found", False)),
        iqi_marker_found=bool(plate.get("ok", False)),
        fields=ctx.general_fields_data.get("fields") if isinstance(ctx.general_fields_data.get("fields"), dict) else {},
        field_statistics=dict(ctx.field_statistics),
        correction=dict(ctx.correction_info),
        full_image_preprocess=build_full_image_preprocess(ctx),
        preprocess=build_roi_preprocess(ctx),
        ocr=ctx.full_ocr_result,
        full_image_ocr=ctx.full_ocr_result,
        full_image_plate=ctx.full_plate_result,
        roi=ctx.roi_info,
        roi_ocr=ctx.roi_ocr_result,
        roi_plate=ctx.roi_plate_result,
        plate=rec_plate,
        wire=wire,
        grade_rule=ctx.grade_result,
        warnings=list(ctx.warnings),
        errors=list(ctx.record_errors),
        visualization=build_visualization(ctx),
        timings_ms=dict(ctx.timings_ms),
    )
```

Then move the remaining helpers from `StageContext` as module functions:

```python
def build_full_image_preprocess(ctx: Any) -> Dict[str, Any]:
    ...

def build_roi_preprocess(ctx: Any) -> Dict[str, Any]:
    ...

def build_visualization(ctx: Any) -> Dict[str, Any]:
    ...

def build_iqi_statistics(results: Sequence[Dict[str, Any]], topk: int = 200) -> Dict[str, Any]:
    ...

def build_delivery_record(record: Dict[str, Any]) -> Dict[str, Any]:
    ...
```

Use the current implementations from `StageContext` and `iqi_inferencer.py` unchanged except for replacing `self` with `ctx` where needed.

- [ ] **Step 4: 修改 `StageContext` 默认值和 `to_record()`**

In `src/gauge/stages/base.py`, change mutable defaults:

```python
correction_info: Dict[str, Any] = Field(default_factory=dict)
full_plate_vis_items: List[Dict[str, Any]] = Field(default_factory=list)
general_fields_data: Dict[str, Any] = Field(default_factory=dict)
field_statistics: Dict[str, Any] = Field(default_factory=dict)
roi_plate_vis_items: List[Dict[str, Any]] = Field(default_factory=list)
record_errors: List[Dict[str, Any]] = Field(default_factory=list)
```

Replace `to_record()` with:

```python
    def to_record(self) -> IQIRecord:
        """Build the final IQIRecord from current context state."""
        from gauge.record_builders import build_iqi_record

        record = build_iqi_record(self)
        if self.debug_artifacts is not None:
            record._debug_artifacts = self.debug_artifacts
        return record
```

Remove helper methods from `StageContext` after confirming equivalent functions exist in `record_builders.py`.

- [ ] **Step 5: 兼容导出 delivery/statistics**

In `src/gauge/iqi_inferencer.py`, replace module-level `build_iqi_statistics` and `build_delivery_record` function bodies with imports:

```python
from gauge.record_builders import build_delivery_record, build_iqi_statistics
```

Remove now-unused imports from `collections import Counter` and result-summary helpers if they are no longer used in this file.

- [ ] **Step 6: 运行 record 测试**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_record_builders.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_delivery_record.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/gauge/record_builders.py src/gauge/stages/base.py src/gauge/iqi_inferencer.py tests/test_record_builders.py
git commit -m "refactor: centralize IQI record builders"
```

## Task 5: 拆出 visualization 并收窄 IQIInferencer

**Files:**
- Create: `src/gauge/visualization.py`
- Modify: `src/gauge/iqi_inferencer.py`
- Modify: `run_iqi_grade_infer.py`
- Test: `tests/test_iqi_delivery_record.py`
- Test: `tests/test_iqi_inferencer.py`

- [ ] **Step 1: 创建可视化模块**

Create `src/gauge/visualization.py` by moving these existing functions from `src/gauge/iqi_inferencer.py` without behavior changes:

```python
build_wire_vis_image
build_final_result_vis_image
save_debug_visualizations
```

The top imports in `visualization.py` must be:

```python
#!/usr/bin/env python3
"""Debug and final-result visualization helpers for IQI inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from gauge.pipeline_utils import ensure_dir
from gauge.services.ocr_stage import build_ocr_item_debug_images, draw_ocr_on_roi
from gauge.services.roi_stage import build_roi_vis_image
```

- [ ] **Step 2: 在 `iqi_inferencer.py` 中保留兼容导出**

In `src/gauge/iqi_inferencer.py`, remove moved function definitions and add:

```python
from gauge.visualization import (
    build_final_result_vis_image,
    build_wire_vis_image,
    save_debug_visualizations,
)
```

If `run_iqi_grade_infer.py` still imports `save_debug_visualizations` from `gauge.iqi_inferencer`, keep that import working through this re-export.

- [ ] **Step 3: 清理 `iqi_inferencer.py` 未使用 imports**

Run:

```bash
python - <<'PY'
from pathlib import Path
p = Path("src/gauge/iqi_inferencer.py")
text = p.read_text()
for name in ["build_wire_vis_image", "build_final_result_vis_image", "save_debug_visualizations"]:
    print(name, text.count(name))
PY
```

Expected: each moved function name appears in import/export use, not as full function definitions.

- [ ] **Step 4: 运行兼容测试**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_delivery_record.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_inferencer.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/gauge/visualization.py src/gauge/iqi_inferencer.py run_iqi_grade_infer.py tests/test_iqi_delivery_record.py tests/test_iqi_inferencer.py
git commit -m "refactor: move IQI visualization helpers out of inferencer"
```

## Task 6: 迁移 OCR subprocess client 到 ocr_runtime

**Files:**
- Create: `src/gauge/ocr_runtime.py`
- Create: `tests/test_ocr_runtime.py`
- Modify: `src/gauge/services/ocr_stage.py`
- Modify: `src/gauge/services/region_ocr_service.py`
- Modify: `src/gauge/iqi_inferencer.py`

- [ ] **Step 1: 写 OCR runtime 测试**

Create `tests/test_ocr_runtime.py`:

```python
import io
import json
import threading
import unittest
from unittest import mock

from gauge.ocr_runtime import PaddleOCRSubprocessClient


class _FakeProcess:
    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.returncode = None
        self.wait_called = False
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_called = True
        self.returncode = 0
        return 0

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class PaddleOCRSubprocessClientRuntimeTest(unittest.TestCase):
    def _client_without_start(self):
        client = object.__new__(PaddleOCRSubprocessClient)
        client.process = _FakeProcess()
        client._lock = threading.Lock()
        client.request_timeout_s = 1.0
        return client

    def test_request_uses_lock_and_returns_result(self) -> None:
        client = self._client_without_start()
        with mock.patch.object(
            client,
            "_readline_with_timeout",
            return_value=json.dumps({"ok": True, "result": {"rec_text": "X"}}) + "\n",
        ):
            result = client._request({"op": "recognize", "image": {"format": "png_base64", "data": ""}})

        self.assertEqual(result["result"]["rec_text"], "X")
        self.assertIn('"op": "recognize"', client.process.stdin.getvalue())

    def test_request_timeout_closes_process(self) -> None:
        client = self._client_without_start()
        with mock.patch.object(client, "_readline_with_timeout", side_effect=TimeoutError("OCR worker response timed out")):
            with self.assertRaises(RuntimeError) as caught:
                client._request({"op": "detect", "image": {"format": "png_base64", "data": ""}})

        self.assertIn("timed out", str(caught.exception))
        self.assertIsNone(client.process)

    def test_close_is_idempotent(self) -> None:
        client = self._client_without_start()

        client.close()
        client.close()

        self.assertIsNone(client.process)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认当前失败**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_ocr_runtime.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gauge.ocr_runtime'`.

- [ ] **Step 3: 创建 `ocr_runtime.py`**

Move `PaddleOCRSubprocessClient` from `src/gauge/services/ocr_stage.py` to `src/gauge/ocr_runtime.py`.

At the top of `src/gauge/ocr_runtime.py`, use:

```python
#!/usr/bin/env python3
"""Subprocess runtime client for PaddleOCR worker protocol."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import selectors
import subprocess
import sys
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np
```

Add constructor parameters:

```python
        startup_timeout_s: float = 120.0,
        request_timeout_s: float = 60.0,
```

Initialize:

```python
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self._lock = threading.Lock()
```

Implement timeout read:

```python
    def _readline_with_timeout(self, timeout_s: float) -> str:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("OCR worker stdout is not available.")
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(timeout=float(timeout_s))
            if not events:
                raise TimeoutError("OCR worker response timed out")
            return self.process.stdout.readline()
        finally:
            selector.close()
```

Update `_read_response`:

```python
    def _read_response(self, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        timeout = self.request_timeout_s if timeout_s is None else float(timeout_s)
        try:
            line = self._readline_with_timeout(timeout)
        except TimeoutError as exc:
            self.close()
            raise RuntimeError(str(exc)) from exc
        if not line:
            returncode = self.process.poll() if self.process is not None else None
            raise RuntimeError(f"OCR worker exited unexpectedly with code {returncode}.")
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse OCR worker response: {line.strip()}") from exc
```

Update `_request`:

```python
    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self.process is None or self.process.stdin is None:
                raise RuntimeError("OCR worker stdin is not available.")
            if self.process.poll() is not None:
                raise RuntimeError(f"OCR worker has already exited with code {self.process.returncode}.")
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            response = self._read_response(timeout_s=self.request_timeout_s)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "OCR worker request failed."))
            return response
```

In `_start`, change:

```python
ready = self._read_response()
```

to:

```python
ready = self._read_response(timeout_s=self.startup_timeout_s)
```

- [ ] **Step 4: Keep old import path compatible**

In `src/gauge/services/ocr_stage.py`, remove the class definition and add:

```python
from gauge.ocr_runtime import PaddleOCRSubprocessClient
```

Keep all OCR normalization and drawing functions in `ocr_stage.py`.

- [ ] **Step 5: Update direct imports**

In `src/gauge/services/region_ocr_service.py` and `src/gauge/iqi_inferencer.py`, replace:

```python
from gauge.services.ocr_stage import PaddleOCRSubprocessClient
```

with:

```python
from gauge.ocr_runtime import PaddleOCRSubprocessClient
```

- [ ] **Step 6: Run OCR runtime tests**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_ocr_runtime.py
```

Expected: PASS.

- [ ] **Step 7: Run OCR stage compile**

Run:

```bash
python -m py_compile src/gauge/ocr_runtime.py src/gauge/services/ocr_stage.py src/gauge/services/region_ocr_service.py src/gauge/iqi_inferencer.py
```

Expected: no output.

- [ ] **Step 8: Commit**

Run:

```bash
git add src/gauge/ocr_runtime.py src/gauge/services/ocr_stage.py src/gauge/services/region_ocr_service.py src/gauge/iqi_inferencer.py tests/test_ocr_runtime.py
git commit -m "refactor: move OCR worker client into runtime module"
```

## Task 7: 统一区域 API runtime

**Files:**
- Create: `src/gauge/region_runtime.py`
- Create: `tests/test_region_runtime.py`
- Modify: `src/gauge/services/region_ocr_api.py`
- Modify: `src/gauge/services/region_snr_api.py`
- Modify: `src/gauge/services/base.py`

- [ ] **Step 1: 写区域 runtime 测试**

Create `tests/test_region_runtime.py`:

```python
import base64
import unittest

import cv2
import numpy as np

from gauge.region_runtime import decode_base64, register_region_service_shutdown, shutdown_region_runtime


class _Closable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class RegionRuntimeTest(unittest.TestCase):
    def test_decode_base64_accepts_data_url_prefix(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        payload = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

        decoded = decode_base64(payload)

        self.assertEqual(decoded.shape[:2], (4, 5))

    def test_decode_base64_rejects_invalid_payload(self) -> None:
        with self.assertRaises(Exception) as caught:
            decode_base64("not-valid-base64")

        self.assertTrue(hasattr(caught.exception, "status_code"))
        self.assertEqual(caught.exception.status_code, 400)

    def test_registered_services_are_closed(self) -> None:
        service = _Closable()
        register_region_service_shutdown(lambda: service.close())

        shutdown_region_runtime()

        self.assertTrue(service.closed)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认当前失败**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_runtime.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gauge.region_runtime'`.

- [ ] **Step 3: 创建 `region_runtime.py`**

Create `src/gauge/region_runtime.py`:

```python
#!/usr/bin/env python3
"""Shared runtime utilities for region OCR and SNR APIs."""

from __future__ import annotations

import atexit
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List

import cv2
import numpy as np

try:
    from fastapi import HTTPException
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = int(status_code)
            self.detail = str(detail)


executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="region-svc")
_shutdown_callbacks: List[Callable[[], None]] = []


def decode_base64(image_base64: str) -> np.ndarray:
    b64_data = str(image_base64 or "")
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="无法解码图片") from exc
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图片")
    return img


def register_region_service_shutdown(callback: Callable[[], None]) -> None:
    _shutdown_callbacks.append(callback)


def shutdown_region_runtime() -> None:
    while _shutdown_callbacks:
        callback = _shutdown_callbacks.pop()
        try:
            callback()
        except Exception:
            pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)


atexit.register(shutdown_region_runtime)
```

- [ ] **Step 4: 更新区域 API 使用新 runtime**

In `src/gauge/services/region_ocr_api.py`, replace:

```python
from gauge.services.base import BaseRegionService
```

with:

```python
from gauge.region_runtime import decode_base64, executor, register_region_service_shutdown
```

After `close_region_ocr_api()` definition, add:

```python
register_region_service_shutdown(close_region_ocr_api)
```

Replace:

```python
img = BaseRegionService.decode_base64(request.image_base64)
result = await loop.run_in_executor(BaseRegionService._executor, _sync_ocr_recognize, img)
```

with:

```python
img = decode_base64(request.image_base64)
result = await loop.run_in_executor(executor, _sync_ocr_recognize, img)
```

Apply the same pattern to `src/gauge/services/region_snr_api.py`.

- [ ] **Step 5: Keep BaseRegionService as compatibility wrapper**

In `src/gauge/services/base.py`, replace `decode_base64` implementation with:

```python
    @staticmethod
    def decode_base64(image_base64: str) -> np.ndarray:
        from gauge.region_runtime import decode_base64

        return decode_base64(image_base64)
```

Leave `BaseRegionService` available for imports, but do not use it in region APIs.

- [ ] **Step 6: Run region runtime tests**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_runtime.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_snr_service.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/gauge/region_runtime.py src/gauge/services/region_ocr_api.py src/gauge/services/region_snr_api.py src/gauge/services/base.py tests/test_region_runtime.py
git commit -m "refactor: share region API runtime utilities"
```

## Task 8: 清理包入口、README 和训练脚本

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/gauge/README.md`
- Modify: `src/gauge/training/OBBtraintest.py`
- Delete: `src/gauge/training/infer.py`
- Delete: `src/gauge/training/valid.py`
- Modify: `tests/test_script_layout.py`

- [ ] **Step 1: 写脚本布局测试**

In `tests/test_script_layout.py`, add this test method:

```python
    def test_training_helpers_do_not_execute_at_import(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "src" / "gauge" / "training" / "OBBtraintest.py"
        source = script.read_text(encoding="utf-8")

        self.assertIn('if __name__ == "__main__":', source)
        self.assertIn("main()", source)
```

Add this assertion to `test_debug_scripts_live_under_scripts_debug_and_compile`:

```python
        self.assertFalse((repo_root / "src" / "gauge" / "training" / "infer.py").exists())
        self.assertFalse((repo_root / "src" / "gauge" / "training" / "valid.py").exists())
```

- [ ] **Step 2: 运行脚本布局测试并确认当前失败**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_script_layout.py
```

Expected: FAIL because `OBBtraintest.py` has top-level execution and `infer.py` / `valid.py` still exist.

- [ ] **Step 3: 修正 pyproject console script**

In `pyproject.toml`, delete:

```toml
[project.scripts]
iqi-grade-infer = "gauge.cli:main"
```

This avoids advertising a console entrypoint that does not exist in the installed package.

- [ ] **Step 4: 更新 `src/gauge/README.md`**

Replace `src/gauge/README.md` with:

```markdown
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
```

- [ ] **Step 5: 给 OBBtraintest 增加 main guard**

Replace `src/gauge/training/OBBtraintest.py` with:

```python
#!/usr/bin/env python3
"""Local validation helper for the gauge OBB detector."""

from __future__ import annotations

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo26n-obb.pt")
    model.val(data="IQIdata/gauge_obb/data.yaml")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 删除未实现训练入口文件**

Run:

```bash
git rm src/gauge/training/infer.py src/gauge/training/valid.py
```

Expected: two files staged for deletion.

- [ ] **Step 7: 运行脚本布局测试**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_script_layout.py
python -m py_compile src/gauge/training/OBBtraintest.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add pyproject.toml src/gauge/README.md src/gauge/training/OBBtraintest.py tests/test_script_layout.py
git commit -m "chore: align gauge package scripts and docs"
```

## Task 9: 全量轻量验证和回归输出比较

**Files:**
- No code changes expected.

- [ ] **Step 1: Run py_compile**

Run:

```bash
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py tests/test_iqi_delivery_record.py tests/test_iqi_inferencer.py tests/test_region_snr_service.py tests/test_script_layout.py tests/test_import_boundaries.py tests/test_pipeline_config.py tests/test_geometry.py tests/test_record_builders.py tests/test_ocr_runtime.py tests/test_region_runtime.py
```

Expected: no output.

- [ ] **Step 2: Run focused tests**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_import_boundaries.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_pipeline_config.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_geometry.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_record_builders.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_ocr_runtime.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_runtime.py
```

Expected: all PASS.

- [ ] **Step 3: Run existing lightweight tests**

Run:

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_delivery_record.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_inferencer.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_snr_service.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_script_layout.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_rules.py
```

Expected: all PASS.

- [ ] **Step 4: Run regression output comparison**

Run:

```bash
python scripts/compare_results.py
```

Expected:

```text
ALL MATCH — 8 image(s), 0 differences.
```

If output starts with `SKIP: missing prerequisites`, do not mark this step complete for release readiness. The local baseline or image directory must be restored, then the command must be rerun until it reports `ALL MATCH`.

- [ ] **Step 5: Commit verification updates if any test-only fixes were needed**

Run only if the previous steps required test-command or doc adjustments:

```bash
git status --short
git add docs/superpowers/plans/2026-05-30-gauge-architecture-dedup-plan.md docs/superpowers/specs/2026-05-30-gauge-architecture-dedup-design.md
git commit -m "docs: add gauge architecture dedup implementation plan"
```

Expected: commit contains only plan/spec documentation if no code changes were pending.

## Self-Review

- Spec coverage:
  - 导入边界由 Task 1 覆盖。
  - 配置验证由 Task 3 覆盖。
  - 几何去重由 Task 1 和 Task 2 覆盖。
  - record 构建由 Task 4 覆盖。
  - `IQIInferencer` 职责收敛由 Task 4 和 Task 5 覆盖。
  - OCR runtime 由 Task 6 覆盖。
  - 区域服务 runtime 由 Task 7 覆盖。
  - 包入口、README、脚本清理由 Task 8 覆盖。
  - `compare_results.py` 输出一致性由 Task 9 覆盖。

- Red-flag marker scan:
  - 本计划已扫描常见未决标记词，当前没有命中。

- Type consistency:
  - 新函数名 `project_roi_box_to_image`、`project_ocr_items_to_image`、`build_iqi_record`、`decode_base64`、`PaddleOCRSubprocessClient` 在任务间保持一致。
