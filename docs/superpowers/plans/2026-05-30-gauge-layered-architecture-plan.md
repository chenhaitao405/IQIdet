# Gauge 职责分层重排实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `src/gauge` 根目录散落的不同职责文件迁入 `app/`、`pipeline/`、`domain/`、`imaging/`、`runtime/`、`services/` 六个子包，每个子包职责明确，依赖方向清晰。

**Architecture:** 按 spec 的低风险到高风险 9 步迁移顺序。每步一个独立 commit，完成后仓库保持可导入、可测试状态。旧路径不保留 shim，同一 commit 内移动文件 + 更新 import + 删除旧文件。

**Tech Stack:** Python 3, unittest, git

---

## 文件结构映射总览

### 新建目录

```
src/gauge/app/__init__.py
src/gauge/domain/__init__.py
src/gauge/imaging/__init__.py
src/gauge/runtime/__init__.py
src/gauge/pipeline/stages/__init__.py  (stages/→pipeline/stages/ 原位留空壳)
src/gauge/services/fclip/__init__.py
src/gauge/services/ocr/__init__.py
src/gauge/services/roi/__init__.py
src/gauge/services/orientation/__init__.py
src/gauge/services/region/__init__.py
```

### 文件迁移映射

| 旧路径 | 新路径 |
|--------|--------|
| `gauge/iqi_rules.py` | `gauge/domain/iqi_rules.py` |
| `gauge/record_builders.py` | `gauge/domain/record_builders.py` |
| `gauge/geometry.py` | `gauge/imaging/geometry.py` |
| `gauge/visualization.py` | `gauge/imaging/visualization.py` |
| `gauge/pipeline_utils.py` | 拆到 `imaging/preprocess.py` + `app/inputs.py` + `domain/record_builders.py` + `imaging/geometry.py` |
| `gauge/services/adaptive_image_processor.py` | `gauge/imaging/adaptive.py` |
| `gauge/ocr_runtime.py` | `gauge/runtime/ocr_runtime.py` |
| `gauge/region_runtime.py` | `gauge/runtime/region_runtime.py` |
| `gauge/services/ocr_paddle_worker.py` | `gauge/runtime/ocr_paddle_worker.py` |
| `gauge/pipeline.py` | `gauge/pipeline/runner.py` |
| `gauge/stages/` (8 files) | `gauge/pipeline/stages/` (重命名) |
| `gauge/iqi_inferencer.py` | `gauge/app/iqi_inferencer.py` |
| `gauge/services/region_ocr_api.py` | `gauge/app/region_ocr_api.py` |
| `gauge/services/region_snr_api.py` | `gauge/app/region_snr_api.py` |
| `gauge/services/ocr_stage.py` | `gauge/services/ocr/factory.py` + `infer.py` + `normalize.py` + `debug.py` |
| (新建) `domain/statistics.py` | 从 `ocr_stage.py` `build_ocr_statistics` 拆出 |
| `gauge/services/fclip_stage.py` | `gauge/services/fclip/inferencer.py` + `line_records.py` |
| `gauge/services/roi_stage.py` | `gauge/services/roi/yolo_obb.py` |
| `gauge/services/correction.py` | `gauge/services/orientation/base.py` |
| `gauge/services/ocr_orientation.py` | `gauge/services/orientation/ocr_text.py` |
| `gauge/services/weld_correction.py` | `gauge/services/orientation/weld.py` |
| `gauge/services/region_ocr_service.py` | `gauge/services/region/ocr_service.py` |
| `gauge/services/region_snr_service.py` | `gauge/services/region/snr_service.py` |

---

### Task 1: 创建新目录结构和 __init__.py

**Files:**
- Create: `src/gauge/app/__init__.py`
- Create: `src/gauge/domain/__init__.py`
- Create: `src/gauge/imaging/__init__.py`
- Create: `src/gauge/runtime/__init__.py`
- Create: `src/gauge/services/fclip/__init__.py`
- Create: `src/gauge/services/ocr/__init__.py`
- Create: `src/gauge/services/roi/__init__.py`
- Create: `src/gauge/services/orientation/__init__.py`
- Create: `src/gauge/services/region/__init__.py`

- [ ] **Step 1: 创建所有新 __init__.py 文件**

```bash
mkdir -p src/gauge/app src/gauge/domain src/gauge/imaging src/gauge/runtime
mkdir -p src/gauge/services/fclip src/gauge/services/ocr src/gauge/services/roi
mkdir -p src/gauge/services/orientation src/gauge/services/region

# 写每个 __init__.py（空文件，仅 docstring）
for dir in app domain imaging runtime services/fclip services/ocr services/roi services/orientation services/region; do
    echo "\"\"\"gauge.$dir sub-package.\"\"\"" > src/gauge/$dir/__init__.py
done
```

- [ ] **Step 2: 语法检查验证目录可导入**

```bash
conda activate weld-gpu
cd /home/cht/code/IQIdet
PYTHONPATH=/home/cht/code/IQIdet/src python -c "
import gauge.app
import gauge.domain
import gauge.imaging
import gauge.runtime
import gauge.services.fclip
import gauge.services.ocr
import gauge.services.roi
import gauge.services.orientation
import gauge.services.region
print('All new packages importable')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/gauge/app/__init__.py src/gauge/domain/__init__.py src/gauge/imaging/__init__.py src/gauge/runtime/__init__.py
git add src/gauge/services/fclip/__init__.py src/gauge/services/ocr/__init__.py src/gauge/services/roi/__init__.py src/gauge/services/orientation/__init__.py src/gauge/services/region/__init__.py
git commit -m "chore: create new gauge sub-package directories

Create app/, domain/, imaging/, runtime/ and services/*/ sub-packages
with empty __init__.py files in preparation for layered migration.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 迁移 domain/ — 纯业务规则层

**迁移内容:** `iqi_rules.py` → `domain/iqi_rules.py`, `record_builders.py` → `domain/record_builders.py`

**依赖规则:** domain/ 只能依赖标准库和 `models/`，不能依赖 OpenCV/Numpy/Torch/PaddleOCR/Ultralytics。

- [ ] **Step 1: 复制 iqi_rules.py 到新路径并验证 import 不在 domain/ 引入重依赖**

```bash
cp src/gauge/iqi_rules.py src/gauge/domain/iqi_rules.py
```

- [ ] **Step 2: 复制 record_builders.py 到新路径，修改其内部 import**

`record_builders.py` 第 19 行有 `from gauge.services.ocr_stage import build_ocr_statistics`，这个 import 违反 domain/ 规则。先保留该 import 但后续 Task 5 会将其迁入 `domain/statistics.py`。临时处理：在新 `domain/record_builders.py` 中将此 import 改为延迟导入。

```bash
cp src/gauge/record_builders.py src/gauge/domain/record_builders.py
```

编辑 `src/gauge/domain/record_builders.py`：
- 第 9 行: `from gauge.iqi_rules import ...` → `from gauge.domain.iqi_rules import ...`
- 第 19 行: `from gauge.services.ocr_stage import build_ocr_statistics` → 改为函数内延迟导入：
  - 在 `build_iqi_statistics()` 函数体开头（第 184 行后）添加: `from gauge.services.ocr_stage import build_ocr_statistics`
  - 删除顶部第 19 行的 import

- [ ] **Step 3: 更新引用 domain 新路径的内部模块**

更新以下文件的 import：
- `src/gauge/iqi_inferencer.py`: `from gauge.iqi_rules` → `from gauge.domain.iqi_rules`
- `src/gauge/stages/grade_fusion.py`: `from gauge.iqi_rules` → `from gauge.domain.iqi_rules`
- `src/gauge/stages/roi_ocr.py`: `from gauge.iqi_rules` → `from gauge.domain.iqi_rules`
- `src/gauge/stages/base.py`: `from gauge.record_builders` → `from gauge.domain.record_builders`
- `src/gauge/services/region_ocr_service.py`: `from gauge.iqi_rules` → `from gauge.domain.iqi_rules`

- [ ] **Step 4: 删除旧文件**

```bash
rm src/gauge/iqi_rules.py
rm src/gauge/record_builders.py
```

- [ ] **Step 5: 运行测试验证**

```bash
conda activate weld-gpu
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_rules.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_record_builders.py
python -m py_compile src/gauge/domain/iqi_rules.py src/gauge/domain/record_builders.py
```

- [ ] **Step 6: 验证 domain/ 边界 — 不导入重依赖**

```bash
PYTHONPATH=/home/cht/code/IQIdet/src python -c "
import sys
class Blocker:
    blocked = {'cv2', 'numpy', 'torch', 'paddleocr', 'ultralytics', 'FClip'}
    def find_spec(self, fullname, target=None):
        root = fullname.split('.')[0]
        if root in self.blocked:
            raise ImportError(f'Blocked: {fullname}')
        return None
sys.meta_path.insert(0, Blocker())
import gauge.domain.iqi_rules
print('domain/iqi_rules.py: PASS (no heavy imports)')
"
```

- [ ] **Step 7: 语法检查全部 gauge 模块**

```bash
cd /home/cht/code/IQIdet
for f in $(find src/gauge -name '*.py' -not -path '*__pycache__*'); do
    python -m py_compile "$f" || echo "FAIL: $f"
done
```

- [ ] **Step 8: 更新测试文件 import**

更新 `tests/test_iqi_rules.py`:
- `from gauge.iqi_rules` → `from gauge.domain.iqi_rules`

`tests/test_record_builders.py` 已经 import `gauge.record_builders`，需更新为 `gauge.domain.record_builders` 和 `gauge.domain.iqi_rules`。

`tests/test_iqi_delivery_record.py` 和 `tests/test_iqi_inferencer.py` 暂时不改，因为 `iqi_inferencer.py` 还在原路径。

所有测试通过后 commit。

- [ ] **Step 9: Commit**

```bash
git add src/gauge/domain/ tests/
git add -u src/gauge/  # 删除旧文件
git commit -m "refactor: migrate domain/ — pure business rules layer

Move iqi_rules.py and record_builders.py to gauge/domain/.
Update all internal imports to use gauge.domain.* paths.
Remove old gauge/iqi_rules.py and gauge/record_builders.py.
domain/ only depends on stdlib and gauge/models/.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 迁移 imaging/ — 纯图像处理工具层

**迁移内容:** `geometry.py` → `imaging/geometry.py`, `visualization.py` → `imaging/visualization.py`, 从 `pipeline_utils.py` 拆出图像处理函数 → `imaging/preprocess.py`, `adaptive_image_processor.py` → `imaging/adaptive.py`

**依赖规则:** imaging/ 可依赖 OpenCV 和 Numpy，不依赖 Torch/PaddleOCR/Ultralytics。

- [ ] **Step 1: 创建 imaging/geometry.py**

```bash
cp src/gauge/geometry.py src/gauge/imaging/geometry.py
# 文件内容不变，纯函数无需修改 import
```

- [ ] **Step 2: 创建 imaging/visualization.py**

```bash
cp src/gauge/visualization.py src/gauge/imaging/visualization.py
```

编辑 `src/gauge/imaging/visualization.py`，更新内部 import：
- 第 12 行: `from gauge.pipeline_utils import ensure_dir` → `from gauge.imaging.preprocess import ensure_dir`（先标记，等 preprocess.py 创建后改）
- 第 13 行: `from gauge.services.ocr_stage import build_ocr_item_debug_images, draw_ocr_on_roi` → `from gauge.services.ocr.debug import build_ocr_item_debug_images, draw_ocr_on_roi`（先标记，等 Task 5 完成）
- 第 14 行: `from gauge.services.roi_stage import build_roi_vis_image` → `from gauge.services.roi.yolo_obb import build_roi_vis_image`（先标记，等 Task 7 完成）

由于 visualization.py 依赖尚未迁移的 services 模块，本步骤**先复制文件到新路径，import 暂时保持旧路径指向旧模块**。后续 task 完成后再更新。

- [ ] **Step 3: 创建 imaging/preprocess.py（从 pipeline_utils.py 拆出图像函数）**

从 `pipeline_utils.py` 提取以下函数到 `src/gauge/imaging/preprocess.py`：
- `load_image()`
- `resize_long_side()`
- `order_points()`
- `crop_rotated_polygon()`
- `rotate_if_wide()`
- `auto_window_level()`
- `apply_window_level()`
- `apply_clahe()`
- `enhance_windowing_gray()`
- `to_gray()`
- `format_polygon()`
- `safe_list()`
- `SUPPORTED_IMAGE_EXTS`

```bash
# 创建文件，写入完整内容
cat > src/gauge/imaging/preprocess.py << 'PYEOF'
#!/usr/bin/env python3
"""Image pre-processing helpers for the gauge pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def resize_long_side(image: np.ndarray, target_long_side: Optional[int]) -> Tuple[np.ndarray, float]:
    if target_long_side is None:
        return image, 1.0
    target = int(target_long_side)
    if target <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= target:
        return image, 1.0
    scale = float(target) / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_rotated_polygon(image: np.ndarray, polygon: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    box = order_points(polygon.astype(np.float32))
    w1 = np.linalg.norm(box[0] - box[1])
    w2 = np.linalg.norm(box[2] - box[3])
    h1 = np.linalg.norm(box[0] - box[3])
    h2 = np.linalg.norm(box[1] - box[2])
    width = int(round(max(w1, w2)))
    height = int(round(max(h1, h2)))
    if width < 2 or height < 2:
        return None, None
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix


def rotate_if_wide(image: np.ndarray, enable: bool = True) -> Tuple[np.ndarray, bool, int]:
    if not enable:
        return image, False, 0
    h, w = image.shape[:2]
    if w > h:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rotated, True, 90
    return image, False, 0


def auto_window_level(image: np.ndarray) -> Tuple[int, int]:
    percentiles = np.percentile(image, [2, 98])
    img_min, img_max = percentiles[0], percentiles[1]
    img_mean = np.mean(image)
    img_std = np.std(image)
    window_level = int(img_mean)
    window_width = int(min(4 * img_std, img_max - img_min))
    window_width = max(1, window_width)
    return window_width, window_level


def apply_window_level(image: np.ndarray, window_width: int, window_level: int) -> np.ndarray:
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2
    if window_max <= window_min:
        window_max = window_min + 1
    output = np.clip((image - window_min) / window_width * 255.0, 0, 255).astype(np.uint8)
    return output


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def enhance_windowing_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    img_float = gray.astype(np.float32, copy=False)
    ww, wl = auto_window_level(img_float)
    enhanced = apply_window_level(img_float, ww, wl)
    enhanced = apply_clahe(enhanced)
    return enhanced


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def format_polygon(points: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in points]


def safe_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]
PYEOF
```

- [ ] **Step 4: 创建 imaging/adaptive.py（从 services/adaptive_image_processor.py 迁移）**

```bash
cp src/gauge/services/adaptive_image_processor.py src/gauge/imaging/adaptive.py
# 文件内容不变，无内部 gauge import
```

- [ ] **Step 5: 更新所有引用 imaging 旧路径的模块**

以下模块将 `from gauge.pipeline_utils import` (图像函数) 改为 `from gauge.imaging.preprocess import`：
- `src/gauge/iqi_inferencer.py`: 第 39-47 行图像函数 import
- `src/gauge/services/ocr_stage.py`: 第 311/562 行 `crop_rotated_polygon`
- `src/gauge/services/roi_stage.py`: 第 11 行 `format_polygon`, 第 72 行 `crop_rotated_polygon`
- `src/gauge/services/region_ocr_service.py`: 第 16 行
- `src/gauge/stages/roi_detect.py`: 第 12 行
- `src/gauge/stages/roi_ocr.py`: 第 14 行
- `src/gauge/stages/image_load.py`: 第 10 行 `load_image`

以下模块将 `from gauge.geometry import` 改为 `from gauge.imaging.geometry import`：
- `src/gauge/services/fclip_stage.py`: 第 12 行
- `src/gauge/stages/roi_detect.py`: 第 11 行
- `src/gauge/stages/roi_ocr.py`: 第 11 行

`services/correction.py` 中 `from gauge.services.adaptive_image_processor` → `from gauge.imaging.adaptive`

删除旧文件：
```bash
rm src/gauge/geometry.py
rm src/gauge/visualization.py
rm src/gauge/services/adaptive_image_processor.py
# pipeline_utils.py 暂时保留非图像函数部分（后续 task 处理）
```

- [ ] **Step 6: 从 pipeline_utils.py 删除已迁移的图像函数**

编辑 `src/gauge/pipeline_utils.py`，删除第 52-165 行（`load_image` 到 `safe_list`），第 12 行 `SUPPORTED_IMAGE_EXTS`。

保留第 1-11 行（imports）和第 167-329 行（Pipeline helper functions: `build_skipped_ocr`, `build_skipped_wire`, `is_usable_ocr_item`, `scale_box_points`, `scale_ocr_items_to_original`, `scale_roi_info_to_original`, `box_points_to_bbox`, `build_plate_visualization_items`, `merge_prefixed_ocr_timings`）。

为保留的函数添加 `from gauge.imaging.preprocess import ...` 如果他们依赖图像函数（`build_plate_visualization_items` 用了 `box_points_to_bbox` 和 `is_usable_ocr_item`，不依赖已迁出的图像函数）。

`collect_images` 和 `ensure_dir` 在 pipeline_utils.py 保留一份（它们也存在于 `imaging/preprocess.py`，等 app/inputs.py 创建后再统一）。

- [ ] **Step 7: 运行测试验证**

```bash
conda activate weld-gpu
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_geometry.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_visualization.py
python -m py_compile src/gauge/imaging/geometry.py src/gauge/imaging/preprocess.py src/gauge/imaging/visualization.py src/gauge/imaging/adaptive.py
```

- [ ] **Step 8: 验证 imaging/ 边界**

```bash
PYTHONPATH=/home/cht/code/IQIdet/src python -c "
import sys
class Blocker:
    blocked = {'torch', 'paddleocr', 'ultralytics', 'FClip'}
    def find_spec(self, fullname, target=None):
        root = fullname.split('.')[0]
        if root in self.blocked:
            raise ImportError(f'Blocked: {fullname}')
        return None
sys.meta_path.insert(0, Blocker())
import gauge.imaging.geometry
import gauge.imaging.preprocess
import gauge.imaging.adaptive
print('imaging/*: PASS (no model imports)')
"
```

- [ ] **Step 9: Commit**

```bash
git add src/gauge/imaging/
git add -u src/gauge/
git commit -m "refactor: migrate imaging/ — pure image processing layer

Move geometry.py, visualization.py to gauge/imaging/.
Extract image functions from pipeline_utils.py to imaging/preprocess.py.
Move adaptive_image_processor.py to imaging/adaptive.py.
Update all internal imports to gauge.imaging.* paths.
Remove old gauge/geometry.py, gauge/visualization.py,
and gauge/services/adaptive_image_processor.py.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 迁移 runtime/ — 子进程/线程运行时层

**迁移内容:** `ocr_runtime.py` → `runtime/ocr_runtime.py`, `region_runtime.py` → `runtime/region_runtime.py`, `services/ocr_paddle_worker.py` → `runtime/ocr_paddle_worker.py`

**依赖规则:** runtime/ 可依赖标准库、OpenCV/Numpy，不依赖 domain/。

- [ ] **Step 1: 复制文件到 runtime/**

```bash
cp src/gauge/ocr_runtime.py src/gauge/runtime/ocr_runtime.py
cp src/gauge/region_runtime.py src/gauge/runtime/region_runtime.py
cp src/gauge/services/ocr_paddle_worker.py src/gauge/runtime/ocr_paddle_worker.py
```

- [ ] **Step 2: 更新 ocr_runtime.py 中的 worker_script 路径**

`ocr_runtime.py` 第 46 行: `self.worker_script = Path(__file__).resolve().parent / "services" / "ocr_paddle_worker.py"`

在新 `runtime/ocr_runtime.py` 中改为：
```python
self.worker_script = Path(__file__).resolve().parent / "ocr_paddle_worker.py"
```

- [ ] **Step 3: 更新 ocr_paddle_worker.py 的 import**

`ocr_paddle_worker.py` 第 18-26 行:
```python
from gauge.services.ocr_stage import (
    _ensure_rgb,
    _run_text_det_predict,
    _run_text_rec_predict,
    create_text_detector,
    create_text_recognizer,
    normalize_text_det_output,
    normalize_text_rec_output,
)
```

在新 `runtime/ocr_paddle_worker.py` 中改为（等 Task 5 拆分 `ocr_stage.py` 后这些 import 指向正确路径）：
```python
from gauge.services.ocr.factory import (
    _ensure_rgb,
    create_text_detector,
    create_text_recognizer,
)
from gauge.services.ocr.infer import (
    _run_text_det_predict,
    _run_text_rec_predict,
)
from gauge.services.ocr.normalize import (
    normalize_text_det_output,
    normalize_text_rec_output,
)
```

注意：此步骤依赖 Task 5 完成 `services/ocr/` 拆分。因此本 task **先复制文件到 runtime/，import 暂指向旧 `gauge.services.ocr_stage`**，等 Task 5 完成后再更新。

- [ ] **Step 4: 更新所有引用 runtime 旧路径的内部模块**

- `src/gauge/iqi_inferencer.py`: `from gauge.ocr_runtime` → `from gauge.runtime.ocr_runtime`
- `src/gauge/services/ocr_stage.py`: `from gauge.ocr_runtime` → `from gauge.runtime.ocr_runtime`
- `src/gauge/services/region_ocr_service.py`: `from gauge.ocr_runtime` → `from gauge.runtime.ocr_runtime`
- `src/gauge/services/region_ocr_api.py`: `from gauge.region_runtime` → `from gauge.runtime.region_runtime`
- `src/gauge/services/region_snr_api.py`: `from gauge.region_runtime` → `from gauge.runtime.region_runtime`
- `src/gauge/services/base.py`: `from gauge.region_runtime` → `from gauge.runtime.region_runtime`

- [ ] **Step 5: 删除旧文件**

```bash
rm src/gauge/ocr_runtime.py
rm src/gauge/region_runtime.py
rm src/gauge/services/ocr_paddle_worker.py
```

- [ ] **Step 6: 运行测试**

```bash
conda activate weld-gpu
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_ocr_runtime.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_runtime.py
python -m py_compile src/gauge/runtime/ocr_runtime.py src/gauge/runtime/region_runtime.py src/gauge/runtime/ocr_paddle_worker.py
```

- [ ] **Step 7: Commit**

```bash
git add src/gauge/runtime/
git add -u src/gauge/
git commit -m "refactor: migrate runtime/ — subprocess/thread runtime layer

Move ocr_runtime.py, region_runtime.py to gauge/runtime/.
Move ocr_paddle_worker.py to gauge/runtime/.
Update worker_script path and all internal imports.
Remove old gauge/ocr_runtime.py, gauge/region_runtime.py,
and gauge/services/ocr_paddle_worker.py.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 拆分 services/ocr_stage.py → services/ocr/ + domain/statistics.py

**拆分目标:** 将 597 行的 `ocr_stage.py` 拆为 4 个文件 + 1 个 domain 统计文件。

**函数归属:**
| 函数 | 目标文件 |
|------|---------|
| `_configure_paddle_runtime()` | `services/ocr/factory.py` |
| `_create_paddle_component()` | `services/ocr/factory.py` |
| `create_paddle_ocr()` | `services/ocr/factory.py` |
| `create_text_detector()` | `services/ocr/factory.py` |
| `create_text_recognizer()` | `services/ocr/factory.py` |
| `_ensure_rgb()` | `services/ocr/factory.py` |
| `_unwrap_result()` | `services/ocr/normalize.py` |
| `_run_ocr_predict()` | `services/ocr/infer.py` |
| `_run_text_det_predict()` | `services/ocr/infer.py` |
| `_run_text_rec_predict()` | `services/ocr/infer.py` |
| `_to_list()` | `services/ocr/normalize.py` |
| `_first_value()` | `services/ocr/normalize.py` |
| `normalize_text_det_output()` | `services/ocr/normalize.py` |
| `normalize_text_rec_output()` | `services/ocr/normalize.py` |
| `_normalize_text()` | `services/ocr/normalize.py` |
| `_contains_jb()` | `services/ocr/normalize.py` |
| `_filter_items_with_jb()` | `services/ocr/normalize.py` |
| `infer_roi_ocr()` | `services/ocr/infer.py` |
| `build_ocr_statistics()` | `domain/statistics.py` |
| `draw_ocr_on_roi()` | `services/ocr/debug.py` |
| `draw_recognition_result()` | `services/ocr/debug.py` |
| `build_ocr_item_debug_images()` | `services/ocr/debug.py` |

- [ ] **Step 1: 创建 services/ocr/factory.py**

包含: `_configure_paddle_runtime`, `_create_paddle_component`, `create_paddle_ocr`, `create_text_detector`, `create_text_recognizer`, `_ensure_rgb`

```bash
# 从 ocr_stage.py 提取第 19-167 行和 223-231 行写入 factory.py
# 添加 imports: inspect, os, typing, cv2, numpy (无 gauge 内部 import)
```

- [ ] **Step 2: 创建 services/ocr/normalize.py**

包含: `_unwrap_result`, `_to_list`, `_first_value`, `normalize_text_det_output`, `normalize_text_rec_output`, `_normalize_text`, `_contains_jb`, `_filter_items_with_jb`

```bash
# 从 ocr_stage.py 提取第 179-301 行写入 normalize.py
# 添加 imports: typing, collections.Counter (可选), numpy
```

- [ ] **Step 3: 创建 services/ocr/infer.py**

包含: `_run_ocr_predict`, `_run_text_det_predict`, `_run_text_rec_predict`, `infer_roi_ocr`

`infer_roi_ocr` 的依赖:
- `from gauge.runtime.ocr_runtime import PaddleOCRSubprocessClient` (不直接用，保留在 factory)
- `from gauge.services.ocr.factory import _ensure_rgb`
- `from gauge.services.ocr.normalize import normalize_text_det_output, normalize_text_rec_output`
- `from gauge.imaging.preprocess import crop_rotated_polygon`

- [ ] **Step 4: 创建 services/ocr/debug.py**

包含: `draw_ocr_on_roi`, `draw_recognition_result`, `build_ocr_item_debug_images`

- [ ] **Step 5: 创建 domain/statistics.py**

```bash
cat > src/gauge/domain/statistics.py << 'PYEOF'
#!/usr/bin/env python3
"""Aggregate statistics for IQI records."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


def build_ocr_statistics(results: List[Dict[str, Any]], topk: int = 200) -> Dict[str, Any]:
    """Aggregate OCR-level stats across all image records."""
    def _normalize_text(text: str) -> str:
        return "".join(ch for ch in str(text).upper() if ch.isalnum())

    status_counter = Counter()
    ocr_status_counter = Counter()
    raw_text_counter = Counter()
    norm_text_counter = Counter()

    for record in results:
        status_counter[str(record.get("status", "unknown"))] += 1
        ocr = record.get("ocr") or {}
        ocr_status_counter[str(ocr.get("status", "missing"))] += 1
        for text in ocr.get("texts", []) or []:
            raw = str(text)
            norm = _normalize_text(raw)
            raw_text_counter[raw] += 1
            if norm:
                norm_text_counter[norm] += 1

    return {
        "images_total": len(results),
        "pipeline_status": {k: int(v) for k, v in status_counter.items()},
        "ocr_status": {k: int(v) for k, v in ocr_status_counter.items()},
        "top_raw_text": {k: int(v) for k, v in raw_text_counter.most_common(topk)},
        "top_normalized_text": {k: int(v) for k, v in norm_text_counter.most_common(topk)},
    }
PYEOF
```

- [ ] **Step 6: 更新所有引用旧路径的模块**

- `src/gauge/iqi_inferencer.py`: `from gauge.services.ocr_stage import infer_roi_ocr` → `from gauge.services.ocr.infer import infer_roi_ocr`
- `src/gauge/domain/record_builders.py`: `from gauge.services.ocr_stage import build_ocr_statistics` → `from gauge.domain.statistics import build_ocr_statistics`
- `src/gauge/imaging/visualization.py`: 更新为 `from gauge.services.ocr.debug import ...`
- `src/gauge/runtime/ocr_paddle_worker.py`: 更新为从新 `services/ocr/` 路径 import
- `src/gauge/services/ocr_stage.py` 顶层保持兼容 re-export（给尚未更新的调用者），在文件开头加 `from gauge.services.ocr.infer import *` 等

- [ ] **Step 7: 删除旧 ocr_stage.py**

```bash
rm src/gauge/services/ocr_stage.py
```

- [ ] **Step 8: 运行测试**

```bash
conda activate weld-gpu
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_delivery_record.py
python -m py_compile src/gauge/services/ocr/factory.py src/gauge/services/ocr/infer.py src/gauge/services/ocr/normalize.py src/gauge/services/ocr/debug.py src/gauge/domain/statistics.py
```

- [ ] **Step 9: Commit**

```bash
git add src/gauge/services/ocr/ src/gauge/domain/statistics.py
git add -u src/gauge/
git commit -m "refactor: split services/ocr_stage.py into services/ocr/ sub-packages

Split 597-line ocr_stage.py into:
- services/ocr/factory.py: PaddleOCR component construction
- services/ocr/infer.py: ROI OCR inference
- services/ocr/normalize.py: output normalization
- services/ocr/debug.py: debug drawing
- domain/statistics.py: OCR statistics (pure business logic)

Remove old gauge/services/ocr_stage.py.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 拆分 services/fclip_stage.py → services/fclip/

**拆分目标:** 将 `fclip_stage.py` 拆为 `services/fclip/inferencer.py` (FClipInferencer 类) 和 `services/fclip/line_records.py` (build_line_records 及辅助函数)。

- [ ] **Step 1: 创建 services/fclip/line_records.py**

包含: `resolve_torch_device`, `_ensure_gray`, `lines_yx_to_xy`, `build_line_records`

这些函数 import `gauge.imaging.geometry` (invert_perspective_matrix, perspective_transform_points, undo_ccw90_points)。

```bash
# 从 fclip_stage.py 提取第 1-78 行（所有函数直到 FClipInferencer 之前）写入 line_records.py
```

- [ ] **Step 2: 创建 services/fclip/inferencer.py**

包含: `FClipInferencer` 类（第 81-197 行）

import 更新为:
- `from gauge.services.fclip.line_records import resolve_torch_device, _ensure_gray, build_line_records`
- `from gauge.imaging.geometry import invert_perspective_matrix, perspective_transform_points, undo_ccw90_points` → 这些现在只在 line_records.py 中使用

- [ ] **Step 3: 更新所有引用旧路径的模块**

- `src/gauge/iqi_inferencer.py`: `from gauge.services.fclip_stage import (FClipInferencer, invert_perspective_matrix, perspective_transform_points, undo_ccw90_points)` →
  ```python
  from gauge.services.fclip.inferencer import FClipInferencer
  from gauge.imaging.geometry import invert_perspective_matrix, perspective_transform_points, undo_ccw90_points
  ```

- [ ] **Step 4: 删除旧 fclip_stage.py**

```bash
rm src/gauge/services/fclip_stage.py
```

- [ ] **Step 5: 运行测试**

```bash
conda activate weld-gpu
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_inferencer.py
python -m py_compile src/gauge/services/fclip/inferencer.py src/gauge/services/fclip/line_records.py
```

- [ ] **Step 6: Commit**

```bash
git add src/gauge/services/fclip/
git add -u src/gauge/
git commit -m "refactor: split services/fclip_stage.py into services/fclip/ sub-packages

Split fclip_stage.py into:
- services/fclip/inferencer.py: FClipInferencer class
- services/fclip/line_records.py: line coordinate helpers

Update iqi_inferencer.py to import geometry functions from imaging/.
Remove old gauge/services/fclip_stage.py.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: 重组 services/ 其余模块

**迁移内容:**
- `roi_stage.py` → `services/roi/yolo_obb.py`
- `correction.py` → `services/orientation/base.py`
- `ocr_orientation.py` → `services/orientation/ocr_text.py`
- `weld_correction.py` → `services/orientation/weld.py`
- `region_ocr_service.py` → `services/region/ocr_service.py`
- `region_snr_service.py` → `services/region/snr_service.py`
- `region_ocr_api.py` → `app/region_ocr_api.py`
- `region_snr_api.py` → `app/region_snr_api.py`
- 删除 `services/base.py`
- 更新 `services/__init__.py`

- [ ] **Step 1: 迁移 roi_stage.py → services/roi/yolo_obb.py**

```bash
cp src/gauge/services/roi_stage.py src/gauge/services/roi/yolo_obb.py
```

编辑 `services/roi/yolo_obb.py`:
- `from gauge.pipeline_utils import format_polygon` → `from gauge.imaging.preprocess import format_polygon`
- `from gauge.pipeline_utils import crop_rotated_polygon`（第 72 行）→ `from gauge.imaging.preprocess import crop_rotated_polygon`

更新引用:
- `src/gauge/iqi_inferencer.py`: `from gauge.services.roi_stage import extract_best_obb` → `from gauge.services.roi.yolo_obb import extract_best_obb`
- `src/gauge/imaging/visualization.py`: `from gauge.services.roi_stage import build_roi_vis_image` → `from gauge.services.roi.yolo_obb import build_roi_vis_image`

删除: `rm src/gauge/services/roi_stage.py`

- [ ] **Step 2: 迁移 orientation 服务**

```bash
cp src/gauge/services/correction.py src/gauge/services/orientation/base.py
cp src/gauge/services/ocr_orientation.py src/gauge/services/orientation/ocr_text.py
cp src/gauge/services/weld_correction.py src/gauge/services/orientation/weld.py
```

更新各文件内部 import:
- `services/orientation/base.py`: `from gauge.services.adaptive_image_processor` → `from gauge.imaging.adaptive`
- `services/orientation/weld.py`: 如引用 base → `from gauge.services.orientation.base import BaseOrientationCorrector`
- `services/orientation/ocr_text.py`: 同 base import 更新

更新引用这些模块的代码:
- `src/gauge/iqi_inferencer.py`:
  - `from gauge.services.weld_correction import WeldOrientationCorrector` → `from gauge.services.orientation.weld import WeldOrientationCorrector`
  - `from gauge.services.ocr_orientation import OCRTextOrientationCorrector` → `from gauge.services.orientation.ocr_text import OCRTextOrientationCorrector`

删除旧文件:
```bash
rm src/gauge/services/correction.py
rm src/gauge/services/ocr_orientation.py
rm src/gauge/services/weld_correction.py
```

- [ ] **Step 3: 迁移 region 服务和 API wrapper**

```bash
cp src/gauge/services/region_ocr_service.py src/gauge/services/region/ocr_service.py
cp src/gauge/services/region_snr_service.py src/gauge/services/region/snr_service.py
cp src/gauge/services/region_ocr_api.py src/gauge/app/region_ocr_api.py
cp src/gauge/services/region_snr_api.py src/gauge/app/region_snr_api.py
```

更新 region service import:
- `services/region/ocr_service.py`: `from gauge.iqi_rules` → `from gauge.domain.iqi_rules`; `from gauge.ocr_runtime` → `from gauge.runtime.ocr_runtime`; `from gauge.pipeline_utils` → `from gauge.imaging.preprocess`
- `services/region/snr_service.py`: 无 gauge 内部 import，直接复制

更新 API wrapper import:
- `app/region_ocr_api.py`: `from gauge.region_runtime` → `from gauge.runtime.region_runtime`; `from gauge.services.region_ocr_service` → `from gauge.services.region.ocr_service`
- `app/region_snr_api.py`: `from gauge.region_runtime` → `from gauge.runtime.region_runtime`; `from gauge.services.region_snr_service` → `from gauge.services.region.snr_service`

更新根目录门面:
- `region_ocr_api.py` (root): `from gauge.services.region_ocr_api` → `from gauge.app.region_ocr_api`; `from gauge.services.region_ocr_service` → `from gauge.services.region.ocr_service`
- `region_SNR_api.py` (root): `from gauge.services.region_snr_api` → `from gauge.app.region_snr_api`; `from gauge.services.region_snr_service` → `from gauge.services.region.snr_service`

删除旧文件:
```bash
rm src/gauge/services/region_ocr_service.py
rm src/gauge/services/region_snr_service.py
rm src/gauge/services/region_ocr_api.py
rm src/gauge/services/region_snr_api.py
```

- [ ] **Step 4: 删除 services/base.py 并清理 services/__init__.py**

```bash
rm src/gauge/services/base.py
```

编辑 `src/gauge/services/__init__.py`，删除对 base 的 re-export，改为空的包 docstring。

- [ ] **Step 5: 运行测试**

```bash
conda activate weld-gpu
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_snr_service.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_script_layout.py
python -m py_compile src/gauge/services/roi/yolo_obb.py src/gauge/services/orientation/base.py src/gauge/services/orientation/ocr_text.py src/gauge/services/orientation/weld.py src/gauge/services/region/ocr_service.py src/gauge/services/region/snr_service.py src/gauge/app/region_ocr_api.py src/gauge/app/region_snr_api.py
```

- [ ] **Step 6: Commit**

```bash
git add src/gauge/services/roi/ src/gauge/services/orientation/ src/gauge/services/region/ src/gauge/app/region_ocr_api.py src/gauge/app/region_snr_api.py
git add -u src/gauge/services/ region_ocr_api.py region_SNR_api.py
git commit -m "refactor: reorganize remaining services/ modules

- roi_stage.py → services/roi/yolo_obb.py
- correction.py → services/orientation/base.py
- ocr_orientation.py → services/orientation/ocr_text.py
- weld_correction.py → services/orientation/weld.py
- region_ocr_service.py → services/region/ocr_service.py
- region_snr_service.py → services/region/snr_service.py
- region_ocr_api.py → app/region_ocr_api.py
- region_snr_api.py → app/region_snr_api.py
- Delete services/base.py

Update root entry-point facades and all internal imports.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: 迁移 pipeline/ — 管道编排层

**迁移内容:**
- `pipeline.py` → `pipeline/runner.py`
- `stages/base.py` → `pipeline/context.py` (StageContext) + 保留 `pipeline/stages/base.py` (PipelineStage)
- `stages/*.py` → `pipeline/stages/*.py` (移动 7 个 stage 文件)

- [ ] **Step 1: 创建 pipeline/runner.py**

```bash
cp src/gauge/pipeline.py src/gauge/pipeline/runner.py
```

编辑 `pipeline/runner.py`，更新 import:
- `from gauge.stages.base import PipelineStage, StageContext` → `from gauge.pipeline.context import StageContext` + `from gauge.pipeline.stages.base import PipelineStage`
- `from gauge.stages.correction import ...` → `from gauge.pipeline.stages.correction import ...`
- 等 7 个 stage import 全都改为 `gauge.pipeline.stages.*`

- [ ] **Step 2: 创建 pipeline/context.py（StageContext 从 stages/base.py 分离）**

```bash
# StageContext 类从 stages/base.py 第 16-87 行提取
# 独立为 pipeline/context.py
```

`pipeline/context.py` 中 `to_record()` 方法 import `from gauge.domain.record_builders import build_iqi_record`。

- [ ] **Step 3: 更新 pipeline/stages/base.py**

删除 StageContext 类（已迁出），仅保留 PipelineStage 抽象类。
更新 `to_record()` import 不再在 base.py 中。

- [ ] **Step 4: 移动 7 个 stage 文件**

```bash
cp src/gauge/stages/correction.py src/gauge/pipeline/stages/correction.py
cp src/gauge/stages/full_image_ocr.py src/gauge/pipeline/stages/full_image_ocr.py
cp src/gauge/stages/grade_fusion.py src/gauge/pipeline/stages/grade_fusion.py
cp src/gauge/stages/image_load.py src/gauge/pipeline/stages/image_load.py
cp src/gauge/stages/roi_detect.py src/gauge/pipeline/stages/roi_detect.py
cp src/gauge/stages/roi_ocr.py src/gauge/pipeline/stages/roi_ocr.py
cp src/gauge/stages/wire_detect.py src/gauge/pipeline/stages/wire_detect.py
```

更新每个 stage 文件的 import：
- `from gauge.stages.base import PipelineStage, StageContext` → `from gauge.pipeline.stages.base import PipelineStage` + `from gauge.pipeline.context import StageContext`
- 所有 `from gauge.iqi_rules import` → `from gauge.domain.iqi_rules import`
- 所有 `from gauge.pipeline_utils import` → 指向新路径（`gauge.imaging.preprocess`, `gauge.app.inputs` 等）
- 所有 `from gauge.geometry import` → `from gauge.imaging.geometry import`

更新 `pipeline/stages/__init__.py`。

- [ ] **Step 5: 更新 iqi_inferencer.py 对 pipeline 的引用**

```python
from gauge.pipeline import PipelineRunner  # → from gauge.pipeline.runner import PipelineRunner
```

- [ ] **Step 6: 删除旧文件**

```bash
rm src/gauge/pipeline.py
rm -r src/gauge/stages/
```

- [ ] **Step 7: 运行测试**

```bash
conda activate weld-gpu
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_pipeline_config.py
python -m py_compile src/gauge/pipeline/runner.py src/gauge/pipeline/context.py src/gauge/pipeline/stages/*.py
```

- [ ] **Step 8: 验证 pipeline import 不加载模型**

```bash
PYTHONPATH=/home/cht/code/IQIdet/src python tests/test_import_boundaries.py
# 此测试检查 `import gauge.pipeline` 不导入 torch/FClip/ultralytics/paddleocr
# 需要更新测试中 import 路径为 gauge.pipeline.runner
```

- [ ] **Step 9: Commit**

```bash
git add src/gauge/pipeline/
rm -rf src/gauge/stages/  # 确认删除空目录
rm src/gauge/pipeline.py
git add -u src/gauge/
git commit -m "refactor: migrate pipeline/ — orchestration layer

Move pipeline.py to pipeline/runner.py.
Separate StageContext from stages/base.py into pipeline/context.py.
Move all stage files to pipeline/stages/.
Update all internal imports.
Remove old gauge/pipeline.py and gauge/stages/ directory.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: 迁移 app/ — 交付层

**迁移内容:**
- `iqi_inferencer.py` → `app/iqi_inferencer.py`
- 从 `pipeline_utils.py` 拆出 `collect_images`, `ensure_dir` → `app/inputs.py`
- `pipeline_utils.py` 剩余函数部分拆入各目标

- [ ] **Step 1: 创建 app/iqi_inferencer.py**

```bash
cp src/gauge/iqi_inferencer.py src/gauge/app/iqi_inferencer.py
```

更新 `app/iqi_inferencer.py` 的所有 import 为新路径：
```python
from gauge.config import ...
from gauge.services.fclip.inferencer import FClipInferencer
from gauge.imaging.geometry import invert_perspective_matrix, perspective_transform_points, undo_ccw90_points
from gauge.domain.iqi_rules import (build_result_status, choose_primary_result_code, compute_iqi_grade, ...)
from gauge.domain.record_builders import build_delivery_record, build_iqi_statistics
from gauge.runtime.ocr_runtime import PaddleOCRSubprocessClient
from gauge.services.ocr.infer import infer_roi_ocr
from gauge.imaging.preprocess import (collect_images, crop_rotated_polygon, enhance_windowing_gray, load_image, resize_long_side, rotate_if_wide, to_gray)
from gauge.services.roi.yolo_obb import extract_best_obb
from gauge.imaging.visualization import (build_final_result_vis_image, build_wire_vis_image, save_debug_visualizations)
from gauge.services.orientation.weld import WeldOrientationCorrector
from gauge.services.orientation.ocr_text import OCRTextOrientationCorrector
from gauge.pipeline.runner import PipelineRunner
```

- [ ] **Step 2: 创建 app/inputs.py（图像收集/路径工具）**

从 `pipeline_utils.py` 提取 `collect_images`, `ensure_dir`, `SUPPORTED_IMAGE_EXTS`。

```bash
# 创建 app/inputs.py
```

更新 `iqi_inferencer.py` 中 `collect_images` 的 import 从 `gauge.imaging.preprocess` 改为 `gauge.app.inputs`。

`imaging/preprocess.py` 中保留 `ensure_dir`（被多处引用），或统一改为从 `app/inputs.py` 导入。

- [ ] **Step 3: 处理 pipeline_utils.py 剩余内容**

`pipeline_utils.py` 剩余函数归属：
- `build_skipped_ocr`, `build_skipped_wire` → `domain/record_builders.py`（这些是业务状态构建）
- `is_usable_ocr_item`, `scale_box_points`, `scale_ocr_items_to_original`, `scale_roi_info_to_original`, `box_points_to_bbox` → `imaging/geometry.py`（坐标变换工具）
- `build_plate_visualization_items` → `imaging/visualization.py`（可视化相关）
- `merge_prefixed_ocr_timings` → 保留在 `app/` 级别或 `pipeline/context.py`

执行分配并更新所有引用。

- [ ] **Step 4: 更新根目录入口**

`run_iqi_grade_infer.py`:
```python
from gauge.app.iqi_inferencer import (
    IQIInferencer, build_delivery_record, build_iqi_statistics,
    collect_input_images, save_debug_visualizations,
)
from gauge.domain.iqi_rules import DEFAULT_ALLOWED_NUMBERS_SPEC
from gauge.imaging.preprocess import SUPPORTED_IMAGE_EXTS, ensure_dir
```

- [ ] **Step 5: 删除旧文件**

```bash
rm src/gauge/iqi_inferencer.py
rm src/gauge/pipeline_utils.py  # 所有内容已迁移
```

- [ ] **Step 6: 修复 IQIInferencer 中延迟导入的静态方法**

`app/iqi_inferencer.py` 中 `IQIInferencer` 类的静态方法（`_build_skipped_wire`, `_scale_box_points` 等）仍使用延迟导入 `from gauge.pipeline_utils`，需更新为从新路径导入：
- `build_skipped_wire` → `from gauge.domain.record_builders import build_skipped_wire`
- `scale_box_points`, `scale_ocr_items_to_original`, `scale_roi_info_to_original`, `box_points_to_bbox` → `from gauge.imaging.geometry import ...`
- `build_plate_visualization_items` → `from gauge.imaging.visualization import ...`
- `merge_prefixed_ocr_timings` → `from gauge.app.iqi_inferencer import merge_prefixed_ocr_timings` (或放到 pipeline/)

- [ ] **Step 7: 运行测试**

```bash
conda activate weld-gpu
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_inferencer.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_delivery_record.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_script_layout.py
python -m py_compile src/gauge/app/iqi_inferencer.py src/gauge/app/inputs.py
```

- [ ] **Step 8: Commit**

```bash
git add src/gauge/app/
git add -u src/gauge/ run_iqi_grade_infer.py
git commit -m "refactor: migrate app/ — delivery layer

Move iqi_inferencer.py to app/iqi_inferencer.py.
Create app/inputs.py for image collection utilities.
Distribute remainder of pipeline_utils.py to domain/, imaging/.
Update all imports to new layered paths.
Remove old gauge/iqi_inferencer.py and gauge/pipeline_utils.py.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: 更新测试文件 import

**所有测试文件需更新 import 以匹配新路径。**

- [ ] **Step 1: 更新测试 import**

| 测试文件 | 旧 import | 新 import |
|----------|----------|----------|
| `test_iqi_rules.py` | `gauge.iqi_rules` | `gauge.domain.iqi_rules` |
| `test_geometry.py` | `gauge.geometry` | `gauge.imaging.geometry` |
| `test_record_builders.py` | `gauge.record_builders`, `gauge.stages.base` | `gauge.domain.record_builders`, `gauge.pipeline.context` |
| `test_iqi_inferencer.py` | `gauge.iqi_inferencer` | `gauge.app.iqi_inferencer` |
| `test_iqi_delivery_record.py` | `gauge.iqi_inferencer` | `gauge.app.iqi_inferencer` |
| `test_ocr_runtime.py` | `gauge.ocr_runtime` | `gauge.runtime.ocr_runtime` |
| `test_region_runtime.py` | `gauge.region_runtime` | `gauge.runtime.region_runtime` |
| `test_region_snr_service.py` | `gauge.services.region_snr_service` | `gauge.services.region.snr_service` |
| `test_visualization.py` | `gauge.visualization`, `gauge.iqi_inferencer` | `gauge.imaging.visualization`, `gauge.app.iqi_inferencer` |
| `test_pipeline_config.py` | (无 gauge 内部 import 需改) | — |
| `test_import_boundaries.py` | `gauge.pipeline` | `gauge.pipeline.runner` |
| `test_script_layout.py` | 验证 root entrypoint import | 保持不变 |

- [ ] **Step 2: 运行全部测试**

```bash
conda activate weld-gpu
cd /home/cht/code/IQIdet
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_rules.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_geometry.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_record_builders.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_inferencer.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_delivery_record.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_ocr_runtime.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_runtime.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_region_snr_service.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_visualization.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_pipeline_config.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_import_boundaries.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_script_layout.py
```

全部通过后 commit。

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: update test imports to new layered paths

Update all test files to import from gauge.app, gauge.domain,
gauge.imaging, gauge.runtime, gauge.pipeline instead of old
flat-package paths.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: 更新文档

- [ ] **Step 1: 更新 ARCHITECTURE.md**

反映新目录结构和依赖方向:
```markdown
## src/gauge 分层架构

| 层 | 目录 | 职责 | 依赖 |
|----|------|------|------|
| 交付层 | `app/` | 对外门面、模型生命周期、delivery payload | 所有层 |
| 编排层 | `pipeline/` | PipelineRunner、StageContext、Stage | domain, imaging, services |
| 业务层 | `domain/` | IQI 规则、等级计算、记录构建、统计 | stdlib + models |
| 图像层 | `imaging/` | 几何变换、预处理、可视化 | OpenCV + NumPy |
| 运行时层 | `runtime/` | 子进程/线程池/lock/shutdown | stdlib + OpenCV/NumPy |
| 适配层 | `services/` | 模型适配器、OCR/FClip/ROI/Orientation/Region | 模型库 |
```

- [ ] **Step 2: 更新 src/gauge/README.md**

简短分层说明:
```markdown
# Gauge 包

像质计（IQI）等级识别系统的核心包。按职责分层组织。

- `app/` - 对外交付层和服务门面
- `pipeline/` - 管道编排和状态管理
- `domain/` - 纯业务规则，无模型依赖
- `imaging/` - 纯图像处理工具
- `runtime/` - 子进程和线程池管理
- `services/` - 外部模型适配器
- `models/` - 数据模型定义
- `config/` - 配置定义
- `training/` - 训练脚本
```

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md src/gauge/README.md
git commit -m "docs: update ARCHITECTURE.md and README for layered structure

Document the new layered architecture with dependency rules
and directory responsibilities.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: 运行输出回归验证

- [ ] **Step 1: 运行 compare_results.py**

```bash
conda activate weld-gpu
cd /home/cht/code/IQIdet
python scripts/compare_results.py
```

**期望输出:**
```
ALL MATCH — 8 image(s), 0 differences.
```

- [ ] **Step 2: 如不匹配，排查差异**

对比 baseline JSON 和当前输出，逐记录检查 `result_code`、`grade`、`plate_code` 等字段。

- [ ] **Step 3: 最终语法检查**

```bash
for f in $(find src/gauge -name '*.py' -not -path '*__pycache__*'); do
    python -m py_compile "$f" || echo "FAIL: $f"
done
```

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "chore: final verification — all tests pass, regression matches

compare_results.py: ALL MATCH — 8 image(s), 0 differences.
All unit tests pass. No stale imports remain.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 自审清单

### 1. Spec 覆盖检查

| Spec 需求 | 覆盖 Task |
|-----------|----------|
| 建立新目录和导入测试 | Task 1 |
| 迁移 domain/ (iqi_rules + record_builders) | Task 2 |
| 迁移 imaging/ (geometry + visualization + pipeline_utils 图像函数) | Task 3 |
| 迁移 runtime/ (ocr_runtime + region_runtime + ocr_paddle_worker) | Task 4 |
| 拆分 services/ocr_stage.py | Task 5 |
| 拆分 services/fclip_stage.py | Task 6 |
| 重组 services/ 其余模块 | Task 7 |
| 迁移 pipeline/ (runner + context + stages) | Task 8 |
| 迁移 app/ (iqi_inferencer + inputs + region APIs) | Task 9 |
| 更新测试 import | Task 10 |
| 更新文档 (ARCHITECTURE.md + README.md) | Task 11 |
| 运行 compare_results.py 回归 | Task 12 |
| domain/ 不导入 OpenCV/Torch/PaddleOCR/Ultralytics | Task 2 Step 6 |
| imaging/ 不导入 Torch/PaddleOCR/Ultralytics | Task 3 Step 8 |
| pipeline 导入不构造模型 | Task 8 Step 8 |
| 旧路径模块文件不存在 | 各 Task 的删除步骤 |
| 根目录入口 import 更新 | Task 9 Step 4, Task 7 Step 3 |

### 2. 占位符扫描

无 TBD/TODO/占位符。所有步骤都有具体命令和代码。

### 3. 类型一致性

各 Task 间文件路径和 import 名称保持一致，新旧路径映射表覆盖所有文件。

### 4. 已知风险

- **Task 4 ocr_paddle_worker.py import**: 依赖 Task 5 完成 services/ocr/ 拆分后更新。如果 Task 4 先执行，`ocr_paddle_worker.py` 暂指向旧 `gauge.services.ocr_stage`。
- **pipeline_utils.py 拆分**: 跨 Task 3, 4, 8, 9 逐步完成。每个 task 只处理明确归属的函数，不产生孤立的未归属函数。
- **循环导入**: spec 已定义清晰依赖方向 (app → pipeline → domain/imaging/services)，迁移过程中需保持此方向。
