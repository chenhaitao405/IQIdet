# IQIdet Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor IQIdet core pipeline into staged architecture with Pydantic models, centralized config, structured logging, and deduplicated base classes — without breaking external interfaces.

**Architecture:** 4 sequential phases. Phase 1 creates all new infrastructure files (zero behavioral change). Phase 2 wires Pydantic models and error handling into existing code. Phase 3 splits `infer_image_path` into 7 Stage classes + PipelineRunner. Phase 4 merges orientation correctors. Each phase ends with `py_compile` + `pytest` verification.

**Tech Stack:** Python 3.10+, Pydantic v2, setuptools, stdlib logging, existing torch/ultralytics/paddleocr.

**Spec:** `docs/superpowers/specs/2026-05-30-iqidet-refactor-design.md`

---

## Phase 0: Git Worktree + Branch

### Task 0: Create isolated branch

**Files:**
- Create branch: `refactor/pipeline-stages`

- [ ] **Step 1: Create feature branch**

```bash
git checkout -b refactor/pipeline-stages
```

- [ ] **Step 2: Verify clean working tree**

```bash
git status
```
Expected: clean tree on branch `refactor/pipeline-stages`

---

## Phase 1: Infrastructure (pyproject.toml, models, config, logging, exceptions, services base)

### Task 1.1: Create pyproject.toml

**Files:**
- Create: `/home/cht/code/IQIdet/pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

Create `/home/cht/code/IQIdet/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "IQIdet"
version = "0.1.0"
description = "IQI (Image Quality Indicator) grade inference for weld radiographs"
requires-python = ">=3.10"
dependencies = [
    "torch",
    "numpy",
    "scipy",
    "matplotlib",
    "scikit-image",
    "opencv-python",
    "PyYAML",
    "docopt",
    "tqdm",
    "ultralytics",
    "optuna",
    "paddleocr",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "dvc",
]

[project.scripts]
iqi-grade-infer = "gauge.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Install in dev mode and verify import**

```bash
cd /home/cht/code/IQIdet && pip install -e .
```
Expected: successful install

- [ ] **Step 3: Verify import works without sys.path hack**

```bash
python -c "from gauge.iqi_rules import normalize_text; print(normalize_text('10FEJB'))"
```
Expected: `10FEJB`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml && git commit -m "feat: add pyproject.toml with setuptools build config"
```

### Task 1.2: Remove sys.path hacks from entry points

**Files:**
- Modify: `/home/cht/code/IQIdet/run_iqi_grade_infer.py:17-21`
- Modify: `/home/cht/code/IQIdet/region_ocr_api.py:4-10`
- Modify: `/home/cht/code/IQIdet/region_SNR_api.py:4-10`

- [ ] **Step 1: Remove sys.path hack from run_iqi_grade_infer.py**

In `/home/cht/code/IQIdet/run_iqi_grade_infer.py`, delete lines 17-21:

```python
# DELETE these lines:
REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

And remove the now-unused imports `sys` (if only used for path) and `Path` (if only used for path hack). Keep `Path` if used elsewhere in the file.

- [ ] **Step 2: Remove sys.path hack from region_ocr_api.py**

In `/home/cht/code/IQIdet/region_ocr_api.py`, delete lines 4-10:

```python
# DELETE these lines:
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

Keep the import lines that are still needed (`from gauge.region_ocr_api import ...` etc.).

- [ ] **Step 3: Remove sys.path hack from region_SNR_api.py**

In `/home/cht/code/IQIdet/region_SNR_api.py`, delete lines 4-10:

```python
# DELETE these lines:
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

- [ ] **Step 4: Remove sys.path hack from ocr_paddle_worker.py**

In `/home/cht/code/IQIdet/src/gauge/ocr_paddle_worker.py`, delete lines 19-21:

```python
# DELETE these lines:
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

- [ ] **Step 5: Verify all entry points compile**

```bash
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py src/gauge/ocr_paddle_worker.py
```
Expected: no output (success)

- [ ] **Step 6: Commit**

```bash
git add run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py src/gauge/ocr_paddle_worker.py
git commit -m "refactor: remove sys.path hacks, use pyproject.toml editable install"
```

### Task 1.3: Create Pydantic data models

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/models/__init__.py`
- Create: `/home/cht/code/IQIdet/src/gauge/models/ocr.py`
- Create: `/home/cht/code/IQIdet/src/gauge/models/roi.py`
- Create: `/home/cht/code/IQIdet/src/gauge/models/wire.py`
- Create: `/home/cht/code/IQIdet/src/gauge/models/plate.py`
- Create: `/home/cht/code/IQIdet/src/gauge/models/grade.py`
- Create: `/home/cht/code/IQIdet/src/gauge/models/fields.py`
- Create: `/home/cht/code/IQIdet/src/gauge/models/record.py`

- [ ] **Step 1: Create models/__init__.py**

```python
#!/usr/bin/env python3
"""Pydantic data models for IQI pipeline."""
from gauge.models.ocr import OCRItem, OCRResult, OCRTimings, OrientationInfo
from gauge.models.roi import ROIInfo
from gauge.models.wire import LineRecord, WireResult
from gauge.models.plate import PlateCandidate, PlateResult
from gauge.models.grade import GradeResult
from gauge.models.fields import FieldRecord, GeneralFields, FieldStatistics
from gauge.models.record import IQIRecord

__all__ = [
    "OCRItem", "OCRResult", "OCRTimings", "OrientationInfo",
    "ROIInfo",
    "LineRecord", "WireResult",
    "PlateCandidate", "PlateResult",
    "GradeResult",
    "FieldRecord", "GeneralFields", "FieldStatistics",
    "IQIRecord",
]
```

- [ ] **Step 2: Create models/ocr.py**

```python
#!/usr/bin/env python3
"""OCR-related Pydantic models."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class OrientationInfo(BaseModel):
    label: Optional[int] = None
    confidence: Optional[float] = None
    status: str = "disabled"
    corrected: bool = False
    actions: Optional[str] = None


class OCRItem(BaseModel):
    crop_index: int
    text: str = ""
    score: Optional[float] = None
    box: List[List[float]] = Field(default_factory=list)
    det_score: Optional[float] = None
    crop_size: Optional[List[int]] = None
    status: str = "ok"
    accepted_by_score: bool = True
    orientation: OrientationInfo = Field(default_factory=OrientationInfo)
    box_image: Optional[List[List[float]]] = None
    box_roi_unrotated: Optional[List[List[float]]] = None
    error: Optional[str] = None


class OCRTimings(BaseModel):
    text_det_ms: float = 0.0
    text_orientation_ms: float = 0.0
    text_rec_ms: float = 0.0
    text_total_ms: float = 0.0


class OCRResult(BaseModel):
    status: str = "error"
    error: Optional[str] = None
    texts: List[str] = Field(default_factory=list)
    scores: List[Optional[float]] = Field(default_factory=list)
    items: List[OCRItem] = Field(default_factory=list)
    all_items: List[OCRItem] = Field(default_factory=list)
    num_items: int = 0
    selected_variant: str = "det_rec"
    all_texts_original: List[str] = Field(default_factory=list)
    all_texts_mirror: List[str] = Field(default_factory=list)
    det_box_count: int = 0
    rec_item_count: int = 0
    jb_items: List[OCRItem] = Field(default_factory=list)
    jb_texts: List[str] = Field(default_factory=list)
    jb_item_count: int = 0
    item_errors: List[Dict[str, Any]] = Field(default_factory=list)
    timings_ms: OCRTimings = Field(default_factory=OCRTimings)
    all_items_original: Optional[List[OCRItem]] = None
    items_original: Optional[List[OCRItem]] = None
    all_items_image: Optional[List[OCRItem]] = None
    items_image: Optional[List[OCRItem]] = None

    @property
    def ok(self) -> bool:
        return self.status in ("ok",)
```

- [ ] **Step 3: Create models/roi.py**

```python
#!/usr/bin/env python3
"""ROI-related Pydantic models."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class ROIInfo(BaseModel):
    polygon: List[List[float]]
    bbox: List[float]
    conf: Optional[float] = None
    class_id: Optional[int] = None
    crop_size_before_rotate: Optional[List[int]] = None
    crop_inverse_matrix: Optional[List[List[float]]] = None
```

- [ ] **Step 4: Create models/wire.py**

```python
#!/usr/bin/env python3
"""Wire detection Pydantic models."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class LineRecord(BaseModel):
    index: int
    score: Optional[float] = None
    roi_xy: List[List[float]]
    roi_unrotated_xy: Optional[List[List[float]]] = None
    image_xy: Optional[List[List[float]]] = None


class WireResult(BaseModel):
    status: str = "error"
    error: Optional[str] = None
    wire_count: Optional[int] = None
    parsed_line_count: int = 0
    lines: List[LineRecord] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.wire_count is not None
```

- [ ] **Step 5: Create models/plate.py**

```python
#!/usr/bin/env python3
"""Plate marker Pydantic models."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PlateCandidate(BaseModel):
    code: str
    iqi_type: str
    number: int
    corrections: List[str] = Field(default_factory=list)
    source_text: str = ""


class PlateResult(BaseModel):
    ok: bool = False
    result_code: int = 9001
    result_name: str = "internal_error"
    result_message: str = "内部异常"
    iqi_type: Optional[str] = None
    number: Optional[int] = None
    plate_code: Optional[str] = None
    raw_texts: List[str] = Field(default_factory=list)
    normalized_texts: List[str] = Field(default_factory=list)
    candidate_codes: List[str] = Field(default_factory=list)
    corrections: List[str] = Field(default_factory=list)
    sequence_candidates: Optional[List[str]] = None
    raw_text_items: Optional[List[Dict[str, Any]]] = None
```

- [ ] **Step 6: Create models/grade.py**

```python
#!/usr/bin/env python3
"""Grade computation Pydantic model."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class GradeResult(BaseModel):
    ok: bool = False
    result_code: int = 9001
    result_name: str = "internal_error"
    result_message: str = "内部异常"
    grade: int = 0
    wire_count: Optional[int] = None
```

- [ ] **Step 7: Create models/fields.py**

```python
#!/usr/bin/env python3
"""General fields extraction Pydantic models."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FieldRecord(BaseModel):
    text: str = ""
    match_text: str = ""
    score: Optional[float] = None
    box: Optional[List[List[float]]] = None
    crop_index: Optional[int] = None
    value: Optional[str] = None


class WeldFilmPair(BaseModel):
    text: str = ""
    match_text: str = ""
    score: Optional[float] = None
    box: Optional[List[List[float]]] = None
    crop_index: Optional[int] = None
    weld_no: str = ""
    film_no: str = ""
    separator: str = ""


class PipeSpec(BaseModel):
    text: str = ""
    match_text: str = ""
    score: Optional[float] = None
    box: Optional[List[List[float]]] = None
    crop_index: Optional[int] = None
    value: str = ""
    outer_diameter: str = ""
    wall_thickness: str = ""


class GeneralFields(BaseModel):
    component_codes: List[FieldRecord] = Field(default_factory=list)
    weld_film_pairs: List[WeldFilmPair] = Field(default_factory=list)
    weld_numbers: List[FieldRecord] = Field(default_factory=list)
    film_numbers: List[FieldRecord] = Field(default_factory=list)
    pipe_specs: List[PipeSpec] = Field(default_factory=list)


class FieldStatistics(BaseModel):
    component_code_count: int = 0
    weld_film_pair_count: int = 0
    weld_number_count: int = 0
    film_number_count: int = 0
    pipe_spec_count: int = 0
    general_fields_found: bool = False
    full_image_marker_found: bool = False
    roi_marker_found: bool = False
    iqi_marker_found: bool = False
```

- [ ] **Step 8: Create models/record.py**

```python
#!/usr/bin/env python3
"""Top-level IQI record Pydantic model."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from gauge.models.fields import GeneralFields, FieldStatistics
from gauge.models.ocr import OCRResult
from gauge.models.roi import ROIInfo
from gauge.models.wire import WireResult
from gauge.models.plate import PlateResult
from gauge.models.grade import GradeResult


class IQIRecord(BaseModel):
    image_path: str
    ok: bool = False
    status: str = "error"
    result_code: int = 9001
    result_name: str = "internal_error"
    result_message: str = "内部异常"
    grade: Optional[int] = None
    iqi_type: Optional[str] = None
    plate_code: Optional[str] = None
    plate_number: Optional[int] = None
    plate_source: Optional[str] = None
    wire_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    general_fields_found: bool = False
    iqi_marker_found: bool = False

    fields: GeneralFields = Field(default_factory=GeneralFields)
    field_statistics: FieldStatistics = Field(default_factory=FieldStatistics)
    correction: Dict[str, Any] = Field(default_factory=dict)
    full_image_preprocess: Dict[str, Any] = Field(default_factory=dict)
    preprocess: Dict[str, Any] = Field(default_factory=dict)
    ocr: Optional[OCRResult] = None
    full_image_ocr: Optional[OCRResult] = None
    full_image_plate: Optional[PlateResult] = None
    roi: Optional[ROIInfo] = None
    roi_ocr: Optional[OCRResult] = None
    roi_plate: Optional[PlateResult] = None
    plate: PlateResult = Field(default_factory=PlateResult)
    wire: WireResult = Field(default_factory=WireResult)
    grade_rule: Optional[GradeResult] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    visualization: Dict[str, Any] = Field(default_factory=dict)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    final_result_vis_path: Optional[str] = None
    status_vis_dir: Optional[str] = None

    _debug_artifacts: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create_error(cls, code: int, message: str, image_path: str = "") -> "IQIRecord":
        from gauge.iqi_rules import build_result_status
        status = build_result_status(code, message)
        return cls(
            image_path=image_path,
            ok=False,
            status="error",
            result_code=code,
            result_name=status.get("result_name", "unknown_error"),
            result_message=message,
        )
```

- [ ] **Step 9: Verify models compile**

```bash
python -c "from gauge.models import OCRItem, OCRResult, ROIInfo, WireResult, PlateResult, GradeResult, IQIRecord; print('OK')"
```
Expected: `OK`

- [ ] **Step 10: Commit**

```bash
git add src/gauge/models/ && git commit -m "feat: add Pydantic data models for IQI pipeline"
```

### Task 1.4: Create exception hierarchy

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/exceptions.py`

- [ ] **Step 1: Write exceptions.py**

```python
#!/usr/bin/env python3
"""Unified exception hierarchy for IQI pipeline."""


class IQIError(Exception):
    """Base for all IQI pipeline business errors."""
    result_code: int = 9001
    result_name: str = "internal_error"


class IQIStageSkipped(Exception):
    """Control-flow signal: stage was skipped (not an error)."""


class ImageReadError(IQIError):
    result_code = 1001
    result_name = "image_read_failed"


class ROINotFoundError(IQIError):
    result_code = 1101
    result_name = "roi_not_found"


class ROIInvalidError(IQIError):
    result_code = 1102
    result_name = "roi_invalid"


class MarkerError(IQIError):
    result_code = 2003
    result_name = "marker_format_invalid"


class MarkerMissingJBError(MarkerError):
    result_code = 2002
    result_name = "marker_missing_jb"


class MarkerAmbiguousError(MarkerError):
    result_code = 2006
    result_name = "marker_ambiguous"


class MarkerNumberOutOfRangeError(MarkerError):
    result_code = 2007
    result_name = "marker_number_out_of_range"


class WireInferenceError(IQIError):
    result_code = 3001
    result_name = "wire_infer_failed"


class WireCountMissingError(IQIError):
    result_code = 3002
    result_name = "wire_count_missing"


class GradeError(IQIError):
    result_code = 3005
    result_name = "grade_out_of_range"
```

- [ ] **Step 2: Verify import**

```bash
python -c "from gauge.exceptions import IQIError, ROINotFoundError; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/gauge/exceptions.py && git commit -m "feat: add unified IQI exception hierarchy"
```

### Task 1.5: Create PipelineConfig

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/config/__init__.py`

- [ ] **Step 1: Write config/__init__.py**

```python
#!/usr/bin/env python3
"""Centralized configuration for IQI pipeline."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class GaugeConfig(BaseModel):
    weights: str = "models/guagerotation.pt"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: Optional[str] = None
    select: Literal["conf", "area"] = "conf"
    class_filter: Optional[int] = Field(default=None, alias="gauge_class")


class FClipConfig(BaseModel):
    ckpt: Optional[str] = None
    device: Optional[str] = None
    model_config: str = "config/model.yaml"
    params: str = "params.yaml"
    threshold: Optional[float] = None


class OCRConfig(BaseModel):
    device: Literal["cpu", "gpu"] = "gpu"
    det_model_name: str = "PP-OCRv5_server_det"
    det_model_dir: Optional[str] = None
    rec_model_name: str = "en_PP-OCRv5_mobile_rec"
    rec_model_dir: Optional[str] = None
    det_limit_side_len: int = 960
    det_limit_type: Literal["max", "min"] = "max"
    min_score: float = 0.0
    number_range: str = "1-19"
    enable_orientation: bool = False
    orientation_model: Optional[str] = "models/ocr_orientation_model.pth"
    orientation_device: Optional[str] = None
    orientation_verbose: bool = False


class CorrectionConfig(BaseModel):
    enabled: bool = False
    model: Optional[str] = None
    device: Optional[str] = None
    verbose: bool = False


class EnhanceConfig(BaseModel):
    mode: Literal["original", "windowing"] = "windowing"
    rotate_roi: bool = True


class PipelineConfig(BaseSettings):
    gauge: GaugeConfig = Field(default_factory=GaugeConfig)
    fclip: FClipConfig = Field(default_factory=FClipConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    correction: CorrectionConfig = Field(default_factory=CorrectionConfig)
    enhance: EnhanceConfig = Field(default_factory=EnhanceConfig)

    class Config:
        env_prefix = "IQIDET_"
        env_nested_delimiter = "__"

    def apply_cli_overrides(self, args) -> "PipelineConfig":
        """Apply argparse Namespace overrides. CLI args take precedence over defaults/env."""
        overrides: dict = {}
        # Map CLI arg names to nested config paths
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
            "fclip_config": ("fclip", "model_config"),
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
            if value is not None and value is not False:
                section, key = config_path
                if section not in overrides:
                    overrides[section] = {}
                # Handle no_rotate inversion
                if attr_name == "no_rotate" and value:
                    overrides[section][key] = False
                else:
                    overrides[section][key] = value
        # Handle ocr_orientation_verbose separately
        if getattr(args, "ocr_orientation_verbose", False):
            overrides.setdefault("ocr", {})["orientation_verbose"] = True
        if overrides:
            return self.model_copy(update=overrides, deep=True)
        return self
```

- [ ] **Step 2: Verify config loads**

```bash
python -c "from gauge.config import PipelineConfig; c = PipelineConfig(); print(c.gauge.conf)"
```
Expected: `0.25`

- [ ] **Step 3: Commit**

```bash
git add src/gauge/config/ && git commit -m "feat: add centralized PipelineConfig with CLI override support"
```

### Task 1.6: Create structured logging

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/logging_setup.py`

- [ ] **Step 1: Write logging_setup.py**

```python
#!/usr/bin/env python3
"""Structured logging setup for IQI pipeline."""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """JSON-lines formatter for machine-parseable logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge extra fields from logger.info("msg", extra={...})
        for key in ("stage", "image", "elapsed_ms", "result_code",
                     "wire_count", "plate_code", "error", "candidates",
                     "iqi_type", "grade"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = str(record.exc_info[1])
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO, json_output: bool = True) -> None:
    root = logging.getLogger("gauge")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            StructuredFormatter()
            if json_output
            else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(handler)
    # Suppress noisy third-party loggers
    for name in ("ultralytics", "paddleocr", "paddle", "matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
```

- [ ] **Step 2: Verify logging works in a test script**

```bash
python -c "
from gauge.logging_setup import setup_logging
import logging
setup_logging(level=logging.DEBUG, json_output=True)
logger = logging.getLogger('gauge.test')
logger.info('test_msg', extra={'stage': 'test'})
print('Logging OK')
"
```
Expected: a JSON line then `Logging OK`

- [ ] **Step 3: Commit**

```bash
git add src/gauge/logging_setup.py && git commit -m "feat: add structured logging with JSON formatter"
```

### Task 1.7: Create BaseRegionService

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/services/__init__.py`
- Create: `/home/cht/code/IQIdet/src/gauge/services/base.py`

- [ ] **Step 1: Create services/__init__.py**

```python
#!/usr/bin/env python3
"""Shared service base classes."""
from gauge.services.base import BaseRegionService
from gauge.services.correction import BaseOrientationCorrector

__all__ = ["BaseRegionService", "BaseOrientationCorrector"]
```

- [ ] **Step 2: Write services/base.py**

```python
#!/usr/bin/env python3
"""Base class for region services (OCR and SNR) with base64 decode + singleton lifecycle."""
from __future__ import annotations
import base64
import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generic, Optional, TypeVar
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

T = TypeVar("T")


class BaseRegionService(Generic[T]):
    """Base for region processing services with shared base64 decode + thread pool."""

    _service: Optional[T] = None
    _executor: ThreadPoolExecutor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="region-svc"
    )

    @staticmethod
    def decode_base64(image_base64: str) -> np.ndarray:
        """Decode base64 image (supports data URL prefix)."""
        b64_data = str(image_base64 or "")
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="无法解码图片")
        return img

    @classmethod
    def get_service(cls) -> T:
        if cls._service is None:
            cls._service = cls._create_service()
        return cls._service

    @classmethod
    def close_service(cls) -> None:
        if cls._service is not None:
            cls._service.close()
            cls._service = None

    @classmethod
    def _create_service(cls) -> T:
        raise NotImplementedError("Subclass must implement _create_service()")

    @staticmethod
    def _shutdown_executor() -> None:
        try:
            BaseRegionService._executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            BaseRegionService._executor.shutdown(wait=False)

    def close(self) -> None:
        pass


atexit.register(BaseRegionService.close_service)
atexit.register(BaseRegionService._shutdown_executor)
```

- [ ] **Step 3: Create placeholder services/correction.py (for now)**

```python
#!/usr/bin/env python3
"""Base class for orientation correction — implemented in Phase 4."""
# Placeholder; real implementation in Phase 4
```

- [ ] **Step 4: Verify service base imports**

```bash
python -c "from gauge.services.base import BaseRegionService; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/gauge/services/ && git commit -m "feat: add BaseRegionService with base64 decode and singleton lifecycle"
```

### Task 1.8: Phase 1 verification

- [ ] **Step 1: Compile-check all entry points**

```bash
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py
```
Expected: no output (success)

- [ ] **Step 2: Run existing tests**

```bash
PYTHONPATH=src python -m pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 3: Phase 1 commit marker**

```bash
git commit --allow-empty -m "Phase 1 complete: infrastructure (pyproject.toml, models, config, logging, exceptions, service base)"
```

---

## Phase 2: Data Types + Error Unification

### Task 2.1: Wire Pydantic models into ocr_stage.py

**Files:**
- Modify: `/home/cht/code/IQIdet/src/gauge/ocr_stage.py`

**What to change:** `infer_roi_ocr()` currently returns `Dict[str, Any]`. Change it to build and return `OCRResult.model_dump()`. The function signature stays the same (still returns `Dict[str, Any]` for backward compatibility). All existing Dict access patterns in callers continue to work because `.model_dump()` produces identical structure.

- [ ] **Step 1: Add import at top of ocr_stage.py**

After existing imports, add:
```python
from gauge.models.ocr import OCRItem, OCRResult, OCRTimings, OrientationInfo
```

- [ ] **Step 2: Replace dict literal in infer_roi_ocr with OCRResult construction**

In `infer_roi_ocr()`, replace the `all_items.append({...})` dict literal with `OCRItem(...).model_dump()`. Replace the return dict literal with `OCRResult(...).model_dump()`.

Key changes:
- Line ~535: `all_items.append(OCRItem(crop_index=idx, text=text, ...).model_dump())`
- Line ~550: error items also use `OCRItem(...).model_dump()`
- Line ~583: return `OCRResult(status="ok", texts=texts, ...).model_dump()`
- Line ~601: error return `OCRResult(status="error", error=str(exc), ...).model_dump()`

Exact code blocks — for each return site in `infer_roi_ocr()`:

```python
# No-text return (line ~479):
return OCRResult(
    status="no_text",
    det_box_count=0,
    rec_item_count=0,
    timings_ms=OCRTimings(),
).model_dump()

# Normal return (line ~583):
return OCRResult(
    status=status,
    texts=texts,
    scores=scores,
    items=scored_items,
    all_items=all_items,
    num_items=len(scored_items),
    selected_variant="det_rec",
    all_texts_original=all_texts,
    det_box_count=len(dt_polys),
    rec_item_count=len(all_items),
    jb_items=jb_items,
    jb_texts=[str(item.get("text", "")) for item in jb_items],
    jb_item_count=len(jb_items),
    item_errors=item_errors,
    timings_ms=OCRTimings(
        text_det_ms=round(float(det_ms), 3),
        text_orientation_ms=round(float(orientation_ms), 3),
        text_rec_ms=round(float(rec_ms), 3),
        text_total_ms=round(float(det_ms + orientation_ms + rec_ms), 3),
    ),
).model_dump()

# Error return (line ~601):
return OCRResult(
    status="error",
    error=str(exc),
    det_box_count=0,
    rec_item_count=0,
    item_errors=[{"crop_index": None, "error": str(exc)}],
    timings_ms=OCRTimings(),
).model_dump()
```

- [ ] **Step 3: Verify ocr_stage.py compiles**

```bash
python -m py_compile src/gauge/ocr_stage.py
```
Expected: no output

- [ ] **Step 4: Commit**

```bash
git add src/gauge/ocr_stage.py && git commit -m "refactor: wire Pydantic OCRResult into ocr_stage.py"
```

### Task 2.2: Wire Pydantic models into roi_stage.py

**Files:**
- Modify: `/home/cht/code/IQIdet/src/gauge/roi_stage.py`

- [ ] **Step 1: Add import**

```python
from gauge.models.roi import ROIInfo
```

- [ ] **Step 2: Change extract_best_obb return to ROIInfo.model_dump()**

Replace the `return {...}` dict at line ~53 with:

```python
return ROIInfo(
    polygon=format_polygon(poly),
    bbox=[x_min, y_min, x_max, y_max],
    conf=conf,
    class_id=cls_id,
).model_dump()
```

- [ ] **Step 3: Commit**

```bash
git add src/gauge/roi_stage.py && git commit -m "refactor: wire Pydantic ROIInfo into roi_stage.py"
```

### Task 2.3: Wire Pydantic models into fclip_stage.py

**Files:**
- Modify: `/home/cht/code/IQIdet/src/gauge/fclip_stage.py`

- [ ] **Step 1: Add imports**

```python
from gauge.models.wire import LineRecord, WireResult
```

- [ ] **Step 2: Change build_line_records and FClipInferencer.infer return values**

In `build_line_records()`, change the `record = {...}` dict to `LineRecord(...).model_dump()`.

In `FClipInferencer.infer()`, change the return dicts to `WireResult(...).model_dump()`:

```python
# Error return (line ~157):
return WireResult(
    status="error",
    error="FClip outputs missing count head.",
).model_dump()

# Success return (line ~200):
return WireResult(
    status="ok",
    wire_count=wire_count,
    parsed_line_count=int(len(line_records)),
    lines=line_records,
    warnings=warnings,
).model_dump()

# Exception return (line ~207):
return WireResult(
    status="error",
    error=str(exc),
).model_dump()
```

- [ ] **Step 3: Commit**

```bash
git add src/gauge/fclip_stage.py && git commit -m "refactor: wire Pydantic WireResult into fclip_stage.py"
```

### Task 2.4: Wire Pydantic models into iqi_rules.py

**Files:**
- Modify: `/home/cht/code/IQIdet/src/gauge/iqi_rules.py`

- [ ] **Step 1: Add imports at top**

```python
from gauge.models.plate import PlateCandidate, PlateResult
from gauge.models.grade import GradeResult
from gauge.models.fields import GeneralFields, FieldStatistics, FieldRecord, WeldFilmPair, PipeSpec
```

- [ ] **Step 2: Change infer_plate_from_texts return values**

Replace inline Dict construction with `PlateResult(...).model_dump()`:

```python
# Success return (~line 506):
return PlateResult(
    ok=True,
    result_code=0,
    result_name="success",
    result_message="识别成功",
    iqi_type=chosen.iqi_type,
    number=chosen.number,
    plate_code=chosen.code,
    raw_texts=raw_texts,
    normalized_texts=normalized_texts,
    candidate_codes=unique_codes,
    corrections=list(chosen.corrections),
).model_dump()

# Error returns similarly replace build_result_status + extras with PlateResult().model_dump()
```

- [ ] **Step 3: Change compute_iqi_grade return values**

Replace with `GradeResult(...).model_dump()`.

- [ ] **Step 4: Change extract_general_fields_from_ocr_items return**

Replace with `GeneralFields(...).model_dump()` and `FieldStatistics(...)`.

- [ ] **Step 5: Commit**

```bash
git add src/gauge/iqi_rules.py && git commit -m "refactor: wire Pydantic PlateResult/GradeResult/GeneralFields into iqi_rules.py"
```

### Task 2.5: Wire error handling into run_iqi_grade_infer.py batch loop

**Files:**
- Modify: `/home/cht/code/IQIdet/run_iqi_grade_infer.py`

- [ ] **Step 1: Add import for logging and setup**

```python
import logging
from gauge.logging_setup import setup_logging
logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Add logging setup and --log-level / --log-json CLI args**

In `parse_args()`, add:
```python
parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging level.")
parser.add_argument("--log-json", action="store_true", help="Output JSON-line logs.")
```

- [ ] **Step 3: Strengthen batch loop error isolation**

In `main()`, wrap the per-image loop body:

```python
setup_logging(
    level=getattr(logging, args.log_level),
    json_output=args.log_json,
)

for image_path in tqdm(image_paths, desc="IQI Grade"):
    try:
        full_record, artifacts = inferencer.infer_image_path(
            image_path, return_debug_artifacts=want_vis
        )
    except Exception as exc:
        logger.error("image_failed", extra={"image": str(image_path), "error": str(exc)})
        full_record = {
            "image_path": str(image_path),
            "ok": False,
            "status": "error",
            "result_code": 9001,
            "result_name": "internal_error",
            "result_message": str(exc),
            "grade": None,
            "iqi_type": None,
            "plate_code": None,
            "plate_number": None,
            "plate_source": None,
            "wire_count": None,
            "fields": {"component_codes": [], "weld_film_pairs": [], "weld_numbers": [], "film_numbers": [], "pipe_specs": []},
            "field_statistics": {"general_fields_found": False, "iqi_marker_found": False},
            "warnings": [],
            "errors": [{"stage": "pipeline", "result_code": 9001, "result_name": "internal_error", "result_message": str(exc)}],
        }
        artifacts = None
    full_results.append(full_record)
    ...
```

- [ ] **Step 4: Commit**

```bash
git add run_iqi_grade_infer.py && git commit -m "feat: add structured logging CLI args and per-image error isolation in batch loop"
```

### Task 2.6: Wire BaseRegionService into region_ocr_api.py and region_snr_api.py

**Files:**
- Modify: `/home/cht/code/IQIdet/src/gauge/region_ocr_api.py`
- Modify: `/home/cht/code/IQIdet/src/gauge/region_snr_api.py`

Note: The root-level `region_ocr_api.py` and `region_SNR_api.py` import from `gauge.region_ocr_api` and `gauge.region_snr_api`. We modify the `src/gauge/` versions, not the root facades.

- [ ] **Step 1: Refactor region_ocr_api.py to use BaseRegionService**

Import `BaseRegionService` and remove the duplicated `_decode_base64_image`, `_shutdown_executor`, atexit registrations.
Use `BaseRegionService.decode_base64()` and `BaseRegionService._executor` instead.

- [ ] **Step 2: Refactor region_snr_api.py similarly**

- [ ] **Step 3: Commit**

```bash
git add src/gauge/region_ocr_api.py src/gauge/region_snr_api.py && git commit -m "refactor: use BaseRegionService in region_ocr_api and region_snr_api"
```

### Task 2.7: Phase 2 verification

- [ ] **Step 1: Compile-check all entry points**

```bash
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py
```

- [ ] **Step 2: Run existing tests**

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

- [ ] **Step 3: Commit phase marker**

```bash
git commit --allow-empty -m "Phase 2 complete: Pydantic models wired, error handling unified"
```

---

## Phase 3: Pipeline Refactoring

### Task 3.1: Create StageContext and PipelineStage base

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/stages/__init__.py`
- Create: `/home/cht/code/IQIdet/src/gauge/stages/base.py`

- [ ] **Step 1: Create stages/__init__.py**

```python
#!/usr/bin/env python3
"""Pipeline stage implementations."""
from gauge.stages.base import PipelineStage, StageContext
```

- [ ] **Step 2: Write stages/base.py**

```python
#!/usr/bin/env python3
"""Pipeline stage ABC and shared context."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from gauge.models.ocr import OCRResult
from gauge.models.roi import ROIInfo
from gauge.models.wire import WireResult
from gauge.models.plate import PlateResult
from gauge.models.grade import GradeResult
from gauge.models.fields import GeneralFields, FieldStatistics
from gauge.models.record import IQIRecord
from gauge.config import PipelineConfig


class StageContext(BaseModel):
    """Mutable context passed through pipeline stages."""
    image_path: str
    config: PipelineConfig
    image: Optional[Any] = None  # np.ndarray
    height: int = 0
    width: int = 0

    # Intermediate products
    sampled_image: Optional[Any] = None
    resize_scale: float = 1.0
    correction_info: Dict[str, Any] = Field(default_factory=dict)
    full_ocr_result: Optional[OCRResult] = None
    full_plate_result: Optional[PlateResult] = None
    roi_info: Optional[ROIInfo] = None
    roi_cropped: Optional[Any] = None
    roi_crop_matrix: Optional[Any] = None
    pre_rotate_size: Optional[List[int]] = None
    roi_image: Optional[Any] = None
    roi_gray: Optional[Any] = None
    rotated: bool = False
    roi_ocr_result: Optional[OCRResult] = None
    roi_plate_result: Optional[PlateResult] = None
    wire_result: Optional[WireResult] = None
    grade_result: Optional[GradeResult] = None
    general_fields: Optional[GeneralFields] = None
    field_statistics: FieldStatistics = Field(default_factory=FieldStatistics)

    # Accumulated state
    warnings: List[str] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    record_errors: List[Dict[str, Any]] = Field(default_factory=list)
    debug_artifacts: Optional[Dict[str, Any]] = None

    # Visualization buffers
    full_plate_vis_items: List[Dict[str, Any]] = Field(default_factory=list)
    roi_plate_vis_items: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def to_record(self) -> IQIRecord:
        """Build IQIRecord from context state."""
        from gauge.iqi_rules import build_result_status
        primary_code = self._choose_primary_code()
        status = build_result_status(primary_code)
        record = IQIRecord(
            image_path=self.image_path,
            ok=primary_code == 0,
            status="ok" if primary_code == 0 else "error",
            result_code=primary_code,
            result_name=status.get("result_name", "internal_error"),
            result_message=status.get("result_message", "内部异常"),
            width=self.width,
            height=self.height,
            correction=self.correction_info,
            warnings=self.warnings,
            errors=self.record_errors,
            timings_ms=self.timings_ms,
            _debug_artifacts=self.debug_artifacts,
        )
        # ... populate fields, plate, wire, grade from context ...
        return record

    @staticmethod
    def _choose_primary_code() -> int:
        # delegate to iqi_rules.choose_primary_result_code
        ...


class PipelineStage(ABC):
    """One reasoning stage in the pipeline."""

    name: str

    def __init__(self, config: PipelineConfig):
        self.config = config

    def should_run(self, ctx: StageContext) -> bool:
        return True

    @abstractmethod
    def run(self, ctx: StageContext) -> StageContext:
        ...
```

- [ ] **Step 3: Verify compiles**

```bash
python -m py_compile src/gauge/stages/base.py
```

- [ ] **Step 4: Commit**

```bash
git add src/gauge/stages/ && git commit -m "feat: add StageContext and PipelineStage base classes"
```

### Task 3.2-3.8: Create 7 Stage classes + PipelineRunner

**Note:** These 7 tasks are structurally identical — each creates one stage file. For brevity, only the first is shown in full detail. The others follow the same pattern: extract the corresponding code block from `iqi_inferencer.py`, wrap in a Stage class.

### Task 3.2: Create ImageLoadStage

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/stages/image_load.py`

- [ ] **Step 1: Write image_load.py**

```python
#!/usr/bin/env python3
"""Image loading stage."""
from __future__ import annotations
import time
import logging
from pathlib import Path
from gauge.stages.base import PipelineStage, StageContext
from gauge.config import PipelineConfig
from gauge.exceptions import ImageReadError

logger = logging.getLogger(__name__)


class ImageLoadStage(PipelineStage):
    name = "image_load"

    def run(self, ctx: StageContext) -> StageContext:
        from gauge.pipeline_utils import load_image
        t0 = time.perf_counter()
        try:
            image = load_image(Path(ctx.image_path))
        except FileNotFoundError as exc:
            raise ImageReadError(str(exc))
        ctx.image = image
        ctx.height, ctx.width = int(image.shape[0]), int(image.shape[1])
        if ctx.debug_artifacts is not None:
            ctx.debug_artifacts["image"] = image
        ctx.timings_ms[self.name] = (time.perf_counter() - t0) * 1000
        logger.debug("done", extra={"stage": self.name, "width": ctx.width, "height": ctx.height})
        return ctx
```

- [ ] **Step 2: Commit**

```bash
git add src/gauge/stages/image_load.py && git commit -m "feat: add ImageLoadStage"
```

### Task 3.3: Create CorrectionStage

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/stages/correction.py`

- [ ] **Step 1: Write correction.py — wrap the correction block from infer_image_path (lines ~567-583)**

Extract: `self.corrector.correct_image(image, verbose=...)` block.

`should_run()` returns `self.config.correction.enabled`.

- [ ] **Step 2: Commit**

```bash
git add src/gauge/stages/correction.py && git commit -m "feat: add CorrectionStage"
```

### Task 3.4: Create FullImageOCRStage

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/stages/full_image_ocr.py`

- [ ] **Step 1: Write full_image_ocr.py — wrap lines ~585-652 from infer_image_path**

Extract: resize → enhance → `infer_roi_ocr` → `extract_general_fields_from_ocr_items` → `infer_plate_from_ocr_items`.

- [ ] **Step 2: Commit**

```bash
git add src/gauge/stages/full_image_ocr.py && git commit -m "feat: add FullImageOCRStage"
```

### Task 3.5: Create ROIDetectStage

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/stages/roi_detect.py`

- [ ] **Step 1: Write roi_detect.py — wrap lines ~663-711 from infer_image_path**

Extract: YOLO predict → `extract_best_obb` → `crop_rotated_polygon` → `rotate_if_wide` → `enhance_windowing_gray`.

Raises `ROINotFoundError` if no ROI detected; raises `ROIInvalidError` if crop fails.

- [ ] **Step 2: Commit**

```bash
git add src/gauge/stages/roi_detect.py && git commit -m "feat: add ROIDetectStage"
```

### Task 3.6: Create ROIOCRStage

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/stages/roi_ocr.py`

- [ ] **Step 1: Write roi_ocr.py — wrap lines ~717-753 from infer_image_path**

Extract: `infer_roi_ocr` on roi_gray → `infer_plate_from_ocr_items` → `_project_ocr_items_to_image`.

`should_run()` returns `ctx.roi_gray is not None`.

- [ ] **Step 2: Commit**

```bash
git add src/gauge/stages/roi_ocr.py && git commit -m "feat: add ROIOCRStage"
```

### Task 3.7: Create WireDetectStage

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/stages/wire_detect.py`

- [ ] **Step 1: Write wire_detect.py — wrap lines ~821-839 from infer_image_path**

Extract: `fclip_inferencer.infer(roi_gray, ...)`.

`should_run()` returns `ctx.roi_gray is not None and self.config.fclip.ckpt is not None`.

- [ ] **Step 2: Commit**

```bash
git add src/gauge/stages/wire_detect.py && git commit -m "feat: add WireDetectStage"
```

### Task 3.8: Create GradeFusionStage

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/stages/grade_fusion.py`

- [ ] **Step 1: Write grade_fusion.py — wrap lines ~774-881 from infer_image_path**

Extract: select best plate → `compute_iqi_grade` → `_finalize_record` → `_attach_visualization_payload` → `_attach_timing`.

- [ ] **Step 2: Commit**

```bash
git add src/gauge/stages/grade_fusion.py && git commit -m "feat: add GradeFusionStage"
```

### Task 3.9: Create PipelineRunner

**Files:**
- Create: `/home/cht/code/IQIdet/src/gauge/pipeline.py`

- [ ] **Step 1: Write pipeline.py**

```python
#!/usr/bin/env python3
"""PipelineRunner: orchestrates stages for single-image IQI inference."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from gauge.config import PipelineConfig
from gauge.stages.base import PipelineStage, StageContext
from gauge.models.record import IQIRecord
from gauge.exceptions import IQIError, IQIStageSkipped

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(self, config: PipelineConfig, stages: List[PipelineStage]):
        self.config = config
        self.stages = stages

    def run(self, image_path: Path,
            return_debug_artifacts: bool = False) -> IQIRecord:
        ctx = StageContext(
            image_path=str(image_path),
            config=self.config,
        )
        if return_debug_artifacts:
            ctx.debug_artifacts = {}

        for stage in self.stages:
            if not stage.should_run(ctx):
                logger.debug("stage_skipped", extra={"stage": stage.name})
                continue
            try:
                ctx = stage.run(ctx)
            except IQIStageSkipped:
                logger.debug("stage_skipped_signal", extra={"stage": stage.name})
                continue
            except IQIError as exc:
                logger.warning("stage_error", extra={"stage": stage.name, "error": str(exc), "result_code": exc.result_code})
                ctx.record_errors.append({
                    "stage": stage.name,
                    "result_code": exc.result_code,
                    "result_name": exc.result_name,
                    "result_message": str(exc),
                })
            except Exception as exc:
                logger.exception("stage_fatal", extra={"stage": stage.name})
                return IQIRecord.create_error(
                    9001,
                    f"{stage.name}: {str(exc)}",
                    image_path=str(image_path),
                )
        return ctx.to_record()

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "PipelineRunner":
        from gauge.stages.image_load import ImageLoadStage
        from gauge.stages.correction import CorrectionStage
        from gauge.stages.full_image_ocr import FullImageOCRStage
        from gauge.stages.roi_detect import ROIDetectStage
        from gauge.stages.roi_ocr import ROIOCRStage
        from gauge.stages.wire_detect import WireDetectStage
        from gauge.stages.grade_fusion import GradeFusionStage

        stages: List[PipelineStage] = [
            ImageLoadStage(config),
            CorrectionStage(config),
            FullImageOCRStage(config),
            ROIDetectStage(config),
            ROIOCRStage(config),
            WireDetectStage(config),
            GradeFusionStage(config),
        ]
        return cls(config, stages)
```

- [ ] **Step 2: Commit**

```bash
git add src/gauge/pipeline.py && git commit -m "feat: add PipelineRunner with 7-stage orchestration"
```

### Task 3.10: Simplify IQIInferencer to delegate to PipelineRunner

**Files:**
- Modify: `/home/cht/code/IQIdet/src/gauge/iqi_inferencer.py`

- [ ] **Step 1: Rewrite IQIInferencer.__init__ to accept PipelineConfig**

```python
class IQIInferencer:
    def __init__(self, config: Optional[PipelineConfig] = None, **kwargs):
        if config is None:
            # Legacy constructor: accept all kwargs and build config from them
            config = self._build_config_from_kwargs(kwargs)
        self.config = config
        self.runner = PipelineRunner.from_config(config)
        # Initialize model resources (same as before but using config)
        self._init_models()
```

**IMPORTANT**: The legacy constructor (with all 32 keyword args) MUST continue to work because `run_iqi_grade_infer.py` calls `IQIInferencer(gauge_weights=..., fclip_ckpt=..., ...)`. Add a `_build_config_from_kwargs` static method that maps kwargs to `PipelineConfig.apply_cli_overrides` equivalent.

- [ ] **Step 2: Rewrite infer_image_path to delegate**

```python
def infer_image_path(self, image_path: Path, return_debug_artifacts=False, debug_timer=False) -> Tuple[Dict, Optional[Dict]]:
    record = self.runner.run(image_path, return_debug_artifacts=return_debug_artifacts)
    result = record.model_dump()
    if debug_timer:
        # timings already in record
        pass
    return result, record._debug_artifacts
```

- [ ] **Step 3: Keep all helper methods for visualization**

Keep `build_wire_vis_image`, `build_final_result_vis_image`, `save_debug_visualizations`, `build_delivery_record`, `build_iqi_statistics`, `collect_input_images` — these are used by `run_iqi_grade_infer.py` and remain unchanged.

- [ ] **Step 4: Commit**

```bash
git add src/gauge/iqi_inferencer.py && git commit -m "refactor: simplify IQIInferencer to delegate to PipelineRunner"
```

### Task 3.11: Phase 3 verification

- [ ] **Step 1: Compile-all**

```bash
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py
```

- [ ] **Step 2: Run tests**

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Fix test mocking to account for PipelineRunner. In test files, mock `gauge.iqi_inferencer.PipelineRunner` instead of individual pipeline functions.

- [ ] **Step 3: Commit phase marker**

```bash
git commit --allow-empty -m "Phase 3 complete: pipeline split into 7 Stages + PipelineRunner"
```

---

## Phase 4: Orientation Merge + Final Polish

### Task 4.1: Create BaseOrientationCorrector

**Files:**
- Replace: `/home/cht/code/IQIdet/src/gauge/services/correction.py` (currently placeholder)

- [ ] **Step 1: Write full services/correction.py**

```python
#!/usr/bin/env python3
"""Base class for orientation correction (8-class rotation/mirror detection)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class BaseOrientationCorrector(ABC):
    """Detect and restore image orientation across 8 direction classes."""

    STATUS_MAP = {
        0: "Normal (OK)",
        1: "Rotated 90 CW",
        2: "Upside Down (180)",
        3: "Rotated 90 CCW",
        4: "Mirrored",
        5: "Mirrored + 90 CW",
        6: "Mirrored + 180",
        7: "Mirrored + 90 CCW",
    }

    def __init__(
        self,
        model_path: Union[str, Path],
        model_type: str,
        device: Optional[str] = None,
        num_classes: int = 8,
        use_adaptive_processor: bool = True,
    ):
        self.model_path = Path(model_path)
        self.model_type = str(model_type)
        self.num_classes = int(num_classes)
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        from gauge.adaptive_image_processor import AdaptiveImageProcessor
        self.adaptive_processor = (
            AdaptiveImageProcessor(use_negative=True) if use_adaptive_processor else None
        )
        self.model = self._load_model()
        self.preprocess = self._get_preprocess()

    # --- Subclass hooks ---

    @abstractmethod
    def _get_model_architecture(self) -> nn.Module:
        """Build the ResNet variant + Linear(num_classes) head."""
        ...

    @abstractmethod
    def _get_preprocess(self) -> transforms.Compose:
        """Return torchvision transforms pipeline."""
        ...

    def _prepare_pil_image(self, image: np.ndarray) -> Image.Image:
        """Convert np.ndarray to PIL Image. Override for custom preprocessing."""
        if self.adaptive_processor is not None:
            processed = self.adaptive_processor.process_image(image)
            return Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))
        if image.ndim == 2:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        if image.ndim == 3 and image.shape[2] == 1:
            return Image.fromarray(cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB))
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # --- Shared logic ---

    def _load_model(self) -> nn.Module:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        model = self._get_model_architecture()
        model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=False)
        )
        model.to(self.device)
        model.eval()
        return model

    def predict_orientation(self, image: np.ndarray) -> Tuple[int, float]:
        pil_img = self._prepare_pil_image(image)
        input_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        return prediction.item(), confidence.item()

    def correct_image(self, image: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, dict]:
        label_idx, confidence = self.predict_orientation(image)
        info = {
            "label": label_idx,
            "confidence": confidence,
            "status": self.STATUS_MAP.get(label_idx, "Unknown"),
            "corrected": False,
            "actions": None,
        }
        if verbose:
            print(f"  Orientation: [{label_idx}] {info['status']} (Conf: {confidence:.4f})")
        if label_idx == 0:
            if verbose:
                print("  Image is already correct.")
            return image.copy(), info
        corrected_img, actions = self.restore_image(image, label_idx)
        info["corrected"] = True
        info["actions"] = actions
        if verbose:
            print(f"  Fix Actions: {actions}")
        return corrected_img, info

    @staticmethod
    def restore_image(image: np.ndarray, label_idx: int) -> Tuple[np.ndarray, str]:
        img = image.copy()
        label_idx = int(label_idx)
        rot_state = label_idx % 4
        if rot_state == 1:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            action_rot = "Rotate -90 (CCW)"
        elif rot_state == 2:
            img = cv2.rotate(img, cv2.ROTATE_180)
            action_rot = "Rotate 180"
        elif rot_state == 3:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            action_rot = "Rotate +90 (CW)"
        else:
            action_rot = "No Rotation"
        if label_idx >= 4:
            img = cv2.flip(img, 1)
            action_mirror = "Mirror Flip"
        else:
            action_mirror = "No Mirror"
        return img, f"{action_rot} + {action_mirror}"
```

- [ ] **Step 2: Commit**

```bash
git add src/gauge/services/correction.py && git commit -m "feat: add BaseOrientationCorrector with shared 8-class rotation/mirror logic"
```

### Task 4.2: Refactor WeldOrientationCorrector as subclass

**Files:**
- Modify: `/home/cht/code/IQIdet/src/gauge/weld_correction.py`

- [ ] **Step 1: Rewrite weld_correction.py**

Remove duplicated code (~100 lines). Inherit from `BaseOrientationCorrector`:

```python
class WeldOrientationCorrector(BaseOrientationCorrector):
    def _get_model_architecture(self) -> nn.Module:
        # ResNet50/101/152 → Linear(num_classes)
        ...

    def _get_preprocess(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
```

- [ ] **Step 2: Commit**

```bash
git add src/gauge/weld_correction.py && git commit -m "refactor: derive WeldOrientationCorrector from BaseOrientationCorrector"
```

### Task 4.3: Refactor OCRTextOrientationCorrector as subclass

**Files:**
- Modify: `/home/cht/code/IQIdet/src/gauge/ocr_orientation.py`

- [ ] **Step 1: Rewrite ocr_orientation.py**

Inherit from `BaseOrientationCorrector`, keep `SquarePadResize` helper:

```python
class OCRTextOrientationCorrector(BaseOrientationCorrector):
    def _get_model_architecture(self) -> nn.Module:
        # ResNet18/34/50 → Linear(num_classes)
        ...

    def _get_preprocess(self) -> transforms.Compose:
        return transforms.Compose([
            SquarePadResize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
```

- [ ] **Step 2: Commit**

```bash
git add src/gauge/ocr_orientation.py && git commit -m "refactor: derive OCRTextOrientationCorrector from BaseOrientationCorrector"
```

### Task 4.4: Add structured logging calls throughout

**Files:**
- Modify: all Stage files from Phase 3, pipeline.py, services

- [ ] **Step 1: Add logger lines to each Stage**

Each stage logs at start (`logger.debug("stage_started")`) and completion (`logger.info("stage_done", extra={...})`).

- [ ] **Step 2: Commit**

```bash
git add src/gauge/ && git commit -m "feat: add structured logging calls to all stages and services"
```

### Task 4.5: Update ARCHITECTURE.md

**Files:**
- Modify: `/home/cht/code/IQIdet/ARCHITECTURE.md`

- [ ] **Step 1: Rewrite ARCHITECTURE.md**

Make it concise and reflect the new structure. Keep the architecture diagram but update file paths. Add new constraints:

Key updates:
- Document `stages/`, `models/`, `config/`, `services/` directories
- Document Stage pipeline flow (`PipelineRunner` → 7 Stages)
- Document configuration management (`PipelineConfig`)
- Document error handling (`IQIError` hierarchy)
- New rules:
  - "新增模块使用 Pydantic BaseModel 定义数据结构，通过 `.model_dump()` 输出 Dict"
  - "新增推理能力以 PipelineStage 子类形式实现，不要直接在 IQIInferencer 中添加逻辑"
  - "对外接口（CLI 参数、JSON schema、import 路径）变更前需检查 run_iqi_grade_infer.py / region_ocr_api.py / region_SNR_api.py"
  - "使用 logging.getLogger(__name__) 获取 logger，通过 extra 字典传递结构化上下文"
  - "不要直接在主进程创建 PaddleOCR 实例，始终使用子进程隔离"
  - "异常处理：业务错误 raise IQIError 子类，非预期错误 raise 普通 Exception"

- [ ] **Step 2: Commit**

```bash
git add ARCHITECTURE.md && git commit -m "docs: update ARCHITECTURE.md for refactored pipeline, add new coding rules"
```

### Task 4.6: Phase 4 verification

- [ ] **Step 1: Compile-all**

```bash
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py
```

- [ ] **Step 2: Run tests**

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

- [ ] **Step 3: Pass all tests**

Ensure ALL tests pass. Update any broken mocks.

- [ ] **Step 4: Final commit**

```bash
git commit --allow-empty -m "Phase 4 complete: orientation merge, logging, ARCHITECTURE.md update"
```

---

## Final Verification Checklist

- [ ] `python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py` — no errors
- [ ] `PYTHONPATH=src python -m pytest tests/ -v` — all tests pass
- [ ] `pip install -e .` succeeds, `from gauge.xxx` works without `PYTHONPATH=src`
- [ ] CLI help unchanged: `python run_iqi_grade_infer.py --help` shows same arguments
- [ ] `python -c "from region_ocr_api import RecognizeRequest, RecognizeResponse, init_region_ocr_api, recognize_region"` — works
- [ ] `python -c "from region_SNR_api import SNRRequest, SNRResponse, init_region_snr_api, compute_region_snr"` — works
- [ ] `python -c "from gauge.iqi_inferencer import IQIInferencer, build_delivery_record"` — works
- [ ] `git diff main -- run_iqi_grade_infer.py` — only internal changes, no CLI arg changes
