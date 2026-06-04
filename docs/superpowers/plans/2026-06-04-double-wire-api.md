# 双丝分辨率核心算法统一封装 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encapsulate double-wire IQI analysis behind a single `analyze_double_wire(strip) → DoubleWireResult` interface, split `profile.py` into generic profile + double-wire algorithm modules, and unify annotate/validate/API consumers.

**Architecture:** Core algorithm `analyze_double_wire()` in new `src/gauge/imaging/double_wire.py` takes a 2D strip image, column-averages to 1D profile, delegates to existing `compute_contrast()` + `find_first_unresolved_group()`, and returns a structured `DoubleWireResult`. Service/API layers wrap it for HTTP; annotate and validate call it directly.

**Tech Stack:** Python 3.13, numpy, scipy, FastAPI, Pydantic

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/gauge/imaging/double_wire.py` | **Create** | All double-wire algorithm + `DoubleWireResult` + `analyze_double_wire()` |
| `src/gauge/imaging/profile.py` | **Modify** | Keep only generic profile/OBB functions; delete moved code |
| `scripts/double_wire/_dwlib/annotation.py` | **Modify** | Update import: `build_groundtruth_payload` from `double_wire` |
| `scripts/double_wire/_dwlib/profile_view.py` | **Modify** | Update import: `bam_pair_marker_indices` from `double_wire` |
| `scripts/double_wire/annotate.py` | **Modify** | Use `analyze_double_wire()` instead of 3 separate calls |
| `scripts/double_wire/validate_bam_gt.py` | **Modify** | Re-extract strip from original image, use `analyze_double_wire()` |
| `src/gauge/services/double_wire/__init__.py` | **Create** | Package init |
| `src/gauge/services/double_wire/service.py` | **Create** | `DoubleWireService` — pure computation, strip → dict |
| `src/gauge/app/double_wire_api.py` | **Create** | FastAPI wrapper, Pydantic Request/Response |
| `double_wire_api.py` | **Create** | Root import facade |
| `docs/contract/DOUBLE_WIRE_ANALYSIS_API.md` | Exists | JSON schema contract (already committed) |

---

### Task 1: Create `double_wire.py` — move all double-wire algorithm code

**Files:**
- Create: `src/gauge/imaging/double_wire.py`
- Modify: `src/gauge/imaging/profile.py`

Functions moving from `profile.py` to `double_wire.py` (with line numbers in current `profile.py`):

| Lines | Symbol |
|-------|--------|
| 19-21 | `_DEFAULT_WIRE_SPACINGS` |
| 268-283 | `ComputeContrastResult` |
| 285-341 | `_detect_film_type` |
| 344-383 | `_fit_quadratic_background` |
| 386-433 | `_compute_dip` |
| 436-502 | `_pair_wires_and_compute_dips` |
| 505-527 | `_trim_pairs_to_stable_center_prefix` |
| 530-581 | `_remove_overlapping_pairs` |
| 584-598 | `_pair_direction_scores` |
| 601-708 | `_recover_tail_pair` |
| 711-769 | `_pair_adjacent_wires_with_gaps` |
| 772-799 | `_cleanup_dips_monotonic` |
| 802-878 | `_find_crossing_group` |
| 922-1086 | `compute_contrast` |
| 1089-1141 | `find_first_unresolved_group` |
| 1182-1200 | `bam_pair_marker_indices` |
| 1207-1254 | `pair_wire_markers` |
| 1257-1322 | `build_groundtruth_payload` |

Functions staying in `profile.py`:

| Lines | Symbol |
|-------|--------|
| 24-38 | `_match_box_to_point_order` |
| 41-110 | `extract_profile_band` |
| 113-176 | `extract_profile_strip` |
| 179-225 | `fit_obb_and_midline` |
| 228-264 | `unwarp_obb_region` |
| 881-915 | `detect_peaks_valleys` |
| 1148-1179 | `normalize_profile_obb` |

- [ ] **Step 1: Create `src/gauge/imaging/double_wire.py`**

Read the full current `profile.py` to get exact code for each moving section. Build `double_wire.py` with the structure below. All moved functions are copied verbatim — no logic changes.

```python
"""Double-wire IQI analysis — dip computation, pairing, and resolution.

Pure numpy/scipy signal processing.  No OpenCV HighGUI, matplotlib,
or FastAPI dependency.  The primary entry point is :func:`analyze_double_wire`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from gauge.imaging.profile import detect_peaks_valleys

# JBT 7902-2025 D1~D13 wire spacings (mm)
_DEFAULT_WIRE_SPACINGS: Tuple[float, ...] = (
    0.80, 0.63, 0.50, 0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10, 0.08, 0.063, 0.05,
)


@dataclass
class ComputeContrastResult:
    """Result of :func:`compute_contrast`.

    Attributes:
        dips: Dip (modulation depth) for each wire pair, in percent [0, 100].
        pairs: Detected (wire_a_idx, gap_idx, wire_b_idx) triplets.
        background: Quadratic background fit values, same length as profile.
        film_type: ``"positive"`` or ``"negative"``.
    """
    dips: List[float]
    pairs: List[Tuple[int, int, int]]
    background: np.ndarray
    film_type: str


# [COPY EXACTLY from profile.py: _detect_film_type (lines 285-341)]
# [COPY EXACTLY from profile.py: _fit_quadratic_background (lines 344-383)]
# [COPY EXACTLY from profile.py: _compute_dip (lines 386-433)]
# [COPY EXACTLY from profile.py: _pair_wires_and_compute_dips (lines 436-502)]
# [COPY EXACTLY from profile.py: _trim_pairs_to_stable_center_prefix (lines 505-527)]
# [COPY EXACTLY from profile.py: _remove_overlapping_pairs (lines 530-581)]
# [COPY EXACTLY from profile.py: _pair_direction_scores (lines 584-598)]
# [COPY EXACTLY from profile.py: _recover_tail_pair (lines 601-708)]
# [COPY EXACTLY from profile.py: _pair_adjacent_wires_with_gaps (lines 711-769)]
# [COPY EXACTLY from profile.py: _cleanup_dips_monotonic (lines 772-799)]
# [COPY EXACTLY from profile.py: _find_crossing_group (lines 802-878)]
# [COPY EXACTLY from profile.py: compute_contrast (lines 922-1086)]
# [COPY EXACTLY from profile.py: find_first_unresolved_group (lines 1089-1141)]
# [COPY EXACTLY from profile.py: bam_pair_marker_indices (lines 1182-1200)]
# [COPY EXACTLY from profile.py: pair_wire_markers (lines 1207-1254)]
# [COPY EXACTLY from profile.py: build_groundtruth_payload (lines 1257-1322)]


@dataclass
class DoubleWireResult:
    """Complete result of :func:`analyze_double_wire`.

    Contains everything needed to render a profile chart with annotations:
    raw profile curve, background fit, peak/valley positions, wire-pair
    dip values, and resolution.

    ``pairs`` stores (wire_a, gap, wire_b) index triplets — compatible with
    :func:`bam_pair_marker_indices` and :meth:`Annotator.load_bam_baseline`.
    Call :meth:`pairs_as_dicts` for JSON-friendly dict form.
    """
    strip_shape: Tuple[int, int]
    profile: np.ndarray
    peaks: list         # [{"idx": int, "gray": float}]
    valleys: list       # [{"idx": int, "gray": float}]
    pairs: List[Tuple[int, int, int]]
    dips: List[float]
    film_type: str           # "positive" | "negative"
    background: np.ndarray
    first_unresolved_group: Optional[int]

    def pairs_as_dicts(self) -> list:
        """Convert pairs to JSON-friendly dict form with gray values."""
        return [
            {
                "group": i + 1,
                "wire_a_idx": int(w1),
                "gap_idx": int(gap),
                "wire_b_idx": int(w2),
                "wire_a_gray": float(self.profile[w1]),
                "gap_gray": float(self.profile[gap]),
                "wire_b_gray": float(self.profile[w2]),
                "dip_percent": float(dip),
            }
            for i, ((w1, gap, w2), dip) in enumerate(zip(self.pairs, self.dips))
        ]


def analyze_double_wire(
    strip: np.ndarray,
    *,
    min_distance: int = 5,
    prominence: float = 0.03,
) -> DoubleWireResult:
    """Analyze a double-wire IQI strip image.

    Column-averages the 2D strip -> 1D profile, then delegates to
    :func:`compute_contrast` and :func:`find_first_unresolved_group`.

    Args:
        strip: 2D float64 array of shape ``(band_height, num_samples)``.
            Each column is a perpendicular slice across the wires;
            row-averaging produces the 1D profile.
        min_distance: Minimum pixel distance between adjacent peaks.
        prominence: Relative peak prominence.

    Returns:
        :class:`DoubleWireResult` with all analysis outputs.
    """
    if strip.ndim != 2 or strip.shape[0] < 1 or strip.shape[1] < 3:
        return DoubleWireResult(
            strip_shape=(0, 0) if strip.ndim != 2 else strip.shape,
            profile=np.array([], dtype=np.float64),
            peaks=[],
            valleys=[],
            pairs=[],
            dips=[],
            film_type="positive",
            background=np.array([], dtype=np.float64),
            first_unresolved_group=None,
        )

    profile = strip.mean(axis=0).astype(np.float64)

    result = compute_contrast(
        profile, film_type="auto",
        min_distance=min_distance, prominence=prominence,
    )
    unresolved = find_first_unresolved_group(result.dips)

    # Raw peak/valley positions on the original profile (for display)
    peaks_idx, valleys_idx = detect_peaks_valleys(
        profile, min_distance=min_distance, prominence=prominence,
    )
    peaks = [
        {"idx": int(p), "gray": float(profile[p])}
        for p in peaks_idx
    ]
    valleys = [
        {"idx": int(v), "gray": float(profile[v])}
        for v in valleys_idx
    ]

    return DoubleWireResult(
        strip_shape=strip.shape,
        profile=profile,
        peaks=peaks,
        valleys=valleys,
        pairs=result.pairs,
        dips=result.dips,
        film_type=result.film_type,
        background=result.background,
        first_unresolved_group=unresolved,
    )
```

- [ ] **Step 2: Fill in the moved functions**

Copy each function listed in the "Functions moving" table above from `profile.py` into `double_wire.py`. Insert them between `ComputeContrastResult` and `DoubleWireResult`. Copy verbatim — do not modify function bodies.

Note: `compute_contrast` calls `detect_peaks_valleys`, which now lives in `profile.py`. The import `from gauge.imaging.profile import detect_peaks_valleys` at the top resolves this.

- [ ] **Step 3: Syntax check `double_wire.py`**

```bash
python -m py_compile src/gauge/imaging/double_wire.py
```

Expected: no output (success).

- [ ] **Step 4: Strip moved code from `profile.py`**

Remove from `profile.py`:
- Lines 19-21: `_DEFAULT_WIRE_SPACINGS`
- Lines 268-283: `ComputeContrastResult`
- Lines 285-878: All private `_*` functions (`_detect_film_type` through `_find_crossing_group`)
- Lines 922-1141: `compute_contrast` and `find_first_unresolved_group`
- Lines 1182-1322: `bam_pair_marker_indices`, `pair_wire_markers`, `build_groundtruth_payload`

Also clean up unused imports in `profile.py`:
- Remove `from dataclasses import dataclass` (only `ComputeContrastResult` used it)
- Remove `peak_widths` from `from scipy.signal import find_peaks, peak_widths, savgol_filter`
- Remove `savgol_filter` from the same import line
- Remove `from scipy.optimize import curve_fit` (was unused)

After removal, `profile.py` should contain only:
- Module docstring
- `from __future__ import annotations` + typing imports
- `import cv2`, `import numpy as np`
- `from scipy.ndimage import map_coordinates`
- `from scipy.signal import find_peaks`
- `_match_box_to_point_order`
- `extract_profile_band`
- `extract_profile_strip`
- `fit_obb_and_midline`
- `unwarp_obb_region`
- `detect_peaks_valleys`
- `normalize_profile_obb`

- [ ] **Step 5: Syntax check `profile.py`**

```bash
python -m py_compile src/gauge/imaging/profile.py
```

Expected: no output.

- [ ] **Step 6: Run existing tests**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_rules.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_inferencer.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/gauge/imaging/double_wire.py src/gauge/imaging/profile.py
git commit -m "refactor: split double-wire algorithm from profile.py into double_wire.py"
```

---

### Task 2: Update imports in consumer files

**Files:**
- Modify: `scripts/double_wire/_dwlib/annotation.py`
- Modify: `scripts/double_wire/_dwlib/profile_view.py`
- Modify: `scripts/double_wire/validate_bam_gt.py`

- [ ] **Step 1: Update `annotation.py`**

In `scripts/double_wire/_dwlib/annotation.py`, change:
```python
        from gauge.imaging.profile import build_groundtruth_payload
```
To:
```python
        from gauge.imaging.double_wire import build_groundtruth_payload
```
(This import is inside `build_groundtruth` method, around line 81.)

- [ ] **Step 2: Update `profile_view.py`**

In `scripts/double_wire/_dwlib/profile_view.py`, change line 51:
```python
        from gauge.imaging.profile import bam_pair_marker_indices
```
To:
```python
        from gauge.imaging.double_wire import bam_pair_marker_indices
```

- [ ] **Step 3: Update `validate_bam_gt.py`**

In `scripts/double_wire/validate_bam_gt.py`, change lines 36-46:
```python
from gauge.imaging.profile import (
    compute_contrast,
    find_first_unresolved_group,
    detect_peaks_valleys,
    _compute_dip,
    _fit_quadratic_background,
    _detect_film_type,
    _pair_adjacent_wires_with_gaps,
    _pair_direction_scores,
    _pair_wires_and_compute_dips,
)
```
To:
```python
from gauge.imaging.profile import detect_peaks_valleys
from gauge.imaging.double_wire import (
    compute_contrast,
    find_first_unresolved_group,
    _compute_dip,
    _fit_quadratic_background,
    _detect_film_type,
    _pair_adjacent_wires_with_gaps,
    _pair_direction_scores,
    _pair_wires_and_compute_dips,
)
```

- [ ] **Step 4: Syntax check all three files**

```bash
python -m py_compile scripts/double_wire/_dwlib/annotation.py scripts/double_wire/_dwlib/profile_view.py scripts/double_wire/validate_bam_gt.py
```

Expected: no output.

- [ ] **Step 5: Run validate script in dry-run mode (no args) to catch import errors**

```bash
python scripts/double_wire/validate_bam_gt.py --help
```

Expected: help text, no import errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/double_wire/_dwlib/annotation.py scripts/double_wire/_dwlib/profile_view.py scripts/double_wire/validate_bam_gt.py
git commit -m "refactor: update imports after profile.py / double_wire.py split"
```

---

### Task 3: Add `DoubleWireService` — pure computation layer

**Files:**
- Create: `src/gauge/services/double_wire/__init__.py`
- Create: `src/gauge/services/double_wire/service.py`

- [ ] **Step 1: Create package init**

```bash
mkdir -p src/gauge/services/double_wire
```

`src/gauge/services/double_wire/__init__.py`:
```python
"""Double-wire IQI analysis service layer."""
```

- [ ] **Step 2: Create `service.py`**

`src/gauge/services/double_wire/service.py`:
```python
#!/usr/bin/env python3
"""Double-wire IQI analysis service — pure computation, no web framework."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from gauge.imaging.double_wire import analyze_double_wire


class DoubleWireService:
    """Compute double-wire IQI analysis on a strip image.

    Thin wrapper around :func:`analyze_double_wire` that handles
    serialization of numpy arrays and timing instrumentation.
    """

    RESULT_TABLE = {
        0: ("ok", "分析成功"),
        5001: ("invalid_image", "图像解码失败或为空"),
        5002: ("profile_too_short", "strip 宽度 < 3，无法分析"),
        5003: ("no_extrema_found", "未检出峰或谷"),
        5999: ("internal_error", "内部异常"),
    }

    def __init__(
        self,
        min_distance: int = 5,
        prominence: float = 0.03,
    ):
        self.min_distance = int(min_distance)
        self.prominence = float(prominence)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @classmethod
    def _build_status(cls, result_code: int, message: Optional[str] = None) -> Dict[str, Any]:
        result_name, default_message = cls.RESULT_TABLE.get(
            result_code, ("unknown_error", "未知错误"),
        )
        return {
            "ok": result_code == 0,
            "status": "ok" if result_code == 0 else "error",
            "result_code": result_code,
            "result_name": result_name,
            "message": str(message or default_message),
        }

    @staticmethod
    def _serialize_result(r) -> Dict[str, Any]:
        """Convert DoubleWireResult fields to JSON-serializable dict."""
        return {
            "film_type": r.film_type,
            "num_pairs": len(r.pairs),
            "first_unresolved_group": r.first_unresolved_group,
            "profile": r.profile.tolist(),
            "background": r.background.tolist(),
            "peaks": r.peaks,
            "valleys": r.valleys,
            "pairs": r.pairs_as_dicts(),
        }

    def compute(self, strip: np.ndarray) -> Dict[str, Any]:
        """Run analysis on a strip image.

        Args:
            strip: 2D ndarray of shape (band_height, num_samples).

        Returns:
            Dict matching the DOUBLE_WIRE_ANALYSIS_API contract.
        """
        if strip is None or not isinstance(strip, np.ndarray) or strip.size == 0:
            return {
                **self._build_status(5001, message="输入图像为空"),
                "timings_ms": {"total_ms": 0.0},
                "strip_shape": None,
                "result": None,
            }

        total_start = time.perf_counter()

        try:
            r = analyze_double_wire(
                strip,
                min_distance=self.min_distance,
                prominence=self.prominence,
            )
        except Exception as exc:
            total_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                **self._build_status(5999, message=str(exc)),
                "timings_ms": {"total_ms": round(total_ms, 3)},
                "strip_shape": list(strip.shape) if strip.ndim == 2 else None,
                "result": None,
            }

        total_ms = (time.perf_counter() - total_start) * 1000.0

        if len(r.profile) < 3:
            return {
                **self._build_status(
                    5002, message=f"strip 宽度 = {len(r.profile)}，无法分析",
                ),
                "timings_ms": {"total_ms": round(total_ms, 3)},
                "strip_shape": list(r.strip_shape),
                "result": None,
            }

        if len(r.peaks) == 0 and len(r.valleys) == 0:
            return {
                **self._build_status(5003, message="未检出峰或谷"),
                "timings_ms": {"total_ms": round(total_ms, 3)},
                "strip_shape": list(r.strip_shape),
                "result": self._serialize_result(r),
            }

        return {
            **self._build_status(0),
            "timings_ms": {"total_ms": round(total_ms, 3)},
            "strip_shape": list(r.strip_shape),
            "result": self._serialize_result(r),
        }
```

- [ ] **Step 3: Syntax check**

```bash
python -m py_compile src/gauge/services/double_wire/service.py
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add src/gauge/services/double_wire/
git commit -m "feat: add DoubleWireService — pure computation layer"
```

---

### Task 4: Add FastAPI API layer + root facade

**Files:**
- Create: `src/gauge/app/double_wire_api.py`
- Create: `double_wire_api.py` (repo root)

- [ ] **Step 1: Create `src/gauge/app/double_wire_api.py`**

```python
#!/usr/bin/env python3
"""FastAPI-friendly wrapper for double-wire IQI analysis."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import numpy as np

from gauge.runtime.region_runtime import decode_base64, executor
from gauge.services.double_wire.service import DoubleWireService

try:  # pragma: no cover
    from fastapi import HTTPException
except ImportError:  # pragma: no cover
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = int(status_code)
            self.detail = str(detail)

try:  # pragma: no cover
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    class BaseModel:
        def __init__(self, **data: Any):
            for key, value in data.items():
                setattr(self, key, value)
        def model_dump(self) -> Dict[str, Any]:
            return dict(self.__dict__)
    def Field(default: Any = None, **_kwargs: Any) -> Any:
        return default


class DoubleWireRequest(BaseModel):
    image_base64: str = Field(
        ..., description="strip 条带图像 base64 编码，支持 data URL 前缀。",
    )


class DoubleWireResponse(BaseModel):
    ok: bool = Field(..., description="是否分析成功。")
    status: str = Field(..., description="分析状态，ok / error。")
    result_code: int = Field(..., description="结果码，0 表示成功。")
    result_name: str = Field(..., description="结果码名称。")
    message: str = Field(..., description="结果说明。")
    timings_ms: Dict[str, float] = Field(..., description="各阶段耗时。")
    strip_shape: Optional[List[int]] = Field(
        default=None, description="strip 图像尺寸 [height, width]。",
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="分析结果，见 docs/contract/DOUBLE_WIRE_ANALYSIS_API.md。",
    )


_double_wire_service: Optional[DoubleWireService] = None


def init_double_wire_api(
    min_distance: int = 5,
    prominence: float = 0.03,
) -> DoubleWireService:
    global _double_wire_service
    if _double_wire_service is not None:
        _double_wire_service.close()
    _double_wire_service = DoubleWireService(
        min_distance=min_distance, prominence=prominence,
    )
    return _double_wire_service


def get_double_wire_service() -> DoubleWireService:
    global _double_wire_service
    if _double_wire_service is None:
        _double_wire_service = init_double_wire_api()
    return _double_wire_service


def close_double_wire_api() -> None:
    global _double_wire_service
    if _double_wire_service is not None:
        _double_wire_service.close()
        _double_wire_service = None


def _sync_compute(strip: np.ndarray) -> Dict[str, Any]:
    service = get_double_wire_service()
    return service.compute(strip)


async def compute_double_wire(request: DoubleWireRequest) -> DoubleWireResponse:
    """分析双丝像质计 strip 图像（base64 输入）。"""
    img = decode_base64(request.image_base64)
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, _sync_compute, img)
        return DoubleWireResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"双丝像质计分析失败: {str(exc)}",
        )


__all__ = [
    "DoubleWireRequest",
    "DoubleWireResponse",
    "DoubleWireService",
    "close_double_wire_api",
    "compute_double_wire",
    "get_double_wire_service",
    "init_double_wire_api",
]
```

- [ ] **Step 2: Create root facade `double_wire_api.py`**

```python
#!/usr/bin/env python3
"""Root-level import facade for the double-wire analysis API."""

from gauge.app.double_wire_api import (
    DoubleWireRequest,
    DoubleWireResponse,
    close_double_wire_api,
    compute_double_wire,
    get_double_wire_service,
    init_double_wire_api,
)
from gauge.services.double_wire.service import DoubleWireService

__all__ = [
    "DoubleWireRequest",
    "DoubleWireResponse",
    "DoubleWireService",
    "close_double_wire_api",
    "compute_double_wire",
    "get_double_wire_service",
    "init_double_wire_api",
]
```

- [ ] **Step 3: Syntax check**

```bash
python -m py_compile src/gauge/app/double_wire_api.py double_wire_api.py
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add src/gauge/app/double_wire_api.py double_wire_api.py
git commit -m "feat: add double-wire analysis API layer and root facade"
```

---

### Task 5: Update `annotate.py` to use `analyze_double_wire()`

**Files:**
- Modify: `scripts/double_wire/annotate.py`

Currently `_update_profile` and `_save_one_version` each call `extract_profile_band` + `compute_contrast` + `find_first_unresolved_group` independently. Replace with a single `analyze_double_wire(narrow_strip)` call.

- [ ] **Step 1: Update imports in `annotate.py`**

Change lines 59-64:
```python
from gauge.imaging.profile import (
    extract_profile_band,
    extract_profile_strip,
    compute_contrast,
    find_first_unresolved_group,
)
```
To:
```python
from gauge.imaging.profile import (
    extract_profile_strip,
)
from gauge.imaging.double_wire import (
    analyze_double_wire,
    bam_pair_marker_indices,
)
```

- [ ] **Step 2: Replace `_update_profile` method**

Replace the current method (lines 116-158) with:

```python
    def _update_profile(self) -> None:
        if (self.line_selector is None or self.view is None
                or not self.line_selector.is_locked):
            return

        line_start = self.line_selector.line_start
        line_end = self.line_selector.line_end

        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        line_length = max(1, int(np.ceil(np.hypot(dx, dy))))

        # Full strip for visualization
        strip = extract_profile_strip(
            self.image_raw, line_start, line_end,
            expand=self.expand, num_samples=line_length,
        )

        # Narrow strip for analysis: center band_width rows around midline
        half_h = strip.shape[0] // 2
        half_bw = self.band_width // 2
        narrow = strip[half_h - half_bw : half_h + half_bw + 1, :]

        result = analyze_double_wire(
            narrow, min_distance=5, prominence=0.03,
        )

        self.view.update(
            strip, result.profile, self.expand, self.band_width,
            result, result.first_unresolved_group, self.image_path.stem,
        )

        # Redraw annotation markers if active
        if self._annotating and self.annotator is not None:
            self.annotator.profile_values = result.profile
            self.annotator.draw_markers(self.view.profile_axes)
            self.view.fig.canvas.draw()
            self.view.fig.canvas.flush_events()

        self._profile = result.profile
        self._strip = strip
        self._bam_result = result
        self._unresolved = result.first_unresolved_group
```

- [ ] **Step 3: Update `_save_one_version` method**

Replace the profile extraction + BAM analysis block (lines 224-256, the part inside `_save_one_version` that does `extract_profile_band` + `compute_contrast` + `find_first_unresolved_group`) with `analyze_double_wire`.

The current block:
```python
        # Profile extraction + BAM analysis
        profile = extract_profile_band(
            image, line_start, line_end,
            band_width=self.band_width, num_samples=line_length,
        )
        result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
        unresolved = find_first_unresolved_group(result.dips)
```

Replace with:
```python
        # Strip extraction + BAM analysis (via unified core API)
        strip_save = extract_profile_strip(
            image, line_start, line_end,
            expand=self.expand, num_samples=line_length,
        )
        half_h_save = strip_save.shape[0] // 2
        half_bw_save = self.band_width // 2
        narrow_save = strip_save[half_h_save - half_bw_save : half_h_save + half_bw_save + 1, :]
        result = analyze_double_wire(
            narrow_save, min_distance=5, prominence=0.03,
        )
        profile = result.profile
        unresolved = result.first_unresolved_group
```

Then update the profile JSON payload to use `result.pairs` and `result.dips` (which are `list[tuple]` and `list[float]` respectively — same types as before).

- [ ] **Step 4: Syntax check**

```bash
python -m py_compile scripts/double_wire/annotate.py
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add scripts/double_wire/annotate.py
git commit -m "refactor(annotate): use unified analyze_double_wire() instead of 3 separate calls"
```

---

### Task 6: Update `validate_bam_gt.py` — re-extract strip from original image

**Files:**
- Modify: `scripts/double_wire/validate_bam_gt.py`

Currently reads `profile_values` from JSON. Change to: read image_path + line + expand + band_width from JSON, load image, extract narrow strip, call `analyze_double_wire`.

- [ ] **Step 1: Add import for `extract_profile_strip` and `analyze_double_wire`**

In the imports section, change:
```python
from gauge.imaging.profile import detect_peaks_valleys
from gauge.imaging.double_wire import (
    compute_contrast,
    find_first_unresolved_group,
    _compute_dip,
    _fit_quadratic_background,
    _detect_film_type,
    _pair_adjacent_wires_with_gaps,
    _pair_direction_scores,
    _pair_wires_and_compute_dips,
)
```
To:
```python
from gauge.imaging.profile import detect_peaks_valleys, extract_profile_strip
from gauge.imaging.double_wire import (
    analyze_double_wire,
    compute_contrast,
    find_first_unresolved_group,
    _compute_dip,
    _fit_quadratic_background,
    _detect_film_type,
    _pair_adjacent_wires_with_gaps,
    _pair_direction_scores,
    _pair_wires_and_compute_dips,
)
```

- [ ] **Step 2: Replace the algorithm call in `_validate_one`**

In `_validate_one()`, replace the `compute_contrast` call block (around line 143):
```python
    result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
```

With:
```python
    # Re-extract narrow strip from original image to use unified core API
    image_path = profile_data.get("image_path")
    profile_line = profile_data.get("profile_line", {})
    expand = profile_data.get("expand", 60)
    band_width = profile_data.get("band_width", 21)

    if image_path and profile_line:
        import cv2
        raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if raw is not None:
            if raw.ndim == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            start = (profile_line["start"][0], profile_line["start"][1])
            end = (profile_line["end"][0], profile_line["end"][1])
            line_length = profile_data.get("line_length_px",
                          max(1, int(np.ceil(np.hypot(end[0]-start[0], end[1]-start[1])))))
            strip_full = extract_profile_strip(
                raw, start, end, expand=expand, num_samples=line_length,
            )
            half_h = strip_full.shape[0] // 2
            half_bw = band_width // 2
            narrow = strip_full[half_h - half_bw : half_h + half_bw + 1, :]
            r = analyze_double_wire(narrow, min_distance=5, prominence=0.03)
            result = ComputeContrastResult(
                dips=r.dips, pairs=r.pairs,
                background=r.background, film_type=r.film_type,
            )
        else:
            # Fallback: use profile_values from JSON (original path)
            result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
    else:
        # Fallback: profile_values only
        result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
```

Note: `ComputeContrastResult` needs to be imported. Add to the double_wire imports:
```python
from gauge.imaging.double_wire import (
    ComputeContrastResult,
    analyze_double_wire,
    ...
)
```

Actually, the validate script uses `result.pairs`, `result.dips`, `result.film_type`, `result.background` extensively. Rather than constructing a `ComputeContrastResult`, we should just use `DoubleWireResult` directly — it has all the same attributes.

Let me revise: just use `r` (DoubleWireResult) directly instead of `result` (ComputeContrastResult). All attribute names match.

- [ ] **Step 3: Replace step 1 with correct version using DoubleWireResult**

```python
    # Re-extract narrow strip from original image to use unified core API
    image_path = profile_data.get("image_path")
    profile_line = profile_data.get("profile_line", {})
    expand = profile_data.get("expand", 60)
    band_width = profile_data.get("band_width", 21)
    line_length = profile_data.get("line_length_px")

    if image_path and profile_line:
        import cv2
        raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if raw is not None:
            if raw.ndim == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            start = (profile_line["start"][0], profile_line["start"][1])
            end = (profile_line["end"][0], profile_line["end"][1])
            if line_length is None:
                line_length = max(1, int(np.ceil(np.hypot(
                    end[0] - start[0], end[1] - start[1],
                ))))
            strip_full = extract_profile_strip(
                raw, start, end, expand=expand, num_samples=line_length,
            )
            half_h = strip_full.shape[0] // 2
            half_bw = band_width // 2
            narrow = strip_full[half_h - half_bw : half_h + half_bw + 1, :]
            result = analyze_double_wire(narrow, min_distance=5, prominence=0.03)
            used_fallback = False
        else:
            result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
            used_fallback = True
    else:
        result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
        used_fallback = True

    if used_fallback:
        log("WARNING: Could not re-extract strip from original image.")
        log("  Falling back to profile_values from JSON (legacy path).")
        log()
```

- [ ] **Step 4: Syntax check**

```bash
python -m py_compile scripts/double_wire/validate_bam_gt.py
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add scripts/double_wire/validate_bam_gt.py
git commit -m "refactor(validate): re-extract strip from original image, use analyze_double_wire()"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run all tests**

```bash
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_rules.py
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python tests/test_iqi_inferencer.py
```

Expected: all tests pass.

- [ ] **Step 2: Run annotate tool to verify no import/crash errors**

```bash
python scripts/double_wire/annotate.py --help
```

Expected: help text, no import errors.

- [ ] **Step 3: Run validate tool to verify**

```bash
python scripts/double_wire/validate_bam_gt.py --help
```

Expected: help text, no import errors.

- [ ] **Step 4: Commit final state**

```bash
git status
# Verify no unexpected changes
git add -A
git commit -m "chore: final verification — all tests pass, imports clean"
```
