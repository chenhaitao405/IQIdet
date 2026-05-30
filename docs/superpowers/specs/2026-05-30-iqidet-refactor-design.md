# IQIdet Refactor Design Spec

**Date:** 2026-05-30
**Status:** Approved

## Objectives

Refactor 8 aspects of the IQIdet codebase without breaking external interfaces:
`run_iqi_grade_infer.py`, `region_ocr_api.py`, `region_SNR_api.py`.

1. Split `infer_image_path` into independent Stage pipeline
2. Introduce Pydantic dataclasses to replace bare `Dict[str, Any]`
3. Extract common base64/singleton service base class (~150 lines dedup)
4. Unify error handling patterns (single-image failure must not block batch)
5. Merge `WeldOrientationCorrector` + `OCRTextOrientationCorrector` (~200 lines dedup)
6. Centralize configuration management
7. Add `pyproject.toml` (setuptools) to eliminate `sys.path` hacks
8. Add structured logging (stdlib `logging`)

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Stage pipeline | Hybrid (PipelineRunner orchestrates; Stages self-skip minor conditions) | Closest to current behavior, lowest refactor risk |
| Data types | Pydantic BaseModel | Built-in serialization/validation, already semi-used in API layer |
| Build backend | setuptools | Max compatibility with existing conda env |
| Logging | stdlib `logging` + JSON formatter | Zero new dependencies |

## File Structure Changes

### New Directories

```
src/gauge/
├── stages/              # Stage implementations (7 stages)
│   ├── __init__.py
│   ├── base.py          # PipelineStage ABC + StageContext
│   ├── image_load.py
│   ├── correction.py
│   ├── full_image_ocr.py
│   ├── roi_detect.py
│   ├── roi_ocr.py
│   ├── wire_detect.py
│   └── grade_fusion.py
├── models/              # Pydantic data models
│   ├── __init__.py
│   ├── ocr.py           # OCRItem, OrientationInfo, OCRResult, OCRTimings
│   ├── roi.py           # ROIInfo
│   ├── wire.py          # WireResult, LineRecord
│   ├── plate.py         # PlateResult, PlateCandidate
│   ├── grade.py         # GradeResult
│   ├── record.py        # IQIRecord (top-level aggregation)
│   └── fields.py        # GeneralFields, WeldFilmPair, etc.
├── config/
│   └── __init__.py      # PipelineConfig (GaugeConfig, FClipConfig, OCRConfig, etc.)
├── services/
│   ├── __init__.py
│   ├── base.py          # BaseRegionService[T] (base64 decode + executor + singleton)
│   └── correction.py    # BaseOrientationCorrector (shared 8-class rotation/mirror logic)
├── exceptions.py         # IQIError hierarchy
├── logging_setup.py      # StructuredFormatter + setup_logging()
└── pipeline.py           # PipelineRunner (stages orchestration)
```

### Root-Level Changes

- **New:** `pyproject.toml` — setuptools build config, package name `IQIdet`
- **Removed:** `sys.path.insert` hacks from `run_iqi_grade_infer.py`, `region_ocr_api.py`, `region_SNR_api.py`, `ocr_paddle_worker.py`
- **Unchanged:** CLI arguments, output JSON schema, class names, import paths for external consumers

## Data Models (Pydantic)

### OCR Models (`models/ocr.py`)

- `OrientationInfo`: label, confidence, status, corrected, actions
- `OCRItem`: crop_index, text, score, box, det_score, status, accepted_by_score, orientation, box_image, box_roi_unrotated
- `OCRTimings`: text_det_ms, text_orientation_ms, text_rec_ms, text_total_ms
- `OCRResult`: status, texts, scores, items, all_items, det_box_count, rec_item_count, jb_items, jb_texts, timings_ms, item_errors

### ROI Model (`models/roi.py`)

- `ROIInfo`: polygon, bbox, conf, class_id, crop_size_before_rotate, crop_inverse_matrix

### Wire Model (`models/wire.py`)

- `LineRecord`: index, score, roi_xy, roi_unrotated_xy, image_xy
- `WireResult`: status, error, wire_count, parsed_line_count, lines, warnings

### Plate Model (`models/plate.py`)

- `PlateCandidate`: code, iqi_type, number, corrections, source_text
- `PlateResult`: ok, result_code, result_name, result_message, iqi_type, number, plate_code, raw_texts, candidate_codes, corrections

### Fields Model (`models/fields.py`)

- `FieldRecord`: text, match_text, score, box, value
- `GeneralFields`: component_codes, weld_film_pairs, weld_numbers, film_numbers, pipe_specs
- `FieldStatistics`: counts per field type, general_fields_found, etc.

### Top-Level Record (`models/record.py`)

- `IQIRecord`: image_path, ok, status, result_code, result_name, result_message, grade, iqi_type, plate_code, plate_number, plate_source, wire_count, fields, field_statistics, warnings, errors, visualization, timings_ms

All models produce identical JSON via `.model_dump()` compared to existing Dict output.

## Stage Pipeline Design

### Stage Interface

```python
class PipelineStage(ABC):
    name: str

    def should_run(self, ctx: StageContext) -> bool:  # default True
        ...

    def run(self, ctx: StageContext) -> StageContext:
        ...
```

### 7 Stages

| Stage | `should_run` condition | Action |
|-------|----------------------|--------|
| `ImageLoadStage` | always | Read image, record dimensions |
| `CorrectionStage` | `config.correction.enabled` | Weld orientation correction |
| `FullImageOCRStage` | always | Resize→enhance→full OCR→field extraction→plate marker matching |
| `ROIDetectStage` | always | YOLO→select ROI→perspective crop→rotate→enhance |
| `ROIOCRStage` | `ctx.roi_info is not None` | ROI OCR + plate marker matching |
| `WireDetectStage` | `ctx.roi_gray is not None and config.fclip.ckpt` | FClip wire inference |
| `GradeFusionStage` | always | Select best plate→grade→visualization→error priority |

### PipelineRunner

- Factory: `PipelineRunner.from_config(PipelineConfig)` builds default stage sequence
- Execution: iterates stages, catches `IQIError` (records, continues), catches `Exception` (returns error record, stops)
- Single-image failure produces `IQIRecord` with error fields; batch continues

### Backward Compatibility

`IQIInferencer.infer_image_path()` signature unchanged:

```python
def infer_image_path(self, image_path: Path, return_debug_artifacts=False, debug_timer=False) -> Tuple[Dict, Optional[Dict]]:
    record = self.runner.run(image_path)
    return record.model_dump(), record._debug_artifacts
```

## Configuration Management

### Single Config Class

`PipelineConfig` is a Pydantic `BaseSettings` with nested config groups:
`GaugeConfig`, `FClipConfig`, `OCRConfig`, `CorrectionConfig`, `EnhanceConfig`.

### Loading Priority

1. Environment variables (`IQIDET_GAUGE__CONF=0.5`)
2. CLI args (`--gauge-conf 0.5`)
3. YAML files (`config/...`)
4. Code defaults

### CLI Integration

`run_iqi_grade_infer.py` args unchanged. Internally:

```python
config = PipelineConfig().apply_cli_overrides(args)
inferencer = IQIInferencer(config)
```

## Base Class Extraction

### BaseRegionService (`services/base.py`)

Shared by `RegionOCRAPI` and `RegionSNRAPI`:
- `decode_base64(image_base64)` → `np.ndarray` (handles data URL prefix)
- ThreadPoolExecutor singleton (max_workers=2)
- `get_service()` / `close_service()` lifecycle
- `atexit` cleanup registration

### BaseOrientationCorrector (`services/correction.py`)

Shared by `WeldOrientationCorrector` and `OCRTextOrientationCorrector`:
- `STATUS_MAP` (8-class rotation/mirror labels)
- `predict_orientation(image)` → `Tuple[int, float]`
- `correct_image(image, verbose)` → `Tuple[np.ndarray, dict]`
- `restore_image(image, label_idx)` (static, rotation + mirror logic)

Subclasses only override:
- `_get_model_architecture()` → `nn.Module` (resnet variant + Linear head)
- `_get_preprocess()` → `transforms.Compose` (Resize vs SquarePadResize)
- `_prepare_pil_image()` → `Image` (adaptive processor + color conversion)

## Error Handling

### Exception Hierarchy (`exceptions.py`)

```
IQIError (base, result_code=9001)
├── ImageReadError (1001)
├── ROINotFoundError (1101)
├── ROIInvalidError (1102)
├── MarkerError (2003)
│   ├── MarkerMissingJBError (2002)
│   ├── MarkerAmbiguousError (2006)
│   └── MarkerNumberOutOfRangeError (2007)
├── WireInferenceError (3001)
├── WireCountMissingError (3002)
└── GradeError (3005)
IQIStageSkipped — control-flow signal, not an error
```

### PipelineRunner Contract

- `IQIStageSkipped` → skip stage, continue pipeline
- `IQIError` → record error entry, continue pipeline
- `Exception` → record as `internal_error` (9001), STOP pipeline
- Batch loop catches all exceptions, produces error records, continues

## Logging

### Setup (`logging_setup.py`)

- `StructuredFormatter`: JSON lines output with `ts`, `level`, `logger`, `msg` + extra fields
- `setup_logging(level, json_output)`: configures `gauge` logger tree
- Suppresses third-party noise (ultralytics, paddleocr, matplotlib)
- New CLI args: `--log-level` (default INFO), `--log-json`

### Usage Pattern

```python
logger = logging.getLogger(__name__)
logger.info("stage_done", extra={"stage": "roi_detect", "elapsed_ms": 123.4})
```

Logging and `record.warnings`/`record.errors` coexist — logging for ops visibility, record fields for delivery JSON.

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "IQIdet"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch", "numpy", "scipy", "matplotlib", "scikit-image",
    "opencv-python", "PyYAML", "docopt", "tqdm",
    "ultralytics", "optuna", "paddleocr",
    "pydantic>=2.0", "pydantic-settings>=2.0",
]

[project.scripts]
iqi-grade-infer = "gauge.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

After `pip install -e .`, `from gauge.xxx import ...` works without `sys.path` manipulation.

## Non-Breaking Guarantees

1. **`run_iqi_grade_infer.py`**: all 32+ CLI args unchanged; output JSON schema `iqi_grade_batch_v1` unchanged
2. **`region_ocr_api.py`**: `init_region_ocr_api()`, `recognize_region()`, `RecognizeRequest`, `RecognizeResponse`, `RegionOCRService` import paths unchanged
3. **`region_SNR_api.py`**: same pattern, all public names unchanged
4. **`IQIInferencer.infer_image_path()`**: return type `Tuple[Dict, Optional[Dict]]` unchanged
5. **`iqi_rules.py`**: all public functions unchanged
6. **`build_delivery_record()`**: output field names and types unchanged

## Execution Phases

### Phase 1: Infrastructure
- `pyproject.toml`, remove `sys.path` hacks
- `models/` directory with all Pydantic models
- `config/` with `PipelineConfig`
- `logging_setup.py` with structured logging
- `services/base.py` with `BaseRegionService`
- `exceptions.py` with error hierarchy

### Phase 2: Data Types + Error Unification
- Wire existing code to use Pydantic models (return `.model_dump()` for Dict compat)
- Apply unified error handling in `iqi_inferencer.py` batch loop
- Wire `BaseRegionService` into `region_ocr_api.py` and `region_snr_api.py`

### Phase 3: Pipeline Refactoring
- Create 7 Stage classes in `stages/`
- Create `PipelineRunner` in `pipeline.py`
- Simplify `IQIInferencer` to delegate to `PipelineRunner`
- Verify same behavior via existing tests

### Phase 4: Orientation Merge
- Create `services/correction.py` with `BaseOrientationCorrector`
- Refactor `WeldOrientationCorrector` and `OCRTextOrientationCorrector` as subclasses
- Add logging calls throughout

### Verification After Each Phase
```bash
python -m py_compile run_iqi_grade_infer.py region_ocr_api.py region_SNR_api.py
PYTHONPATH=src python -m pytest tests/ -v
```
