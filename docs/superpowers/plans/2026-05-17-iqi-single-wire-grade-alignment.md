# IQI 单丝型标记与等级规则对齐实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将单丝型像质计的标记解析与等级计算改为与 `docs/contract/IQI_SINGLE_WIRE_MARKER_GRADE_RULE.md` 一致，使用 `general/special` 语义并支持“专用型识别到 1 根即可输出标记丝号等级”。

**Architecture:** 规则集中在 `src/gauge/iqi_rules.py`。先把标记解析从“材料代号决定类型”改成“标记顺序决定类型”，再把 `compute_iqi_grade` 改成按 `general/special` 计算。`src/gauge/iqi_inferencer.py` 只负责透传规则结果，`run_iqi_grade_infer.py` 只保留现有参数流，不引入兼容分支。

**Tech Stack:** Python, unittest, existing `gauge` pipeline.

---

### Task 1: Add regression tests for the new marker contract

**Files:**
- Create: `tests/test_iqi_rules.py`
- Modify: `tests/test_iqi_inferencer.py:1-220` if needed for new `iqi_type` values

- [ ] **Step 1: Write the failing tests**

```python
from gauge.iqi_rules import compute_iqi_grade, infer_plate_from_texts


def test_infer_plate_from_texts_general_marker():
    result = infer_plate_from_texts(["10FEJB"], require_jb=True, allowed_numbers={1, 6, 10, 13})
    assert result["ok"] is True
    assert result["iqi_type"] == "general"
    assert result["number"] == 10
    assert result["plate_code"] == "10FEJB"


def test_infer_plate_from_texts_special_marker():
    result = infer_plate_from_texts(["FE10JB"], require_jb=True, allowed_numbers=set(range(1, 20)))
    assert result["ok"] is True
    assert result["iqi_type"] == "special"
    assert result["number"] == 10
    assert result["plate_code"] == "FE10JB"


def test_compute_iqi_grade_special_requires_only_one_wire():
    result = compute_iqi_grade("special", 10, 1, allowed_numbers=set(range(1, 20)))
    assert result["ok"] is True
    assert result["grade"] == 10
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `PYTHONPATH=/home/cht/code/IQIdet python -m unittest tests.test_iqi_rules -v`

Expected: fail because `general/special` is not yet implemented.

- [ ] **Step 3: Keep the new tests focused on contract behavior**

Add one more regression case for ambiguous or invalid order, for example a text like `FE10` without `JB` should still fail when `require_jb=True`.

- [ ] **Step 4: Leave implementation for Task 2**

### Task 2: Rewrite marker parsing and grade fusion in `src/gauge/iqi_rules.py`

**Files:**
- Modify: `src/gauge/iqi_rules.py:1-700`

- [ ] **Step 1: Replace the old type evidence model**

```python
@dataclass(frozen=True)
class MarkerEvidence:
    iqi_type: str
    material: str
    number: int
    corrections: Tuple[str, ...]
    raw_text: str
    position: int
```

- [ ] **Step 2: Parse ordered markers instead of FE/NI-as-type**

```python
GENERAL_MATERIALS = ("FE", "NI", "SS", "CU", "TI", "AL", "ZR")

# general: number + material + JB
# special: material + number + JB
```

- [ ] **Step 3: Return `iqi_type` as `general` or `special`**

```python
return {
    **build_result_status(0),
    "iqi_type": chosen.iqi_type,
    "number": chosen.number,
    "plate_code": chosen.code,
}
```

- [ ] **Step 4: Update grade calculation**

```python
def compute_iqi_grade(iqi_type, number, wire_count, allowed_numbers=None):
    if iqi_type not in {"general", "special"} or number is None:
        ...
    if iqi_type == "general":
        if wire_count >= 1 and wire_count <= 7:
            grade = int(number) + int(wire_count) - 1
        else:
            ...
    else:
        if wire_count >= 1:
            grade = int(number)
        else:
            ...
```

- [ ] **Step 5: Run the unit tests**

Run: `PYTHONPATH=/home/cht/code/IQIdet python -m unittest tests.test_iqi_rules -v`

Expected: pass.

### Task 3: Sync docs and inferencer-facing expectations

**Files:**
- Modify: `docs/README_IQI_GRADE_INFERENCE_DELIVERY.md:153-166`
- Modify: `tests/test_iqi_delivery_record.py:1-420` only if assertions depend on `iqi_type`

- [ ] **Step 1: Replace the old `FE -> uniform` wording**

```md
- `10FEJB` -> `general`
- `FE10JB` -> `special`
```

- [ ] **Step 2: Keep the pipeline untouched where it only forwards the rule result**

```python
grade_result = compute_iqi_grade(
    selected_plate.get("iqi_type"),
    selected_plate.get("number"),
    wire_result.get("wire_count"),
    allowed_numbers=self.ocr_allowed_numbers,
)
```

- [ ] **Step 3: Run the delivery and inferencer tests**

Run: `PYTHONPATH=/home/cht/code/IQIdet python -m unittest tests.test_iqi_inferencer tests.test_iqi_delivery_record -v`

Expected: pass after test fixtures are updated to `general/special`.

- [ ] **Step 4: Commit**

```bash
git add src/gauge/iqi_rules.py tests/test_iqi_rules.py tests/test_iqi_inferencer.py tests/test_iqi_delivery_record.py docs/README_IQI_GRADE_INFERENCE_DELIVERY.md docs/superpowers/plans/2026-05-17-iqi-single-wire-grade-alignment.md
git commit -m "Align IQI single-wire grading with marker order"
```

