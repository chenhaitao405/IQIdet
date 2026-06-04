# 双丝分辨率核心算法统一封装设计

**日期：** 2026-06-04
**状态：** 设计中
**分支：** `refactor/line-profile-selection`

## 目标

将双丝像质计分析算法封装为统一核心接口，供三类消费者共用：

1. **对外 API** — base64 strip 图像输入，JSON 输出
2. **annotate.py** — 交互标注工具
3. **validate_bam_gt.py** — GT 验证工具

算法优化后所有消费者自动受益，稳定版本可直接合并分支交付。

## 核心接口

### `analyze_double_wire()`

位置：`src/gauge/imaging/profile.py`

```python
@dataclass
class DoubleWireResult:
    strip_shape: tuple[int, int]
    profile: np.ndarray
    peaks: list[dict]       # [{"idx": int, "gray": float}]
    valleys: list[dict]
    pairs: list[dict]       # [{"group": int, "wire_a_idx": int, "gap_idx": int,
                            #   "wire_b_idx": int, "wire_a_gray": float,
                            #   "gap_gray": float, "wire_b_gray": float,
                            #   "dip_percent": float}]
    film_type: str          # "positive" | "negative"
    background: np.ndarray
    first_unresolved_group: int | None


def analyze_double_wire(
    strip: np.ndarray,
    *,
    min_distance: int = 5,
    prominence: float = 0.03,
) -> DoubleWireResult:
    """strip 图像 → 列平均 → compute_contrast → find_first_unresolved_group"""
```

逻辑：`strip.mean(axis=0)` → `compute_contrast(profile)` → `find_first_unresolved_group(dips)` → 组装 `DoubleWireResult`。

## 架构

```
                    ┌──────────────────────────────┐
                    │  src/gauge/imaging/profile.py │
                    │  analyze_double_wire()         │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
    ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
    │ API 层       │  │ annotate.py  │  │ validate.py  │
    │ Service +    │  │ strip 切片   │  │ 读 profile   │
    │ FastAPI      │  │ → analyze    │  │ → 重提 strip │
    │              │  │ → 显示/保存  │  │ → analyze    │
    └──────────────┘  └──────────────┘  └──────────────┘
```

### 文件清单

| 层 | 文件 | 变更 |
|----|------|------|
| 核心算法 | `src/gauge/imaging/profile.py` | 新增 `DoubleWireResult` + `analyze_double_wire()` |
| Service | `src/gauge/services/double_wire/service.py` | 新增 `DoubleWireService` |
| API | `src/gauge/app/double_wire_api.py` | 新增 FastAPI 包装 + Pydantic 模型 |
| 门面 | `double_wire_api.py`（根目录） | 新增 import facade |
| Contract | `docs/contract/DOUBLE_WIRE_ANALYSIS_API.md` | 新增 JSON schema |
| annotate | `scripts/double_wire/annotate.py` | 改用 `analyze_double_wire()` |
| validate | `scripts/double_wire/validate_bam_gt.py` | 从原图重提 strip → `analyze_double_wire()` |

### annotate 改动

```
当前: extract_profile_band + compute_contrast + find_first_unresolved_group
      （三个独立调用，分散在 _update_profile 和 _save_one_version）

改为: extract_profile_strip(img, line, expand) → full_strip
      narrow = full_strip[center ± band_width//2]
      analyze_double_wire(narrow) → DoubleWireResult
      view.update(result)   ← 直接用 result 数据画图
      annotator.load_bam_baseline(result.pairs, result.film_type)
```

`_update_profile` 和 `_save_one_version` 都走 `analyze_double_wire`，消除重复调用。

### validate 改动

```
当前: 读 _profile.json → profile_values 数组 → compute_contrast

改为: 读 profile.json → image_path, profile_line, expand, band_width
      load_image(image_path) → raw
      extract_profile_strip(raw, line, expand) → full_strip
      narrow = full_strip[center ± band_width//2]
      analyze_double_wire(narrow) → 与 GT pairs 对比
```

### API 层结构

参考 `region_SNR_api.py` 模式：

```
DoubleWireService.compute(strip: np.ndarray) → dict
    ↓
double_wire_api.py
    DoubleWireRequest:  image_base64: str
    DoubleWireResponse: ok, result_code, result_name, message, timings_ms,
                        strip_shape, film_type, num_pairs, first_unresolved_group,
                        profile, background, peaks, valleys, pairs
    ↓
double_wire_api.py（根目录门面）
```

## JSON 输出契约

详细 schema 见 `docs/contract/DOUBLE_WIRE_ANALYSIS_API.md`。

顶层结构参考 `run_iqi_grade_infer.py`：

```json
{
  "schema": "double_wire_analysis_v1",
  "ok": true,
  "meta": { "created_at": "...", "strip_shape": [42, 519], "algorithm": {...} },
  "result": {
    "film_type": "negative",
    "num_pairs": 7,
    "first_unresolved_group": 8,
    "profile": [...],
    "background": [...],
    "peaks": [...],
    "valleys": [...],
    "pairs": [...]
  }
}
```

`profile`、`background`、`peaks`、`valleys`、`pairs` 全部输出，前端可完整渲染 annotate 中的灰度曲线图及标注。

## 约束

- 核心算法 `analyze_double_wire()` 是纯函数，不依赖 matplotlib / OpenCV HighGUI / FastAPI
- `DoubleWireResult` 含 `np.ndarray`，交给 service 层序列化
- 现有 `compute_contrast()` / `find_first_unresolved_group()` 签名和行为不变
- 不修改 `profile.py` 现有函数的对外接口
