# Line-Based Profile Selection — Design Spec

**Date:** 2026-06-03
**Branch:** `refactor/line-profile-selection`
**Status:** implemented

## Overview

Replace 4-point OBB-based profile line selection with direct 2-point line drawing.

### Motivation

- 4-point OBB + offset trackbar is unnecessarily complex — users just want to draw a line across the wires
- OBB unwarp visualization doesn't directly correspond to the profile curve
- Strip image (±expand pixels along the line) is a more natural visual

## Module Changes

| File | Change | Description |
|------|--------|-------------|
| `_dwlib/obb_ui.py` → `_dwlib/line_ui.py` | Rename + rewrite | `OBBSelector` → `LineSelector` |
| `_dwlib/profile_view.py` | Modify | Top panel: OBB → strip image |
| `annotate.py` | Modify | Remove trackbar/offset, add `--expand` |
| `_dwlib/io_utils.py` | Modify | Payload fields, new `save_strip_image()` |
| `validate_bam_gt.py` | Modify | Visual titles, remove OBB references |
| `_dwlib/annotation.py` | No change | Annotation interaction unchanged |
| `src/gauge/imaging/profile.py` | Add function | `extract_profile_strip()` — band grid without averaging |

## Data Flow

```
Old: 4 clicks → fit_obb_and_midline() → compute_profile_line(offset) → extract_profile_band()
New: 2 clicks → extract_profile_band(line_start, line_end)
```

## Component Designs

### LineSelector (`_dwlib/line_ui.py`)

State machine: `IDLE → COLLECTING → LOCKED → (R) → IDLE`

No CONFIRM state — auto-locks on 2nd click.

```python
class LineSelector:
    STATE_IDLE = "idle"
    STATE_COLLECTING = "collecting"
    STATE_LOCKED = "locked"

    # Key attributes
    line_start: tuple[float, float] | None   # raw coords
    line_end: tuple[float, float] | None     # raw coords
    expand: int                                # ±pixels for strip view
    scale_x, scale_y: float                    # raw→display scale
```

**Mouse callback:**
- L-click in IDLE/COLLECTING: add point as line endpoint
- 1st click → COLLECTING; 2nd click → LOCKED (auto-locks)
- RMB in COLLECTING: undo last point, back to IDLE
- RMB in LOCKED: back to COLLECTING (remove endpoint)

**Overlay drawing:**
- Crosshair in IDLE/COLLECTING
- Line between the two points in LOCKED
- Expand zone: two dashed lines at ±expand pixels perpendicular to the profile line
- Status bar text

### Strip Image Extraction (`profile.py`)

New function `extract_profile_strip()`:

```python
def extract_profile_strip(
    image, start_point, end_point,
    expand: int = 100,
    num_samples: int | None = None,
) -> np.ndarray:
    """Extract a 2D strip image along a line, expanded ±expand pixels.

    Returns: float64 array of shape (2*expand, num_samples)
    """
```

Implementation: identical sampling grid as `extract_profile_band`, but returns the 2D grid before averaging.

### ProfileView (`_dwlib/profile_view.py`)

**Top panel (Figure 1):**
- `imshow(strip_image, cmap="gray", aspect="auto")`
- Center line at row `expand` (the profile line)
- Dashed lines at `expand ± band_width/2` showing the averaging band
- Y-axis label: "Perpendicular (px)", centered at 0

**Bottom panel (Figure 2):** Unchanged — profile curve + BAM overlay.

**Title:** Replace "OBB: WxH" with "line: Lpx | expand: N"

### annotate.py

```python
class BAMAnnotator:
    def __init__(self, image_path, output_dir, window_size, band_width, expand=100):
        self.expand = expand
        # No trackbar, no on_trackbar_change callback

    def _update_profile(self):
        line_start = self.line_selector.line_start
        line_end = self.line_selector.line_end
        line_length = int(np.hypot(dx, dy))
        num_samples = max(1, line_length)

        profile = extract_profile_band(
            self.image_raw, line_start, line_end,
            band_width=self.band_width, num_samples=num_samples,
        )
        strip = extract_profile_strip(
            self.image_raw, line_start, line_end,
            expand=self.expand, num_samples=num_samples,
        )
        # ... BAM analysis unchanged ...
        self.view.update(strip, profile, ...)

    def run(self):
        # LineSelector(display, scale_x, scale_y, expand, window_name)
        # No cv2.createTrackbar
        # No profile_offset_pct
```

### io_utils.py

**New:** `save_strip_image(strip, output_path)` — save 2D strip as 8-bit PNG.

**`save_profile_json` payload changes:**

Removed: `obb_corners_raw`, `obb_size`, `obb_points`, `profile_midline`, `profile_offset_pct`

Added:
```json
{
  "profile_line": {"start": [x, y], "end": [x, y]},
  "expand": 100
}
```

Kept: `image_path`, `band_width`, `profile_values`, `bam_*`

### validate_bam_gt.py

- Panel titles: remove OBB dimensions, show line length + expand
- Remove `obb_corners_raw` / `obb_size` references
- Panel 1 uses strip image or just profile curve (no OBB reference needed)
- All validation logic unchanged

## CLI Changes (annotate.py)

```
Options:
    --expand <px>             剖面线上下扩展像素数 [default: 100]
```

## Backward Compatibility

**None.** This is a breaking change:
- Old `*_profile.json` files with `obb_corners_raw` are not readable by new code
- New `*_profile.json` files use `profile_line` instead
- Old outputs should be re-annotated or a migration script written if needed
