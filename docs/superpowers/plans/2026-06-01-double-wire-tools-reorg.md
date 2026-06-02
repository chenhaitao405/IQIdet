# Double-Wire Tools Reorganization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize three double-wire scripts from `scripts/debug/` into `scripts/double_wire/`, merge OBB selection + annotation into one tool, and lift general-purpose functions to `src/gauge/imaging/profile.py`.

**Architecture:** Four general functions (`normalize_profile_obb`, `bam_pair_marker_indices`, `pair_wire_markers`, `build_groundtruth_payload`) move to `src/gauge/imaging/profile.py`. UI/IO code goes into `scripts/double_wire/src/` modules (`obb_ui`, `profile_view`, `annotation`, `io_utils`). The merged `annotate.py` orchestrates OBB selection → profile view → annotation in a single workflow with OpenCV + matplotlib windows.

**Tech Stack:** Python 3.10+, OpenCV (cv2), matplotlib (TkAgg), numpy, scipy, docopt

---

### Task 1: Add `normalize_profile_obb` and `bam_pair_marker_indices` to profile.py

**Files:**
- Modify: `src/gauge/imaging/profile.py` (append at end of file)

- [ ] **Step 1: Append two functions to profile.py**

Add the following code after line 881 (end of `find_first_unresolved_group`):

```python


# ---------------------------------------------------------------------------
# OBB geometry helpers
# ---------------------------------------------------------------------------

def normalize_profile_obb(
    corners: np.ndarray,
) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Normalize double-wire OBB so profile width follows the long edge.

    The interactive clicks may start on either the long or short rectangle edge.
    The profile scan direction should follow the longer edge across the wires.

    Args:
        corners: 4×2 array of OBB corners in any cyclic order.

    Returns:
        (normalized_corners, midline) where midline is ((start_x, start_y), (end_x, end_y)).
    """
    normalized = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    edge_01 = float(np.linalg.norm(normalized[1] - normalized[0]))
    edge_12 = float(np.linalg.norm(normalized[2] - normalized[1]))

    if edge_01 < edge_12:
        normalized = np.roll(normalized, -1, axis=0)

    tl, tr, br, bl = normalized
    start = (
        float((bl[0] + tl[0]) / 2.0),
        float((bl[1] + tl[1]) / 2.0),
    )
    end = (
        float((tr[0] + br[0]) / 2.0),
        float((tr[1] + br[1]) / 2.0),
    )
    return normalized, (start, end)


def bam_pair_marker_indices(
    pairs: list[tuple[int, int, int]] | list[list[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return BAM wire and gap marker indices for profile plotting.

    Args:
        pairs: List of (wire_a_idx, gap_idx, wire_b_idx) triplets.

    Returns:
        (wire_indices, gap_indices) as sorted numpy arrays.
    """
    if not pairs:
        return np.array([], dtype=int), np.array([], dtype=int)
    wire_indices: list[int] = []
    gap_indices: list[int] = []
    for w1, gap, w2 in pairs:
        wire_indices.extend([int(w1), int(w2)])
        gap_indices.append(int(gap))
    return np.array(sorted(set(wire_indices)), dtype=int), np.array(gap_indices, dtype=int)
```

- [ ] **Step 2: Verify compilation**

```bash
cd /home/cht/code/IQIdet && python -m py_compile src/gauge/imaging/profile.py
```

- [ ] **Step 3: Commit**

```bash
git add src/gauge/imaging/profile.py
git commit -m "feat(profile): add normalize_profile_obb and bam_pair_marker_indices

Extracted from scripts/debug/double_wire_demo.py — these are general-purpose
OBB geometry and BAM data utilities suitable for the shared profile module.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Add `pair_wire_markers` and `build_groundtruth_payload` to profile.py

**Files:**
- Modify: `src/gauge/imaging/profile.py` (append after Task 1 additions)

- [ ] **Step 1: Append GT construction functions to profile.py**

Add after the functions from Task 1:

```python


# ---------------------------------------------------------------------------
# Ground truth construction
# ---------------------------------------------------------------------------

def pair_wire_markers(
    profile_values: np.ndarray,
    wire_markers: list[dict],
    gap_markers: list[dict],
    *,
    wire_type: str,
) -> list[dict]:
    """Pair adjacent wire markers and choose the strongest gap between them.

    Args:
        profile_values: 1-D profile grayscale values.
        wire_markers: Sorted list of ``{"idx": int, "type": "peak"|"valley"}``.
        gap_markers: Sorted list of ``{"idx": int, "type": "peak"|"valley"}``.
        wire_type: ``"peak"`` for positive film (bright wires), ``"valley"``
            for negative film (dark wires). Determines whether the gap is
            selected as the minimum or maximum between two wires.

    Returns:
        List of wire-pair dicts with keys ``group``, ``wire_a_idx``,
        ``gap_idx``, ``wire_b_idx``, ``wire_a_gray``, ``gap_gray``,
        ``wire_b_gray``.
    """
    wire_markers = sorted(wire_markers, key=lambda m: m["idx"])
    gap_markers = sorted(gap_markers, key=lambda m: m["idx"])

    wire_pairs = []
    for i in range(len(wire_markers) - 1):
        w1 = wire_markers[i]
        w2 = wire_markers[i + 1]
        between = [g for g in gap_markers if w1["idx"] < g["idx"] < w2["idx"]]
        if not between:
            continue

        if wire_type == "peak":
            gap_idx = min(between, key=lambda g: profile_values[g["idx"]])["idx"]
        else:
            gap_idx = max(between, key=lambda g: profile_values[g["idx"]])["idx"]

        wire_pairs.append({
            "group": len(wire_pairs) + 1,
            "wire_a_idx": int(w1["idx"]),
            "gap_idx": int(gap_idx),
            "wire_b_idx": int(w2["idx"]),
            "wire_a_gray": float(profile_values[w1["idx"]]),
            "gap_gray": float(profile_values[gap_idx]),
            "wire_b_gray": float(profile_values[w2["idx"]]),
        })

    return wire_pairs


def build_groundtruth_payload(
    profile_values: np.ndarray,
    markers: list[dict],
    *,
    source_profile: str,
    band_width: int,
    film_type: str | None = None,
) -> dict:
    """Build neutral wire/gap ground-truth payload from manual markers.

    Positive film uses bright wires around a dark gap (peak-valley-peak).
    Negative film uses dark wires around a bright gap (valley-peak-valley).

    Args:
        profile_values: 1-D profile grayscale values.
        markers: List of ``{"type": "peak"|"valley", "idx": int}`` dicts.
        source_profile: Path to the source profile JSON (for provenance).
        band_width: Band width used when extracting the profile.
        film_type: ``"positive"``, ``"negative"``, or ``"auto"`` (default).
            Auto selects the type that produces more pairs.

    Returns:
        Ground-truth payload dict with keys ``source_profile``, ``band_width``,
        ``film_type``, ``num_wire_pairs``, ``wire_pairs``, ``all_peaks``,
        ``all_valleys``, ``annotated_at``.
    """
    from datetime import datetime, timezone

    profile_values = np.asarray(profile_values, dtype=np.float64)
    peaks = sorted([m for m in markers if m["type"] == "peak"], key=lambda m: m["idx"])
    valleys = sorted([m for m in markers if m["type"] == "valley"], key=lambda m: m["idx"])

    positive_pairs = pair_wire_markers(
        profile_values, peaks, valleys, wire_type="peak",
    )
    negative_pairs = pair_wire_markers(
        profile_values, valleys, peaks, wire_type="valley",
    )

    if film_type is None or film_type == "auto":
        if len(positive_pairs) >= len(negative_pairs) and positive_pairs:
            film_type = "positive"
            wire_pairs = positive_pairs
        elif negative_pairs:
            film_type = "negative"
            wire_pairs = negative_pairs
        else:
            film_type = "unknown"
            wire_pairs = []
    elif film_type == "positive":
        wire_pairs = positive_pairs
    elif film_type == "negative":
        wire_pairs = negative_pairs
    else:
        raise ValueError(f"Unsupported film_type: {film_type}")

    return {
        "source_profile": str(source_profile),
        "band_width": band_width,
        "film_type": film_type,
        "num_wire_pairs": len(wire_pairs),
        "wire_pairs": wire_pairs,
        "all_peaks": [{"idx": int(m["idx"]), "gray": float(profile_values[m["idx"]])}
                      for m in peaks],
        "all_valleys": [{"idx": int(m["idx"]), "gray": float(profile_values[m["idx"]])}
                        for m in valleys],
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }
```

- [ ] **Step 2: Verify compilation**

```bash
cd /home/cht/code/IQIdet && python -m py_compile src/gauge/imaging/profile.py
```

- [ ] **Step 3: Commit**

```bash
git add src/gauge/imaging/profile.py
git commit -m "feat(profile): add pair_wire_markers and build_groundtruth_payload

Extracted from scripts/debug/annotate_profile.py — GT construction logic
is general-purpose and belongs in the shared profile module.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Create `scripts/double_wire/src/__init__.py` and `io_utils.py`

**Files:**
- Create: `scripts/double_wire/src/__init__.py`
- Create: `scripts/double_wire/src/io_utils.py`

- [ ] **Step 1: Create directory and __init__.py**

```bash
mkdir -p /home/cht/code/IQIdet/scripts/double_wire/src
touch /home/cht/code/IQIdet/scripts/double_wire/src/__init__.py
```

- [ ] **Step 2: Write io_utils.py**

```python
"""File I/O and path utilities for the double-wire annotation tool."""

import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_image(
    image_path: str, window_size: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Load grayscale image and create resized display copy.

    Returns:
        (image_raw, image_display, scale_x, scale_y)
        image_raw: original grayscale (never resized)
        image_display: 8-bit BGR resized for OpenCV window
        scale_x, scale_y: raw -> display scale factors
    """
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    if raw.dtype == np.uint16:
        disp = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif raw.dtype == np.uint8:
        disp = raw.copy()
    else:
        disp = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    h_raw, w_raw = raw.shape[:2]
    long_side = max(disp.shape[0], disp.shape[1])
    if long_side > window_size:
        scale = window_size / long_side
        new_w = max(1, int(disp.shape[1] * scale))
        new_h = max(1, int(disp.shape[0] * scale))
        disp = cv2.resize(disp, (new_w, new_h))

    display = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    scale_x = display.shape[1] / w_raw
    scale_y = display.shape[0] / h_raw

    return raw, display, scale_x, scale_y


def save_obb_image(
    unwarped: np.ndarray,
    obb_size: Tuple[int, int],
    output_path: Path,
) -> None:
    """Save unwarped OBB region as 8-bit PNG."""
    uw, uh = obb_size
    if unwarped.dtype in (np.uint16, np.int32):
        obb_8u = cv2.normalize(unwarped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif unwarped.dtype == np.uint8:
        obb_8u = unwarped
    else:
        obb_8u = cv2.normalize(unwarped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(str(output_path), obb_8u)
    print(f"[save] OBB image ({uw}x{uh}): {output_path}")


def save_overlay_image(overlay: np.ndarray, output_path: Path) -> None:
    """Save overlay visualization as PNG."""
    cv2.imwrite(str(output_path), overlay)
    print(f"[save] Overlay: {output_path}")


def save_profile_json(output_path: Path, payload: dict) -> None:
    """Save profile data as JSON."""
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[save] Profile data: {output_path}")


def save_groundtruth_json(output_path: Path, payload: dict) -> None:
    """Save ground truth data as JSON."""
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[save] Ground truth: {output_path}")


def default_groundtruth_path(profile_json_path: str | Path) -> Path:
    """Return the default GT path next to the input profile JSON.

    ``<stem>_profile.json`` -> ``<stem>_groundtruth.json``
    """
    profile_path = Path(profile_json_path)
    stem = profile_path.stem
    if stem.endswith("_profile"):
        stem = stem[:-len("_profile")] + "_groundtruth"
    else:
        stem = stem + "_groundtruth"
    return profile_path.with_name(stem + ".json")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/double_wire/src/__init__.py scripts/double_wire/src/io_utils.py
git commit -m "feat: add scripts/double_wire/src/io_utils.py

File I/O and path utilities for the double-wire annotation tool.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Create `scripts/double_wire/src/obb_ui.py`

**Files:**
- Create: `scripts/double_wire/src/obb_ui.py`

- [ ] **Step 1: Write obb_ui.py**

```python
"""OBB selection UI — OpenCV window for interactive region-of-interest selection."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import cv2
import numpy as np

# Color constants (BGR for OpenCV)
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


def compute_profile_line(
    obb_corners: np.ndarray,
    midline: Tuple[Tuple[float, float], Tuple[float, float]],
    offset_pct: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute the profile scan line from OBB geometry and trackbar offset.

    offset_pct=50 yields the centerline; 0/100 are at the edges.
    """
    obb = np.asarray(obb_corners, dtype=np.float64)
    start, end = midline
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return ((sx, sy), (ex, ey))

    px = -dy / length
    py = dx / length

    side_e0 = float(np.linalg.norm(obb[0] - obb[3]))
    side_e1 = float(np.linalg.norm(obb[1] - obb[2]))
    offset_side = (side_e0 + side_e1) / 2.0
    offset_frac = (offset_pct - 50) / 50.0
    offset_amount = offset_frac * offset_side * 0.45

    sx_off = sx + offset_amount * px
    sy_off = sy + offset_amount * py
    ex_off = ex + offset_amount * px
    ey_off = ey + offset_amount * py

    return ((sx_off, sy_off), (ex_off, ey_off))


class OBBSelector:
    """Manages OpenCV window for OBB point selection and band visualization.

    State machine: IDLE -> COLLECTING -> CONFIRM -> LOCKED -> (R key) -> IDLE
    """

    STATE_IDLE = "idle"
    STATE_COLLECTING = "collecting"
    STATE_CONFIRM = "confirm"
    STATE_LOCKED = "locked"

    def __init__(
        self,
        image_display: np.ndarray,
        scale_x: float,
        scale_y: float,
        band_width: int = 21,
        on_trackbar_change: Optional[Callable[[int], None]] = None,
        window_name: str = "Double Wire Demo",
    ):
        self.image_display = image_display
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.band_width = int(band_width)
        if self.band_width < 1:
            raise ValueError(f"band_width must be >= 1, got {self.band_width}")
        self.window_name = window_name
        self.trackbar_name = "offset%"

        self.state: str = self.STATE_IDLE
        self.obb_points: list[tuple[float, float]] = []
        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None
        self.profile_offset_pct: int = 50
        self.show_help: bool = False

        self.obb_corners_raw: Optional[np.ndarray] = None
        self.obb_midline_raw: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.obb_corners_disp: Optional[list[tuple[float, float]]] = None

        self._on_trackbar_change = on_trackbar_change

    # -- Mouse callback -----------------------------------------------------

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.state in (self.STATE_IDLE, self.STATE_COLLECTING):
                self.obb_points.append((float(x), float(y)))
                if self.state == self.STATE_IDLE:
                    self.state = self.STATE_COLLECTING
                if len(self.obb_points) >= 4:
                    self.state = self.STATE_CONFIRM
                    self._fit_obb()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.state == self.STATE_COLLECTING and self.obb_points:
                self.obb_points.pop()
                if not self.obb_points:
                    self.state = self.STATE_IDLE
            elif self.state == self.STATE_CONFIRM:
                self.obb_points.pop()
                self.obb_corners_raw = None
                self.obb_midline_raw = None
                self.obb_corners_disp = None
                self.state = self.STATE_COLLECTING if self.obb_points else self.STATE_IDLE

    # -- Trackbar -----------------------------------------------------------

    def on_trackbar(self, value: int) -> None:
        self.profile_offset_pct = value
        if self._on_trackbar_change is not None:
            self._on_trackbar_change(value)

    # -- OBB fitting --------------------------------------------------------

    def _fit_obb(self) -> None:
        from gauge.imaging.profile import fit_obb_and_midline, normalize_profile_obb

        raw_pts = [(x / self.scale_x, y / self.scale_y) for x, y in self.obb_points]
        pts = np.array(raw_pts, dtype=np.float32)
        self.obb_corners_raw, self.obb_midline_raw = fit_obb_and_midline(pts)
        self.obb_corners_raw, self.obb_midline_raw = normalize_profile_obb(
            self.obb_corners_raw
        )
        self.obb_corners_disp = [
            (x * self.scale_x, y * self.scale_y)
            for x, y in self.obb_corners_raw
        ]

    # -- Reset --------------------------------------------------------------

    def reset(self) -> None:
        self.state = self.STATE_IDLE
        self.obb_points.clear()
        self.obb_corners_raw = None
        self.obb_midline_raw = None
        self.obb_corners_disp = None
        self.profile_offset_pct = 50
        print("[annotate] Reset")

    # -- Overlay drawing ----------------------------------------------------

    def draw_overlay(self, annotating: bool = False) -> np.ndarray:
        """Draw the current overlay (crosshair, OBB polygon, band, status)."""
        vis = self.image_display.copy()

        if self.state in (self.STATE_IDLE, self.STATE_COLLECTING):
            if self.mouse_x is not None and self.mouse_y is not None:
                h_img, w_img = vis.shape[:2]
                cv2.line(vis, (self.mouse_x, 0), (self.mouse_x, h_img - 1),
                         COLOR_YELLOW, 1, cv2.LINE_AA)
                cv2.line(vis, (0, self.mouse_y), (w_img - 1, self.mouse_y),
                         COLOR_YELLOW, 1, cv2.LINE_AA)

        if len(self.obb_points) >= 1:
            pts_int = [(int(x), int(y)) for x, y in self.obb_points]
            for pt in pts_int:
                cv2.circle(vis, pt, 5, COLOR_YELLOW, -1, cv2.LINE_AA)
            for i in range(len(pts_int) - 1):
                cv2.line(vis, pts_int[i], pts_int[i + 1], COLOR_YELLOW, 1, cv2.LINE_AA)

        if self.state in (self.STATE_CONFIRM, self.STATE_LOCKED) and self.obb_corners_disp:
            pts_int = [(int(x), int(y)) for x, y in self.obb_corners_disp]
            cv2.polylines(vis, [np.array(pts_int)], isClosed=True, color=COLOR_GREEN,
                          thickness=2, lineType=cv2.LINE_AA)
            for x, y in self.obb_points:
                cv2.circle(vis, (int(x), int(y)), 3, COLOR_YELLOW, -1, cv2.LINE_AA)

        if self.state in (self.STATE_CONFIRM, self.STATE_LOCKED) and self.obb_midline_raw is not None:
            (sx, sy), (ex, ey) = self.obb_midline_raw
            sx_d = sx * self.scale_x
            sy_d = sy * self.scale_y
            ex_d = ex * self.scale_x
            ey_d = ey * self.scale_y
            ddx = ex_d - sx_d
            ddy = ey_d - sy_d
            plen = np.hypot(ddx, ddy)
            if plen > 1e-6:
                ppx = -ddy / plen
                ppy = ddx / plen
                half_band = (self.band_width - 1) / 2.0
                s1 = (int(sx_d + half_band * ppx), int(sy_d + half_band * ppy))
                e1 = (int(ex_d + half_band * ppx), int(ey_d + half_band * ppy))
                s2 = (int(sx_d - half_band * ppx), int(sy_d - half_band * ppy))
                e2 = (int(ex_d - half_band * ppx), int(ey_d - half_band * ppy))
                overlay = vis.copy()
                band_pts = np.array([s1, e1, e2, s2], dtype=np.int32)
                cv2.fillPoly(overlay, [band_pts], (0, 0, 255))
                vis = cv2.addWeighted(overlay, 0.2, vis, 0.8, 0)
                cv2.line(vis, s1, e1, COLOR_RED, 1, cv2.LINE_AA)
                cv2.line(vis, s2, e2, COLOR_RED, 1, cv2.LINE_AA)
                mid_s = (int(sx_d), int(sy_d))
                mid_e = (int(ex_d), int(ey_d))
                cv2.line(vis, mid_s, mid_e, COLOR_RED, 1, cv2.LINE_AA)

        status_map = {
            self.STATE_IDLE: "Ready",
            self.STATE_COLLECTING: f"Points {len(self.obb_points)}/4",
            self.STATE_CONFIRM: "CONFIRM - Enter to accept, R to retry, RMB to undo",
            self.STATE_LOCKED: (
                f"Locked | offset={self.profile_offset_pct}% | band={self.band_width}"
                f"{' | [ANNOTATING]' if annotating else ''}"
                f" | A=annotate  S=save  Q=quit"
            ),
        }
        status = status_map.get(self.state, "")
        cv2.putText(vis, status, (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

        if self.show_help:
            help_lines = [
                "L-click: add OBB vertex (4 to confirm)",
                "R-click: undo last vertex",
                "R: reset OBB   A: annotate   S: save   Q/ESC: quit",
                "H: hide help",
                f"Trackbar: adjust offset | band_width={self.band_width}",
            ]
            overlay = vis.copy()
            panel_h = 20 * len(help_lines) + 20
            h_img, w_img = overlay.shape[:2]
            cv2.rectangle(overlay, (10, 30), (min(520, w_img - 10), 30 + panel_h),
                          COLOR_BLACK, -1)
            vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)
            for i, line in enumerate(help_lines):
                cv2.putText(vis, line, (20, 55 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

        return vis
```

- [ ] **Step 2: Commit**

```bash
git add scripts/double_wire/src/obb_ui.py
git commit -m "feat: add scripts/double_wire/src/obb_ui.py

OBBSelector class for OpenCV-based OBB selection and compute_profile_line
utility. Extracted from scripts/debug/double_wire_demo.py.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Create `scripts/double_wire/src/profile_view.py`

**Files:**
- Create: `scripts/double_wire/src/profile_view.py`

- [ ] **Step 1: Write profile_view.py**

```python
"""Profile visualization — matplotlib figure for OBB image + profile curve."""

from __future__ import annotations

from typing import Optional

import matplotlib
try:
    matplotlib.use("TkAgg", force=True)
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np

PROFILE_COLOR = "#4C78A8"


class ProfileView:
    """Manages the matplotlib figure for profile + OBB image display.

    Owns figure creation, profile curve rendering, and BAM dip overlay.
    Annotation markers are drawn separately by the Annotator.
    """

    def __init__(self):
        self.fig: plt.Figure = plt.figure(figsize=(10, 6))
        self.ax_top: Optional[plt.Axes] = None
        self.ax_bottom: Optional[plt.Axes] = None

    def update(
        self,
        image_raw: np.ndarray,
        obb_corners: np.ndarray,
        profile: np.ndarray,
        bam_result,  # ComputeContrastResult
        unresolved_group: Optional[int],
        band_width: int,
        offset_pct: int,
        image_stem: str,
    ) -> None:
        """Update the figure with current profile and BAM analysis results."""
        from gauge.imaging.profile import unwarp_obb_region, bam_pair_marker_indices

        self.fig.clear()
        self.ax_top = self.fig.add_subplot(2, 1, 1)
        self.ax_bottom = self.fig.add_subplot(2, 1, 2, sharex=self.ax_top)

        # Top: unwarped OBB band image
        unwarped, (uw, uh) = unwarp_obb_region(image_raw, obb_corners)
        self.ax_top.imshow(unwarped, cmap="gray", aspect="auto")

        band_y = uh / 2 + (offset_pct - 50) / 50.0 * uh * 0.45
        half_band = (band_width - 1) / 2.0
        self.ax_top.axhline(y=band_y, color="red", linewidth=1.0)
        self.ax_top.axhline(y=band_y - half_band, color="red", linewidth=0.5, linestyle="--")
        self.ax_top.axhline(y=band_y + half_band, color="red", linewidth=0.5, linestyle="--")
        self.ax_top.set_ylabel("Across wires (px)")

        # Bottom: profile curve
        x = np.arange(len(profile))
        self.ax_bottom.plot(x, profile, color=PROFILE_COLOR, linewidth=1.2, label="Profile")

        prof_range = max(float(profile.max() - profile.min()), 1.0)
        y_min = float(profile.min()) - 0.05 * prof_range
        y_max = float(profile.max()) + 0.05 * prof_range

        if bam_result is not None and len(bam_result.pairs) > 0:
            dips = bam_result.dips
            pairs = bam_result.pairs
            bam_wires, bam_gaps = bam_pair_marker_indices(pairs)
            self.ax_bottom.plot(
                bam_wires, profile[bam_wires],
                "co", markersize=5, fillstyle="none", markeredgewidth=1.2,
                label="BAM wires",
            )
            self.ax_bottom.plot(
                bam_gaps, profile[bam_gaps],
                "mo", markersize=5, fillstyle="none", markeredgewidth=1.2,
                label="BAM gaps",
            )
            for i, ((w1, g, w2), dip) in enumerate(zip(pairs, dips)):
                color = "green" if dip >= 20.0 else "orange"
                self.ax_bottom.axvspan(w1, w2, alpha=0.12, color=color)
                mid = (w1 + w2) // 2
                self.ax_bottom.annotate(
                    f"D{i+1}:{dip:.0f}%", (mid, profile[g]),
                    textcoords="offset points", xytext=(0, 16),
                    fontsize=6, color=color, ha="center",
                )

        self.ax_bottom.set_ylim(y_min, y_max)
        self.ax_bottom.set_xlabel("Profile position (px)")
        self.ax_bottom.set_ylabel("Gray value")
        self.ax_bottom.legend(fontsize=7, loc="upper right")

        if bam_result is not None:
            title = (
                f"{image_stem} | OBB: {uw}x{uh} | offset:{offset_pct}%"
                f" | band:{band_width} | film:{bam_result.film_type}"
                f" | pairs:{len(bam_result.pairs)}"
            )
            if unresolved_group is not None:
                title += f" | 1st unres.:D{unresolved_group}"
        else:
            title = f"{image_stem} | OBB: {uw}x{uh} | offset:{offset_pct}% | band:{band_width}"
        self.fig.suptitle(title, fontsize=9)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @property
    def profile_axes(self):
        return self.ax_bottom

    @property
    def profile_ylim(self) -> Optional[Tuple[float, float]]:
        if self.ax_bottom is not None:
            return self.ax_bottom.get_ylim()
        return None

    def clear(self) -> None:
        self.fig.clear()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.ax_top = None
        self.ax_bottom = None
```

- [ ] **Step 2: Commit**

```bash
git add scripts/double_wire/src/profile_view.py
git commit -m "feat: add scripts/double_wire/src/profile_view.py

ProfileView class for matplotlib-based profile + OBB image display.
Extracted from scripts/debug/double_wire_demo.py plot_profile method.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Create `scripts/double_wire/src/annotation.py`

**Files:**
- Create: `scripts/double_wire/src/annotation.py`

- [ ] **Step 1: Write annotation.py**

```python
"""Ground truth annotation interaction — matplotlib event handlers."""

from __future__ import annotations

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

PEAK_COLOR = "#d62728"
VALLEY_COLOR = "#1f77b4"
PEAK_MARKER = "v"
VALLEY_MARKER = "^"


class Annotator:
    """Manages annotation markers and matplotlib event handlers.

    Designed to be activated/deactivated by the orchestrator. When active,
    click and key events on the matplotlib figure are intercepted for
    peak/valley marking.
    """

    MODE_PEAK = "peak"
    MODE_VALLEY = "valley"

    def __init__(
        self,
        profile_values: np.ndarray,
        band_width: int,
        on_save: Optional[Callable[[], None]] = None,
        on_toggle: Optional[Callable[[], None]] = None,
        on_quit: Optional[Callable[[], None]] = None,
    ):
        self.profile_values = np.asarray(profile_values, dtype=np.float64)
        self.band_width = band_width
        self.mode: str = self.MODE_VALLEY
        self.markers: list[dict] = []

        # Callbacks
        self._on_save = on_save
        self._on_toggle = on_toggle
        self._on_quit = on_quit

        # Matplotlib event connection IDs (set on activate, cleared on deactivate)
        self._click_cid: Optional[int] = None
        self._key_cid: Optional[int] = None
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None

    # -- Marker management --------------------------------------------------

    def add_marker(self, idx: int) -> None:
        self.markers.append({"type": self.mode, "idx": int(idx)})

    def remove_nearest(self, x: float) -> bool:
        if not self.markers:
            return False
        dists = [abs(m["idx"] - x) for m in self.markers]
        nearest = int(np.argmin(dists))
        threshold = max(3, len(self.profile_values) * 0.01)
        if dists[nearest] < threshold:
            removed = self.markers.pop(nearest)
            print(f"[annotate] Removed {removed['type']} at idx={removed['idx']}")
            return True
        return False

    def undo_last(self) -> None:
        if self.markers:
            removed = self.markers.pop()
            print(f"[annotate] Undo: removed {removed['type']} at idx={removed['idx']}")

    def clear_markers(self) -> None:
        self.markers.clear()

    # -- Ground truth -------------------------------------------------------

    def build_groundtruth(self, source_profile: str) -> dict:
        from gauge.imaging.profile import build_groundtruth_payload
        return build_groundtruth_payload(
            self.profile_values,
            self.markers,
            source_profile=source_profile,
            band_width=self.band_width,
        )

    # -- Drawing ------------------------------------------------------------

    def draw_markers(self, ax: plt.Axes) -> None:
        """Draw annotation markers on the given axes."""
        peak_idxs = [m["idx"] for m in self.markers if m["type"] == "peak"]
        valley_idxs = [m["idx"] for m in self.markers if m["type"] == "valley"]

        if peak_idxs:
            ax.plot(
                peak_idxs, self.profile_values[peak_idxs],
                PEAK_MARKER, color=PEAK_COLOR, markersize=10,
                markeredgecolor="black", markeredgewidth=0.5,
                label="peaks (manual)",
            )
            for idx in peak_idxs:
                ax.annotate(
                    str(idx), (idx, self.profile_values[idx]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, color=PEAK_COLOR, ha="center",
                )

        if valley_idxs:
            ax.plot(
                valley_idxs, self.profile_values[valley_idxs],
                VALLEY_MARKER, color=VALLEY_COLOR, markersize=10,
                markeredgecolor="black", markeredgewidth=0.5,
                label="valleys (manual)",
            )
            for idx in valley_idxs:
                ax.annotate(
                    str(idx), (idx, self.profile_values[idx]),
                    textcoords="offset points", xytext=(0, -12),
                    fontsize=7, color=VALLEY_COLOR, ha="center",
                )

    # -- Activation ---------------------------------------------------------

    def activate(self, fig: plt.Figure, ax: plt.Axes) -> None:
        """Connect event handlers and start annotation mode."""
        self._fig = fig
        self._ax = ax
        self._click_cid = fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._key_cid = fig.canvas.mpl_connect("key_press_event", self._on_key)
        print("[annotate] Annotation mode ON (p=peak, v=valley, u=undo)")

    def deactivate(self) -> None:
        """Disconnect event handlers and exit annotation mode."""
        if self._click_cid is not None and self._fig is not None:
            self._fig.canvas.mpl_disconnect(self._click_cid)
        if self._key_cid is not None and self._fig is not None:
            self._fig.canvas.mpl_disconnect(self._key_cid)
        self._click_cid = None
        self._key_cid = None
        self._fig = None
        self._ax = None
        print("[annotate] Annotation mode OFF")

    @property
    def is_active(self) -> bool:
        return self._click_cid is not None

    # -- Event handlers -----------------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes != self._ax:
            return
        if event.xdata is None:
            return
        idx = int(round(event.xdata))
        idx = max(0, min(idx, len(self.profile_values) - 1))

        if event.button == MouseButton.LEFT:
            self.add_marker(idx)
            print(f"[annotate] Added {self.mode} at idx={idx}, gray={self.profile_values[idx]:.2f}")
            # Redraw: orchestrator calls redraw after event
        elif event.button == MouseButton.RIGHT:
            if self.remove_nearest(event.xdata):
                pass  # Redraw handled by orchestrator

    def _on_key(self, event) -> None:
        if event.key == "p":
            self.mode = self.MODE_PEAK
            print("[annotate] Mode: PEAK")
        elif event.key == "v":
            self.mode = self.MODE_VALLEY
            print("[annotate] Mode: VALLEY")
        elif event.key == "u":
            self.undo_last()
        elif event.key in ("a", "escape"):
            if self._on_toggle:
                self._on_toggle()
        elif event.key == "s":
            if self._on_save:
                self._on_save()
        elif event.key in ("q",):
            if self._on_quit:
                self._on_quit()

    # -- Title info ---------------------------------------------------------

    def build_title_suffix(self, image_stem: str) -> str:
        n_peaks = sum(1 for m in self.markers if m["type"] == "peak")
        n_valleys = sum(1 for m in self.markers if m["type"] == "valley")
        mode_label = {"peak": "PEAK", "valley": "VALLEY"}[self.mode]
        return (
            f"{image_stem} | mode: [{mode_label}] | "
            f"peaks: {n_peaks}, valleys: {n_valleys} | "
            f"band_width={self.band_width}"
        )
```

- [ ] **Step 2: Commit**

```bash
git add scripts/double_wire/src/annotation.py
git commit -m "feat: add scripts/double_wire/src/annotation.py

Annotator class for interactive peak/valley marking on profile plots.
Extracted and adapted from scripts/debug/annotate_profile.py.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Create `scripts/double_wire/annotate.py` (merged orchestrator)

**Files:**
- Create: `scripts/double_wire/annotate.py`

- [ ] **Step 1: Write annotate.py**

```python
#!/usr/bin/env python3
"""Interactive BAM double-wire IQI annotation tool.

Combines OBB selection, profile visualization, and ground-truth annotation
in a single unified workflow.  Press **A** after locking the OBB to enter
annotation mode; press **S** to save both profile data and ground truth.

Usage:
    annotate.py <image_path> [options]
    annotate.py (-h | --help)

Arguments:
    <image_path>              双丝像质计图像路径

Options:
    -h --help                 显示帮助信息
    --output-dir <dir>        输出目录 [default: outputs/double_wire_demo]
    --window-size <size>      显示窗口最大尺寸 [default: 1200]
    --band-width <N>          剖面带平行线数量 [default: 21]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np

try:
    matplotlib.use("TkAgg", force=True)
except Exception:
    pass
import matplotlib.pyplot as plt
from docopt import docopt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Add scripts/double_wire/src to path for local modules
_LOCAL_SRC = Path(__file__).resolve().parent / "src"
if str(_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SRC))

from src.io_utils import (
    load_image,
    save_obb_image,
    save_overlay_image,
    save_profile_json,
    save_groundtruth_json,
)
from src.obb_ui import OBBSelector, compute_profile_line
from src.profile_view import ProfileView
from src.annotation import Annotator

from gauge.imaging.profile import (
    extract_profile_band,
    unwarp_obb_region,
    compute_contrast,
    find_first_unresolved_group,
)


class BAMAnnotator:
    """Orchestrates OBB selection, profile view, and ground-truth annotation."""

    def __init__(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        window_size: int = 1200,
        band_width: int = 21,
    ):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir) if output_dir else None
        self.window_size = int(window_size)
        self.band_width = int(band_width)

        # Image data (loaded in run())
        self.image_raw: Optional[np.ndarray] = None

        # Components (created in run())
        self.obb: Optional[OBBSelector] = None
        self.view: Optional[ProfileView] = None
        self.annotator: Optional[Annotator] = None

        # Cached profile data for save
        self._profile: Optional[np.ndarray] = None
        self._bam_result = None
        self._unresolved: Optional[int] = None
        self._profile_line = None
        self._uw: int = 0
        self._uh: int = 0

        # Annotation flag
        self._annotating: bool = False

    # -- Profile update ----------------------------------------------------

    def _update_profile(self) -> None:
        """Extract profile, run BAM analysis, update plot."""
        if (self.obb is None or self.view is None
                or self.obb.obb_corners_raw is None
                or self.obb.obb_midline_raw is None):
            return

        profile_line = compute_profile_line(
            self.obb.obb_corners_raw,
            self.obb.obb_midline_raw,
            self.obb.profile_offset_pct,
        )
        _, (uw, uh) = unwarp_obb_region(self.image_raw, self.obb.obb_corners_raw)

        profile = extract_profile_band(
            self.image_raw, profile_line[0], profile_line[1],
            band_width=self.band_width, num_samples=uw,
        )
        result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
        unresolved = find_first_unresolved_group(result.dips)

        self.view.update(
            self.image_raw, self.obb.obb_corners_raw, profile,
            result, unresolved, self.band_width,
            self.obb.profile_offset_pct, self.image_path.stem,
        )

        if self._annotating and self.annotator is not None:
            self.annotator.draw_markers(self.view.profile_axes)
            self.view.fig.canvas.draw()
            self.view.fig.canvas.flush_events()

        self._profile = profile
        self._bam_result = result
        self._unresolved = unresolved
        self._profile_line = profile_line
        self._uw = uw
        self._uh = uh

    # -- Annotation toggle -------------------------------------------------

    def _toggle_annotation(self) -> None:
        if self.view is None or self.annotator is None:
            return
        if self._annotating:
            self.annotator.deactivate()
            self._annotating = False
            # Restore normal profile view
            self._update_profile()
        else:
            if self.view.profile_axes is None:
                print("[annotate] No profile axes available. Lock OBB first.")
                return
            self.annotator.activate(self.view.fig, self.view.profile_axes)
            self._annotating = True
            self.annotator.draw_markers(self.view.profile_axes)
            self.view.fig.canvas.draw()
            self.view.fig.canvas.flush_events()

    # -- Save --------------------------------------------------------------

    def _save(self) -> None:
        if self.output_dir is None:
            print("[save] No output directory configured. Skipping.")
            return
        if self.obb is None or self.obb.obb_corners_raw is None:
            print("[save] No OBB fitted. Lock OBB first before saving.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem

        # OBB image
        unwarped, (uw, uh) = unwarp_obb_region(self.image_raw, self.obb.obb_corners_raw)
        obb_path = self.output_dir / f"{stem}_obb.png"
        save_obb_image(unwarped, (uw, uh), obb_path)

        # Overlay image
        overlay_path = self.output_dir / f"{stem}_overlay.png"
        overlay = self.obb.draw_overlay(annotating=self._annotating)
        save_overlay_image(overlay, overlay_path)

        # Profile JSON
        if self._profile is not None:
            profile_path = self.output_dir / f"{stem}_profile.json"
            payload = {
                "image_path": str(self.image_path),
                "obb_corners_raw": self.obb.obb_corners_raw.tolist(),
                "obb_size": {"width": uw, "height": uh},
                "obb_points": [[float(x), float(y)] for x, y in self.obb.obb_points],
                "profile_midline": {
                    "start": list(self._profile_line[0]) if self._profile_line else None,
                    "end": list(self._profile_line[1]) if self._profile_line else None,
                },
                "band_width": self.band_width,
                "profile_offset_pct": self.obb.profile_offset_pct,
                "profile_values": self._profile.tolist(),
            }
            if self._bam_result is not None:
                payload["bam_film_type"] = self._bam_result.film_type
                payload["bam_dips"] = self._bam_result.dips
                payload["bam_pairs"] = [
                    [int(w1), int(g), int(w2)]
                    for w1, g, w2 in self._bam_result.pairs
                ]
                payload["bam_unresolved_group"] = self._unresolved
            save_profile_json(profile_path, payload)

            # Ground truth JSON (if markers exist)
            if self.annotator is not None and self.annotator.markers:
                gt_path = self.output_dir / f"{stem}_groundtruth.json"
                gt_payload = self.annotator.build_groundtruth(str(profile_path))
                save_groundtruth_json(gt_path, gt_payload)
                print(f"  Film type: {gt_payload['film_type']}")
                print(f"  Wire pairs: {gt_payload['num_wire_pairs']}")
                for wp in gt_payload["wire_pairs"]:
                    print(
                        f"    D{wp['group']:2d}: wire_a={wp['wire_a_idx']:4d}, "
                        f"gap={wp['gap_idx']:4d}, wire_b={wp['wire_b_idx']:4d}"
                    )
            else:
                print("[save] No annotation markers — skipping groundtruth.json")

    # -- Main loop ---------------------------------------------------------

    def run(self) -> None:
        # Load image
        self.image_raw, display, scale_x, scale_y = load_image(
            str(self.image_path), self.window_size,
        )

        # Create components
        self.obb = OBBSelector(
            display, scale_x, scale_y, self.band_width,
            on_trackbar_change=lambda v: self._update_profile(),
        )
        self.view = ProfileView()

        # Create annotator (profile_values set after first profile extraction)
        self.annotator = Annotator(
            profile_values=np.array([]),  # Will be updated
            band_width=self.band_width,
            on_save=self._save,
            on_toggle=self._toggle_annotation,
            on_quit=lambda: None,  # Handled by OpenCV loop
        )

        # OpenCV window setup
        cv2.namedWindow(self.obb.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.obb.window_name, self.obb.mouse_callback)
        cv2.createTrackbar(
            self.obb.trackbar_name, self.obb.window_name, 50, 100, self.obb.on_trackbar,
        )

        plt.ion()

        print(f"[annotate] Loaded: {self.image_path.name}")
        print(f"[annotate] Shape: {self.image_raw.shape}, dtype: {self.image_raw.dtype}")
        print(f"[annotate] band_width: {self.band_width}")
        print("[annotate] L-click=add point  Enter=lock  R=reset  A=annotate  S=save  Q=quit  H=help")

        while True:
            overlay = self.obb.draw_overlay(annotating=self._annotating)
            cv2.imshow(self.obb.window_name, overlay)

            key = cv2.waitKey(30) & 0xFF

            if key == 13 or key == 32:  # Enter or Space
                if self.obb.state == OBBSelector.STATE_CONFIRM:
                    self.obb.state = OBBSelector.STATE_LOCKED
                    self._update_profile()
            elif key == ord("r"):
                if self._annotating:
                    self._toggle_annotation()  # Exit annotation first
                self.obb.reset()
                self.view.clear()
                self._annotating = False
                cv2.setTrackbarPos(self.obb.trackbar_name, self.obb.window_name, 50)
            elif key == ord("a"):
                if self.obb.state == OBBSelector.STATE_LOCKED:
                    self._toggle_annotation()
            elif key == ord("s"):
                if self.obb.state == OBBSelector.STATE_LOCKED:
                    self._save()
                else:
                    print("[annotate] Lock OBB first (complete 4 points, press Enter)")
            elif key == ord("q") or key == 27:
                print("[annotate] Quit")
                break
            elif key == ord("h"):
                self.obb.show_help = not self.obb.show_help

        cv2.destroyAllWindows()
        plt.close("all")


def main() -> None:
    args = docopt(__doc__)
    image_path = args["<image_path>"]
    output_dir = args["--output-dir"]
    window_size = int(args["--window-size"])
    band_width = int(args["--band-width"])

    if not Path(image_path).is_file():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    app = BAMAnnotator(
        image_path=image_path,
        output_dir=output_dir,
        window_size=window_size,
        band_width=band_width,
    )
    app.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify compilation**

```bash
cd /home/cht/code/IQIdet && python -m py_compile scripts/double_wire/annotate.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/double_wire/annotate.py
git commit -m "feat: add scripts/double_wire/annotate.py

Merged OBB selection (double_wire_demo.py) + GT annotation (annotate_profile.py)
into a single unified tool. Press A to annotate after locking OBB, S saves both
profile and groundtruth JSON.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Move `validate_bam_gt.py` to `scripts/double_wire/`

**Files:**
- Create: `scripts/double_wire/validate_bam_gt.py`
- Delete: `scripts/debug/validate_bam_gt.py`

- [ ] **Step 1: Copy and update the script**

The script is copied verbatim — `REPO_ROOT = Path(__file__).resolve().parents[2]` resolves correctly from `scripts/double_wire/` (3 levels up → repo root).

```bash
cp /home/cht/code/IQIdet/scripts/debug/validate_bam_gt.py /home/cht/code/IQIdet/scripts/double_wire/validate_bam_gt.py
```

- [ ] **Step 2: Verify compilation**

```bash
cd /home/cht/code/IQIdet && python -m py_compile scripts/double_wire/validate_bam_gt.py
```

- [ ] **Step 3: Delete old file and commit**

```bash
git rm scripts/debug/validate_bam_gt.py
git add scripts/double_wire/validate_bam_gt.py
git commit -m "refactor: move validate_bam_gt.py to scripts/double_wire/

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Update tests

**Files:**
- Modify: `tests/test_double_wire_demo.py`
- Modify: `tests/test_annotate_profile.py`

- [ ] **Step 1: Update test_double_wire_demo.py**

Replace the import block (lines 1-11):

```python
import unittest

import numpy as np

try:
    from gauge.imaging.profile import bam_pair_marker_indices, normalize_profile_obb
except ImportError as exc:
    bam_pair_marker_indices = None
    normalize_profile_obb = None


@unittest.skipIf(bam_pair_marker_indices is None, "gauge.imaging.profile not importable")
class TestDoubleWireDemoMarkers(unittest.TestCase):
    def test_bam_pair_marker_indices_include_both_wire_points(self):
        pairs = [[10, 16, 22], [86, 90, 94], [243, 244, 245]]

        wires, gaps = bam_pair_marker_indices(pairs)

        self.assertEqual(wires.tolist(), [10, 22, 86, 94, 243, 245])
        self.assertEqual(gaps.tolist(), [16, 90, 244])


@unittest.skipIf(normalize_profile_obb is None, "gauge.imaging.profile not importable")
class TestDoubleWireDemoOBB(unittest.TestCase):
    # ... rest of test methods unchanged ...
```

The test methods (lines 27-84) remain identical — only the import and skip condition change.

- [ ] **Step 2: Update test_annotate_profile.py**

Replace lines 1-6:

```python
import unittest

import numpy as np
from docopt import docopt

from gauge.imaging.profile import build_groundtruth_payload
from scripts.double_wire.src.io_utils import default_groundtruth_path


class AnnotateProfilePathTest(unittest.TestCase):
    def test_default_groundtruth_path_is_next_to_profile_json(self) -> None:
        path = default_groundtruth_path(
            "outputs/double_wire_demo_3/sample_profile.json"
        )

        self.assertEqual(
            str(path),
            "outputs/double_wire_demo_3/sample_groundtruth.json",
        )

    def test_docopt_does_not_treat_help_text_as_output_default(self) -> None:
        # Test the annotate.py docstring instead
        from scripts.double_wire import annotate
        args = docopt(
            annotate.__doc__,
            argv=["outputs/double_wire_demo_3/sample.jpg"],
        )

        self.assertIsNone(args["--output-dir"])

    def test_build_groundtruth_payload_positive_uses_peak_valley_peak(self) -> None:
        profile = np.full(80, 100.0, dtype=np.float64)
        profile[[10, 30]] = 180.0
        profile[20] = 80.0
        markers = [
            {"type": "peak", "idx": 10},
            {"type": "valley", "idx": 20},
            {"type": "peak", "idx": 30},
        ]

        payload = build_groundtruth_payload(
            profile,
            markers,
            source_profile="sample_profile.json",
            band_width=21,
            film_type="positive",
        )

        self.assertEqual(payload["film_type"], "positive")
        self.assertEqual(payload["num_wire_pairs"], 1)
        self.assertEqual(
            payload["wire_pairs"][0],
            {
                "group": 1,
                "wire_a_idx": 10,
                "gap_idx": 20,
                "wire_b_idx": 30,
                "wire_a_gray": 180.0,
                "gap_gray": 80.0,
                "wire_b_gray": 180.0,
            },
        )

    def test_build_groundtruth_payload_negative_uses_valley_peak_valley(self) -> None:
        profile = np.full(80, 100.0, dtype=np.float64)
        profile[[10, 30]] = 80.0
        profile[20] = 180.0
        markers = [
            {"type": "valley", "idx": 10},
            {"type": "peak", "idx": 20},
            {"type": "valley", "idx": 30},
        ]

        payload = build_groundtruth_payload(
            profile,
            markers,
            source_profile="sample_profile.json",
            band_width=21,
            film_type="negative",
        )

        self.assertEqual(payload["film_type"], "negative")
        self.assertEqual(payload["num_wire_pairs"], 1)
        self.assertEqual(payload["wire_pairs"][0]["wire_a_idx"], 10)
        self.assertEqual(payload["wire_pairs"][0]["gap_idx"], 20)
        self.assertEqual(payload["wire_pairs"][0]["wire_b_idx"], 30)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run updated tests**

```bash
cd /home/cht/code/IQIdet && PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m pytest tests/test_double_wire_demo.py tests/test_annotate_profile.py -v 2>&1
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_double_wire_demo.py tests/test_annotate_profile.py
git commit -m "test: update imports for double-wire tools reorg

Import bam_pair_marker_indices/normalize_profile_obb from gauge.imaging.profile,
build_groundtruth_payload from gauge.imaging.profile, and default_groundtruth_path
from scripts.double_wire.src.io_utils.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: Delete old scripts

**Files:**
- Delete: `scripts/debug/double_wire_demo.py`
- Delete: `scripts/debug/annotate_profile.py`

- [ ] **Step 1: Delete old scripts**

```bash
cd /home/cht/code/IQIdet
git rm scripts/debug/double_wire_demo.py
git rm scripts/debug/annotate_profile.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "refactor: remove old double-wire scripts (merged into scripts/double_wire/)

double_wire_demo.py + annotate_profile.py merged into scripts/double_wire/annotate.py
validate_bam_gt.py moved to scripts/double_wire/validate_bam_gt.py

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: Update SOP docs

**Files:**
- Modify: `docs/sop/BAM-algorithm-dev-workflow.md`
- Modify: `docs/sop/BAM-algorithm-validation.md`

- [ ] **Step 1: Update BAM-algorithm-dev-workflow.md**

Replace script paths in the Step 1, Step 2, Step 3→4 loop, and Step 4 sections:

- Line 27-31 (flow diagram): `double_wire_demo.py` → `scripts/double_wire/annotate.py`, remove Step 2 line
- Line 46: `python scripts/debug/double_wire_demo.py` → `python scripts/double_wire/annotate.py`
- Lines 64-73 (Step 2): Replace with "在 annotate.py 中按 A 进入标注模式"
- Lines 124-156: `scripts/debug/validate_bam_gt.py` → `scripts/double_wire/validate_bam_gt.py`

- [ ] **Step 2: Update BAM-algorithm-validation.md**

- Line 20-21: Update reference to mention `scripts/double_wire/annotate.py` instead of separate scripts
- Validation command examples: `scripts/debug/validate_bam_gt.py` → `scripts/double_wire/validate_bam_gt.py`
- Fast iteration loop: update script paths

- [ ] **Step 3: Commit**

```bash
git add docs/sop/BAM-algorithm-dev-workflow.md docs/sop/BAM-algorithm-validation.md
git commit -m "docs: update SOP paths for double-wire tools reorg

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 12: Final verification

- [ ] **Step 1: Compile all new files**

```bash
cd /home/cht/code/IQIdet
python -m py_compile src/gauge/imaging/profile.py
python -m py_compile scripts/double_wire/annotate.py
python -m py_compile scripts/double_wire/validate_bam_gt.py
python -m py_compile scripts/double_wire/src/io_utils.py
python -m py_compile scripts/double_wire/src/obb_ui.py
python -m py_compile scripts/double_wire/src/profile_view.py
python -m py_compile scripts/double_wire/src/annotation.py
echo "All files compile OK"
```

- [ ] **Step 2: Run all relevant tests**

```bash
cd /home/cht/code/IQIdet
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_demo -v
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_annotate_profile -v
PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src python -m unittest tests.test_double_wire_profile -v
echo "All tests pass"
```

- [ ] **Step 3: Final commit (if any fixes needed)**

```bash
git add -A && git commit -m "chore: final verification fixes for double-wire reorg

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
