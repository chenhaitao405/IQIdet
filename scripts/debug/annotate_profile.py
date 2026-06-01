#!/usr/bin/env python3
"""Interactive ground-truth annotation tool for double-wire IQI profile data.

Reads a profile JSON saved by ``double_wire_demo.py``, displays the unwarped
OBB image and grayscale profile curve side-by-side, and lets the user manually
mark peak and valley positions.  Output is a ``_groundtruth.json`` file that
can later be used to validate the BAM peak-detection algorithm.

Usage:
    annotate_profile.py <profile_json> [options]
    annotate_profile.py (-h | --help)

Arguments:
    <profile_json>           Path to the *_profile.json file saved by double_wire_demo.py

Options:
    -h --help                显示帮助信息
    --output <path>          输出 ground truth JSON 路径；默认与输入 *_profile.json 同目录

Interaction:
    Left-click on the profile curve  → add a marker at the nearest profile index.
    p / v                            → switch to **peak** or **valley** mode.
    Right-click near a marker        → remove that marker.
    u                                → undo last marker.
    s                                → save ground truth (auto-pairs valleys).
    q / Esc                          → quit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── matplotlib setup ──
import matplotlib

try:
    matplotlib.use("TkAgg", force=True)
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from docopt import docopt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2

# ── Marker display constants ──
PEAK_COLOR = "#d62728"     # red
VALLEY_COLOR = "#1f77b4"   # blue
PEAK_MARKER = "v"          # downward triangle (profile peaks point down visually)
VALLEY_MARKER = "^"        # upward triangle


def default_groundtruth_path(profile_json_path: str | Path) -> Path:
    """Return the default GT path next to the input profile JSON."""
    profile_path = Path(profile_json_path)
    stem = profile_path.stem
    if stem.endswith("_profile"):
        stem = stem[: -len("_profile")] + "_groundtruth"
    else:
        stem = stem + "_groundtruth"
    return profile_path.with_name(stem + ".json")


class ProfileAnnotator:
    """Interactive profile annotation tool."""

    def __init__(self, profile_json_path: str, output_path: Optional[str] = None):
        self.profile_json_path = Path(profile_json_path)
        self.output_path = (
            Path(output_path)
            if output_path
            else default_groundtruth_path(self.profile_json_path)
        )

        # Load data
        with open(self.profile_json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.profile_values = np.array(self.data.get("profile_values", []), dtype=np.float64)
        self.obb_corners_raw = self.data.get("obb_corners_raw")
        self.obb_size = self.data.get("obb_size", {})
        self.band_width = self.data.get("band_width", 21)
        self.auto_peak_indices = np.array(self.data.get("peak_indices", []), dtype=int)
        self.auto_valley_indices = np.array(self.data.get("valley_indices", []), dtype=int)

        # OBB image path (alongside the JSON)
        obb_path = self.profile_json_path.with_name(
            self.profile_json_path.stem.replace("_profile", "_obb") + ".png"
        )
        self.obb_image: Optional[np.ndarray] = None
        if obb_path.is_file():
            raw = cv2.imread(str(obb_path), cv2.IMREAD_UNCHANGED)
            if raw is not None:
                if raw.ndim == 3:
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                self.obb_image = raw

        # Annotation state
        self.mode: str = "valley"  # "peak" or "valley"
        self.markers: list[dict] = []  # [{"type": "peak"|"valley", "idx": int}, ...]

        # Matplotlib handles
        self.fig: Optional[plt.Figure] = None
        self.ax_img: Optional[plt.Axes] = None
        self.ax_profile: Optional[plt.Axes] = None
        self.scatter_peaks: Optional[plt.PathCollection] = None
        self.scatter_valleys: Optional[plt.PathCollection] = None

    # ── Marker Management ──

    def add_marker(self, idx: int) -> None:
        """Add a marker at the nearest profile index."""
        self.markers.append({"type": self.mode, "idx": int(idx)})

    def remove_nearest(self, x: float) -> bool:
        """Remove the marker nearest to display coordinate *x*, return True if removed."""
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

    # ── Rendering ──

    def _build_title(self) -> str:
        stem = self.profile_json_path.stem.replace("_profile", "")
        n_peaks = sum(1 for m in self.markers if m["type"] == "peak")
        n_valleys = sum(1 for m in self.markers if m["type"] == "valley")
        mode_label = {"peak": "PEAK", "valley": "VALLEY"}[self.mode]
        return (
            f"{stem} | mode: [{mode_label}] | "
            f"peaks: {n_peaks}, valleys: {n_valleys} | "
            f"band_width={self.band_width}"
        )

    def redraw(self) -> None:
        """Update the plot with current markers."""
        if self.fig is None:
            return

        self.ax_profile.clear()

        x = np.arange(len(self.profile_values))
        self.ax_profile.plot(x, self.profile_values, color="#4C78A8", linewidth=1.0)

        # Profile range for annotations
        prof_min, prof_max = self.profile_values.min(), self.profile_values.max()
        prof_range = max(prof_max - prof_min, 1.0)
        self.ax_profile.set_ylim(prof_min - 0.05 * prof_range, prof_max + 0.05 * prof_range)

        # Auto-detected markers (faded, for reference)
        if len(self.auto_peak_indices) > 0:
            self.ax_profile.plot(
                self.auto_peak_indices, self.profile_values[self.auto_peak_indices],
                "v", color="lightcoral", markersize=10, alpha=0.4, label="auto peaks"
            )
        if len(self.auto_valley_indices) > 0:
            self.ax_profile.plot(
                self.auto_valley_indices, self.profile_values[self.auto_valley_indices],
                "^", color="lightsteelblue", markersize=10, alpha=0.4, label="auto valleys"
            )

        # Manual markers
        peak_idxs = [m["idx"] for m in self.markers if m["type"] == "peak"]
        valley_idxs = [m["idx"] for m in self.markers if m["type"] == "valley"]

        if peak_idxs:
            self.ax_profile.plot(
                peak_idxs, self.profile_values[peak_idxs],
                PEAK_MARKER, color=PEAK_COLOR, markersize=10,
                markeredgecolor="black", markeredgewidth=0.5,
                label="peaks (manual)"
            )
            for idx in peak_idxs:
                self.ax_profile.annotate(
                    str(idx), (idx, self.profile_values[idx]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, color=PEAK_COLOR, ha="center"
                )

        if valley_idxs:
            self.ax_profile.plot(
                valley_idxs, self.profile_values[valley_idxs],
                VALLEY_MARKER, color=VALLEY_COLOR, markersize=10,
                markeredgecolor="black", markeredgewidth=0.5,
                label="valleys (manual)"
            )
            for idx in valley_idxs:
                self.ax_profile.annotate(
                    str(idx), (idx, self.profile_values[idx]),
                    textcoords="offset points", xytext=(0, -12),
                    fontsize=7, color=VALLEY_COLOR, ha="center"
                )

        self.ax_profile.set_xlabel("Profile position (px)")
        self.ax_profile.set_ylabel("Gray value")
        self.ax_profile.legend(fontsize=7, loc="upper right")
        self.ax_profile.set_xlim(0, len(self.profile_values) - 1)

        # Sync image axis width
        if self.obb_image is not None and self.ax_img is not None:
            self.ax_img.set_xlim(self.ax_profile.get_xlim())

        self.fig.suptitle(self._build_title(), fontsize=9)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ── Event Handlers ──

    def on_click(self, event) -> None:
        """Handle mouse click on the profile axes."""
        if event.inaxes != self.ax_profile:
            return
        if event.xdata is None:
            return

        idx = int(round(event.xdata))
        idx = max(0, min(idx, len(self.profile_values) - 1))

        if event.button == MouseButton.LEFT:
            self.add_marker(idx)
            mtype = self.mode
            val = self.profile_values[idx]
            print(f"[annotate] Added {mtype} at idx={idx}, gray={val:.2f}")
            self.redraw()
        elif event.button == MouseButton.RIGHT:
            if self.remove_nearest(event.xdata):
                self.redraw()

    def on_key(self, event) -> None:
        """Handle keyboard shortcuts."""
        if event.key == "p":
            self.mode = "peak"
            print("[annotate] Mode: PEAK")
            self.redraw()
        elif event.key == "v":
            self.mode = "valley"
            print("[annotate] Mode: VALLEY")
            self.redraw()
        elif event.key == "u":
            self.undo_last()
            self.redraw()
        elif event.key == "s":
            self.save_groundtruth()
        elif event.key in ("q", "escape"):
            print("[annotate] Quit")
            plt.close("all")

    # ── Save ──

    def save_groundtruth(self) -> None:
        """Auto-pair valleys into wire groups and save ground truth JSON."""
        valleys = sorted([m for m in self.markers if m["type"] == "valley"], key=lambda m: m["idx"])
        peaks = sorted([m for m in self.markers if m["type"] == "peak"], key=lambda m: m["idx"])

        if len(valleys) < 2:
            print("[annotate] Need at least 2 valleys to form wire pairs. Not saving.")
            return

        # Auto-pair: valley[i] + peak_between + valley[i+1]
        peak_idx_map = {p["idx"]: p for p in peaks}
        wire_pairs = []
        for i in range(len(valleys) - 1):
            v1 = valleys[i]
            v2 = valleys[i + 1]
            # Find all peaks between v1 and v2
            between = [
                p for p in peaks if v1["idx"] < p["idx"] < v2["idx"]
            ]
            if len(between) == 1:
                p_idx = between[0]["idx"]
            elif len(between) > 1:
                # Take highest gray value (positive film: gap is bright)
                p_idx = max(between, key=lambda p: self.profile_values[p["idx"]])["idx"]
                print(
                    f"[annotate] Pair {i+1}: multiple peaks between valley "
                    f"{v1['idx']}-{v2['idx']}, took max-gray at {p_idx}"
                )
            else:
                # No peak between — valleys are adjacent, skip
                print(
                    f"[annotate] Pair {i+1}: no peak between valley "
                    f"{v1['idx']}-{v2['idx']}, skipping"
                )
                continue
            wire_pairs.append({
                "group": len(wire_pairs) + 1,
                "valley_a_idx": v1["idx"],
                "peak_idx": p_idx,
                "valley_b_idx": v2["idx"],
                "valley_a_gray": float(self.profile_values[v1["idx"]]),
                "peak_gray": float(self.profile_values[p_idx]),
                "valley_b_gray": float(self.profile_values[v2["idx"]]),
            })

        # Determine film type from D1 (first pair)
        film_type = "unknown"
        if wire_pairs:
            pair1 = wire_pairs[0]
            # Positive film: gap (peak) > wire (valley) → peak gray > valley gray
            if pair1["peak_gray"] > pair1["valley_a_gray"]:
                film_type = "positive"
            else:
                film_type = "negative"

        payload = {
            "source_profile": str(self.profile_json_path),
            "band_width": self.band_width,
            "film_type": film_type,
            "num_wire_pairs": len(wire_pairs),
            "wire_pairs": wire_pairs,
            "all_peaks": [{"idx": m["idx"], "gray": float(self.profile_values[m["idx"]])}
                          for m in peaks],
            "all_valleys": [{"idx": m["idx"], "gray": float(self.profile_values[m["idx"]])}
                            for m in valleys],
            "annotated_at": None,  # filled below
        }

        from datetime import datetime, timezone
        payload["annotated_at"] = datetime.now(timezone.utc).isoformat()

        self.output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[annotate] Ground truth saved: {self.output_path}")
        print(f"  Film type: {film_type}")
        print(f"  Wire pairs: {len(wire_pairs)}")
        for wp in wire_pairs:
            print(
                f"    D{wp['group']:2d}: valley_a={wp['valley_a_idx']:4d}, "
                f"peak={wp['peak_idx']:4d}, valley_b={wp['valley_b_idx']:4d}"
            )

    # ── Main ──

    def run(self) -> None:
        if len(self.profile_values) == 0:
            print("[annotate] Error: no profile values in JSON", file=sys.stderr)
            sys.exit(1)

        print(f"[annotate] Loaded: {self.profile_json_path.name}")
        print(f"[annotate] Profile length: {len(self.profile_values)}")
        print(f"[annotate] band_width: {self.band_width}")
        print(f"[annotate] OBB image: {'yes' if self.obb_image is not None else 'not found'}")
        print(f"[annotate] Auto-detected: {len(self.auto_peak_indices)} peaks, "
              f"{len(self.auto_valley_indices)} valleys")
        print("[annotate] ──────────────────────────────────────")
        print("[annotate] Left-click profile → add marker")
        print("[annotate] p/v → switch Peak/Valley mode")
        print("[annotate] Right-click near marker → remove")
        print("[annotate] u → undo last   s → save   q → quit")
        print("[annotate] ──────────────────────────────────────")

        plt.ion()
        nrows = 2 if self.obb_image is not None else 1
        self.fig = plt.figure(figsize=(12, 7))

        if self.obb_image is not None:
            self.ax_img = self.fig.add_subplot(nrows, 1, 1)
            self.ax_img.imshow(self.obb_image, cmap="gray", aspect="auto",
                               extent=[0, len(self.profile_values) - 1, 0, 1])
            self.ax_img.set_ylabel("OBB image")
            self.ax_img.set_yticks([])
            self.ax_profile = self.fig.add_subplot(nrows, 1, 2, sharex=self.ax_img)
        else:
            self.ax_profile = self.fig.add_subplot(1, 1, 1)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

        # Block until figure is closed
        plt.show(block=True)


def main() -> None:
    args = docopt(__doc__)
    profile_json = args["<profile_json>"]
    output = args["--output"] or None

    if not Path(profile_json).is_file():
        print(f"Error: file not found: {profile_json}", file=sys.stderr)
        sys.exit(1)

    annotator = ProfileAnnotator(profile_json, output_path=output)
    annotator.run()


if __name__ == "__main__":
    main()
