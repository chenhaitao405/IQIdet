#!/usr/bin/env python3
"""Interactive double-wire IQI profile band visualization tool.

Usage:
    double_wire_demo.py <image_path> [options]
    double_wire_demo.py (-h | --help)

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
from typing import Optional, Tuple

import cv2
import numpy as np

# ── matplotlib setup (must happen before pyplot import) ──
import matplotlib

# Try TkAgg first; fall back to default if unavailable
try:
    matplotlib.use("TkAgg", force=True)
except Exception:
    pass  # use default backend

import matplotlib.pyplot as plt
from docopt import docopt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gauge.imaging.profile import (
    extract_profile_band,
    fit_obb_and_midline,
    unwarp_obb_region,
    detect_peaks_valleys,
)

# ── Color constants (BGR for OpenCV) ──
COLOR_YELLOW = (0, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

PROFILE_COLOR = "#4C78A8"


class DoubleWireDemo:
    """Interactive double-wire IQI profile band visualization tool.

    State machine: IDLE -> COLLECTING -> CONFIRM -> LOCKED -> (R key) -> IDLE
    """

    STATE_IDLE = "idle"
    STATE_COLLECTING = "collecting"
    STATE_CONFIRM = "confirm"
    STATE_LOCKED = "locked"

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
        if self.band_width < 1:
            raise ValueError(f"band_width must be >= 1, got {self.band_width}")

        # Image data
        self.image_raw: Optional[np.ndarray] = None
        self.image_display: Optional[np.ndarray] = None
        self.display_scale_x: float = 1.0
        self.display_scale_y: float = 1.0

        # Interaction state
        self.state: str = self.STATE_IDLE
        self.obb_points: list = []
        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None

        # Profile parameters
        self.profile_offset_pct: int = 50
        self.show_help: bool = False

        # OBB fitted data (raw image coords)
        self.obb_corners_raw: Optional[np.ndarray] = None
        self.obb_midline_raw: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.obb_corners_disp: Optional[list] = None

        # Profile data
        self.profile_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.profile: Optional[np.ndarray] = None
        self.peak_indices: Optional[np.ndarray] = None
        self.valley_indices: Optional[np.ndarray] = None

        # UI handles
        self.window_name = "Double Wire Demo"
        self.trackbar_name = "offset%"
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    # ── Image Loading ──

    def load_image(self) -> None:
        raw = cv2.imread(str(self.image_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Failed to read image: {self.image_path}")
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        self.image_raw = raw  # NEVER resize -- keep original resolution
        # Build 8-bit display image
        if self.image_raw.dtype == np.uint16:
            disp = cv2.normalize(self.image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        elif self.image_raw.dtype == np.uint8:
            disp = self.image_raw.copy()
        else:
            disp = cv2.normalize(self.image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        h_raw, w_raw = self.image_raw.shape[:2]
        # Resize only the display copy
        long_side = max(disp.shape[0], disp.shape[1])
        if long_side > self.window_size:
            scale = self.window_size / long_side
            new_w = max(1, int(disp.shape[1] * scale))
            new_h = max(1, int(disp.shape[0] * scale))
            disp = cv2.resize(disp, (new_w, new_h))
        self.image_display = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        # Scale factor: raw -> display
        self.display_scale_x = self.image_display.shape[1] / w_raw
        self.display_scale_y = self.image_display.shape[0] / h_raw

    # ── Mouse Callback ──

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
                # Undo last point and go back to COLLECTING
                self.obb_points.pop()
                self.obb_corners_raw = None
                self.obb_midline_raw = None
                self.obb_corners_disp = None
                self.state = self.STATE_COLLECTING if self.obb_points else self.STATE_IDLE

    # ── OBB Fitting ──

    def _fit_obb(self) -> None:
        """Fit OBB from clicked points and compute fitted corners + midline."""
        raw_pts = [(x / self.display_scale_x, y / self.display_scale_y) for x, y in self.obb_points]
        pts = np.array(raw_pts, dtype=np.float32)
        self.obb_corners_raw, self.obb_midline_raw = fit_obb_and_midline(pts)
        # Convert fitted corners back to display for overlay drawing
        self.obb_corners_disp = [
            (x * self.display_scale_x, y * self.display_scale_y)
            for x, y in self.obb_corners_raw
        ]

    # ── Trackbar Callback ──

    def on_trackbar(self, value: int) -> None:
        self.profile_offset_pct = value
        if self.state == self.STATE_LOCKED and len(self.obb_points) >= 4:
            self.update_profile()

    # ── Profile Update ──

    def update_profile(self) -> None:
        if self.obb_corners_raw is None or self.obb_midline_raw is None:
            return
        obb = self.obb_corners_raw
        midline = self.obb_midline_raw
        start, end = midline
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        length = np.hypot(dx, dy)
        if length < 1e-6:
            return
        px = -dy / length
        py = dx / length
        # Compute OBB short edge from fitted corners
        short_e0 = float(np.linalg.norm(obb[0] - obb[3]))
        short_e1 = float(np.linalg.norm(obb[1] - obb[2]))
        short_side = (short_e0 + short_e1) / 2.0
        offset_frac = (self.profile_offset_pct - 50) / 50.0
        offset_amount = offset_frac * short_side * 0.45
        sx_off = sx + offset_amount * px
        sy_off = sy + offset_amount * py
        ex_off = ex + offset_amount * px
        ey_off = ey + offset_amount * py
        self.profile_line = ((sx_off, sy_off), (ex_off, ey_off))
        # Get unwarped width for 1:1 alignment
        _, (uw, uh) = unwarp_obb_region(self.image_raw, obb)
        self.profile = extract_profile_band(
            self.image_raw, self.profile_line[0], self.profile_line[1],
            band_width=self.band_width, num_samples=uw,
        )
        self.peak_indices, self.valley_indices = detect_peaks_valleys(
            self.profile, min_distance=10, prominence=0.05,
        )
        self.plot_profile()

    # ── Overlay Drawing ──

    def draw_overlay(self) -> np.ndarray:
        vis = self.image_display.copy()
        # Crosshair in IDLE/COLLECTING states
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

        # Fitted OBB overlay for CONFIRM and LOCKED states
        if self.state in (self.STATE_CONFIRM, self.STATE_LOCKED) and self.obb_corners_disp:
            pts_int = [(int(x), int(y)) for x, y in self.obb_corners_disp]
            cv2.polylines(vis, [np.array(pts_int)], isClosed=True, color=COLOR_GREEN,
                          thickness=2, lineType=cv2.LINE_AA)
            # Draw original clicked points as small yellow dots
            for x, y in self.obb_points:
                cv2.circle(vis, (int(x), int(y)), 3, COLOR_YELLOW, -1, cv2.LINE_AA)

        # Profile midline and band (CONFIRM and LOCKED)
        if self.state in (self.STATE_CONFIRM, self.STATE_LOCKED) and self.obb_midline_raw is not None:
            (sx, sy), (ex, ey) = self.obb_midline_raw
            sx_d = sx * self.display_scale_x
            sy_d = sy * self.display_scale_y
            ex_d = ex * self.display_scale_x
            ey_d = ey * self.display_scale_y
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
        # Status text
        status_map = {
            self.STATE_IDLE: "Ready",
            self.STATE_COLLECTING: f"Points {len(self.obb_points)}/4",
            self.STATE_CONFIRM: "CONFIRM — Enter to accept, R to retry, RMB to undo",
            self.STATE_LOCKED: f"Locked | offset={self.profile_offset_pct}% | band={self.band_width}",
        }
        status = status_map.get(self.state, "")
        cv2.putText(vis, status, (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
        # Help overlay
        if self.show_help:
            help_lines = [
                "L-click: add OBB vertex (4 to confirm)",
                "R-click: undo last vertex",
                "R: reset OBB   S: save   Q/ESC: quit",
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

    # ── Matplotlib Plot ──

    def plot_profile(self) -> None:
        if self.profile is None or self.obb_corners_raw is None:
            return

        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 6))

        self.fig.clear()
        self.ax_top = self.fig.add_subplot(2, 1, 1)
        self.ax_bottom = self.fig.add_subplot(2, 1, 2, sharex=self.ax_top)

        # Top: unwarped OBB band image
        unwarped, (uw, uh) = unwarp_obb_region(self.image_raw, self.obb_corners_raw)
        self.ax_top.imshow(unwarped, cmap='gray', aspect='auto')

        # Draw profile band indicators on unwarped image
        # Profile midline in unwarped coords: y = uh/2 (center by default)
        # Trackbar offset: map 0-100% to Y position in unwarped image
        band_y = uh / 2 + (self.profile_offset_pct - 50) / 50.0 * uh * 0.45
        half_band = (self.band_width - 1) / 2.0
        self.ax_top.axhline(y=band_y, color='red', linewidth=1.0)
        self.ax_top.axhline(y=band_y - half_band, color='red', linewidth=0.5, linestyle='--')
        self.ax_top.axhline(y=band_y + half_band, color='red', linewidth=0.5, linestyle='--')
        self.ax_top.set_ylabel("Across wires (px)")

        # Bottom: profile curve
        x = np.arange(len(self.profile))
        self.ax_bottom.plot(x, self.profile, color=PROFILE_COLOR, linewidth=1.2, label="Profile")

        prof_range = max(self.profile.max() - self.profile.min(), 1.0)
        y_min = self.profile.min() - 0.05 * prof_range
        y_max = self.profile.max() + 0.05 * prof_range

        # Peak and valley markers
        if self.peak_indices is not None and len(self.peak_indices) > 0:
            peaks = self.peak_indices
            self.ax_bottom.plot(peaks, self.profile[peaks], "r^", markersize=8, label="Peaks")
            for p in peaks:
                val = self.profile[p]
                self.ax_bottom.annotate(f"{val:.2f}", (p, val), textcoords="offset points",
                                        xytext=(0, 8), fontsize=7, color="red", ha="center")
        if self.valley_indices is not None and len(self.valley_indices) > 0:
            valleys = self.valley_indices
            self.ax_bottom.plot(valleys, self.profile[valleys], "bv", markersize=8, label="Valleys")
            for v in valleys:
                val = self.profile[v]
                self.ax_bottom.annotate(f"{val:.2f}", (v, val), textcoords="offset points",
                                        xytext=(0, -12), fontsize=7, color="blue", ha="center")

        self.ax_bottom.set_ylim(y_min, y_max)
        self.ax_bottom.set_xlabel("Profile position (px) — aligned with image above")
        self.ax_bottom.set_ylabel("Gray value")
        self.ax_bottom.legend(fontsize=7, loc="upper right")

        # Title
        stem = self.image_path.stem
        title = (f"{stem} | OBB: {uw}x{uh} | offset:{self.profile_offset_pct}%"
                 f" | band:{self.band_width}")
        self.fig.suptitle(title, fontsize=9)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ── Save ──

    def save_results(self) -> None:
        if self.output_dir is None:
            print("[save] No output directory configured. Skipping.")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem
        overlay_path = self.output_dir / f"{stem}_overlay.png"
        overlay = self.draw_overlay()
        cv2.imwrite(str(overlay_path), overlay)
        print(f"[save] Overlay: {overlay_path}")
        json_path = self.output_dir / f"{stem}_profile.json"
        payload = {
            "image_path": str(self.image_path),
            "obb_points": [[float(x), float(y)] for x, y in self.obb_points],
            "profile_midline": {
                "start": list(self.profile_line[0]) if self.profile_line else None,
                "end": list(self.profile_line[1]) if self.profile_line else None,
            },
            "band_width": self.band_width,
            "profile_offset_pct": self.profile_offset_pct,
            "profile_values": self.profile.tolist() if self.profile is not None else [],
            "peak_indices": self.peak_indices.tolist() if self.peak_indices is not None else [],
            "valley_indices": self.valley_indices.tolist() if self.valley_indices is not None else [],
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[save] Profile data: {json_path}")

    # ── Main Loop ──

    def run(self) -> None:
        self.load_image()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.createTrackbar(self.trackbar_name, self.window_name, 50, 100, self.on_trackbar)
        plt.ion()
        self.fig = plt.figure(figsize=(10, 6))
        print(f"[demo] Loaded: {self.image_path.name}")
        print(f"[demo] Shape: {self.image_raw.shape}, dtype: {self.image_raw.dtype}")
        print(f"[demo] band_width: {self.band_width}")
        print("[demo] Keys: L-click=add point, R-click=undo, R=reset, Enter/Space=confirm, S=save, Q=quit, H=help")
        while True:
            overlay = self.draw_overlay()
            cv2.imshow(self.window_name, overlay)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 or key == 32:  # Enter or Space
                if self.state == self.STATE_CONFIRM:
                    self.state = self.STATE_LOCKED
                    self.update_profile()
            elif key == ord("r"):
                self.state = self.STATE_IDLE
                self.obb_points.clear()
                self.obb_corners_raw = None
                self.obb_midline_raw = None
                self.obb_corners_disp = None
                self.profile_line = None
                self.profile = None
                self.peak_indices = None
                self.valley_indices = None
                cv2.setTrackbarPos(self.trackbar_name, self.window_name, 50)
                self.profile_offset_pct = 50
                self.fig.clear()
                self.fig.canvas.draw()
                print("[demo] Reset")
            elif key == ord("s"):
                if self.state == self.STATE_LOCKED:
                    self.save_results()
                else:
                    print("[demo] Lock OBB first (complete 4 points) before saving")
            elif key == ord("q") or key == 27:
                print("[demo] Quit")
                break
            elif key == ord("h"):
                self.show_help = not self.show_help
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
    demo = DoubleWireDemo(
        image_path=image_path,
        output_dir=output_dir,
        window_size=window_size,
        band_width=band_width,
    )
    demo.run()


if __name__ == "__main__":
    main()
