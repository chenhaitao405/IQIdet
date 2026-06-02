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

    Parameters:
        image_display: 8-bit BGR display image.
        scale_x, scale_y: raw -> display coordinate scale factors.
        band_width: number of parallel profile lines in the band.
        on_trackbar_change: callback(offset_pct) called when trackbar moves.
        window_name: OpenCV window title.
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

        locked_status = f"Locked | offset={self.profile_offset_pct}% | band={self.band_width}"
        if annotating:
            locked_status += " | [ANNOTATING] P/V mode  U undo  S save  Esc exit  N next  Q quit"
        else:
            locked_status += " | A=annotate  S=save  Q=quit"
        status_map = {
            self.STATE_IDLE: "Ready",
            self.STATE_COLLECTING: f"Points {len(self.obb_points)}/4",
            self.STATE_CONFIRM: "CONFIRM - Enter to accept, R to retry, RMB to undo",
            self.STATE_LOCKED: locked_status,
        }
        status = status_map.get(self.state, "")
        cv2.putText(vis, status, (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

        if self.show_help:
            help_lines = [
                "L-click: add OBB vertex (4 to confirm)",
                "R-click: undo last vertex",
                "R: reset OBB   A: annotate   S: save   Q/ESC: quit",
                "Annotating: P/V mode   U undo   S save   Esc exit   N next   Q quit",
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
