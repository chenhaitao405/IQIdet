"""Profile line selection UI — OpenCV window for interactive line-drawing."""

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
COLOR_CYAN = (255, 255, 0)


class LineSelector:
    """Manages OpenCV window for 2-point profile line selection.

    State machine: IDLE -> COLLECTING -> LOCKED -> (R key) -> IDLE

    Two clicks define the profile line. The line auto-locks on the 2nd click.
    The expand zone (±expand pixels perpendicular to the line) is visualized.

    Parameters:
        image_display: 8-bit BGR display image.
        scale_x, scale_y: raw -> display coordinate scale factors.
        expand: pixels to expand perpendicularly for strip visualization.
        window_name: OpenCV window title.
    """

    STATE_IDLE = "idle"
    STATE_COLLECTING = "collecting"
    STATE_LOCKED = "locked"

    def __init__(
        self,
        image_display: np.ndarray,
        scale_x: float,
        scale_y: float,
        expand: int = 60,
        on_lock: Optional[Callable[[], None]] = None,
        window_name: str = "Double Wire Demo",
    ):
        self.image_display = image_display
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.expand = int(expand)
        self.window_name = window_name
        self._on_lock = on_lock

        self.state: str = self.STATE_IDLE
        self.mouse_x: Optional[int] = None
        self.mouse_y: Optional[int] = None
        self.show_help: bool = False

        # Display-space points (as clicked)
        self._pts_disp: list[Tuple[float, float]] = []

        # Raw-image-space line endpoints (computed on lock)
        self.line_start: Optional[Tuple[float, float]] = None
        self.line_end: Optional[Tuple[float, float]] = None

    # -- Mouse callback -----------------------------------------------------

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.state in (self.STATE_IDLE, self.STATE_COLLECTING):
                self._pts_disp.append((float(x), float(y)))
                if self.state == self.STATE_IDLE:
                    self.state = self.STATE_COLLECTING
                if len(self._pts_disp) >= 2:
                    self._lock()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.state == self.STATE_COLLECTING and self._pts_disp:
                self._pts_disp.pop()
                if not self._pts_disp:
                    self.state = self.STATE_IDLE
            elif self.state == self.STATE_LOCKED:
                self._pts_disp.pop()
                self.line_start = None
                self.line_end = None
                self.state = self.STATE_COLLECTING if self._pts_disp else self.STATE_IDLE

    # -- Lock ----------------------------------------------------------------

    def _lock(self) -> None:
        """Lock the profile line from the two clicked points."""
        self.state = self.STATE_LOCKED
        raw_pts = [
            (x / self.scale_x, y / self.scale_y)
            for x, y in self._pts_disp
        ]
        self.line_start = raw_pts[0]
        self.line_end = raw_pts[1]
        print(f"[annotate] Line locked: ({raw_pts[0][0]:.1f},{raw_pts[0][1]:.1f})"
              f" -> ({raw_pts[1][0]:.1f},{raw_pts[1][1]:.1f})")
        if self._on_lock is not None:
            self._on_lock()

    # -- Reset ---------------------------------------------------------------

    def reset(self) -> None:
        self.state = self.STATE_IDLE
        self._pts_disp.clear()
        self.line_start = None
        self.line_end = None
        print("[annotate] Reset")

    @property
    def is_locked(self) -> bool:
        return self.state == self.STATE_LOCKED and self.line_start is not None

    # -- Overlay drawing ----------------------------------------------------

    def draw_overlay(self, annotating: bool = False) -> np.ndarray:
        """Draw the current overlay (crosshair, profile line, expand zone, status)."""
        vis = self.image_display.copy()

        # Crosshair
        if self.state in (self.STATE_IDLE, self.STATE_COLLECTING):
            if self.mouse_x is not None and self.mouse_y is not None:
                h_img, w_img = vis.shape[:2]
                cv2.line(vis, (self.mouse_x, 0), (self.mouse_x, h_img - 1),
                         COLOR_YELLOW, 1, cv2.LINE_AA)
                cv2.line(vis, (0, self.mouse_y), (w_img - 1, self.mouse_y),
                         COLOR_YELLOW, 1, cv2.LINE_AA)

        # Clicked points
        if self._pts_disp:
            pts_int = [(int(x), int(y)) for x, y in self._pts_disp]
            for pt in pts_int:
                cv2.circle(vis, pt, 5, COLOR_YELLOW, -1, cv2.LINE_AA)

            # Rubber-band preview: line from last placed point to mouse
            if (self.state == self.STATE_COLLECTING
                    and len(pts_int) == 1
                    and self.mouse_x is not None
                    and self.mouse_y is not None):
                cv2.line(vis, pts_int[0], (self.mouse_x, self.mouse_y),
                         COLOR_YELLOW, 1, cv2.LINE_AA)

            if len(pts_int) >= 2:
                cv2.line(vis, pts_int[0], pts_int[1], COLOR_YELLOW, 1, cv2.LINE_AA)

        # Locked: draw profile line + expand zone
        if self.state == self.STATE_LOCKED and len(self._pts_disp) >= 2:
            sx_d, sy_d = self._pts_disp[0]
            ex_d, ey_d = self._pts_disp[1]
            ddx = ex_d - sx_d
            ddy = ey_d - sy_d
            plen = np.hypot(ddx, ddy)

            if plen > 1e-6:
                # Perpendicular unit vector
                ppx = -ddy / plen
                ppy = ddx / plen

                # Profile line (green)
                cv2.line(vis, (int(sx_d), int(sy_d)), (int(ex_d), int(ey_d)),
                         COLOR_GREEN, 2, cv2.LINE_AA)

                # Expand zone edges (dashed cyan)
                expand_disp = self.expand * min(self.scale_x, self.scale_y)
                s_top = (int(sx_d + expand_disp * ppx), int(sy_d + expand_disp * ppy))
                e_top = (int(ex_d + expand_disp * ppx), int(ey_d + expand_disp * ppy))
                s_bot = (int(sx_d - expand_disp * ppx), int(sy_d - expand_disp * ppy))
                e_bot = (int(ex_d - expand_disp * ppx), int(ey_d - expand_disp * ppy))

                for (s, e) in [(s_top, e_top), (s_bot, e_bot)]:
                    cv2.line(vis, s, e, COLOR_CYAN, 1, cv2.LINE_AA)

                # Semi-transparent fill for expand zone
                overlay = vis.copy()
                zone_pts = np.array([s_top, e_top, e_bot, s_bot], dtype=np.int32)
                cv2.fillPoly(overlay, [zone_pts], (255, 255, 0))
                vis = cv2.addWeighted(overlay, 0.08, vis, 0.92, 0)

        # Endpoint markers on locked line
        if self.state == self.STATE_LOCKED and len(self._pts_disp) >= 2:
            for pt_disp in self._pts_disp:
                cv2.circle(vis, (int(pt_disp[0]), int(pt_disp[1])),
                           6, COLOR_GREEN, 2, cv2.LINE_AA)

        # Status bar
        locked_status = f"Locked | expand={self.expand}px"
        if annotating:
            locked_status += " | [ANNOTATING] P/V mode  U undo  S save  Esc exit  N next  Q quit"
        else:
            locked_status += " | A=annotate  S=save  Q=quit"
        status_map = {
            self.STATE_IDLE: "Ready — click 2 points to define profile line",
            self.STATE_COLLECTING: f"Point 1/2 placed — click 2nd point (RMB to undo)",
            self.STATE_LOCKED: locked_status,
        }
        status = status_map.get(self.state, "")
        cv2.putText(vis, status, (10, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

        # Help overlay
        if self.show_help:
            help_lines = [
                "L-click: place profile line endpoint (2 to lock)",
                "R-click: undo last point",
                "R: reset line   A: annotate   S: save   Q/ESC: quit",
                "Annotating: P/V mode   U undo   S save   Esc exit   N next   Q quit",
                "H: hide help",
                f"expand={self.expand}px (--expand CLI arg)",
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
