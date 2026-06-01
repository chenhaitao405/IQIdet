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

        self._on_save = on_save
        self._on_toggle = on_toggle
        self._on_quit = on_quit

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
        elif event.button == MouseButton.RIGHT:
            self.remove_nearest(event.xdata)

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
