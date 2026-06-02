"""Profile visualization — matplotlib figure for OBB image + profile curve."""

from __future__ import annotations

from typing import Optional, Tuple

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

    def flush_events(self) -> None:
        """Process pending GUI events for the matplotlib window."""
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
