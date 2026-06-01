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
_LOCAL_SRC = Path(__file__).resolve().parent / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT), str(_LOCAL_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

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

        self.image_raw: Optional[np.ndarray] = None
        self.obb: Optional[OBBSelector] = None
        self.view: Optional[ProfileView] = None
        self.annotator: Optional[Annotator] = None

        self._profile: Optional[np.ndarray] = None
        self._bam_result = None
        self._unresolved: Optional[int] = None
        self._profile_line = None
        self._uw: int = 0
        self._annotating: bool = False

    # -- Profile update ----------------------------------------------------

    def _update_profile(self) -> None:
        if (self.obb is None or self.view is None
                or self.obb.obb_corners_raw is None
                or self.obb.obb_midline_raw is None):
            return

        profile_line = compute_profile_line(
            self.obb.obb_corners_raw,
            self.obb.obb_midline_raw,
            self.obb.profile_offset_pct,
        )
        _, (uw, _uh) = unwarp_obb_region(self.image_raw, self.obb.obb_corners_raw)

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

        # Redraw annotation markers if active
        if self._annotating and self.annotator is not None:
            self.annotator.profile_values = profile
            self.annotator.draw_markers(self.view.profile_axes)
            self.view.fig.canvas.draw()
            self.view.fig.canvas.flush_events()

        self._profile = profile
        self._bam_result = result
        self._unresolved = unresolved
        self._profile_line = profile_line
        self._uw = uw

    # -- Annotation toggle -------------------------------------------------

    def _toggle_annotation(self) -> None:
        if self.view is None or self.annotator is None:
            return
        if self._annotating:
            self.annotator.deactivate()
            self._annotating = False
            self._update_profile()
        else:
            if self.view.profile_axes is None:
                print("[annotate] No profile axes. Lock OBB first.")
                return
            self.annotator.profile_values = self._profile
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
                print("[save] No annotation markers - skipping groundtruth.json")

    # -- Main loop ---------------------------------------------------------

    def run(self) -> None:
        self.image_raw, display, scale_x, scale_y = load_image(
            str(self.image_path), self.window_size,
        )

        self.obb = OBBSelector(
            display, scale_x, scale_y, self.band_width,
            on_trackbar_change=lambda v: self._update_profile(),
        )
        self.view = ProfileView()

        self.annotator = Annotator(
            profile_values=np.array([]),
            band_width=self.band_width,
            on_save=self._save,
            on_toggle=self._toggle_annotation,
        )

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

            if key in (13, 32):  # Enter / Space
                if self.obb.state == OBBSelector.STATE_CONFIRM:
                    self.obb.state = OBBSelector.STATE_LOCKED
                    self._update_profile()
            elif key == ord("r"):
                if self._annotating:
                    self._toggle_annotation()
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
            elif key in (ord("q"), 27):
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
