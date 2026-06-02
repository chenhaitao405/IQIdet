#!/usr/bin/env python3
"""Interactive BAM double-wire IQI annotation tool.

Combines OBB selection, profile visualization, and ground-truth annotation
in a single unified workflow.  Annotation mode starts automatically after
locking the OBB; press **S** to save both profile data and ground truth.

Usage:
    annotate.py <image_path> [options]
    annotate.py (-h | --help)

Arguments:
    <image_path>              双丝像质计图像路径，或包含图像的目录（批量模式）

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
_DW_DIR = str(Path(__file__).resolve().parent)
for p in (str(REPO_ROOT), str(SRC_ROOT), _DW_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from _dwlib.io_utils import (
    double_wire_artifact_paths,
    load_image,
    save_obb_image,
    save_overlay_image,
    save_profile_json,
    save_groundtruth_json,
)
from _dwlib.obb_ui import OBBSelector, compute_profile_line
from _dwlib.profile_view import ProfileView
from _dwlib.annotation import Annotator

from gauge.imaging.profile import (
    extract_profile_band,
    unwarp_obb_region,
    compute_contrast,
    find_first_unresolved_group,
)


DEFAULT_OUTPUT_DIR = "outputs/double_wire_demo"


def _cv2_key_to_annotation_key(key: int) -> Optional[str]:
    """Translate OpenCV waitKey codes used by annotation mode."""
    if key < 0:
        return None
    if key == 27:
        return "escape"
    if ord("A") <= key <= ord("Z"):
        key = key + 32
    if 0 <= key <= 255:
        ch = chr(key)
        if ch in {"p", "v", "u", "s", "q"}:
            return ch
    return None


class BAMAnnotator:
    """Orchestrates OBB selection, profile view, and ground-truth annotation."""

    def __init__(
        self,
        image_path: str,
        output_dir: Optional[str] = DEFAULT_OUTPUT_DIR,
        window_size: int = 1200,
        band_width: int = 21,
    ):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
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
        self._pending_result: Optional[str] = None

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
            # Bring matplotlib figure to front
            try:
                self.view.fig.canvas.manager.window.raise_()
            except Exception:
                pass

    def _request_result(self, status: str) -> None:
        self._pending_result = status

    def _finish(self, status: str) -> str:
        if status == "next":
            if self._annotating:
                self._toggle_annotation()
            print("[annotate] Next image")
        else:
            print("[annotate] Quit")
        cv2.destroyAllWindows()
        plt.close("all")
        return status

    # -- Save --------------------------------------------------------------

    def _save_one_version(
        self, image: np.ndarray, *, inverted: bool = False
    ) -> Optional[str]:
        """Save OBB image, profile JSON, and GT for one image version.

        Args:
            image: Raw grayscale image to extract profile from.
            inverted: Whether this is the photometric-inverted version.

        Returns:
            Path to the saved profile JSON, or None if profile is unavailable.
        """
        paths = double_wire_artifact_paths(self.output_dir, self.image_path, inverted=inverted)
        paths.variant_dir.mkdir(parents=True, exist_ok=True)
        obb = self.obb.obb_corners_raw

        # OBB unwarp + image
        unwarped, (uw, uh) = unwarp_obb_region(image, obb)
        save_obb_image(unwarped, (uw, uh), paths.obb)

        # Profile extraction + BAM analysis
        profile_line = compute_profile_line(
            obb, self.obb.obb_midline_raw, self.obb.profile_offset_pct,
        )
        profile = extract_profile_band(
            image, profile_line[0], profile_line[1],
            band_width=self.band_width, num_samples=uw,
        )
        result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
        unresolved = find_first_unresolved_group(result.dips)

        # Profile JSON
        payload = {
            "image_path": str(self.image_path),
            "obb_corners_raw": obb.tolist(),
            "obb_size": {"width": uw, "height": uh},
            "obb_points": [[float(x), float(y)] for x, y in self.obb.obb_points],
            "profile_midline": {
                "start": list(profile_line[0]),
                "end": list(profile_line[1]),
            },
            "band_width": self.band_width,
            "profile_offset_pct": self.obb.profile_offset_pct,
            "profile_values": profile.tolist(),
            "bam_film_type": result.film_type,
            "bam_dips": result.dips,
            "bam_pairs": [
                [int(w1), int(g), int(w2)] for w1, g, w2 in result.pairs
            ],
            "bam_unresolved_group": unresolved,
        }
        save_profile_json(paths.profile, payload)

        # Ground truth JSON (rebuilt with this version's profile values)
        if self.annotator is not None and self.annotator.markers:
            gt_payload = self.annotator.build_groundtruth(str(paths.profile))
            save_groundtruth_json(paths.groundtruth, gt_payload)
            print(f"  Film type: {gt_payload['film_type']}")
            print(f"  Wire pairs: {gt_payload['num_wire_pairs']}")
            for wp in gt_payload["wire_pairs"]:
                print(
                    f"    D{wp['group']:2d}: wire_a={wp['wire_a_idx']:4d}, "
                    f"gap={wp['gap_idx']:4d}, wire_b={wp['wire_b_idx']:4d}"
                )
        else:
            print("[save] No annotation markers - skipping groundtruth.json")

        return str(paths.profile)

    def _invert_image(self) -> np.ndarray:
        """Create a photometric-inverted copy of the raw image."""
        raw = self.image_raw
        if raw.dtype == np.uint8:
            return 255 - raw
        elif raw.dtype == np.uint16:
            return 65535 - raw
        else:
            return raw.max() - raw

    def _save(self) -> None:
        if self.output_dir is None:
            print("[save] No output directory configured. Skipping.")
            return
        if self.obb is None or self.obb.obb_corners_raw is None:
            print("[save] No OBB fitted. Lock OBB first before saving.")
            return

        paths = double_wire_artifact_paths(self.output_dir, self.image_path, inverted=False)
        paths.image_dir.mkdir(parents=True, exist_ok=True)

        # Overlay image (once, shared)
        overlay = self.obb.draw_overlay(annotating=self._annotating)
        save_overlay_image(overlay, paths.overlay)

        # Save original (positive/negative as-is)
        print("[save] --- Original ---")
        self._save_one_version(self.image_raw, inverted=False)

        # Save inverted (255 - x)
        print("[save] --- Inverted (255-x) ---")
        inverted = self._invert_image()
        self._save_one_version(inverted, inverted=True)

    # -- Main loop ---------------------------------------------------------

    def run(self) -> None:
        plt.ion()  # Must be before ProfileView() — enables interactive figure windows

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
            on_quit=lambda: self._request_result("quit"),
            on_next=lambda: self._request_result("next"),
        )

        cv2.namedWindow(self.obb.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.obb.window_name, self.obb.mouse_callback)
        cv2.createTrackbar(
            self.obb.trackbar_name, self.obb.window_name, 50, 100, self.obb.on_trackbar,
        )

        print(f"[annotate] Loaded: {self.image_path.name}")
        print(f"[annotate] Shape: {self.image_raw.shape}, dtype: {self.image_raw.dtype}")
        print(f"[annotate] band_width: {self.band_width}")
        print("[annotate] L-click=add point  Enter=lock  R=reset  S=save  N=next  Q=quit  H=help")
        print("[annotate] Annotation: click profile to mark, P/V=mode, U=undo, S=save, Esc=exit")

        while True:
            overlay = self.obb.draw_overlay(annotating=self._annotating)
            cv2.imshow(self.obb.window_name, overlay)
            self.view.flush_events()
            if self._pending_result is not None:
                pending = self._pending_result
                self._pending_result = None
                return self._finish(pending)

            key = cv2.waitKey(30) & 0xFF

            # ── When annotating, keyboard belongs to matplotlib figure ──
            # Route OpenCV keys too, because matplotlib focus is WM/backend
            # dependent. Matplotlib events remain connected as a fallback.
            if self._annotating:
                if key == ord("n"):
                    return self._finish("next")
                elif key == ord("q"):
                    return self._finish("quit")
                elif self.annotator is not None:
                    self.annotator.handle_key(_cv2_key_to_annotation_key(key))
                continue

            # ── When NOT annotating, OpenCV owns keyboard ──
            if key in (13, 32):  # Enter / Space
                if self.obb.state == OBBSelector.STATE_CONFIRM:
                    self.obb.state = OBBSelector.STATE_LOCKED
                    self._update_profile()
                    self._toggle_annotation()
                elif self.obb.state == OBBSelector.STATE_LOCKED:
                    self._toggle_annotation()
            elif key == ord("r"):
                self.obb.reset()
                self.view.clear()
                cv2.setTrackbarPos(self.obb.trackbar_name, self.obb.window_name, 50)
            elif key == ord("s"):
                if self.obb.state == OBBSelector.STATE_LOCKED:
                    self._save()
                else:
                    print("[annotate] Lock OBB first (complete 4 points, press Enter)")
            elif key == ord("n"):
                return self._finish("next")
            elif key in (ord("q"), 27):
                return self._finish("quit")
            elif key == ord("h"):
                self.obb.show_help = not self.obb.show_help

        cv2.destroyAllWindows()
        plt.close("all")
        return "quit"


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _collect_images(path: Path) -> list[Path]:
    """Collect image files from a file or directory path."""
    if path.is_file():
        return [path]
    if path.is_dir():
        images = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS and p.is_file()
        )
        if not images:
            print(f"Error: no images found in directory: {path}", file=sys.stderr)
            sys.exit(1)
        return images
    print(f"Error: path not found: {path}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    args = docopt(__doc__)
    input_path = Path(args["<image_path>"])
    output_dir = args["--output-dir"]
    window_size = int(args["--window-size"])
    band_width = int(args["--band-width"])

    images = _collect_images(input_path)

    if len(images) > 1:
        print(f"[annotate] Batch mode: {len(images)} images found")
        print(f"[annotate] Output dir: {output_dir}")
        print(f"[annotate] N=next image, Q=quit batch")
        print()

    processed = 0
    for i, img_path in enumerate(images):
        if len(images) > 1:
            print(f"\n[annotate] === Image {i + 1}/{len(images)}: {img_path.name} ===")

        app = BAMAnnotator(
            image_path=str(img_path),
            output_dir=output_dir,
            window_size=window_size,
            band_width=band_width,
        )
        status = app.run()
        if status == "next":
            processed += 1
        elif status == "quit":
            break
        else:
            processed += 1

    if len(images) > 1:
        print(f"\n[annotate] Batch complete: {processed}/{len(images)} images processed")


if __name__ == "__main__":
    main()
