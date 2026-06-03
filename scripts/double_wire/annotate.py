#!/usr/bin/env python3
"""Interactive BAM double-wire IQI annotation tool.

Draw a profile line across the wires with 2 clicks, then annotate
peaks and valleys on the extracted profile curve in a linked
matplotlib figure.

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
    --expand <px>             剖面线上下扩展像素数 [default: 60]
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
    save_strip_image,
    save_overlay_image,
    save_profile_json,
    save_groundtruth_json,
)
from _dwlib.line_ui import LineSelector
from _dwlib.profile_view import ProfileView
from _dwlib.annotation import Annotator

from gauge.imaging.profile import (
    extract_profile_band,
    extract_profile_strip,
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
    """Orchestrates line selection, profile view, and ground-truth annotation."""

    def __init__(
        self,
        image_path: str,
        output_dir: Optional[str] = DEFAULT_OUTPUT_DIR,
        window_size: int = 1200,
        band_width: int = 21,
        expand: int = 60,
    ):
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.window_size = int(window_size)
        self.band_width = int(band_width)
        self.expand = int(expand)

        self.image_raw: Optional[np.ndarray] = None
        self.line_selector: Optional[LineSelector] = None
        self.view: Optional[ProfileView] = None
        self.annotator: Optional[Annotator] = None

        self._profile: Optional[np.ndarray] = None
        self._strip: Optional[np.ndarray] = None
        self._bam_result = None
        self._unresolved: Optional[int] = None
        self._annotating: bool = False
        self._pending_result: Optional[str] = None

    # -- Profile update ----------------------------------------------------

    def _update_profile(self) -> None:
        if (self.line_selector is None or self.view is None
                or not self.line_selector.is_locked):
            return

        line_start = self.line_selector.line_start
        line_end = self.line_selector.line_end

        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        line_length = max(1, int(np.ceil(np.hypot(dx, dy))))

        # Strip image (for visualization)
        strip = extract_profile_strip(
            self.image_raw, line_start, line_end,
            expand=self.expand, num_samples=line_length,
        )

        # Band-averaged profile (for analysis)
        profile = extract_profile_band(
            self.image_raw, line_start, line_end,
            band_width=self.band_width, num_samples=line_length,
        )

        result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
        unresolved = find_first_unresolved_group(result.dips)

        self.view.update(
            strip, profile, self.expand, self.band_width,
            result, unresolved, self.image_path.stem,
        )

        # Redraw annotation markers if active
        if self._annotating and self.annotator is not None:
            self.annotator.profile_values = profile
            self.annotator.draw_markers(self.view.profile_axes)
            self.view.fig.canvas.draw()
            self.view.fig.canvas.flush_events()

        self._profile = profile
        self._strip = strip
        self._bam_result = result
        self._unresolved = unresolved

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
                print("[annotate] No profile axes. Lock line first.")
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
        """Save strip image, profile JSON, and GT for one image version.

        Args:
            image: Raw grayscale image to extract profile from.
            inverted: Whether this is the photometric-inverted version.

        Returns:
            Path to the saved profile JSON, or None if profile is unavailable.
        """
        paths = double_wire_artifact_paths(self.output_dir, self.image_path, inverted=inverted)
        paths.variant_dir.mkdir(parents=True, exist_ok=True)
        ls = self.line_selector
        line_start = ls.line_start
        line_end = ls.line_end

        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        line_length = max(1, int(np.ceil(np.hypot(dx, dy))))

        # Strip image
        strip = extract_profile_strip(
            image, line_start, line_end,
            expand=self.expand, num_samples=line_length,
        )
        save_strip_image(strip, paths.strip)

        # Profile extraction + BAM analysis
        profile = extract_profile_band(
            image, line_start, line_end,
            band_width=self.band_width, num_samples=line_length,
        )
        result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
        unresolved = find_first_unresolved_group(result.dips)

        # Profile JSON
        payload = {
            "image_path": str(self.image_path),
            "profile_line": {
                "start": [float(line_start[0]), float(line_start[1])],
                "end": [float(line_end[0]), float(line_end[1])],
            },
            "line_length_px": line_length,
            "expand": self.expand,
            "band_width": self.band_width,
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
        if self.line_selector is None or not self.line_selector.is_locked:
            print("[save] No profile line locked. Lock line first before saving.")
            return

        paths = double_wire_artifact_paths(self.output_dir, self.image_path, inverted=False)
        paths.image_dir.mkdir(parents=True, exist_ok=True)

        # Overlay image (once, shared)
        overlay = self.line_selector.draw_overlay(annotating=self._annotating)
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

        self.line_selector = LineSelector(
            display, scale_x, scale_y, expand=self.expand,
            on_lock=lambda: self._update_profile(),
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

        cv2.namedWindow(self.line_selector.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(
            self.line_selector.window_name, self.line_selector.mouse_callback,
        )

        print(f"[annotate] Loaded: {self.image_path.name}")
        print(f"[annotate] Shape: {self.image_raw.shape}, dtype: {self.image_raw.dtype}")
        print(f"[annotate] band_width: {self.band_width}, expand: {self.expand}")
        print("[annotate] L-click=place endpoint  R-click=undo  R=reset  S=save  N=next  Q=quit  H=help")
        print("[annotate] Annotation: click profile to mark, P/V=mode, U=undo, S=save, Esc=exit")

        while True:
            overlay = self.line_selector.draw_overlay(annotating=self._annotating)
            cv2.imshow(self.line_selector.window_name, overlay)
            self.view.flush_events()
            if self._pending_result is not None:
                pending = self._pending_result
                self._pending_result = None
                return self._finish(pending)

            key = cv2.waitKey(30) & 0xFF

            # ── When annotating, keyboard belongs to matplotlib figure ──
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
                if self.line_selector.state == LineSelector.STATE_LOCKED:
                    self._toggle_annotation()
                # No CONFIRM state — line auto-locks on 2nd click; Enter just toggles annotation
            elif key == ord("r"):
                self.line_selector.reset()
                self.view.clear()
            elif key == ord("s"):
                if self.line_selector.state == LineSelector.STATE_LOCKED:
                    self._save()
                else:
                    print("[annotate] Lock line first (place 2 points)")
            elif key == ord("a"):
                if self.line_selector.state == LineSelector.STATE_LOCKED:
                    self._toggle_annotation()
                else:
                    print("[annotate] Lock line first (place 2 points)")
            elif key == ord("n"):
                return self._finish("next")
            elif key in (ord("q"), 27):
                return self._finish("quit")
            elif key == ord("h"):
                self.line_selector.show_help = not self.line_selector.show_help

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
    expand = int(args["--expand"])

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
            expand=expand,
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
