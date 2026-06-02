"""File I/O and path utilities for the double-wire annotation tool."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

ORIGINAL_VARIANT_DIR = "ori"
INVERTED_VARIANT_DIR = "inver"


@dataclass(frozen=True)
class DoubleWireArtifactPaths:
    """Canonical paths for one image/version's double-wire artifacts."""

    image_dir: Path
    variant_dir: Path
    overlay: Path
    obb: Path
    profile: Path
    groundtruth: Path


def double_wire_artifact_paths(
    output_dir: str | Path,
    image_path: str | Path,
    *,
    inverted: bool = False,
) -> DoubleWireArtifactPaths:
    """Build canonical artifact paths grouped by image stem and variant."""
    output_root = Path(output_dir)
    image_stem = Path(image_path).stem
    image_dir = output_root / image_stem
    variant_dir = image_dir / (INVERTED_VARIANT_DIR if inverted else ORIGINAL_VARIANT_DIR)
    suffix = "_inverted" if inverted else ""
    return DoubleWireArtifactPaths(
        image_dir=image_dir,
        variant_dir=variant_dir,
        overlay=image_dir / f"{image_stem}_overlay.png",
        obb=variant_dir / f"{image_stem}{suffix}_obb.png",
        profile=variant_dir / f"{image_stem}{suffix}_profile.json",
        groundtruth=variant_dir / f"{image_stem}{suffix}_groundtruth.json",
    )


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


def find_profile_groundtruth_pairs(directory: str | Path) -> list[tuple[Path, Path]]:
    """Find matching profile/GT pairs recursively under a directory."""
    root = Path(directory)
    pairs: list[tuple[Path, Path]] = []
    for profile_path in sorted(root.rglob("*_profile.json")):
        gt_path = default_groundtruth_path(profile_path)
        if gt_path.is_file():
            pairs.append((profile_path, gt_path))
    return pairs
