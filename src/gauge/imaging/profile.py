"""Profile band extraction for double-wire IQI analysis.

Pure functions with no OpenCV HighGUI or matplotlib dependency.
Suitable for integration into the automated pipeline (Phase 2).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


def extract_profile_band(
    image: np.ndarray,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    band_width: int = 21,
    num_samples: Optional[int] = None,
) -> np.ndarray:
    """Extract a band-averaged gray profile along a line, with sub-pixel interpolation.

    Samples ``band_width`` parallel lines centered on the profile midline, then
    averages them column-wise to produce a single 1D profile. This satisfies the
    JBT 7902-2025 requirement of >=21 rows/columns averaged.

    Args:
        image: Input image (supports uint8, uint16, float32, float64).
        start_point: Midline start coordinate (x, y).
        end_point: Midline end coordinate (x, y).
        band_width: Number of parallel lines to sample and average (default 21).
        num_samples: Number of samples along the midline direction.
            If None, auto-calculated as ``ceil(line_length)``.

    Returns:
        1D float64 array of band-averaged gray values.
    """
    x0, y0 = float(start_point[0]), float(start_point[1])
    x1, y1 = float(end_point[0]), float(end_point[1])

    dx = x1 - x0
    dy = y1 - y0
    line_length = np.hypot(dx, dy)

    if line_length < 1e-6:
        return np.array([], dtype=np.float64)

    if num_samples is None:
        num_samples = max(1, int(np.ceil(line_length)))

    # Unit vectors
    ux = dx / line_length   # along profile direction
    uy = dy / line_length

    # Perpendicular unit vector (rotated 90 degrees CCW: (-y, x))
    px = -uy
    py = ux

    # Generate sampling coordinates for the band
    # t_vals: parameter along midline [0, 1]
    t_vals = np.linspace(0, 1, num_samples)

    # offsets: perpendicular offsets from midline, symmetric about 0
    offsets = np.arange(band_width) - (band_width - 1) / 2.0

    # Midline coordinates at each t: shape (num_samples,)
    mid_x = x0 + t_vals * dx
    mid_y = y0 + t_vals * dy

    # Full coordinate grid: shape (band_width, num_samples)
    y_coords = mid_y[np.newaxis, :] + offsets[:, np.newaxis] * py
    x_coords = mid_x[np.newaxis, :] + offsets[:, np.newaxis] * px

    # map_coordinates expects (row=y, col=x) order, flattened spatial axes
    coords = np.stack([y_coords.ravel(), x_coords.ravel()], axis=0)

    # Sample using bilinear interpolation (order=1 for good speed/quality)
    sampled = map_coordinates(image.astype(np.float64), coords, order=1, mode="nearest")
    sampled = sampled.reshape(band_width, num_samples)

    # Average across band lines
    profile = sampled.mean(axis=0)
    return profile


def fit_obb_and_midline(
    points: np.ndarray,
) -> Tuple[np.ndarray, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Fit an oriented bounding box to 4 points and compute the profile midline.

    Uses cv2.minAreaRect to fit a true rectangle (correcting for click imprecision),
    then finds the two shortest edges and connects their midpoints -- this line runs
    along the OBB's long-edge direction (perpendicular to wires).

    Args:
        points: (4, 2) array of corner points in any order.

    Returns:
        (obb_corners, (midline_start, midline_end)):
        - obb_corners: (4, 2) rectangle corners in "top-left, top-right, bottom-right, bottom-left" order,
          suitable for cv2.getPerspectiveTransform.
        - midline: endpoints of the profile midline, running parallel to the long edges.
    """
    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    box = cv2.boxPoints(rect)  # (4, 2), CCW from lowest point, may not be TL-TR-BR-BL

    (cx, cy), (w, h), angle = rect  # w >= h guaranteed

    # Identify the four edges of box
    n = len(box)
    edges = []
    for i in range(n):
        a = box[i]
        b = box[(i + 1) % n]
        length_val = float(np.linalg.norm(b - a))
        mid = (a + b) / 2.0
        edges.append({"length": length_val, "mid": mid, "a": a, "b": b, "idx": i})

    # Sort edges by length: shortest first
    edges.sort(key=lambda e: e["length"])

    # The two shortest edges are the OBB height edges (parallel to wires)
    # Their midpoints define the profile midline
    start = (float(edges[0]["mid"][0]), float(edges[0]["mid"][1]))
    end = (float(edges[1]["mid"][0]), float(edges[1]["mid"][1]))

    # --- Reorder box corners to TL-TR-BR-BL for getPerspectiveTransform ---
    # box from minAreaRect is CCW starting from the point with lowest y (or
    # lowest x if tie). Sort by (y, x): topmost first, leftmost within same row.
    ordered = sorted(box, key=lambda p: (p[1], p[0]))
    tl = ordered[0]  # smallest y (and smallest x if tie)
    br = ordered[3]  # largest y (and largest x if tie)
    # Among remaining two, the one with larger x is TR
    mid_pts = ordered[1:3]
    tr = mid_pts[0] if mid_pts[0][0] > mid_pts[1][0] else mid_pts[1]
    bl = mid_pts[1] if mid_pts[0][0] > mid_pts[1][0] else mid_pts[0]

    obb_corners = np.array([tl, tr, br, bl], dtype=np.float32)

    return obb_corners, (start, end)


def unwarp_obb_region(
    image: np.ndarray,
    obb_corners: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Perspective-unwarp an OBB region into an axis-aligned rectangle.

    Args:
        image: Source image (grayscale, any dtype).
        obb_corners: (4, 2) rectangle corners in TL-TR-BR-BL order
            (as returned by fit_obb_and_midline).

    Returns:
        (unwarped, (width, height)):
        - unwarped: Unwarped grayscale image.
        - (width, height): Dimensions in pixels. width = OBB long edge (profile direction),
          height = OBB short edge (across wires).
    """
    corners = np.asarray(obb_corners, dtype=np.float32).reshape(4, 2)

    # Width = distance from TL->TR (or BL->BR) -- the long edge
    w = int(np.ceil(np.linalg.norm(corners[1] - corners[0])))
    # Height = distance from TL->BL (or TR->BR) -- the short edge
    h = int(np.ceil(np.linalg.norm(corners[3] - corners[0])))

    # Ensure w >= h (the long edge should be width)
    if w < h:
        w, h = h, w
        # Reorder corners: rotate 90 so 0-1 is now the long edge
        corners = corners[[3, 0, 1, 2]]

    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    unwarped = cv2.warpPerspective(image, M, (w, h))
    return unwarped, (w, h)


def detect_peaks_valleys(
    profile: np.ndarray,
    min_distance: int = 10,
    prominence: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peak and valley positions in a 1D gray profile.

    Peaks are detected on the profile directly; valleys are detected by
    inverting the profile and applying the same peak-finding algorithm.

    Args:
        profile: 1D gray profile array.
        min_distance: Minimum pixel distance between adjacent peaks.
        prominence: Minimum peak prominence relative to local baseline.
            Scaled to the profile's dynamic range (max - min).

    Returns:
        (peak_indices, valley_indices): Integer index arrays of peak and valley
        positions, sorted ascending.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if len(profile) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    data_range = float(np.max(profile) - np.min(profile))
    if data_range < 1e-10:
        return np.array([], dtype=int), np.array([], dtype=int)

    abs_prominence = prominence * data_range

    peaks, _ = find_peaks(profile, distance=min_distance, prominence=abs_prominence)
    valleys, _ = find_peaks(-profile, distance=min_distance, prominence=abs_prominence)

    return peaks.astype(int), valleys.astype(int)


# ---------------------------------------------------------------------------
# Phase 2 interface stubs (not implemented in this phase)
# ---------------------------------------------------------------------------


def compute_contrast(
    profile: np.ndarray,
    peak_indices: np.ndarray,
    valley_indices: np.ndarray,
) -> float:
    """Compute Contrast = (a + b - 2c) / (a + b) using the peak-two-valley method.

    Args:
        profile: 1D gray profile.
        peak_indices: Peak position indices.
        valley_indices: Valley position indices.

    Returns:
        Contrast value in [0, 1].

    Note:
        Phase 2 implementation. Must handle:
        - Positive/negative film determination
        - Neighborhood averaging of a, b values
        - Mismatched peak/valley counts
    """
    raise NotImplementedError("Phase 2 implementation")


def find_first_unresolved_group(
    profiles: List[np.ndarray],
    threshold: float = 0.2,
) -> Optional[int]:
    """Find the first wire pair group with Contrast < threshold.

    Args:
        profiles: List of 1D gray profiles, one per wire pair group (D1 -> Dn).
        threshold: Contrast threshold (default 0.2 = 20%).

    Returns:
        1-indexed group number of the first unresolved group, or None if all
        groups are resolved.

    Note:
        Phase 2 implementation.
    """
    raise NotImplementedError("Phase 2 implementation")
