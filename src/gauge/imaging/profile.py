"""Profile band extraction for double-wire IQI analysis.

Pure functions with no OpenCV HighGUI or matplotlib dependency.
Suitable for integration into the automated pipeline (Phase 2).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

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


def get_obb_long_edge_midline(
    obb_points: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute the midline connecting the midpoints of an OBB's two longest edges.

    This midline runs perpendicular to the metal wires in a double-wire IQI and
    serves as the center line for profile band extraction.

    Args:
        obb_points: OBB vertices, shape (4, 2), in counterclockwise order.

    Returns:
        (start_point, end_point): Midline endpoints (x, y) tuples.
    """
    pts = np.asarray(obb_points, dtype=np.float64).reshape(4, 2)

    # Four edges: (0->1), (1->2), (2->3), (3->0)
    edges = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        length = float(np.linalg.norm(b - a))
        midpoint = (a + b) / 2.0
        edges.append({"length": length, "midpoint": midpoint, "idx": i})

    # Sort by length descending
    edges.sort(key=lambda e: e["length"], reverse=True)

    # The two longest edges are the long edges of the OBB
    # Their midpoints define the midline
    start = (float(edges[0]["midpoint"][0]), float(edges[0]["midpoint"][1]))
    end = (float(edges[1]["midpoint"][0]), float(edges[1]["midpoint"][1]))

    return start, end


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
