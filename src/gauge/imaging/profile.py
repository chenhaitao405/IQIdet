"""Profile band extraction for double-wire IQI analysis.

Pure functions with no OpenCV HighGUI or matplotlib dependency.
Suitable for integration into the automated pipeline (Phase 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

# JBT 7902-2025 表2 标准双丝型像质计 D1~D13 丝径/间距 (mm)
_DEFAULT_WIRE_SPACINGS: Tuple[float, ...] = (
    0.80, 0.63, 0.50, 0.40, 0.32, 0.25, 0.20, 0.16, 0.13, 0.10, 0.08, 0.063, 0.05,
)


def _match_box_to_point_order(box: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Match fitted rectangle corners to the caller-provided corner order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    box = np.asarray(box, dtype=np.float32).reshape(4, 2)

    best_order = box
    best_score = float("inf")
    for sequence in (box, box[::-1]):
        for shift in range(4):
            candidate = np.roll(sequence, -shift, axis=0)
            score = float(np.sum((candidate - pts) ** 2))
            if score < best_score:
                best_score = score
                best_order = candidate
    return np.asarray(best_order, dtype=np.float32)


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
    then matches the fitted rectangle back to the caller's corner order via
    nearest-neighbor. The fitted corners preserve the user's TL-TR-BR-BL
    clockwise order: corner 0 maps to the unwarped image's top-left, corner 1
    to top-right, etc.

    The profile midline runs parallel to the user's top edge (p0→p1), which
    defines the profile scan direction. The midline connects the left edge
    midpoint to the right edge midpoint.

    Args:
        points: (4, 2) array of corner points. Pass in TL-TR-BR-BL
            clockwise order. The user's p0→p1 edge defines the profile
            scan direction (across wires for double-wire IQI).

    Returns:
        (obb_corners, (midline_start, midline_end)):
        - obb_corners: (4, 2) rectangle corners in
          "top-left, top-right, bottom-right, bottom-left" order, suitable
          for cv2.getPerspectiveTransform. corners[0] corresponds to the
          user's p0 (top-left). Edge 0-1 is the profile scan direction.
        - midline: endpoints of the profile midline, running parallel to
          corners[0]→corners[1] (the profile direction).
    """
    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    box = cv2.boxPoints(rect)
    obb_corners = _match_box_to_point_order(box, points)
    tl, tr, br, bl = obb_corners

    # Midline: connect midpoints of the left (BL→TL) and right (TR→BR) edges.
    # Profile runs LEFT to RIGHT so that profile[0] ≈ left edge
    # (col 0 of unwarped image) and profile[-1] ≈ right edge (col w-1).
    start = (
        float((bl[0] + tl[0]) / 2.0),
        float((bl[1] + tl[1]) / 2.0),
    )
    end = (
        float((tr[0] + br[0]) / 2.0),
        float((tr[1] + br[1]) / 2.0),
    )

    return obb_corners, (start, end)


def unwarp_obb_region(
    image: np.ndarray,
    obb_corners: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Perspective-unwarp an OBB region into an axis-aligned rectangle.

    Maps user-ordered corners directly: corner 0 → top-left, corner 1 →
    top-right, corner 2 → bottom-right, corner 3 → bottom-left of the
    output image. The output width is the TL→TR edge length (profile scan
    direction), and the height is the TL→BL edge length (perpendicular).

    Args:
        image: Source image (grayscale, any dtype).
        obb_corners: (4, 2) rectangle corners in TL-TR-BR-BL order
            (as returned by fit_obb_and_midline).

    Returns:
        (unwarped, (width, height)):
        - unwarped: Unwarped grayscale image.
        - (width, height): Dimensions in pixels. width = TL→TR edge
          (profile direction, across wires), height = TL→BL edge
          (perpendicular direction).
    """
    corners = np.asarray(obb_corners, dtype=np.float32).reshape(4, 2)

    # Width = TL→TR (profile direction, across wires)
    w = int(np.ceil(np.linalg.norm(corners[1] - corners[0])))
    # Height = TL→BL (perpendicular direction)
    h = int(np.ceil(np.linalg.norm(corners[3] - corners[0])))

    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    unwarped = cv2.warpPerspective(image, M, (w, h))
    return unwarped, (w, h)


@dataclass
class ComputeContrastResult:
    """Result of :func:`compute_contrast`.

    Attributes:
        dips: Dip (modulation depth) for each wire pair, in percent [0, 100].
        pairs: Detected (wire_a_idx, gap_idx, wire_b_idx) triplets.
            Conventions follow positive-film semantics (valley, peak, valley),
            matching the naming in ``groundtruth.json``.
        background: Quadratic background fit values, same length as the input
            profile.
        film_type: ``"positive"`` or ``"negative"``.
    """
    dips: List[float]
    pairs: List[Tuple[int, int, int]]
    background: np.ndarray
    film_type: str


def _detect_film_type(
    profile: np.ndarray,
    valleys: np.ndarray,
    peaks: np.ndarray,
) -> str:
    """Determine film type from the first wire pair's gray-level relationship.

    For a positive film wires are dark (low gray) and gaps are bright (high
    gray), so ``profile[valley] < profile[peak]``.  Negative film is the
    inverse.

    Args:
        profile: 1D band-averaged gray profile.
        valleys: Valley index array (positions of profile minima).
        peaks: Peak index array (positions of profile maxima).

    Returns:
        ``"positive"`` or ``"negative"``.  Defaults to ``"positive"`` when
        there are fewer than 2 valleys or 1 peak.
    """
    if len(valleys) >= 2 and len(peaks) >= 1:
        a_val = float(profile[valleys[0]])
        c_val = float(profile[peaks[0]])
        return "negative" if a_val > c_val else "positive"
    return "positive"


def _fit_quadratic_background(
    profile: np.ndarray,
    wire_indices: np.ndarray,
    *,
    inverted: bool = False,
) -> np.ndarray:
    """Fit a quadratic background curve after masking out wire regions.

    Wire regions are masked out by computing their widths with
    :func:`scipy.signal.peak_widths` at ``rel_height=0.9``.  The profile is
    negated before calling ``peak_widths`` when *inverted* is ``True``
    (positive film, where wires are valleys in the original profile).
    A quadratic ``a*x^2 + b*x + c`` is then fitted to the remaining
    (gap-dominated) samples via :func:`scipy.optimize.curve_fit`.

    Args:
        profile: 1D band-averaged gray profile.
        wire_indices: Integer indices of wire positions (valleys for positive
            film, peaks for negative film).
        inverted: If ``True``, negate the profile before calling
            ``peak_widths``.  This is needed for positive film where wires
            appear as valleys (dark) and must be flipped to peaks for width
            measurement.

    Returns:
        1D ndarray of background values, same length as *profile*.
    """
    n = len(profile)
    if len(wire_indices) == 0:
        x = np.arange(n, dtype=np.float64)
        popt, _ = curve_fit(
            lambda x, a, b, c: a * x * x + b * x + c,
            x, profile.astype(np.float64),
        )
        return np.asarray(popt[0] * x * x + popt[1] * x + popt[2], dtype=np.float64)

    # Compute wire widths via peak_widths at rel_height=0.9.
    # For positive film (inverted=True): wires are valleys, so negate profile
    # to turn valleys into peaks for peak_widths.
    # For negative film (inverted=False): wires are peaks, use profile directly.
    target = -profile if inverted else profile
    widths, _, _, _ = peak_widths(target, wire_indices, rel_height=0.9)

    mask = np.ones(n, dtype=bool)
    for i, p in enumerate(wire_indices):
        lo = max(0, int(np.rint(p - widths[i])))
        hi = min(n - 1, int(np.rint(p + widths[i])))
        mask[lo:hi + 1] = False

    if mask.sum() < 3:
        mask[:] = True

    x = np.arange(n, dtype=np.float64)
    popt, _ = curve_fit(
        lambda x, a, b, c: a * x * x + b * x + c,
        x[mask], profile.astype(np.float64)[mask],
    )
    return np.asarray(popt[0] * x * x + popt[1] * x + popt[2], dtype=np.float64)


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
