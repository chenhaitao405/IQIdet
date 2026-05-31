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
    """Determine film type from raw peak/valley positions.

    Works by finding the alternating triple (v-p-v or p-v-p) with the
    largest amplitude swing.  The dominant wire pair produces the largest
    swing, and its pattern reveals the film type:

    - v-p-v with deep valleys → positive film  (wires are dark valleys)
    - p-v-p with tall peaks   → negative film (wires are bright peaks)

    Args:
        profile: 1D band-averaged gray profile.
        valleys: Valley index array (positions of profile minima).
        peaks: Peak index array (positions of profile maxima).

    Returns:
        ``"positive"`` or ``"negative"``.  Defaults to ``"positive"`` when
        no reliable alternating triple is found.
    """
    if len(valleys) < 1 or len(peaks) < 1:
        return "positive"

    # Merge extrema in position order
    extrema: List[Tuple[str, int]] = []
    vi = pi = 0
    while vi < len(valleys) or pi < len(peaks):
        if pi >= len(peaks) or (vi < len(valleys) and int(valleys[vi]) < int(peaks[pi])):
            extrema.append(("v", int(valleys[vi])))
            vi += 1
        else:
            extrema.append(("p", int(peaks[pi])))
            pi += 1

    # Find the alternating triple with the largest total swing
    best_type: Optional[str] = None
    best_swing = -1.0
    for i in range(len(extrema) - 2):
        t1, p1 = extrema[i]
        t2, p2 = extrema[i + 1]
        t3, p3 = extrema[i + 2]
        if t1 == t3 and t1 != t2:
            swing = abs(float(profile[p1]) - float(profile[p2])) + abs(
                float(profile[p2]) - float(profile[p3])
            )
            if swing > best_swing:
                best_swing = swing
                best_type = t1

    if best_type == "v":
        return "positive"
    if best_type == "p":
        return "negative"
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


def _compute_dip(
    profile: np.ndarray,
    wire_a: int,
    gap_c: int,
    wire_b: int,
    background: np.ndarray,
    half_w: int,
) -> float:
    """Compute the modulation depth (dip) for a single wire pair.

    The dip is defined as ``100 * (A + B - 2*C) / (A + B)`` where *A*, *B*
    are the absolute deviations of the two wires from the background and *C*
    is the absolute deviation of the gap from the background.  Each value is
    taken as the neighbourhood mean of width ``2*half_w+1`` around the
    detected position.

    Args:
        profile: 1D band-averaged gray profile.
        wire_a: Index of the first wire.
        gap_c: Index of the gap between the two wires.
        wire_b: Index of the second wire.
        background: Background fit values (same length as *profile*).
        half_w: Half-width of the neighbourhood window.

    Returns:
        Dip value in percent [0, 100].  Returns 0 when the denominator is
        negligible (fully merged pair).
    """
    L = len(profile)

    def _region_mean(center: int) -> float:
        lo = max(0, center - half_w)
        hi = min(L - 1, center + half_w)
        return float(profile[lo:hi + 1].mean())

    a_mean = _region_mean(wire_a)
    c_mean = _region_mean(gap_c)
    b_mean = _region_mean(wire_b)

    A = abs(float(background[wire_a]) - a_mean)
    B = abs(float(background[wire_b]) - b_mean)
    C = abs(float(background[gap_c]) - c_mean)

    denom = A + B
    if denom < 1e-10:
        return 0.0
    dip = 100.0 * (A + B - 2.0 * C) / denom
    return max(0.0, dip)


def _pair_wires_and_compute_dips(
    profile: np.ndarray,
    wire_positions: np.ndarray,
    gap_positions: np.ndarray,
    background: np.ndarray,
    half_w: int,
    *,
    dist_factor: float = 1.05,
    film_type: str = "positive",
) -> Tuple[List[float], List[Tuple[int, int, int]]]:
    """Pair adjacent wires into wire-pair groups and compute each dip.

    Two adjacent wire positions are paired when their distance does not
    exceed ``dist_factor * dist_between_first_two``.  The gap (the profile
    extremum between the two wires) is located in a film-type-aware manner,
    and the dip for the pair is computed via :func:`_compute_dip`.

    Args:
        profile: 1D band-averaged gray profile.
        wire_positions: Sorted indices of wire positions (valleys for
            positive film, peaks for negative).
        gap_positions: Sorted indices of gap positions (peaks for positive,
            valleys for negative).
        background: Quadratic background fit, same length as *profile*.
        half_w: Half-window for neighbourhood-averaged dip computation.
        dist_factor: Maximum allowed multiple of the first-pair spacing
            for two wires to be considered a pair.
        film_type: ``"positive"`` or ``"negative"``.

    Returns:
        ``(dips, pairs)`` where *dips* is a list of float percentages and
        *pairs* is the corresponding list of ``(wire_a, gap, wire_b)``
        index triplets.
    """
    dips: List[float] = []
    pairs: List[Tuple[int, int, int]] = []

    if len(wire_positions) < 2:
        return dips, pairs

    dist = wire_positions[1:] - wire_positions[:-1]
    dist_max = dist_factor * float(dist[0])

    i = 0
    while i < len(wire_positions) - 1:
        if dist[i] <= dist_max:
            w1 = int(wire_positions[i])
            w2 = int(wire_positions[i + 1])
            gap_mask = (gap_positions > w1) & (gap_positions < w2)
            gaps_between = gap_positions[gap_mask]
            if len(gaps_between) >= 1:
                if film_type == "negative":
                    c = int(gaps_between[np.argmin(profile[gaps_between])])
                else:
                    c = int(gaps_between[np.argmax(profile[gaps_between])])
                pairs.append((w1, c, w2))
                dip = _compute_dip(profile, w1, c, w2, background, half_w)
                dips.append(dip)
        i += 1

    return dips, pairs


def _cleanup_dips_monotonic(
    dips: Sequence[float],
    spacings: Sequence[float],
) -> Tuple[List[float], List[float]]:
    """Enforce monotonic decrease of dips from coarse (D1) to fine pairs.

    A dip that is more than 5 percentage points deeper than its
    predecessor is considered a detection anomaly; the shallower
    predecessor is removed.  This matches the ctsimu-toolbox monotonicity
    check.

    Args:
        dips: Dip values (percent) per wire pair, from coarse to fine.
        spacings: Nominal wire-pair spacings (mm), same length as *dips*.

    Returns:
        ``(cleaned_dips, cleaned_spacings)`` as new lists.
    """
    d = list(dips)
    s = list(spacings)
    i = 1
    while i < len(d):
        if (d[i] - d[i - 1]) > 5.0:
            del d[i - 1]
            del s[i - 1]
            i -= 1
        i += 1
    return d, s


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
    wire_spacings: Optional[Sequence[float]] = None,
    window_half_width: int = 3,
    film_type: str = "auto",
    min_distance: int = 10,
    prominence: float = 0.05,
) -> ComputeContrastResult:
    """Compute the modulation depth (dip) for every wire pair in a profile.

    Orchestrates: peak/valley detection -> film-type determination ->
    quadratic background fitting -> wire pairing -> per-pair dip calculation.

    The input *profile* is expected to be the output of
    :func:`extract_profile_band`, i.e. already band-averaged across
    >= 21 pixel rows to satisfy JBT 7902.

    Args:
        profile: 1D band-averaged gray profile.  Each element is the
            column-wise mean of >= 21 pixel rows perpendicular to the
            profile direction.
        wire_spacings: Nominal spacings (mm) of the wire pairs, e.g. the
            JBT 7902 D1-D13 sequence.  Accepted for forward compatibility;
            not used internally by this function.
        window_half_width: Half-width of the neighbourhood window for
            computing the a / b / c region means.  0 degenerates to
            single-pixel values.
        film_type: ``"positive"``, ``"negative"``, or ``"auto"``.  When
            ``"auto"`` the type is detected from the first wire pair.
        min_distance: Minimum pixel distance between adjacent peaks,
            forwarded to :func:`detect_peaks_valleys`.
        prominence: Relative peak prominence, forwarded to
            :func:`detect_peaks_valleys`.

    Returns:
        :class:`ComputeContrastResult` with dips, pairs, background, and
        film_type.
    """
    n = len(profile)
    if n < 3:
        return ComputeContrastResult(
            dips=[], pairs=[],
            background=np.array([], dtype=np.float64),
            film_type=film_type if film_type != "auto" else "positive",
        )

    # 1. Peak / valley detection
    peaks, valleys = detect_peaks_valleys(
        profile, min_distance=min_distance, prominence=prominence,
    )

    if len(peaks) == 0 and len(valleys) == 0:
        return ComputeContrastResult(
            dips=[], pairs=[],
            background=np.zeros(n, dtype=np.float64),
            film_type=film_type if film_type != "auto" else "positive",
        )

    # 2. Film-type determination
    if film_type == "auto":
        ft = _detect_film_type(profile, valleys, peaks)
    else:
        ft = film_type

    is_negative = (ft == "negative")

    # 3. Assign wire / gap roles
    if is_negative:
        wire_positions = peaks
        gap_positions = valleys
    else:
        wire_positions = valleys
        gap_positions = peaks

    # 4. Quadratic background fit (masking wire regions)
    background = _fit_quadratic_background(
        profile, wire_positions, inverted=not is_negative,
    )

    # 5. Pair wires and compute dips
    dips, pairs = _pair_wires_and_compute_dips(
        profile, wire_positions, gap_positions, background,
        half_w=window_half_width,
        dist_factor=1.05,
        film_type=ft,
    )

    return ComputeContrastResult(
        dips=dips,
        pairs=pairs,
        background=background,
        film_type=ft,
    )


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
