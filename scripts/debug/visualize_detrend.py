#!/usr/bin/env python3
"""Visualise detrending: show 2D strip, original profile + trend line, and
detrended profile side-by-side for a single double-wire strip image.

Usage:
    python scripts/debug/visualize_detrend.py \\
        outputs/doubledebug_0604/wqxDR__SHLNG-PED-A06+009-Z-NJ01__01/ori/wqxDR__SHLNG-PED-A06+009-Z-NJ01__01_strip.png \\
        [--output result.png] [--band-width 21] [--interactive]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── reproduce the exact detrend logic from src/gauge/imaging/profile.py ──

_coeffs: list[float] = []


def detrend_profile(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit 2nd-order polynomial and return (trend, detrended)."""
    global _coeffs
    n = len(profile)
    x = np.arange(n, dtype=np.float64)
    y = profile.astype(np.float64)
    coeffs = np.polyfit(x, y, 2)          # y = a·x² + b·x + c
    _coeffs = list(coeffs)                 # store for logging
    trend = np.polyval(coeffs, x)          # evaluated trend
    detrended = y - trend                  # subtract trend
    return trend, detrended


# ── tiny helper: extract 1D band-averaged profile from a 2D strip ──

def band_average(strip: np.ndarray, band_width: int = 21) -> np.ndarray:
    """Column-wise mean of the centre *band_width* rows of *strip*."""
    h = strip.shape[0]
    half = band_width // 2
    centre = h // 2
    lo = max(0, centre - half)
    hi = min(h, centre + half + 1)
    return strip[lo:hi, :].mean(axis=0).astype(np.float64)


# ── main ──────────────────────────────────────────────────────────────────

def main(
    strip_path: str,
    band_width: int = 21,
    output: str | None = None,
    interactive: bool = False,
) -> None:
    strip_path = Path(strip_path)
    if not strip_path.is_file():
        print(f"File not found: {strip_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load the 2D strip image
    strip = cv2.imread(str(strip_path), cv2.IMREAD_UNCHANGED)
    if strip is None:
        print(f"Failed to read: {strip_path}", file=sys.stderr)
        sys.exit(1)
    strip = strip.astype(np.float64)
    strip_h, strip_w = strip.shape
    half_h = strip_h // 2
    print(f"Strip shape: {strip_h} rows × {strip_w} cols  |  centre row = {half_h}")

    # 2. Extract band-averaged 1D profile
    profile = band_average(strip, band_width=band_width)
    print(f"Profile length: {len(profile)}, band_width: {band_width}")

    # 3. Detrend
    trend, detrended = detrend_profile(profile)
    print(f"Trend coeffs (a·x² + b·x + c): a={_coeffs[0]:.6f}  "
          f"b={_coeffs[1]:.6f}  c={_coeffs[2]:.4f}")
    print(f"Raw profile   — min: {profile.min():.2f}  max: {profile.max():.2f}  "
          f"range: {profile.max() - profile.min():.2f}")
    print(f"Trend         — min: {trend.min():.2f}  max: {trend.max():.2f}  "
          f"range: {trend.max() - trend.min():.2f}")
    print(f"Detrended     — min: {detrended.min():.2f}  max: {detrended.max():.2f}  "
          f"range: {detrended.max() - detrended.min():.2f}")

    # 4. Peak/valley detection on detrended profile
    from scipy.signal import find_peaks
    data_range = float(np.max(detrended) - np.min(detrended))
    promin = max(data_range * 0.05, 0.5)
    peaks, _ = find_peaks(detrended, distance=10, prominence=promin)
    valleys, _ = find_peaks(-detrended, distance=10, prominence=promin)
    print(f"Detected on detrended: peaks={len(peaks)}  valleys={len(valleys)}")

    # 5. Also show what happens WITHOUT detrend (peak detection on raw)
    raw_range = float(np.max(profile) - np.min(profile))
    raw_promin = max(raw_range * 0.05, 0.5)
    raw_peaks, _ = find_peaks(profile, distance=10, prominence=raw_promin)
    raw_valleys, _ = find_peaks(-profile, distance=10, prominence=raw_promin)
    print(f"Detected on raw:     peaks={len(raw_peaks)}  valleys={len(raw_valleys)}")

    # ── Plot ───────────────────────────────────────────────────────────
    if not interactive:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg", force=True)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"{strip_path.stem}  |  strip {strip_h}×{strip_w}  |  band_width={band_width}",
        fontsize=10,
    )

    # Panel 1: 2D strip image
    ax_strip = fig.add_subplot(4, 1, 1)
    ax_strip.imshow(strip, cmap="gray", aspect="auto")
    ax_strip.axhline(y=half_h, color="red", linewidth=1.0, label="midline")
    half_bw = (band_width - 1) / 2
    ax_strip.axhline(y=half_h - half_bw, color="red", linewidth=0.5,
                     linestyle="--", label="band edge")
    ax_strip.axhline(y=half_h + half_bw, color="red", linewidth=0.5, linestyle="--")
    y_ticks = [0, half_h, strip_h - 1]
    y_labels = [f"−{half_h}", "0", f"+{half_h}"]
    ax_strip.set_yticks(y_ticks)
    ax_strip.set_yticklabels(y_labels)
    ax_strip.set_title("2D Strip (dashed red = band edges)")
    ax_strip.legend(fontsize=7, loc="upper right")

    # Panel 2: raw profile + quadratic trend
    ax_raw = fig.add_subplot(4, 1, 2)
    x = np.arange(len(profile))
    ax_raw.plot(x, profile, color="#4C78A8", linewidth=1.2, label="raw profile")
    ax_raw.plot(x, trend, color="#d62728", linewidth=1.5,
                linestyle="--", label="quadratic trend (a·x²+b·x+c)")
    ax_raw.fill_between(x, profile, trend, alpha=0.15, color="red")
    # Show raw peak/valley detection on this panel
    if len(raw_peaks) > 0:
        ax_raw.plot(raw_peaks, profile[raw_peaks], "v", color="#d62728",
                    markersize=5, markeredgecolor="black", markeredgewidth=0.5,
                    label=f"raw peaks ({len(raw_peaks)})", alpha=0.6)
    if len(raw_valleys) > 0:
        ax_raw.plot(raw_valleys, profile[raw_valleys], "^", color="#1f77b4",
                    markersize=5, markeredgecolor="black", markeredgewidth=0.5,
                    label=f"raw valleys ({len(raw_valleys)})", alpha=0.6)
    ax_raw.set_title("Raw Profile + Quadratic Trend (dashed red)")
    ax_raw.set_ylabel("Gray value")
    ax_raw.legend(fontsize=6, loc="upper right")

    # Panel 3: detrended profile
    ax_det = fig.add_subplot(4, 1, 3)
    ax_det.plot(x, detrended, color="#2ca02c", linewidth=1.2,
                label="detrended profile")
    ax_det.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    if len(peaks) > 0:
        ax_det.plot(peaks, detrended[peaks], "v", color="#d62728",
                    markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                    label=f"peaks ({len(peaks)})")
    if len(valleys) > 0:
        ax_det.plot(valleys, detrended[valleys], "^", color="#1f77b4",
                    markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                    label=f"valleys ({len(valleys)})")
    ax_det.set_title("Detrended Profile (profile − trend) + Peaks & Valleys")
    ax_det.set_ylabel("Detrended gray value")
    ax_det.legend(fontsize=7, loc="upper right")

    # Panel 4: side-by-side comparison of raw vs detrended peak detection
    ax_cmp = fig.add_subplot(4, 1, 4)
    labels = ["raw peaks", "raw valleys", "detr peaks", "detr valleys"]
    values = [len(raw_peaks), len(raw_valleys), len(peaks), len(valleys)]
    colors = ["#d62728", "#1f77b4", "#d62728", "#1f77b4"]
    alphas = [0.45, 0.45, 0.9, 0.9]
    bars = []
    for i in range(4):
        bars.append(ax_cmp.bar(i, values[i], color=colors[i],
                               alpha=alphas[i], edgecolor="black", linewidth=0.5))
    ax_cmp.set_xticks(range(4))
    ax_cmp.set_xticklabels(labels)
    ax_cmp.set_title("Peak/Valley Count Comparison (same prominence=5% after detrend)")
    ax_cmp.set_ylabel("Count")
    for i, v in enumerate(values):
        ax_cmp.text(i, v + 0.3, str(v), ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()

    if interactive:
        plt.show()
    else:
        out_path = Path(output) if output else strip_path.with_suffix(".detrend.png")
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)




if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Visualise profile detrending")
    ap.add_argument("strip_png", help="Path to a *_strip.png file")
    ap.add_argument("--output", "-o", default=None,
                    help="Output PNG path (default: <strip>.detrend.png)")
    ap.add_argument("--band-width", "-b", type=int, default=21,
                    help="Number of centre rows to average for 1D profile")
    ap.add_argument("--interactive", "-i", action="store_true",
                    help="Show interactive matplotlib window instead of saving")
    a = ap.parse_args()
    main(a.strip_png, band_width=a.band_width, output=a.output,
         interactive=a.interactive)
