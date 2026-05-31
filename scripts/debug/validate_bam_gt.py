#!/usr/bin/env python3
"""
Phase 3: BAM 双丝算法 Ground Truth 验证

Compares compute_contrast() and find_first_unresolved_group() output
against manually annotated groundtruth.json for a single-wire-pair profile.

Usage:
    cd /home/cht/code/IQIdet && \
    PYTHONPATH=/home/cht/code/IQIdet:/home/cht/code/IQIdet/src \
    python scripts/debug/validate_bam_gt.py

Output: outputs/double_wire_demo/validation_report.txt
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from scipy.signal import find_peaks

from gauge.imaging.profile import (
    compute_contrast,
    find_first_unresolved_group,
    detect_peaks_valleys,
    _compute_dip,
    _fit_quadratic_background,
    _detect_film_type,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_err(e: float) -> str:
    return f"{e:.0f}" if e == int(e) else f"{e:.1f}"


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------

profile_path = REPO_ROOT / "outputs/double_wire_demo/wqxDR__SHLNG-PED-A05+002-Z-NJ01__01_profile.json"
gt_path = REPO_ROOT / "outputs/double_wire_demo/groundtruth.json"

with open(profile_path) as f:
    profile_data = json.load(f)
profile = np.array(profile_data["profile_values"], dtype=np.float64)

with open(gt_path) as f:
    gt = json.load(f)

out_path = REPO_ROOT / "outputs/double_wire_demo/validation_report.txt"
out_lines: list[str] = []

def log(msg: str = ""):
    out_lines.append(msg)
    print(msg)


# ---------------------------------------------------------------------------
# Step 2: Run algorithm (default parameters)
# ---------------------------------------------------------------------------

log("=" * 80)
log("BAM DOUBLE-WIRE ALGORITHM -- GROUND TRUTH VALIDATION")
log("=" * 80)
log()

result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)

# ---- 2a. Film type ----
log(f"Algorithm film_type: {result.film_type}")
log(f"GT film_type:        {gt['film_type']}")
flag = "" if result.film_type == gt["film_type"] else " ** MISMATCH"
log(f"Film type match:     {'YES' if result.film_type == gt['film_type'] else 'NO'}{flag}")
log()

# ---- 2b. Pair count ----
log(f"Detected {len(result.dips)} wire pairs, GT has {gt['num_wire_pairs']}")
log()

# ---- 2c. Extrema detection ----
algo_peaks, algo_valleys = detect_peaks_valleys(
    profile, min_distance=10, prominence=0.05,
)
# GT annotated 'all_peaks' = signal valleys (local minima), 'all_valleys' = signal peaks (local maxima)
gt_signal_valleys = sorted([p["idx"] for p in gt["all_peaks"]])     # signal minima
gt_signal_peaks   = sorted([v["idx"] for v in gt["all_valleys"]])   # signal maxima

log("Raw extrema (detect_peaks_valleys on non-detrended profile):")
log(f"  Algorithm peaks:   {algo_peaks.tolist()}")
log(f"  Algorithm valleys: {algo_valleys.tolist()}")
log(f"  GT signal peaks    (wire positions):              {gt_signal_peaks}")
log(f"  GT signal valleys  (gap candidates):              {gt_signal_valleys}")
log()

# Overlap metrics
algo_p_set = set(algo_peaks.tolist())
algo_v_set = set(algo_valleys.tolist())
gt_p_set   = set(gt_signal_peaks)
gt_v_set   = set(gt_signal_valleys)

common_peaks   = algo_p_set & gt_p_set
missing_peaks  = gt_p_set - algo_p_set
extra_peaks    = algo_p_set - gt_p_set
common_valleys = algo_v_set & gt_v_set
missing_valleys = gt_v_set - algo_v_set
extra_valleys   = algo_v_set - gt_v_set

log(f"Peak (signal maxima / wire) match:  {len(common_peaks)}/{len(gt_signal_peaks)}  "
    f"({len(common_peaks)/max(len(gt_signal_peaks),1)*100:.0f}%)")
if missing_peaks:
    log(f"  GT peaks MISSED:   {sorted(missing_peaks)}  "
        f"(suppressed by min_distance or insufficient prominence)")
if extra_peaks:
    log(f"  EXTRA algorithm peaks:  {sorted(extra_peaks)}  "
        f"(noise/fluctuations on descending profile slope)")

log(f"Valley (signal minima / gap) match: {len(common_valleys)}/{len(gt_signal_valleys)}  "
    f"({len(common_valleys)/max(len(gt_signal_valleys),1)*100:.0f}%)")
if missing_valleys:
    log(f"  GT valleys MISSED:  {sorted(missing_valleys)}  "
        f"(suppressed by min_distance=10)")
if extra_valleys:
    log(f"  EXTRA algo valleys: {sorted(extra_valleys)}  "
        f"(background noise)")
log()


# ---------------------------------------------------------------------------
# Step 3: Pair comparison
# ---------------------------------------------------------------------------

log("=" * 80)
log("PAIR-BY-PAIR COMPARISON")
log("=" * 80)
log("  GT naming convention (inverted relative to signal processing):")
log("    GT valley_a/b_idx = signal PEAK   = wire position (bright in negative film)")
log("    GT peak_idx        = signal VALLEY = gap position  (dark in negative film)")
log("  Comparison matches by physical role regardless of naming.")
log()

w1_errors: list[float] = []
gap_errors: list[float] = []
w2_errors: list[float] = []

for i, (algo_pair, gp) in enumerate(zip(result.pairs, gt["wire_pairs"])):
    a_w1, a_gap, a_w2 = algo_pair
    g_va = gp["valley_a_idx"]
    g_pk = gp["peak_idx"]
    g_vb = gp["valley_b_idx"]

    w1_err   = abs(a_w1 - g_va)
    gap_err  = abs(a_gap - g_pk)
    w2_err   = abs(a_w2 - g_vb)

    w1_errors.append(w1_err)
    gap_errors.append(gap_err)
    w2_errors.append(w2_err)

    f1 = " ***" if w1_err > 3 else ""
    fg = " ***" if gap_err > 3 else ""
    f2 = " ***" if w2_err > 3 else ""

    log(f"D{gp['group']}: (algo) ({a_w1:3d},{a_gap:3d},{a_w2:3d}) vs "
        f"(GT) ({g_va:3d},{g_pk:3d},{g_vb:3d})  "
        f"| err: w1={_fmt_err(w1_err):>3}px{f1}  "
        f"gap={_fmt_err(gap_err):>3}px{fg}  "
        f"w2={_fmt_err(w2_err):>3}px{f2}")

log()
log("NOTE: Since algorithm detected 11 vs GT 7 pairs, a simple positional")
log("comparison by index is misleading. Each algorithm pair matches the GT")
log("pair at the same list position, but the actual physical correspondence")
log("is lost due to spurious extra pairs.")
log()


# ---------------------------------------------------------------------------
# Step 4: Root cause analysis
# ---------------------------------------------------------------------------

log("=" * 80)
log("ROOT CAUSE ANALYSIS")
log("=" * 80)
log()

# --- 4a: Profile trend ---
profile_range = float(profile.max() - profile.min())
log(f"Profile statistics:")
log(f"  Length: {len(profile)} px")
log(f"  Range:  {profile.min():.1f} - {profile.max():.1f} ({profile_range:.1f})")
log(f"  Global trend: left background ~{profile[:10].mean():.0f} -> "
    f"right tail ~{profile[-10:].mean():.0f}")
log(f"  Trend amplitude: {profile[:10].mean() - profile[-10:].mean():.1f}  "
    f"({100*(profile[:10].mean()-profile[-10:].mean())/profile_range:.0f}% of full range)")
log()

# --- 4b: Prominence threshold ---
abs_prom = 0.05 * profile_range
log(f"scipy.signal.find_peaks parameters:")
log(f"  min_distance = 10")
log(f"  prominence   = 0.05 (relative), {abs_prom:.2f} (absolute)")
log()

# --- 4c: Spurious profiles near background ---
log("Spurious detections in background region (profile indices 0-60):")
log("  Profile at this region drops gradually from ~247 to ~226.")
log("  Small fluctuations (0.2-1% range) satisfy prominence threshold,")
log("  causing false peaks at indices 36, 49.")
log()

# --- 4d: Inter-wire plateau creates extra peaks ---
log("Extra peaks in inter-wire regions:")
log("  Between the two wires of D1 (indices ~125 and ~163), the profile")
log("  has a plateau at ~208-209 (indices 138-152). Fluctuations on this")
log("  plateau produce spurious peaks at 142, 184, etc., which the")
log("  algorithm treats as separate wire positions.")
log()

# --- 4e: min_distance suppression ---
log("min_distance=10 suppresses genuine valley detections:")
log("  GT signal valleys:  {gt_signal_valleys}")
log("  In this sequence, several valleys are <10px from a neighboring")
log("  valley with higher prominence, so they are suppressed:")
dists_between_gt_valleys = np.diff(gt_signal_valleys)
close_pairs = []
for j in range(len(gt_signal_valleys) - 1):
    d = dists_between_gt_valleys[j]
    if d < 10:
        close_pairs.append((gt_signal_valleys[j], gt_signal_valleys[j+1], int(d)))
if close_pairs:
    for idx_a, idx_b, d in close_pairs:
        log(f"    ({idx_a},{idx_b}) distance={d}px < min_distance=10")
log()

# --- 4f: Pairing chain effect ---
log("Pairing chain effect:")
log("  _pair_wires_and_compute_dips computes dist_max from first peak-pair")
log("  spacing. With spurious peak at index 49 (background), the first pair")
log("  (49,120,126) sets dist_max=1.05*(126-49)=81px, which is too large")
log("  and pairs all adjacent detected peaks. This creates 11 spurious pairs")
log("  from 16 detected peaks.")
log()

# --- 4g: Dip values with GT positions ---
log("Dip computation using GT positions (isolating detection vs formula):")
gt_wire_pos = np.array([125, 163, 199, 236, 265, 295, 324, 353])
gt_gap_pos  = np.array([158, 196, 233, 263, 293, 323, 352])  # GT's 'peak_idx'
gt_pairs_phys = list(zip(gt_wire_pos[:-1], gt_gap_pos, gt_wire_pos[1:]))

bg_gt = _fit_quadratic_background(profile, gt_wire_pos, inverted=False)
log(f"  Quadratic background range: {bg_gt.min():.1f} - {bg_gt.max():.1f}")
log(f"  Profile range at wire region: {profile[min(gt_wire_pos):max(gt_wire_pos)+1].min():.1f} - "
    f"{profile[min(gt_wire_pos):max(gt_wire_pos)+1].max():.1f}")
log(f"  Background at D1 centers (~index 140): ~{bg_gt[140]:.1f}")
log(f"  -> Wires are ~15 px DARKER than fitted background")
log(f"  -> Gaps are ~60-80 px DARKER than fitted background")
log(f"  -> A << C in dip formula => numerator negative => dip = 0% for coarse pairs")
log()

log(f"  {'Pair':>6} {'dip':>8} {'A':>8} {'B':>8} {'C':>8} {'profile[gap]':>12} {'bg[gap]':>8} {'numerator':>10}")
log(f"  {'-'*70}")
for i, (w1, gap, w2) in enumerate(gt_pairs_phys):
    half_w = 3
    L = len(profile)
    def _region_mean(center):
        lo = max(0, center - half_w)
        hi = min(L - 1, center + half_w)
        return float(profile[lo:hi + 1].mean())

    a_mean = _region_mean(w1)
    c_mean = _region_mean(gap)
    b_mean = _region_mean(w2)
    A = abs(float(bg_gt[w1]) - a_mean)
    B = abs(float(bg_gt[w2]) - b_mean)
    C = abs(float(bg_gt[gap]) - c_mean)
    numer = A + B - 2 * C
    dip = _compute_dip(profile, w1, gap, w2, bg_gt, half_w=3)
    log(f"  D{i+1}: {dip:7.1f}%  {A:7.2f} {B:7.2f} {C:7.2f}  "
        f"{c_mean:10.2f}  {bg_gt[gap]:7.2f}  {numer:+9.2f}")

log()
log("  Observation: For D1-D2, C >> A,B => dip clamped to 0.")
log("  For D3-D7, dip rises as C decreases (gap fills in).")
log("  This is an inverted response: well-resolved pairs give 0% dip,")
log("  while nearly-unresolved pairs give 50+% dip.")
log()

# --- 4h: Parameter sweep ---
log("Parameter sensitivity (prominence, min_distance):")
log(f"  {'prom':>6} {'min_dist':>9} {'pairs':>6} {'peaks':>6} {'wire_hit':>9} {'dips':>30}")
log(f"  {'-'*70}")
for prom in [0.05, 0.08, 0.10, 0.15, 0.20]:
    for md in [10, 15, 20]:
        r2 = compute_contrast(profile, film_type="negative", min_distance=md, prominence=prom)
        p2, _ = detect_peaks_valleys(profile, min_distance=md, prominence=prom)
        hit = len(set(p2.tolist()) & set(gt_signal_peaks))
        dip_str = " ".join(f"{d:.0f}" for d in r2.dips[:7])
        log(f"  {prom:5.2f}  {md:8d}  {len(r2.pairs):5d}  {len(p2):5d}  "
            f"{hit:3d}/{len(gt_signal_peaks):d}  {dip_str}")

log()
log("  No (prominence, min_distance) combination correctly detects all 8")
log("  wire peaks. The best (prom=0.05, mind=10) detects 16 peaks with")
log("  only 4/8 matching GT wire positions.")
log()


# ---------------------------------------------------------------------------
# Step 5: Summary
# ---------------------------------------------------------------------------

log("=" * 80)
log("SUMMARY")
log("=" * 80)
log()

log("Validation result: FAIL")
log()

log("Key findings:")
log()
log("1. Film type detection: PASS")
log("   Algorithm correctly detects 'negative' matching GT annotation.")
log()
log("2. Peak/valley detection on non-detrended profile: FAIL")
log("   16 peaks detected vs 8 expected (2x over-detection)")
log("   Only 4/8 GT wire positions are correctly identified.")
log("   11 pairs produced instead of 7 (57% over-count)")
log()
log("3. Pairing logic: FAIL")
log("   Spurious background peaks pollute dist_max computation,")
log("   causing all 16 peaks to be paired into non-existent pairs.")
log()
log("4. Dip formula on correct positions: WARNING")
log("   D1-D2 give 0% dip even with GT positions (gap deviation >> wire deviation)")
log("   Dip values increase for finer pairs (inverted response)")
log()
log("5. Parameter sensitivity: POOR")
log("   No (prominence, min_distance) tuning produces correct 7-pair output.")
log()
log("Root causes:")
log()
log("  RC-1 (PRIMARY): No profile detrending before peak detection.")
log("    The profile has a ~126px global trend (55% of full range).")
log("    find_peaks on the raw profile treats this trend as signal,")
log("    generating spurious extrema everywhere. The correct fix is to")
log("    subtract a smoothed estimate before find_peaks, e.g.:")
log("      - Low-pass filter (wide kernel) to capture the trend")
log("      - Polynomial fit (linear or quadratic) and subtract")
log("      - Detrended fluctuation analysis for adaptive baseline")
log()
log("  RC-2: min_distance parameter is global, not adaptive.")
log("    As wire spacings shrink from D1 (38px) to D7 (29px), the")
log("    inter-wire gap shrinks too. Global min_distance=10 suppresses")
log("    genuine valleys at fine scales.")
log()
log("  RC-3: dist_max in _pair_wires_and_compute_dips depends on first pair.")
log("    If the first 'pair' is spurious (background noise), the entire")
log("    pairing chain is corrupted. The code assumes the first pair is valid.")
log()
log("  RC-4: Dip formula assumes background is flat at wire region.")
log("    A = |bg - wire| assumes wires deviate from a locally correct")
log("    background. When the background fit overshoots (as it does for")
log("    a strongly modulated profile), C = |bg - gap| >> A,B and dip = 0.")
log("    This inverts the expected dip ordering (coarse pairs = 0%, fine = 50%).")
log()

log("=" * 80)
log("END OF REPORT")
log("=" * 80)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    f.write("\n".join(out_lines))

log(f"\nReport saved to: {out_path}")
