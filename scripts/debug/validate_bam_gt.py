#!/usr/bin/env python3
"""
Phase 3: BAM 双丝算法 Ground Truth 验证

Compares compute_contrast() and find_first_unresolved_group() output
against manually annotated groundtruth.json.

Usage:
    python scripts/debug/validate_bam_gt.py <profile_json> <groundtruth_json>
    python scripts/debug/validate_bam_gt.py <profile_json> <groundtruth_json> --vis

Options:
    --vis   输出可视化图表（3-panel PNG），保存在 profile 同目录下
"""

import argparse
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


def _pair_metrics(algo_pairs: list[tuple[int, int, int]], gt_pairs: list[dict]) -> dict:
    point_errors: list[float] = []
    triplet_max_errors: list[float] = []
    for algo_pair, gp in zip(algo_pairs, gt_pairs):
        gt_triplet = (gp["valley_a_idx"], gp["peak_idx"], gp["valley_b_idx"])
        errs = [abs(int(a) - int(g)) for a, g in zip(algo_pair, gt_triplet)]
        point_errors.extend(errs)
        triplet_max_errors.append(max(errs))
    mean_err = float(np.mean(point_errors)) if point_errors else float("inf")
    max_triplet_err = float(max(triplet_max_errors)) if triplet_max_errors else float("inf")
    return {
        "matched_pairs": min(len(algo_pairs), len(gt_pairs)),
        "extra_pairs": max(0, len(algo_pairs) - len(gt_pairs)),
        "missing_pairs": max(0, len(gt_pairs) - len(algo_pairs)),
        "mean_point_error": mean_err,
        "max_triplet_error": max_triplet_err,
    }


# ---------------------------------------------------------------------------
# Step 1: Parse args & load data
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="BAM double-wire GT validation")
parser.add_argument("profile_json", type=str, help="Path to profile JSON")
parser.add_argument("groundtruth_json", type=str, help="Path to groundtruth JSON")
parser.add_argument("--vis", action="store_true", help="Generate visualization PNG")
args = parser.parse_args()

profile_path = Path(args.profile_json)
gt_path = Path(args.groundtruth_json)

with open(profile_path) as f:
    profile_data = json.load(f)
profile = np.array(profile_data["profile_values"], dtype=np.float64)

with open(gt_path) as f:
    gt = json.load(f)

out_path = profile_path.parent / "validation_report.txt"
vis_path = profile_path.parent / "validation_vis.png"
out_lines: list[str] = []

def log(msg: str = ""):
    out_lines.append(msg)
    print(msg)


# ---------------------------------------------------------------------------
# Step 2: Run algorithm
# ---------------------------------------------------------------------------

log("=" * 80)
log("BAM DOUBLE-WIRE ALGORITHM -- GROUND TRUTH VALIDATION")
log("=" * 80)
log()

result = compute_contrast(profile, film_type="auto", min_distance=5, prominence=0.03)
metrics = _pair_metrics(result.pairs, gt["wire_pairs"])
film_type_ok = result.film_type == gt["film_type"]
pair_count_ok = len(result.pairs) == gt["num_wire_pairs"]
no_extra_missing_ok = metrics["extra_pairs"] == 0 and metrics["missing_pairs"] == 0
max_error_ok = metrics["max_triplet_error"] <= 5.0
mean_error_ok = metrics["mean_point_error"] <= 3.0
validation_pass = all([
    film_type_ok,
    pair_count_ok,
    no_extra_missing_ok,
    max_error_ok,
    mean_error_ok,
])

# 2a. Film type
log(f"Algorithm film_type: {result.film_type}")
log(f"GT film_type:        {gt['film_type']}")
flag = "" if result.film_type == gt["film_type"] else " ** MISMATCH"
log(f"Film type match:     {'YES' if result.film_type == gt['film_type'] else 'NO'}{flag}")
log()

# 2b. Pair count
log(f"Detected {len(result.dips)} wire pairs, GT has {gt['num_wire_pairs']}")
log(f"Matched GT pairs: {metrics['matched_pairs']}/{gt['num_wire_pairs']}")
log(f"Extra pairs:      {metrics['extra_pairs']}")
log(f"Missing pairs:    {metrics['missing_pairs']}")
log(f"Mean point error: {metrics['mean_point_error']:.3f}px  (PASS <= 3.000px)")
log(f"Max triplet err:  {metrics['max_triplet_error']:.3f}px  (PASS <= 5.000px)")
log()

# 2c. Extrema detection
algo_peaks, algo_valleys = detect_peaks_valleys(
    profile, min_distance=10, prominence=0.05,
)
gt_signal_valleys = sorted([p["idx"] for p in gt["all_peaks"]])
gt_signal_peaks   = sorted([v["idx"] for v in gt["all_valleys"]])

log("Raw extrema (detect_peaks_valleys on non-detrended profile):")
log(f"  Algorithm peaks:   {algo_peaks.tolist()}")
log(f"  Algorithm valleys: {algo_valleys.tolist()}")
log(f"  GT signal peaks    (wire positions):              {gt_signal_peaks}")
log(f"  GT signal valleys  (gap candidates):              {gt_signal_valleys}")
log()

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
log("  GT naming convention:")
log("    positive film: valley_a/b_idx = dark wire, peak_idx = bright gap")
log("    negative film: valley_a/b_idx = bright wire, peak_idx = dark gap")
log()

w1_errors: list[float] = []
gap_errors: list[float] = []
w2_errors: list[float] = []

for i, (algo_pair, gp) in enumerate(zip(result.pairs, gt["wire_pairs"])):
    a_w1, a_gap, a_w2 = algo_pair
    g_va = gp["valley_a_idx"]
    g_pk = gp["peak_idx"]
    g_vb = gp["valley_b_idx"]

    w1_err  = abs(a_w1 - g_va)
    gap_err = abs(a_gap - g_pk)
    w2_err  = abs(a_w2 - g_vb)

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

if metrics["extra_pairs"] or metrics["missing_pairs"]:
    log()
    log("NOTE: Pair count mismatch makes index-by-index comparison diagnostic only.")
    log()


# ---------------------------------------------------------------------------
# Step 4: Root cause analysis
# ---------------------------------------------------------------------------

log("=" * 80)
log("DIAGNOSTICS")
log("=" * 80)
log()

profile_range = float(profile.max() - profile.min())
log(f"Profile statistics:")
log(f"  Length: {len(profile)} px")
log(f"  Range:  {profile.min():.1f} - {profile.max():.1f} ({profile_range:.1f})")
log(f"  Global trend: left background ~{profile[:10].mean():.0f} -> "
    f"right tail ~{profile[-10:].mean():.0f}")
log(f"  Trend amplitude: {profile[:10].mean() - profile[-10:].mean():.1f}  "
    f"({100*(profile[:10].mean()-profile[-10:].mean())/profile_range:.0f}% of full range)")
log()

abs_prom = 0.05 * profile_range
log(f"scipy.signal.find_peaks parameters:")
log(f"  min_distance = 10")
log(f"  prominence   = 0.05 (relative), {abs_prom:.2f} (absolute)")
log()

# 4c: Spurious in bg region
log("Spurious detections in background region (profile indices 0-60):")
log("  Profile at this region drops gradually from ~247 to ~226.")
log("  Small fluctuations (0.2-1% range) satisfy prominence threshold,")
log("  causing false peaks at indices 36, 49.")
log()

# 4d: Inter-wire plateau
log("Extra peaks in inter-wire regions:")
log("  Between the two wires of D1 (indices ~125 and ~163), the profile")
log("  has a plateau at ~208-209 (indices 138-152). Fluctuations on this")
log("  plateau produce spurious peaks at 142, 184, etc., which the")
log("  algorithm treats as separate wire positions.")
log()

# 4e: min_distance suppression
log("min_distance=10 suppresses genuine valley detections:")
log("  GT signal valleys:  {gt_signal_valleys}")
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

# 4f: Pairing chain
log("Pairing chain effect:")
log("  Current compute_contrast uses fine local extrema for positive-film")
log("  adjacent-wire pairing, then trims the physically ordered prefix.")
log("  Raw extrema above are shown only as diagnostics.")
log()

# 4g: Dip with GT positions
log("Dip computation using GT positions (isolating detection vs formula):")
gt_wire_pos = np.array([125, 163, 199, 236, 265, 295, 324, 353])
if len(result.pairs) > 0 and len(result.pairs[0]) >= 3:
    # Use first detected pair's wire positions to anchor the GT comparison
    gt_wires_alt = []
    for wp in gt["wire_pairs"]:
        gt_wires_alt.append(wp["valley_a_idx"])
        gt_wires_alt.append(wp["valley_b_idx"])
    gt_wire_pos = np.array(sorted(set(gt_wires_alt)))
gt_gap_pos = np.array([wp["peak_idx"] for wp in gt["wire_pairs"]])

bg_gt = _fit_quadratic_background(profile, gt_wire_pos, inverted=False)
log(f"  Quadratic background range: {bg_gt.min():.1f} - {bg_gt.max():.1f}")
log(f"  Profile range at wire region: {profile[min(gt_wire_pos):max(gt_wire_pos)+1].min():.1f} - "
    f"{profile[min(gt_wire_pos):max(gt_wire_pos)+1].max():.1f}")
log(f"  Background at D1 centers (~index {gt_wire_pos[len(gt_wire_pos)//2]}): ~{bg_gt[gt_wire_pos[len(gt_wire_pos)//2]]:.1f}")
log()

log(f"  {'Pair':>6} {'dip':>8} {'A':>8} {'B':>8} {'C':>8} {'profile[gap]':>12} {'bg[gap]':>8} {'numerator':>10}")
log(f"  {'-'*70}")
gt_dips_viz: list[dict] = []
for i, gp in enumerate(gt["wire_pairs"]):
    w1, gap, w2 = gp["valley_a_idx"], gp["peak_idx"], gp["valley_b_idx"]
    dip = _compute_dip(profile, w1, gap, w2, bg_gt, half_w=0)

    L = len(profile)
    def _region_mean(center):
        lo = center
        hi = center
        return float(profile[lo:hi + 1].mean())

    a_mean = _region_mean(w1)
    c_mean = _region_mean(gap)
    b_mean = _region_mean(w2)
    A = abs(float(bg_gt[w1]) - a_mean)
    B = abs(float(bg_gt[w2]) - b_mean)
    C = abs(float(bg_gt[gap]) - c_mean)
    numer = A + B - 2 * C
    log(f"  D{gp['group']}: {dip:7.1f}%  {A:7.2f} {B:7.2f} {C:7.2f}  "
        f"{c_mean:10.2f}  {bg_gt[gap]:7.2f}  {numer:+9.2f}")
    gt_dips_viz.append({"w1": w1, "gap": gap, "w2": w2, "dip": dip})

log()
log("  Observation: This diagnostic uses single extrema points (half_w=0),")
log("  matching the current compute_contrast dip path.")
log()

# 4h: Parameter sweep
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
log("  Parameter sweep is diagnostic only; PASS/FAIL is based on pair triplet")
log("  positions against GT, not raw extrema overlap.")
log()


# ---------------------------------------------------------------------------
# Step 5: Summary
# ---------------------------------------------------------------------------

log("=" * 80)
log("SUMMARY")
log("=" * 80)
log()

log(f"Validation result: {'PASS' if validation_pass else 'FAIL'}")
log()
log("Acceptance checks:")
log(f"  film_type matches GT:        {'PASS' if film_type_ok else 'FAIL'}")
log(f"  pair count equals GT:        {'PASS' if pair_count_ok else 'FAIL'}")
log(f"  no extra/missing pairs:      {'PASS' if no_extra_missing_ok else 'FAIL'}")
log(f"  max triplet error <= 5 px:   {'PASS' if max_error_ok else 'FAIL'} "
    f"({metrics['max_triplet_error']:.3f}px)")
log(f"  mean point error <= 3 px:    {'PASS' if mean_error_ok else 'FAIL'} "
    f"({metrics['mean_point_error']:.3f}px)")

log("=" * 80)
log("END OF REPORT")
log("=" * 80)

# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    f.write("\n".join(out_lines))

log(f"\nReport saved to: {out_path}")


# ---------------------------------------------------------------------------
# Visualization (--vis)
# ---------------------------------------------------------------------------

if args.vis:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)

    fig = plt.figure(figsize=(16, 12))
    x = np.arange(len(profile), dtype=np.float64)
    prof_min, prof_max = float(profile.min()), float(profile.max())
    y_pad = (prof_max - prof_min) * 0.08

    n_compare = min(len(result.pairs), len(gt["wire_pairs"]))
    n_cols = min(4, n_compare)
    n_rows = max(1, (n_compare + n_cols - 1) // n_cols)

    # --- Panel 1: Per-pair GT vs Algorithm comparison ---
    gs = fig.add_gridspec(3, 1, height_ratios=[max(2, n_rows), 3, 2], hspace=0.35)
    top_gs = gs[0].subgridspec(n_rows, n_cols, wspace=0.3, hspace=0.5)

    for pair_idx in range(n_compare):
        row, col = divmod(pair_idx, n_cols)
        ax_pair = fig.add_subplot(top_gs[row, col])

        algo_w1, algo_g, algo_w2 = result.pairs[pair_idx]
        dip = result.dips[pair_idx]
        gp = gt["wire_pairs"][pair_idx]
        gt_w1, gt_g, gt_w2 = gp["valley_a_idx"], gp["peak_idx"], gp["valley_b_idx"]

        # Zoom window around this pair
        margin = 10
        x_lo = max(0, min(algo_w1, gt_w1) - margin)
        x_hi = min(len(profile), max(algo_w2, gt_w2) + margin)
        xs = np.arange(x_lo, x_hi)
        ax_pair.plot(xs, profile[xs], color="#4C78A8", linewidth=1.2, label=None)

        # GT markers (solid vertical lines with labels)
        ax_pair.axvline(gt_w1, color="#54A24B", linewidth=1.5, linestyle="-", alpha=0.8)
        ax_pair.axvline(gt_g, color="#F58518", linewidth=1.5, linestyle="-", alpha=0.8)
        ax_pair.axvline(gt_w2, color="#54A24B", linewidth=1.5, linestyle="-", alpha=0.8)

        # Algorithm markers (dashed vertical lines)
        ax_pair.axvline(algo_w1, color="#E45756", linewidth=1.2, linestyle="--", alpha=0.8)
        ax_pair.axvline(algo_g, color="#4C78A8", linewidth=1.2, linestyle="--", alpha=0.8)
        ax_pair.axvline(algo_w2, color="#E45756", linewidth=1.2, linestyle="--", alpha=0.8)

        err1, errg, err2 = abs(algo_w1 - gt_w1), abs(algo_g - gt_g), abs(algo_w2 - gt_w2)
        grp_label = gp.get("group", pair_idx + 1)
        ax_pair.set_title(f"D{grp_label}  dip={dip:.0f}%  Δ=({err1},{errg},{err2})px", fontsize=7.5)
        ax_pair.tick_params(labelsize=6)

    # Legend on last subplot (or first if only 1)
    legend_ax = fig.add_subplot(top_gs[0, -1]) if n_compare > 1 else fig.add_subplot(top_gs[0, 0])
    legend_ax.plot([], [], color="#54A24B", linewidth=1.5, linestyle="-", label="GT wire")
    legend_ax.plot([], [], color="#F58518", linewidth=1.5, linestyle="-", label="GT gap")
    legend_ax.plot([], [], color="#E45756", linewidth=1.2, linestyle="--", label="Algo wire")
    legend_ax.plot([], [], color="#4C78A8", linewidth=1.2, linestyle="--", label="Algo gap")
    legend_ax.legend(fontsize=6, loc="center")
    legend_ax.axis("off")

    fig.text(0.5, 0.96, "Panel 1 — Per-pair GT vs Algorithm Comparison", fontsize=10,
             ha="center", weight="bold")

    # --- Panel 2: Background + Algorithm pairs (full profile) ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x, profile, color="#4C78A8", linewidth=0.8, alpha=0.6, label="Profile")
    ax2.plot(x, result.background, color="#F58518", linewidth=1.5, label="SG background")

    for i, ((w1, g, w2), dip) in enumerate(zip(result.pairs, result.dips)):
        alpha = 0.15 if dip >= 20 else 0.08
        color = "#54A24B" if dip >= 20 else "#E45756"
        ax2.axvspan(w1, w2, alpha=alpha, color=color)
        mid = (w1 + w2) // 2
        ax2.annotate(f"D{i+1}:{dip:.0f}%", (mid, profile[g]),
                     textcoords="offset points", xytext=(0, 14),
                     fontsize=6.5, color=color, ha="center", weight="bold")

    for w1, g, w2 in result.pairs:
        ax2.plot(w1, profile[w1], "r.", markersize=4, alpha=0.7)
        ax2.plot(g, profile[g], "b.", markersize=4, alpha=0.7)
        ax2.plot(w2, profile[w2], "r.", markersize=4, alpha=0.7)

    unresolved_str = f"D{find_first_unresolved_group(result.dips)}" if find_first_unresolved_group(result.dips) else "none"
    ax2.set_title(f"Panel 2 — Algorithm overview: {len(result.pairs)} pairs, film={result.film_type}, "
                  f"1st unresolved={unresolved_str}", fontsize=9)
    ax2.legend(fontsize=7, loc="upper right")

    # --- Panel 3: GT vs Algorithm full overlay ---
    ax3 = fig.add_subplot(gs[2], sharex=ax2)
    ax3.plot(x, profile, color="#4C78A8", linewidth=0.8, alpha=0.5, label="Profile")

    for gp in gt["wire_pairs"]:
        g_va, g_pk, g_vb = gp["valley_a_idx"], gp["peak_idx"], gp["valley_b_idx"]
        ax3.axvline(g_va, color="#54A24B", alpha=0.6, linewidth=1.2, linestyle="-")
        ax3.axvline(g_pk, color="#F58518", alpha=0.6, linewidth=1.2, linestyle="-")
        ax3.axvline(g_vb, color="#54A24B", alpha=0.6, linewidth=1.2, linestyle="-")

    for i, ((w1, g, w2), dip) in enumerate(zip(result.pairs, result.dips)):
        ax3.axvline(w1, color="#E45756", alpha=0.5, linewidth=0.8, linestyle=":")
        ax3.axvline(g, color="#4C78A8", alpha=0.5, linewidth=0.8, linestyle=":")
        ax3.axvline(w2, color="#E45756", alpha=0.5, linewidth=0.8, linestyle=":")

    ax3.plot([], [], color="#54A24B", linewidth=1.2, linestyle="-", label="GT wires / gaps")
    ax3.plot([], [], color="#E45756", linewidth=0.8, linestyle=":", label="Algo wires / gaps")
    ax3.set_title(f"Panel 3 — GT ({gt['film_type']}, {gt['num_wire_pairs']} pairs) vs Algorithm "
                  f"({result.film_type}, {len(result.pairs)} pairs)", fontsize=9)
    ax3.legend(fontsize=7, loc="upper right", ncol=2)

    if len(result.pairs) > 0:
        wire_start = max(0, result.pairs[0][0] - 20)
        wire_end = min(len(profile), result.pairs[-1][2] + 20)
        ax3.set_xlim(wire_start, wire_end)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(vis_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Visualization saved to: {vis_path}")
