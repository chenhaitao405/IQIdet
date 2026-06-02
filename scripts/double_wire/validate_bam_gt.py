#!/usr/bin/env python3
"""
Phase 3: BAM 双丝算法 Ground Truth 验证

Compares compute_contrast() and find_first_unresolved_group() output
against manually annotated groundtruth.json.

Usage:
    # 单对验证
    python scripts/double_wire/validate_bam_gt.py <profile_json> <groundtruth_json>
    python scripts/double_wire/validate_bam_gt.py <profile_json> <groundtruth_json> --vis

    # 批量验证（自动配对目录下的 *_profile.json 和 *_groundtruth.json）
    python scripts/double_wire/validate_bam_gt.py <directory>
    python scripts/double_wire/validate_bam_gt.py <directory> --vis

Options:
    --vis                 输出可视化图表（3-panel PNG），保存在 profile 同目录下
    --save-intermediates  保存算法中间结果 JSON，用于原理文档
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
    _pair_adjacent_wires_with_gaps,
    _pair_direction_scores,
    _pair_wires_and_compute_dips,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_err(e: float) -> str:
    return f"{e:.0f}" if e == int(e) else f"{e:.1f}"


def _gt_triplet(gp: dict) -> tuple[int, int, int]:
    if "wire_a_idx" in gp:
        return int(gp["wire_a_idx"]), int(gp["gap_idx"]), int(gp["wire_b_idx"])
    return int(gp["valley_a_idx"]), int(gp["peak_idx"]), int(gp["valley_b_idx"])


def _pair_metrics(algo_pairs: list[tuple[int, int, int]], gt_pairs: list[dict]) -> dict:
    point_errors: list[float] = []
    triplet_max_errors: list[float] = []
    for algo_pair, gp in zip(algo_pairs, gt_pairs):
        gt_triplet = _gt_triplet(gp)
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
parser.add_argument("path_a", type=str, help="Profile JSON path, or directory for batch mode")
parser.add_argument("path_b", type=str, nargs="?", default=None,
                    help="Ground truth JSON path (omit if path_a is a directory)")
parser.add_argument("--vis", action="store_true", help="Generate visualization PNG")
parser.add_argument("--save-intermediates", action="store_true",
                    help="Save intermediate algorithm results as JSON")
args = parser.parse_args()

path_a = Path(args.path_a)


def _find_profile_gt_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Find matching (*_profile.json, *_groundtruth.json) pairs in a directory."""
    profiles = sorted(directory.glob("*_profile.json"))
    pairs = []
    for prof in profiles:
        stem = prof.stem
        if stem.endswith("_profile"):
            gt_name = stem[:-len("_profile")] + "_groundtruth.json"
        else:
            gt_name = stem + "_groundtruth.json"
        gt = prof.with_name(gt_name)
        if gt.is_file():
            pairs.append((prof, gt))
    return pairs


def _validate_one(profile_path: Path, gt_path: Path, *,
                  vis: bool = False, save_intermediates: bool = False) -> dict:
    """Validate algorithm against one profile/GT pair.

    Returns a dict with keys: profile, gt, pass, film_type_ok, pair_count_ok,
    no_extra_missing_ok, max_error_ok, mean_error_ok, metrics, film_type,
    num_pairs, dips.
    """

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
    gt_wire_positions = sorted({
        idx
        for gp in gt["wire_pairs"]
        for idx in (_gt_triplet(gp)[0], _gt_triplet(gp)[2])
    })
    gt_gap_positions = sorted({_gt_triplet(gp)[1] for gp in gt["wire_pairs"]})
    gt_signal_peaks = sorted([p["idx"] for p in gt["all_peaks"]])
    gt_signal_valleys = sorted([v["idx"] for v in gt["all_valleys"]])
    
    log("Raw extrema (detect_peaks_valleys on non-detrended profile):")
    log(f"  Algorithm peaks:   {algo_peaks.tolist()}")
    log(f"  Algorithm valleys: {algo_valleys.tolist()}")
    log(f"  GT signal peaks    (manual maxima):               {gt_signal_peaks}")
    log(f"  GT signal valleys  (manual minima):               {gt_signal_valleys}")
    log(f"  GT wire positions:                                {gt_wire_positions}")
    log(f"  GT gap positions:                                 {gt_gap_positions}")
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
    
    log(f"Peak (signal maxima) match:         {len(common_peaks)}/{len(gt_signal_peaks)}  "
        f"({len(common_peaks)/max(len(gt_signal_peaks),1)*100:.0f}%)")
    if missing_peaks:
        log(f"  GT peaks MISSED:   {sorted(missing_peaks)}  "
            f"(suppressed by min_distance or insufficient prominence)")
    if extra_peaks:
        log(f"  EXTRA algorithm peaks:  {sorted(extra_peaks)}  "
            f"(noise/fluctuations on descending profile slope)")
    
    log(f"Valley (signal minima) match:       {len(common_valleys)}/{len(gt_signal_valleys)}  "
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
    log("    positive film: wire_a/b_idx = bright peaks, gap_idx = dark valley")
    log("    negative film: wire_a/b_idx = dark valleys, gap_idx = bright peak")
    log()
    
    w1_errors: list[float] = []
    gap_errors: list[float] = []
    w2_errors: list[float] = []
    
    for i, (algo_pair, gp) in enumerate(zip(result.pairs, gt["wire_pairs"])):
        a_w1, a_gap, a_w2 = algo_pair
        g_va, g_pk, g_vb = _gt_triplet(gp)
    
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
            w1, _gap, w2 = _gt_triplet(wp)
            gt_wires_alt.append(w1)
            gt_wires_alt.append(w2)
        gt_wire_pos = np.array(sorted(set(gt_wires_alt)))
    gt_gap_pos = np.array([_gt_triplet(wp)[1] for wp in gt["wire_pairs"]])
    
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
        w1, gap, w2 = _gt_triplet(gp)
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
    
    if vis:
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
            gt_w1, gt_g, gt_w2 = _gt_triplet(gp)
    
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
            g_va, g_pk, g_vb = _gt_triplet(gp)
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
    
    
    # ---------------------------------------------------------------------------
    # Intermediate results capture (--save-intermediates)
    # ---------------------------------------------------------------------------
    
    if save_intermediates:
        import copy as _copy
        from scipy.signal import savgol_filter
    
        intermediates: dict = {}
        n = len(profile)
    
        # === Step 1: Raw profile ===
        intermediates["step1_raw_profile"] = {
            "length": n,
            "values": profile.tolist(),
            "stats": {
                "min": float(profile.min()),
                "max": float(profile.max()),
                "range": float(profile.max() - profile.min()),
                "mean": float(profile.mean()),
                "std": float(profile.std()),
            },
        }
    
        # === Step 2: Detrending (quadratic polyfit) ===
        x_arr = np.arange(n, dtype=np.float64)
        coeffs_detrend = np.polyfit(x_arr, profile.astype(np.float64), 2)
        trend = np.polyval(coeffs_detrend, x_arr)
        detrended = profile.astype(np.float64) - trend
    
        intermediates["step2_detrending"] = {
            "method": "quadratic polynomial fit (degree=2)",
            "formula": "f(x) = a·x² + b·x + c  →  detrended = profile - f(x)",
            "coefficients": {
                "a": float(coeffs_detrend[0]),
                "b": float(coeffs_detrend[1]),
                "c": float(coeffs_detrend[2]),
            },
            "trend_values": trend.tolist(),
            "detrended_values": detrended.tolist(),
            "detrended_stats": {
                "min": float(detrended.min()),
                "max": float(detrended.max()),
                "range": float(detrended.max() - detrended.min()),
                "mean": float(detrended.mean()),
            },
        }
    
        # === Step 3: Coarse peak/valley detection ===
        md_coarse = 5
        prom_coarse = 0.03
        coarse_peaks, coarse_valleys = detect_peaks_valleys(
            detrended, min_distance=md_coarse, prominence=prom_coarse,
        )
        data_range_detrend = float(detrended.max() - detrended.min())
        abs_prom_coarse = prom_coarse * data_range_detrend
    
        intermediates["step3_coarse_extrema"] = {
            "method": "scipy.signal.find_peaks on detrended profile",
            "parameters": {
                "min_distance": md_coarse,
                "prominence_relative": prom_coarse,
                "prominence_absolute": float(abs_prom_coarse),
            },
            "peaks": [{"idx": int(p), "gray": float(profile[p])} for p in coarse_peaks],
            "valleys": [{"idx": int(v), "gray": float(profile[v])} for v in coarse_valleys],
            "peak_count": int(len(coarse_peaks)),
            "valley_count": int(len(coarse_valleys)),
        }
    
        # === Step 4: Fine peak/valley detection ===
        md_fine = 1
        prom_fine = max(0.005, prom_coarse * 0.5)
        fine_peaks, fine_valleys = detect_peaks_valleys(
            detrended, min_distance=md_fine, prominence=prom_fine,
        )
        abs_prom_fine = prom_fine * data_range_detrend
    
        intermediates["step4_fine_extrema"] = {
            "method": "scipy.signal.find_peaks on detrended profile (relaxed params)",
            "reason": "双丝越往细组，wire-gap 间距仅 1~4px，粗检测 min_distance=5 会漏掉一侧 valley",
            "parameters": {
                "min_distance": md_fine,
                "prominence_relative": float(prom_fine),
                "prominence_absolute": float(abs_prom_fine),
            },
            "peaks": [{"idx": int(p), "gray": float(profile[p])} for p in fine_peaks],
            "valleys": [{"idx": int(v), "gray": float(profile[v])} for v in fine_valleys],
            "peak_count": int(len(fine_peaks)),
            "valley_count": int(len(fine_valleys)),
        }
    
        # === Step 5: Film-type determination ===
        triple_ft = _detect_film_type(profile, coarse_valleys, coarse_peaks)
    
        # Build positive candidate (used for film-type correction)
        pos_bg = _fit_quadratic_background(profile, fine_peaks, inverted=False)
        pos_dips, pos_pairs = _pair_adjacent_wires_with_gaps(
            profile, fine_peaks, fine_valleys, pos_bg,
            half_w=0, film_type="positive",
        )
        pos_scores = _pair_direction_scores(profile, pos_pairs, film_type="positive")
    
        profile_range = float(profile.max() - profile.min())
        min_positive_score = 0.08 * profile_range
        has_strong_positive = (
            len(pos_pairs) >= max(3, len(coarse_peaks) // 3)
            and len(pos_scores) > 0
            and float(np.median(pos_scores)) >= min_positive_score
        )
        final_ft = "positive" if has_strong_positive else triple_ft
    
        intermediates["step5_film_type_detection"] = {
            "method": "基础三元组 + positive 候选序列显著性校正",
            "basic_triplet_result": triple_ft,
            "basic_triplet_rationale": (
                "positive" if triple_ft == "positive"
                else "negative (p-v-p / v-p-v 最大振幅三元组判定)"
            ),
            "positive_candidate_pairs": [
                {"wire_a": int(w1), "gap": int(g), "wire_b": int(w2)}
                for w1, g, w2 in pos_pairs
            ],
            "positive_candidate_count": len(pos_pairs),
            "positive_direction_scores": [float(s) for s in pos_scores],
            "min_positive_score_threshold": float(min_positive_score),
            "has_strong_positive_series": has_strong_positive,
            "correction_applied": has_strong_positive and triple_ft != "positive",
            "final_film_type": final_ft,
        }
    
        # === Step 6: Role assignment ===
        is_negative = (final_ft == "negative")
        if is_negative:
            wire_positions = coarse_valleys
            gap_positions = coarse_peaks
        else:
            wire_positions = coarse_peaks
            gap_positions = coarse_valleys
    
        intermediates["step6_role_assignment"] = {
            "film_type": final_ft,
            "is_negative": is_negative,
            "rule": "正片: wire=peaks, gap=valleys; 负片: wire=valleys, gap=peaks",
            "wire_positions_used": [int(w) for w in wire_positions],
        }
    
        # === Step 7: Background fitting (Savitzky-Golay) ===
        win = min(n // 8 * 2 + 1, 201)
        if win < 9:
            win = 9
        if win % 2 == 0:
            win += 1
        if win > n:
            win = n if n % 2 == 1 else n - 1
        background = savgol_filter(profile.astype(np.float64), win, 2, mode='mirror')
        background = background.astype(np.float64)
    
        intermediates["step7_background_fitting"] = {
            "method": "Savitzky-Golay low-pass filter",
            "reason": "用宽窗口 SG 低通滤波替代全局二次拟合，避免 wire 强调制区 overshoot",
            "window_size": int(win),
            "polynomial_order": 2,
            "formula_window": "min(n // 8 * 2 + 1, 201), forced odd, ≥ 9",
            "background_values": background.tolist(),
            "background_stats": {
                "min": float(background.min()),
                "max": float(background.max()),
            },
        }
    
        # === Step 8: Pairing + Dip computation ===
        dip_half_w = 0
        if final_ft == "positive":
            dips, pairs = _pair_adjacent_wires_with_gaps(
                profile, fine_peaks, fine_valleys, background,
                half_w=dip_half_w, film_type=final_ft,
            )
        else:
            dips, pairs = _pair_wires_and_compute_dips(
                profile, wire_positions, gap_positions, background,
                half_w=dip_half_w, dist_factor=1.05, film_type=final_ft,
            )
    
        # Compute per-pair details (A, B, C for each)
        pair_details = []
        for i, ((w1, g, w2), dip_val) in enumerate(zip(pairs, dips)):
            a_mean = float(profile[w1])
            c_mean = float(profile[g])
            b_mean = float(profile[w2])
            A = abs(float(background[w1]) - a_mean)
            B = abs(float(background[w2]) - b_mean)
            C = abs(float(background[g]) - c_mean)
            numerator = A + B - 2.0 * C
            denom = A + B
    
            pair_details.append({
                "pair_index": i,
                "wire_a_idx": int(w1),
                "gap_idx": int(g),
                "wire_b_idx": int(w2),
                "wire_a_gray": a_mean,
                "gap_gray": c_mean,
                "wire_b_gray": b_mean,
                "bg_wire_a": float(background[w1]),
                "bg_gap": float(background[g]),
                "bg_wire_b": float(background[w2]),
                "A": float(A),
                "B": float(B),
                "C": float(C),
                "numerator": float(numerator),
                "denominator": float(denom),
                "dip_percent": float(dip_val),
                "formula": f"100 × ({A:.4f} + {B:.4f} - 2×{C:.4f}) / ({A:.4f} + {B:.4f}) = {dip_val:.2f}%",
            })
    
        # BAM-style reference distance
        fine_wire_positions = fine_valleys if final_ft == "negative" else fine_peaks
        if len(fine_wire_positions) >= 2:
            dist_wires = np.diff(np.asarray(sorted(int(v) for v in fine_wire_positions), dtype=int))
            k_m = min(5, len(dist_wires))
            ref_dist = float(np.median(dist_wires[:k_m]))
            if ref_dist < 1.0:
                ref_dist = float(dist_wires[0])
        else:
            ref_dist = 0.0
    
        intermediates["step8_pairing_and_dip"] = {
            "pairing_strategy": (
                "positive: 相邻 peak-valley-peak 三元组 + 灰度方向检查 + 首组间距锚点 + 中心距前缀截断"
                if final_ft == "positive"
                else "negative: 相邻 valley-peak-valley 三元组 + 1.05× 首组间距过滤"
            ),
            "dip_half_w": dip_half_w,
            "dip_formula": "100 × (A + B - 2C) / (A + B)",
            "dip_formula_detail": {
                "A": "|background[wire_a] - profile[wire_a]|",
                "B": "|background[wire_b] - profile[wire_b]|",
                "C": "|background[gap] - profile[gap]|",
            },
            "reference_distance_px": float(ref_dist),
            "distance_threshold": f"1.05 × {ref_dist:.1f} = {1.05 * ref_dist:.1f}px",
            "num_pairs": len(pairs),
            "pairs": pair_details,
            "dips": [float(d) for d in dips],
        }
    
        # === Step 9: Resolution determination ===
        from gauge.imaging.profile import _DEFAULT_WIRE_SPACINGS
        unresolved = find_first_unresolved_group(dips)
        if unresolved is None:
            resolution_note = "全部线对可分辨，分辨率优于 D13 (0.05mm)"
        else:
            idx = unresolved - 1
            if idx < len(_DEFAULT_WIRE_SPACINGS):
                spacing = _DEFAULT_WIRE_SPACINGS[idx]
                resolution_note = f"D{unresolved} (丝径 {spacing}mm) 不可分辨，SRb = {spacing}mm"
            else:
                resolution_note = f"D{unresolved} (超出标准 D13 范围)"
    
        intermediates["step9_resolution"] = {
            "threshold_dip": 20.0,
            "criterion": "Dip < 20% → 该线对不可分辨",
            "first_unresolved_group": unresolved,
            "note": resolution_note,
            "gt_num_wire_pairs": gt["num_wire_pairs"],
            "validation_mae_px": metrics["mean_point_error"],
            "validation_max_err_px": metrics["max_triplet_error"],
        }
    
        # --- Save intermediates JSON ---
        intermediates_path = profile_path.parent / "intermediates.json"
        with open(intermediates_path, "w") as f:
            json.dump(intermediates, f, indent=2, ensure_ascii=False)
        log(f"Intermediate results saved to: {intermediates_path}")
    # Return validation summary for batch reporting
    return {
        "profile": str(profile_path),
        "gt": str(gt_path),
        "pass": validation_pass,
        "film_type_ok": film_type_ok,
        "pair_count_ok": pair_count_ok,
        "no_extra_missing_ok": no_extra_missing_ok,
        "max_error_ok": max_error_ok,
        "mean_error_ok": mean_error_ok,
        "film_type": result.film_type,
        "gt_film_type": gt["film_type"],
        "num_pairs": len(result.pairs),
        "gt_num_pairs": gt["num_wire_pairs"],
        "dips": result.dips,
        "mean_point_error": metrics["mean_point_error"],
        "max_triplet_error": metrics["max_triplet_error"],
    }


# ---------------------------------------------------------------------------
# Dispatch: single pair or batch
# ---------------------------------------------------------------------------

if path_a.is_dir():
    pairs = _find_profile_gt_pairs(path_a)
    if not pairs:
        print(f"Error: no profile/groundtruth pairs found in: {path_a}", file=sys.stderr)
        sys.exit(1)

    print(f"Batch validation: {len(pairs)} pair(s) found in {path_a}")
    print()

    results = []
    for i, (prof_path, gt_path) in enumerate(pairs):
        print(f"\n{'='*80}")
        print(f"  [{i+1}/{len(pairs)}] {prof_path.name}")
        print(f"{'='*80}")
        r = _validate_one(prof_path, gt_path, vis=args.vis,
                          save_intermediates=args.save_intermediates)
        results.append(r)

    # Batch summary
    print(f"\n{'='*80}")
    print("BATCH SUMMARY")
    print(f"{'='*80}")
    passed = sum(1 for r in results if r["pass"])
    print(f"Total: {len(results)}  |  PASS: {passed}  |  FAIL: {len(results) - passed}")
    print()
    print(f"{'Profile':<50} {'Pass':>5} {'pairs':>6} {'MAE':>6} {'maxE':>6}")
    print(f"{'-'*80}")
    for r in results:
        name = Path(r["profile"]).name[:48]
        pairs_str = f"{r['num_pairs']}/{r['gt_num_pairs']}"
        mae = f"{r['mean_point_error']:.1f}" if r['mean_point_error'] != float('inf') else "inf"
        maxe = f"{r['max_triplet_error']:.1f}" if r['max_triplet_error'] != float('inf') else "inf"
        print(f"{name:<50} {'PASS' if r['pass'] else 'FAIL':>5} {pairs_str:>6} {mae:>6} {maxe:>6}")

elif args.path_b is not None:
    _validate_one(path_a, Path(args.path_b), vis=args.vis,
                  save_intermediates=args.save_intermediates)
else:
    print("Error: provide both profile_json and groundtruth_json, or a directory",
          file=sys.stderr)
    sys.exit(1)
