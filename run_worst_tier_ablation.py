"""
run_worst_tier_ablation.py
============================
Ablation-tests the 6 new interaction features added to target the
$2.5K-10K worst-dollar-error tier (see
worst_case_analysis_2500_10000/tier_band_feature_correlation.md and
WORST_TIER_FEATURE_COLS in preprocessor.py):

  unknowns_x_mileage_bkt, unknowns_x_age_bkt, mech_severity_x_mileage_bkt,
  cult_x_n_unknowns, vtype_x_mileage_bkt, mileage_unknown_x_n_unknowns

Trains two script21 models via subprocess calls to train_save_script21.py:

  baseline  -- python train_save_script21.py --disable-worst-tier-features
  new       -- python train_save_script21.py   (features enabled, default)

then evaluates both the same way train_save_script21.py always does
(test_metrics.json via evaluate_predictions.evaluate()) and renders a
combined overall + per-tier comparison table, mirroring
run_feature_ablation.py's format. Unlike run_feature_ablation.py this
compares exactly 2 groups (features off vs on) since all 6 are one bundle
gated by a single flag, not independently toggleable columns.

USAGE:
    python run_worst_tier_ablation.py
    python run_worst_tier_ablation.py --gpu
    python run_worst_tier_ablation.py --use-dataone
    python run_worst_tier_ablation.py --enable-new-features true_mileage_unknown,clean_title
"""
import os
import sys
import json
import argparse
import subprocess
import time

import pandas as pd

from evaluate_predictions import evaluate, TIER_LABELS

GROUPS = [
    ("baseline_worst_tier_off", True),   # (name, pass --disable-worst-tier-features)
    ("new_worst_tier_on",       False),
]

METRIC_KEYS = [
    ("N", "N", "{:,.0f}"),
    ("MAE_p50", "MAE", "${:,.0f}"),
    ("RMSE_p50", "RMSE", "${:,.0f}"),
    ("RMSLE_p50", "RMSLE", "{:.4f}"),
    ("coverage_90", "Cov90", "{:.1%}"),
    ("mean_ci_width", "Width", "${:,.0f}"),
    ("bias_p50", "Bias", "${:+,.0f}"),
]


def run_group(name, disable_flag, args):
    out_dir = os.path.join("artifacts", f"script21_ablation_{name}")
    cmd = [
        sys.executable, "train_save_script21.py",
        "--data", args.data,
        "--cult", args.cult,
        "--out", out_dir,
    ]
    if disable_flag:
        cmd.append("--disable-worst-tier-features")
    if args.enable_new_features:
        cmd += ["--enable-new-features", args.enable_new_features]
    if args.gpu:
        cmd.append("--gpu")
    if args.use_dataone:
        cmd.append("--use-dataone")
    print(f"\n{'=' * 100}\nGROUP {name}: worst-tier features "
          f"{'DISABLED' if disable_flag else 'ENABLED'}\n{'=' * 100}")
    print("  $", " ".join(cmd))
    t0 = time.time()
    log_path = f"worst_tier_ablation_{name}.log"
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}) after {elapsed:.1f} min — see {log_path}")
        return None
    print(f"  Done in {elapsed:.1f} min")
    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path) as f:
        return json.load(f)


def metrics_rows(label, metrics):
    """Flatten a metrics dict (as saved by evaluate_predictions.evaluate()) into
    one row per tier + overall, for a combined comparison table."""
    rows = []
    blocks = {"OVERALL": metrics["overall"]}
    blocks.update(metrics.get("by_tier", {}))
    for tier, m in blocks.items():
        if m["N"] == 0:
            continue
        row = {"group": label, "tier": tier}
        row.update({k: m[k] for k, _, _ in METRIC_KEYS})
        rows.append(row)
    return rows


def render_markdown(df, tier_order):
    lines = ["# Worst-tier interaction feature ablation results\n"]
    lines.append("baseline_worst_tier_off = pre-change behavior "
                  "(--disable-worst-tier-features). "
                  "new_worst_tier_on = with the 6 new interaction features "
                  "(default, no flag needed).\n")
    lines.append("Watch the $2.5K-4K / $4K-6K / $6K-10K rows — that's the "
                  "band these features specifically target. A regression in "
                  "$0-200/$200-500/$500-1K would mean the interactions are "
                  "adding noise elsewhere; check those too.\n")
    headers = ["group"] + [label for _, label, _ in METRIC_KEYS]
    for tier in tier_order:
        sub = df[df["tier"] == tier]
        if sub.empty:
            continue
        lines.append(f"\n## {tier}\n")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "---|" * len(headers))
        for _, r in sub.iterrows():
            cells = [str(r["group"])]
            for key, _, fmt in METRIC_KEYS:
                v = r[key]
                cells.append(fmt.format(v) if pd.notna(v) else "n/a")
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="taegram_all_table_merged_2018_2026.csv")
    parser.add_argument("--cult", default="cult_cars_fixed.xlsx",
                        help="cult_cars.xlsx's sheet in this checkout is named 'cult_cars', "
                             "not 'Cult Vehicles' as train_save_script21.py expects — "
                             "cult_cars_fixed.xlsx is a renamed-sheet copy (see repo root).")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--use-dataone", action="store_true",
                        help="Forward --use-dataone to every training subprocess "
                             "(default off, matching train_save_script21.py's default).")
    parser.add_argument("--enable-new-features", default="",
                        help="Forward --enable-new-features to every training subprocess "
                             "(default: none — keeps the true_mileage_unknown/clean_title/"
                             "gvm_range/tonnage/engine_type set at its own baseline so this "
                             "run isolates the worst-tier interactions only).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of group names to run "
                             "(default: both baseline_worst_tier_off and new_worst_tier_on)")
    args = parser.parse_args()

    groups_to_run = GROUPS
    if args.only:
        wanted = set(args.only.split(","))
        groups_to_run = [g for g in GROUPS if g[0] in wanted]

    all_rows = []
    for name, disable_flag in groups_to_run:
        metrics = run_group(name, disable_flag, args)
        if metrics is None:
            continue
        all_rows.extend(metrics_rows(name, metrics))

    df = pd.DataFrame(all_rows)
    tier_order = ["OVERALL"] + TIER_LABELS
    df["tier"] = pd.Categorical(df["tier"], categories=tier_order, ordered=True)
    df = df.sort_values(["tier", "group"])

    out_md = "worst_tier_ablation_results.md"
    with open(out_md, "w") as f:
        f.write(render_markdown(df, tier_order))
    print(f"\nSaved {out_md}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
