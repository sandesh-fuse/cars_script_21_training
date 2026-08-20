"""
run_worst_tier_ablation.py
============================
Ablation-tests the 6 new interaction features added to target
under/overpredicted vehicles (see WORST_TIER_FEATURE_COLS in
preprocessor.py and worst_case_analysis_2500_10000/tier_band_feature_
correlation.md for where the underlying evidence came from):

  unknowns_x_mileage_bkt, unknowns_x_age_bkt, mech_severity_x_mileage_bkt,
  cult_x_n_unknowns, vtype_x_mileage_bkt, mileage_unknown_x_n_unknowns

Tests each ONE AT A TIME against a shared baseline (--enable-worst-tier-
features none) -- not cumulative, not all-6-at-once -- via subprocess
calls to train_save_script21.py's --enable-worst-tier-features flag, then
evaluates every run the same way train_save_script21.py always does
(test_metrics.json via evaluate_predictions.evaluate()) and renders a
combined overall + per-tier comparison table.

Most of the vehicle volume (and the tier this repo's business priority
targets) sits in $100-2000, not the sparser $2.5K-10K tail the features
were originally diagnosed against -- watch the $0-200 / $200-500 /
$500-1K / $1K-2.5K rows in the output first; the $2.5K+ rows are still
printed for reference but are the secondary concern here.

USAGE:
    python run_worst_tier_ablation.py                          # baseline + all 6, one at a time
    python run_worst_tier_ablation.py --only unknowns_x_mileage_bkt
    python run_worst_tier_ablation.py --only baseline,unknowns_x_mileage_bkt,unknowns_x_age_bkt
    python run_worst_tier_ablation.py --include-all-combined    # + a 7th group: all 6 together
    python run_worst_tier_ablation.py --gpu
"""
import os
import sys
import json
import argparse
import subprocess
import time

import pandas as pd

from evaluate_predictions import evaluate, TIER_LABELS
from preprocessor import WORST_TIER_FEATURE_COLS

BASELINE_NAME = "baseline"


def build_groups(include_all_combined):
    groups = [(BASELINE_NAME, "none")]
    groups += [(name, name) for name in WORST_TIER_FEATURE_COLS]
    if include_all_combined:
        groups.append(("all_combined", "all"))
    return groups


METRIC_KEYS = [
    ("N", "N", "{:,.0f}"),
    ("MAE_p50", "MAE", "${:,.0f}"),
    ("RMSE_p50", "RMSE", "${:,.0f}"),
    ("RMSLE_p50", "RMSLE", "{:.4f}"),
    ("coverage_90", "Cov90", "{:.1%}"),
    ("mean_ci_width", "Width", "${:,.0f}"),
    ("bias_p50", "Bias", "${:+,.0f}"),
]

# Tiers where most volume and current business priority sit -- printed
# first in the summary. The rest of TIER_LABELS still appears in the full
# per-tier table below for reference.
PRIORITY_TIERS = ["$0-200", "$200-500", "$500-1K", "$1K-2.5K"]


def run_group(name, enable_value, args):
    out_dir = os.path.join("artifacts", f"script21_ablation_{name}")
    cmd = [
        sys.executable, "train_save_script21.py",
        "--data", args.data,
        "--cult", args.cult,
        "--out", out_dir,
        "--enable-worst-tier-features", enable_value,
    ]
    if args.enable_new_features:
        cmd += ["--enable-new-features", args.enable_new_features]
    if args.gpu:
        cmd.append("--gpu")
    if args.use_dataone:
        cmd.append("--use-dataone")
    print(f"\n{'=' * 100}\nGROUP {name}: --enable-worst-tier-features {enable_value}\n{'=' * 100}")
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
    lines.append(f"`{BASELINE_NAME}` = --enable-worst-tier-features none (pre-change behavior). "
                  "Each other group enables exactly ONE of the 6 new interaction features "
                  "against that same baseline (not cumulative).\n")
    lines.append("**Most vehicle volume sits in $0-200/$200-500/$500-1K/$1K-2.5K** -- check "
                  "those rows first. $0-500 is currently *overpredicted* (positive bias); "
                  "$1K-2.5K is *underpredicted* (negative bias). The $2.5K+ rows are kept "
                  "for reference (that tier has the worst per-row MAE) but are secondary here.\n")
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
                             "(default: none), so this run isolates the worst-tier "
                             "interactions only.")
    parser.add_argument("--include-all-combined", action="store_true",
                        help="Add a 7th group testing all 6 features together "
                             "(--enable-worst-tier-features all), on top of the "
                             "baseline + one-at-a-time groups.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of group names to run "
                             f"(choose from: baseline, {', '.join(WORST_TIER_FEATURE_COLS)}"
                             + (", all_combined" if True else "") +
                             "). Default: baseline + all 6 individually.")
    args = parser.parse_args()

    groups_to_run = build_groups(args.include_all_combined)
    if args.only:
        wanted = set(args.only.split(","))
        groups_to_run = [g for g in groups_to_run if g[0] in wanted]
        if not groups_to_run:
            parser.error(f"--only matched no groups; choose from "
                         f"{[g[0] for g in build_groups(True)]}")

    all_rows = []
    for name, enable_value in groups_to_run:
        metrics = run_group(name, enable_value, args)
        if metrics is None:
            continue
        all_rows.extend(metrics_rows(name, metrics))

    df = pd.DataFrame(all_rows)
    tier_order = ["OVERALL"] + PRIORITY_TIERS + [t for t in TIER_LABELS if t not in PRIORITY_TIERS]
    df["tier"] = pd.Categorical(df["tier"], categories=tier_order, ordered=True)
    df = df.sort_values(["tier", "group"])

    out_md = "worst_tier_ablation_results.md"
    with open(out_md, "w") as f:
        f.write(render_markdown(df, tier_order))
    print(f"\nSaved {out_md}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
