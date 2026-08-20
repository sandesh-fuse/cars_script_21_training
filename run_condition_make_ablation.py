"""
run_condition_make_ablation.py
================================
Ablation-tests the 3 condition-x-make interaction features added to
target the $100-2000 tier (see CONDITION_MAKE_FEATURE_COLS in
preprocessor.py and worst_case_analysis_100_2000/tier_band_feature_
correlation.md for where the underlying evidence came from):

  runs_x_make, mech_severity_x_make, all_cond_combo_x_make

Tests each ONE AT A TIME against a shared baseline (--enable-condition-
make-features none) -- not cumulative -- via subprocess calls to
train_save_script21.py's --enable-condition-make-features flag, then
evaluates every run the same way train_save_script21.py always does
(test_metrics.json via evaluate_predictions.evaluate()) and renders a
combined overall + per-tier comparison table.

These target $0-200/$200-500/$500-1K/$1K-2.5K specifically (most vehicle
volume) -- those rows print first in the output. $2.5K+ is kept for
reference only.

USAGE:
    python run_condition_make_ablation.py                       # baseline + all 3, one at a time
    python run_condition_make_ablation.py --only runs_x_make
    python run_condition_make_ablation.py --only baseline,runs_x_make,mech_severity_x_make
    python run_condition_make_ablation.py --include-all-combined  # + a 4th group: all 3 together
    python run_condition_make_ablation.py --gpu
"""
import os
import sys
import json
import argparse
import subprocess
import time

import pandas as pd

from evaluate_predictions import evaluate, TIER_LABELS
from preprocessor import CONDITION_MAKE_FEATURE_COLS

BASELINE_NAME = "baseline"

# Tiers where most volume and current business priority sit -- printed
# first in the summary. The rest of TIER_LABELS still appears in the full
# per-tier table below for reference.
PRIORITY_TIERS = ["$0-200", "$200-500", "$500-1K", "$1K-2.5K"]

METRIC_KEYS = [
    ("N", "N", "{:,.0f}"),
    ("MAE_p50", "MAE", "${:,.0f}"),
    ("RMSE_p50", "RMSE", "${:,.0f}"),
    ("RMSLE_p50", "RMSLE", "{:.4f}"),
    ("coverage_90", "Cov90", "{:.1%}"),
    ("mean_ci_width", "Width", "${:,.0f}"),
    ("bias_p50", "Bias", "${:+,.0f}"),
]


def build_groups(include_all_combined):
    groups = [(BASELINE_NAME, "none")]
    groups += [(name, name) for name in CONDITION_MAKE_FEATURE_COLS]
    if include_all_combined:
        groups.append(("all_combined", "all"))
    return groups


def run_group(name, enable_value, args):
    out_dir = os.path.join("artifacts", f"script21_ablation_{name}")
    cmd = [
        sys.executable, "train_save_script21.py",
        "--data", args.data,
        "--cult", args.cult,
        "--out", out_dir,
        "--enable-condition-make-features", enable_value,
    ]
    if args.enable_new_features:
        cmd += ["--enable-new-features", args.enable_new_features]
    if args.enable_worst_tier_features != "all":
        cmd += ["--enable-worst-tier-features", args.enable_worst_tier_features]
    if args.gpu:
        cmd.append("--gpu")
    if args.use_dataone:
        cmd.append("--use-dataone")
    print(f"\n{'=' * 100}\nGROUP {name}: --enable-condition-make-features {enable_value}\n{'=' * 100}")
    print("  $", " ".join(cmd))
    t0 = time.time()
    log_path = f"condition_make_ablation_{name}.log"
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
    lines = ["# Condition-x-make interaction feature ablation results\n"]
    lines.append(f"`{BASELINE_NAME}` = --enable-condition-make-features none (pre-change behavior). "
                  "Each other group enables exactly ONE of the 3 new interaction features "
                  "against that same baseline (not cumulative).\n")
    lines.append("**Most vehicle volume sits in $0-200/$200-500/$500-1K/$1K-2.5K** -- check "
                  "those rows first; that's what these 3 features target (worst-overpredicted "
                  "rows there show *better*-looking condition than the rest of the band, not "
                  "worse -- watch `Bias` moving toward $0 and MAE dropping). $2.5K+ rows are "
                  "kept for reference only.\n")
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
                             "(default off — omit this flag to run WITHOUT DataOne "
                             "features, which is train_save_script21.py's own default).")
    parser.add_argument("--enable-new-features", default="",
                        help="Forward --enable-new-features to every training subprocess "
                             "(default: none), so this run isolates the condition-x-make "
                             "interactions only.")
    parser.add_argument("--enable-worst-tier-features", default="all",
                        help="Forward --enable-worst-tier-features to every training "
                             "subprocess (default: all -- the $2.5K-10K-tier batch stays "
                             "on unless you override this), so this run isolates the "
                             "condition-x-make interactions on top of that fixed baseline.")
    parser.add_argument("--include-all-combined", action="store_true",
                        help="Add a 4th group testing all 3 features together "
                             "(--enable-condition-make-features all), on top of the "
                             "baseline + one-at-a-time groups.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of group names to run "
                             f"(choose from: baseline, {', '.join(CONDITION_MAKE_FEATURE_COLS)}"
                             ", all_combined). Default: baseline + all 3 individually.")
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

    out_md = "condition_make_ablation_results.md"
    with open(out_md, "w") as f:
        f.write(render_markdown(df, tier_order))
    print(f"\nSaved {out_md}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
