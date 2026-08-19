"""
run_feature_ablation.py
========================
Ablation-tests the 5 raw features extracted from commit 20c8c17
("added new features true_mileage_unknown, clean_title (bool), gvm_range,
tonnage, engine_type") in 3 groups against the existing 2406a7a baseline,
without retraining the baseline itself:

  Baseline  -- none of the 5 (reuses the existing test_predictions.csv)
  Group A   -- true_mileage_unknown + clean_title (the two bool-flag
               features with bundled interactions from the same commit)
  Group B   -- gvm_range + engine_type (freq-encoded categoricals, no
               interactions)
  Group C   -- tonnage (bare passthrough numeric, no engineering at all)

Each of A/B/C is trained independently against the SAME baseline feature
set (not stacked/cumulative — each group is its own isolated run) via a
subprocess call to train_save_script21.py with --enable-new-features, then
evaluated the same way train_save_script21.py always evaluates
(test_metrics.json via evaluate_predictions.evaluate()).

USAGE:
    python run_feature_ablation.py
    python run_feature_ablation.py --only group_c_tonnage
    python run_feature_ablation.py --gpu
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
    ("group_a_mileage_title", ["true_mileage_unknown", "clean_title"]),
    ("group_b_gvm_engine",    ["gvm_range", "engine_type"]),
    ("group_c_tonnage",       ["tonnage"]),
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


def run_group(name, features, args):
    out_dir = os.path.join("artifacts", f"script21_ablation_{name}")
    cmd = [
        sys.executable, "train_save_script21.py",
        "--data", args.data,
        "--cult", args.cult,
        "--out", out_dir,
        "--enable-new-features", ",".join(features),
    ]
    if args.gpu:
        cmd.append("--gpu")
    if args.use_dataone:
        cmd.append("--use-dataone")
    print(f"\n{'=' * 100}\nGROUP {name}: enabling {features}"
          f" (dataone {'ON' if args.use_dataone else 'OFF'})\n{'=' * 100}")
    print("  $", " ".join(cmd))
    t0 = time.time()
    log_path = f"ablation_{name}.log"
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
    lines = ["# Feature ablation results\n"]
    lines.append("Baseline vs. each group, tested independently (not cumulative).\n")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="taegram_all_table_merged_2018_2026.csv")
    parser.add_argument("--cult", default="cult_cars_fixed.xlsx",
                        help="cult_cars.xlsx's sheet in this checkout is named 'cult_cars', "
                             "not 'Cult Vehicles' as train_save_script21.py expects — "
                             "cult_cars_fixed.xlsx is a renamed-sheet copy (see repo root).")
    parser.add_argument("--baseline-predictions", default="test_predictions.csv")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--use-dataone", action="store_true",
                        help="Forward --use-dataone to every training subprocess "
                             "(default off, matching train_save_script21.py's default).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of group names to run (default: all 3)")
    args = parser.parse_args()

    print(f"Loading baseline predictions ({args.baseline_predictions})...")
    baseline_df = pd.read_csv(args.baseline_predictions)
    baseline_metrics = evaluate(baseline_df, label="baseline (2406a7a, none of the 5 new features)",
                                 verbose=False)

    all_rows = metrics_rows("Baseline (none)", baseline_metrics)

    groups_to_run = GROUPS
    if args.only:
        wanted = set(args.only.split(","))
        groups_to_run = [g for g in GROUPS if g[0] in wanted]

    for name, features in groups_to_run:
        metrics = run_group(name, features, args)
        if metrics is None:
            continue
        all_rows.extend(metrics_rows(f"{name} (+{'/'.join(features)})", metrics))

    df = pd.DataFrame(all_rows)
    tier_order = ["OVERALL"] + TIER_LABELS
    df["tier"] = pd.Categorical(df["tier"], categories=tier_order, ordered=True)
    df = df.sort_values(["tier", "group"])

    out_md = "feature_ablation_results.md"
    with open(out_md, "w") as f:
        f.write(render_markdown(df, tier_order))
    print(f"\nSaved {out_md}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
