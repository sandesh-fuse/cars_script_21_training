"""
evaluate_predictions.py
========================
Reads a predictions file (parquet or csv) and computes:
  - Overall metrics: MAE, RMSE, RMSLE, empirical coverage of 90% CI, mean CI width
  - Per-price-tier metrics: same metrics broken down by salevalue tier
  - For script21 outputs (with is_cult column): also breaks down by cult/non-cult route

Prints a formatted table to stdout AND saves a JSON file alongside the input.

USAGE:
    python evaluate_predictions.py path/to/test_predictions.parquet
    python evaluate_predictions.py artifacts/script21/test_predictions.parquet
    python evaluate_predictions.py artifacts/script17/train_predictions.csv

OR as a library:
    from evaluate_predictions import evaluate
    metrics = evaluate(df, label="test")  # returns dict, also prints/saves if requested

The training scripts call this automatically at the end of training; you can also
re-run it whenever you want to look at metrics again without retraining.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Dict


# Tier definitions: edges = breakpoints; labels names each tier
TIER_EDGES  = [0, 200, 500, 1000, 2500, 4000, 6000, 10000, 15000, 25000, np.inf]
TIER_LABELS = ['$0-200', '$200-500', '$500-1K', '$1K-2.5K', '$2.5K-4K',
               '$4K-6K',  '$6K-10K',  '$10K-15K', '$15K-25K', '$25K+']


# ---------- Metrics ----------
def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def rmsle(y, p):
    p = np.clip(p, 1, None)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))
def coverage(y, lo, hi): return float(np.mean((y >= lo) & (y <= hi)))


def _block_metrics(actual, p5, p50, p95) -> Dict:
    """Compute all metrics for a slice of rows."""
    n = len(actual)
    if n == 0:
        return {
            'N': 0,
            'MAE_p50': None, 'RMSE_p50': None, 'RMSLE_p50': None,
            'coverage_90': None, 'mean_ci_width': None,
            'mean_actual': None, 'mean_pred_p50': None, 'bias_p50': None,
        }
    actual = np.asarray(actual, dtype=float)
    p5     = np.asarray(p5, dtype=float)
    p50    = np.asarray(p50, dtype=float)
    p95    = np.asarray(p95, dtype=float)
    return {
        'N':              int(n),
        'MAE_p50':        mae(actual, p50),
        'RMSE_p50':       rmse(actual, p50),
        'RMSLE_p50':      rmsle(actual, p50),
        'coverage_90':    coverage(actual, p5, p95),
        'mean_ci_width':  float(np.mean(p95 - p5)),
        'mean_actual':    float(np.mean(actual)),
        'mean_pred_p50':  float(np.mean(p50)),
        'bias_p50':       float(np.mean(p50 - actual)),   # +ve: over-prediction
    }


def _print_block(name: str, m: Dict):
    """Pretty-print a single metrics block."""
    if m['N'] == 0:
        print(f"  {name:<14}  N=0  (no rows)")
        return
    print(
        f"  {name:<14}  "
        f"N={m['N']:>6,}  "
        f"MAE=${m['MAE_p50']:>7.0f}  "
        f"RMSE=${m['RMSE_p50']:>7.0f}  "
        f"RMSLE={m['RMSLE_p50']:>6.4f}  "
        f"Cov90={m['coverage_90']*100:>5.1f}%  "
        f"Width=${m['mean_ci_width']:>7.0f}  "
        f"Bias=${m['bias_p50']:>+7.0f}"
    )


def evaluate(df: pd.DataFrame, label: str = "predictions",
             save_json_to: Optional[str] = None, verbose: bool = True) -> Dict:
    """Compute and (optionally) print/save full metrics.

    Required columns: salevalue, p5, p50, p95
    Optional column:  is_cult  (enables route-level breakdown)

    Returns the metrics dict (also saved to JSON if save_json_to is set).
    """
    needed = {'salevalue', 'p5', 'p50', 'p95'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    actual = df['salevalue'].values
    p5     = df['p5'].values
    p50    = df['p50'].values
    p95    = df['p95'].values

    out = {'label': label, 'n_total': len(df)}
    out['overall']    = _block_metrics(actual, p5, p50, p95)
    out['by_tier']    = {}
    out['by_route']   = {}

    # Per-tier
    tiers = pd.cut(actual, bins=TIER_EDGES, labels=TIER_LABELS, right=True, include_lowest=True)
    tier_array = np.asarray(tiers)  # works whether pd.cut returned a Series or Categorical
    for t in TIER_LABELS:
        m = (tier_array == t)
        out['by_tier'][t] = _block_metrics(actual[m], p5[m], p50[m], p95[m])

    # Per-route (cult vs non-cult), if is_cult is in the file
    if 'is_cult' in df.columns:
        is_cult = df['is_cult'].astype(bool).values
        out['by_route']['cult'] = _block_metrics(
            actual[is_cult], p5[is_cult], p50[is_cult], p95[is_cult])
        out['by_route']['non_cult'] = _block_metrics(
            actual[~is_cult], p5[~is_cult], p50[~is_cult], p95[~is_cult])

    if verbose:
        print("=" * 100)
        print(f"METRICS — {label}  (n={len(df):,})")
        print("=" * 100)
        _print_block("OVERALL", out['overall'])
        print()
        print("By price tier:")
        for t in TIER_LABELS:
            _print_block(t, out['by_tier'][t])
        if out['by_route']:
            print()
            print("By route (cult / non-cult):")
            for r in ['cult', 'non_cult']:
                _print_block(r, out['by_route'][r])
        print()
        # Honest reminder if this is a train file
        if label.lower().startswith('train'):
            print("  Note: train metrics are computed on rows the model saw during training.")
            print("        They will look better than test metrics; use for sanity, not evaluation.")
        print()

    if save_json_to is not None:
        with open(save_json_to, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        if verbose:
            print(f"  Metrics JSON written to: {save_json_to}")

    return out


def _load_predictions(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.parquet':
        return pd.read_parquet(path)
    if ext in ('.csv', '.tsv'):
        return pd.read_csv(path, sep=',' if ext == '.csv' else '\t')
    raise ValueError(f"Unsupported file extension: {ext} (use .parquet or .csv)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved predictions file.")
    parser.add_argument('path', help='Path to predictions file (.parquet or .csv)')
    parser.add_argument('--no-save', action='store_true',
                        help="Don't write the metrics JSON next to the input file")
    args = parser.parse_args()

    df = _load_predictions(args.path)

    label = os.path.basename(args.path).rsplit('.', 1)[0]   # e.g. 'test_predictions'
    save_json = None
    if not args.no_save:
        base, _ = os.path.splitext(args.path)
        save_json = f"{base}_metrics.json"

    evaluate(df, label=label, save_json_to=save_json, verbose=True)


if __name__ == '__main__':
    main()
