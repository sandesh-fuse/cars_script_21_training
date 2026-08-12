"""
test_examples.py
================
Demonstrates the /predict endpoint using low-MAE test rows from
the trained artifacts. Loads sample_test_rows.parquet and sends
each row through the API.

Two ways to run:

  1. Against a running service (Docker or local uvicorn):
        # Terminal 1:
        uvicorn app.main:app --port 8000
        # Terminal 2:
        python tests/test_examples.py --base-url http://localhost:8000

  2. As a script that builds the request payload only (no service needed),
     useful for sanity-checking what rows look like:
        python tests/test_examples.py --dry-run

The script prints actual vs predicted for each sample, top SHAP features,
and the natural-language explanation (when --explain is set).
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import pandas as pd
import requests


# Columns added by the training script to the sample parquet — metadata about
# the prediction, not features we send to the API. Send everything else as-is.
META_COLS = {
    'actual', 'p5', 'p50', 'p95', 'ci_width', 'abs_error_p50', 'is_cult',
}


def load_samples(artifacts_dir: Path, model: str) -> pd.DataFrame:
    """Load sample test rows for the given model ('script17' or 'script21')."""
    sample_path = artifacts_dir / model / 'sample_test_rows.parquet'
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample file not found: {sample_path}\n"
            f"Run train_save_{model}.py first to generate it."
        )
    return pd.read_parquet(sample_path)


def row_to_request(row: pd.Series) -> dict:
    """Strip metadata cols and serialize all remaining fields to JSON-safe dict.

    This sends every non-metadata column from the parquet, matching what the
    preprocessor expected at training time. Anything the preprocessor doesn't
    recognize will be silently dropped by SaleValuePreprocessor.transform().
    """
    body = {}
    for col, val in row.items():
        if col in META_COLS:
            continue
        if pd.isna(val):
            continue
        if hasattr(val, 'item'):       # numpy scalar -> python scalar
            val = val.item()
        if isinstance(val, pd.Timestamp):
            val = val.strftime('%Y-%m-%d')
        body[col] = val
    return body


def call_predict(base_url: str, body: dict, model: str, explain: bool,
                  k_pos: int, k_neg: int, explanation_units: str = "dollars",
                  timeout: int = 30) -> dict:
    url = f"{base_url}/predict"
    params = {
        'model': model,
        'explain': 'true' if explain else 'false',
        'explanation_units': explanation_units,
        'shap_quantile': 'p50',
        'k_pos': k_pos,
        'k_neg': k_neg,
    }
    r = requests.post(url, params=params, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fmt_money(v: float) -> str:
    sign = '-' if v < 0 else '+'
    return f"{sign}${abs(v):,.0f}"


def print_row_summary(row: pd.Series, idx: int, total: int):
    print(f"\n{'=' * 72}")
    bucket = "CULT" if row.get('is_cult', False) else "STANDARD"
    label = f"[{bucket}] {row.get('year', '?')} {row.get('make', '?')} {row.get('model', '?')}"
    print(f"Sample {idx+1}/{total}: {label}")
    print(f"  Actual sale value: ${row['actual']:,.0f}")
    print(f"  Trained p50: ${row['p50']:,.0f}  (90% CI: ${row['p5']:,.0f} - ${row['p95']:,.0f})")
    print(f"  Abs error (trained): ${row['abs_error_p50']:,.0f}")


def print_api_response(resp: dict, actual: float):
    p = resp['predictions']
    err = p['p50'] - actual
    print(f"  API     p50: ${p['p50']:,.0f}  (90% CI: ${p['p5']:,.0f} - ${p['p95']:,.0f})")
    print(f"  API error vs actual: {fmt_money(err)}")
    print(f"  Route: {resp.get('route')}, is_cult: {resp.get('is_cult')}")
    elapsed = resp.get('elapsed_ms', {})
    print(f"  Latency: predict={elapsed.get('predict', 0)} ms  "
          f"explain={elapsed.get('explain', 0)} ms  total={elapsed.get('total', 0)} ms")

    if resp.get('shap'):
        s = resp['shap']
        baseline = s.get('baseline_dollars')
        final = s.get('final_pred_dollars')
        if baseline is not None and final is not None:
            print(f"\n  SHAP baseline: ${baseline:,.0f}  -> final: ${final:,.0f}  "
                  f"(quantile={s.get('quantile_explained', 'q50')})")
        else:
            print(f"\n  SHAP attribution (explaining {s.get('quantile_explained', 'q50')}):")
        print(f"  Top positive (raw features driving price up):")
        for r in s['top_positive']:
            val_str = str(r.get('value', ''))[:18] if r.get('value') is not None else '-'
            print(f"    {r['feature_label']:<40}  "
                  f"value={val_str:<18}  "
                  f"{fmt_money(r['dollar_impact']):>9} "
                  f"({r['pct_of_prediction']:+5.1f}%)  "
                  f"[n_underlying={r['n_underlying']}]")
        print(f"  Top negative (raw features driving price down):")
        for r in s['top_negative']:
            val_str = str(r.get('value', ''))[:18] if r.get('value') is not None else '-'
            print(f"    {r['feature_label']:<40}  "
                  f"value={val_str:<18}  "
                  f"{fmt_money(r['dollar_impact']):>9} "
                  f"({r['pct_of_prediction']:+5.1f}%)  "
                  f"[n_underlying={r['n_underlying']}]")

    if resp.get('explanation'):
        print(f"\n  Natural-language explanation:")
        print(f"    \"{resp['explanation']}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://localhost:8000',
                        help='Running service URL')
    parser.add_argument('--artifacts-dir', default='./artifacts',
                        help='Path to artifacts/ directory containing scriptXX/ subfolders')
    parser.add_argument('--model', default='script21',
                        choices=['script17', 'script21'],
                        help='Which model to use')
    parser.add_argument('--explain', action='store_true',
                        help='Request natural-language explanation (slower)')
    parser.add_argument('--n-cult', type=int, default=2,
                        help='Number of cult samples to test')
    parser.add_argument('--n-standard', type=int, default=3,
                        help='Number of standard samples to test')
    parser.add_argument('--k-pos', type=int, default=5)
    parser.add_argument('--k-neg', type=int, default=5)
    parser.add_argument('--units', default='dollars', choices=['dollars', 'percentage'],
                        help='Units for the natural-language explanation '
                             '(only matters when --explain is set)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Skip API calls; print request payloads only')
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    print(f"Loading samples from {artifacts_dir / args.model}/sample_test_rows.parquet ...")
    samples = load_samples(artifacts_dir, args.model)

    # Split into cult and standard if the column exists; otherwise treat all as standard
    if 'is_cult' in samples.columns:
        cult_samples = samples[samples['is_cult'] == True].head(args.n_cult)
        std_samples  = samples[samples['is_cult'] == False].head(args.n_standard)
        selected = pd.concat([cult_samples, std_samples], ignore_index=True)
    else:
        selected = samples.head(args.n_cult + args.n_standard)

    print(f"Selected {len(selected)} samples ({(selected.get('is_cult', pd.Series([False]*len(selected))) == True).sum()} cult, "
          f"{(selected.get('is_cult', pd.Series([False]*len(selected))) == False).sum()} standard)")

    if not args.dry_run:
        # Health check first
        try:
            h = requests.get(f"{args.base_url}/healthz", timeout=5).json()
            print(f"Service health: {h}")
        except Exception as e:
            print(f"WARNING: cannot reach {args.base_url}: {e}")
            print("Run with --dry-run to skip API calls.")
            sys.exit(1)

    total_err = 0.0
    n_calls = 0
    for i, (_, row) in enumerate(selected.iterrows()):
        print_row_summary(row, i, len(selected))
        body = row_to_request(row)

        if args.dry_run:
            print(f"  Request body that would be sent:")
            print(f"    {json.dumps(body, indent=4, default=str)}")
            continue

        try:
            t0 = time.time()
            resp = call_predict(args.base_url, body, args.model, args.explain,
                                args.k_pos, args.k_neg, args.units)
            print_api_response(resp, row['actual'])
            total_err += abs(resp['predictions']['p50'] - row['actual'])
            n_calls += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    if not args.dry_run and n_calls > 0:
        print(f"\n{'=' * 72}")
        print(f"Summary: {n_calls} successful predictions, "
              f"mean |API_p50 - actual| = ${total_err/n_calls:,.0f}")


if __name__ == '__main__':
    main()
