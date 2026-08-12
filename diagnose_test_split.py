"""
diagnose_test_split.py
=======================
Compare model performance on test rows that had salevalue in BOTH datasets
versus rows that were newly labeled in the new dataset.

If the model's MAE is much worse on the "newly labeled" subset, that confirms
the test set got harder (rather than the model getting worse).

USAGE:
    python diagnose_test_split.py \\
        --old-data path/to/old.csv \\
        --new-data path/to/new.csv \\
        --predictions artifacts/script21/test_predictions.parquet

The script joins by stock_id (override with --join-col vin if needed).
"""
import argparse
import os
import numpy as np
import pandas as pd


TIER_EDGES  = [0, 200, 500, 1000, 2500, 4000, 6000, 10000, 15000, 25000, np.inf]
TIER_LABELS = ['$0-200', '$200-500', '$500-1K', '$1K-2.5K', '$2.5K-4K',
               '$4K-6K',  '$6K-10K',  '$10K-15K', '$15K-25K', '$25K+']


def metrics_block(df_sub: pd.DataFrame, label: str):
    """Print MAE/RMSE/coverage for a subset, plus per-tier breakdown."""
    print(f"\n  {label}")
    print(f"  {'-' * len(label)}")
    if len(df_sub) == 0:
        print(f"    (0 rows)")
        return
    actual = df_sub['salevalue'].values
    p5  = df_sub['p5'].values
    p50 = df_sub['p50'].values
    p95 = df_sub['p95'].values
    mae  = float(np.mean(np.abs(actual - p50)))
    rmse = float(np.sqrt(np.mean((actual - p50)**2)))
    cov  = float(np.mean((actual >= p5) & (actual <= p95)))
    print(f"    rows:           {len(df_sub):,}")
    print(f"    actual mean:    ${actual.mean():.0f}")
    print(f"    actual median:  ${np.median(actual):.0f}")
    print(f"    pred mean p50:  ${p50.mean():.0f}")
    print(f"    MAE:            ${mae:.0f}")
    print(f"    RMSE:           ${rmse:.0f}")
    print(f"    Coverage 90:    {cov*100:.1f}%")
    print(f"    Per-tier:")
    tiers = pd.cut(actual, bins=TIER_EDGES, labels=TIER_LABELS, right=True, include_lowest=True)
    tiers = np.asarray(tiers)
    for tname in TIER_LABELS:
        m = (tiers == tname)
        n_t = m.sum()
        if n_t == 0:
            continue
        mae_t = float(np.mean(np.abs(actual[m] - p50[m])))
        mean_actual = float(actual[m].mean())
        print(f"      {tname:>10}: n={n_t:>5,}  MAE=${mae_t:>6.0f}  mean_actual=${mean_actual:>6.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-data", required=True,
                        help="Path to the old training CSV")
    parser.add_argument("--new-data", required=True,
                        help="Path to the new training CSV")
    parser.add_argument("--predictions", required=True,
                        help="Path to the new model's test_predictions.parquet "
                             "(produced by train_save_script21.py)")
    parser.add_argument("--join-col", default="stock_id",
                        help="Column to join old/new datasets (default: stock_id, "
                             "fallback to vin if you see lots of unmatched rows)")
    args = parser.parse_args()

    print(f"Loading old data ({os.path.basename(args.old_data)})...")
    # Load only the columns we need to keep memory manageable
    old = pd.read_csv(args.old_data, low_memory=False,
                      usecols=[args.join_col, 'salevalue'])
    print(f"  {len(old):,} rows")

    print(f"\nLoading new data ({os.path.basename(args.new_data)})...")
    new = pd.read_csv(args.new_data, low_memory=False,
                      usecols=[args.join_col, 'salevalue'])
    print(f"  {len(new):,} rows")

    print(f"\nLoading predictions ({os.path.basename(args.predictions)})...")
    if args.predictions.endswith('.parquet'):
        preds = pd.read_parquet(args.predictions)
    else:
        preds = pd.read_csv(args.predictions)
    print(f"  {len(preds):,} prediction rows")

    # Build the lookup of OLD salevalue, indexed by join key
    if args.join_col not in old.columns:
        print(f"\nERROR: '{args.join_col}' not in old data columns.")
        return
    if args.join_col not in preds.columns:
        print(f"\nERROR: '{args.join_col}' not in predictions columns.")
        print(f"       Available: {list(preds.columns)[:10]}")
        return

    # Build lookup: for each join key, was it in old AND did it have salevalue?
    old_lookup = old.set_index(args.join_col)['salevalue']
    # Pull old salevalue for each test row
    preds['_old_salevalue'] = preds[args.join_col].map(old_lookup)

    # Categorize each test row
    preds['_in_old']         = preds[args.join_col].isin(set(old[args.join_col]))
    preds['_had_label_old']  = preds['_old_salevalue'].notna()

    # Three buckets:
    #   A: was in old AND had salevalue in old   (the "old test set" subset)
    #   B: was in old BUT had no salevalue in old  (backfilled rows)
    #   C: not in old at all                       (genuinely new rows)
    mask_A = preds['_in_old'] & preds['_had_label_old']
    mask_B = preds['_in_old'] & ~preds['_had_label_old']
    mask_C = ~preds['_in_old']

    print("\n" + "=" * 80)
    print(f"TEST SET BREAKDOWN BY ORIGIN  (total: {len(preds):,} rows)")
    print("=" * 80)
    print(f"  A: In OLD data WITH salevalue:    {mask_A.sum():>6,} ({mask_A.mean()*100:5.1f}%)")
    print(f"  B: In OLD data WITHOUT salevalue: {mask_B.sum():>6,} ({mask_B.mean()*100:5.1f}%)  <- newly labeled")
    print(f"  C: NOT in OLD data:               {mask_C.sum():>6,} ({mask_C.mean()*100:5.1f}%)  <- genuinely new")

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE BY SUBSET")
    print("=" * 80)

    metrics_block(preds, "OVERALL (all test rows)")
    metrics_block(preds[mask_A],
                  "A — Rows that had salevalue in BOTH datasets")
    metrics_block(preds[mask_B],
                  "B — Rows that were in OLD but salevalue was NULL then (NEWLY LABELED)")
    metrics_block(preds[mask_C],
                  "C — Rows that weren't in OLD at all (NEW ROWS)")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
  If MAE on (A) is similar to your old model's MAE, but MAE on (B) is much
  higher, then your test set got harder — the newly-labeled rows are
  systematically harder to predict than the rows that had labels promptly.

  This is NOT a model regression; it's a test set composition change.
  The fair comparison is "old model on old test" vs "new model on subset A".

  If MAE on (A) is ALSO worse than your old model's MAE, then the model
  itself regressed and we need to look at training-side issues.
""")


if __name__ == "__main__":
    main()
