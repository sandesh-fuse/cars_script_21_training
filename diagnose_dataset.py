"""
diagnose_dataset.py
====================
Print a comprehensive summary of a training dataset so we can compare two
datasets side-by-side and find what changed.

USAGE:
    python diagnose_dataset.py path/to/data.csv
    python diagnose_dataset.py path/to/data.csv --train-cutoff 2018-01-01 --test-months 3

What it prints:
  1. Shape, schema, raw null rates
  2. Target (salevalue) distribution: overall and by year-month
  3. Key feature null rates (vazipcode, make, model, mileage, conditions, ...)
  4. Train/test split summary (same metrics for each split)
  5. Cap impact at various percentiles
  6. Per-tier target distribution (rows per tier, mean/median salevalue)
  7. Cross-feature checks: missingness correlation, conflicting nav_*/primary pairs

Pipe the output to a file so you can paste it back:
    python diagnose_dataset.py path/to/data.csv > dataset_old.txt
    python diagnose_dataset.py path/to/new_data.csv > dataset_new.txt
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd


TARGET_COL = "salevalue"
TIME_COL   = "record_creation_date"

# Same tiers used in evaluate_predictions.py
TIER_EDGES  = [0, 200, 500, 1000, 2500, 4000, 6000, 10000, 15000, 25000, np.inf]
TIER_LABELS = ['$0-200', '$200-500', '$500-1K', '$1K-2.5K', '$2.5K-4K',
               '$4K-6K',  '$6K-10K',  '$10K-15K', '$15K-25K', '$25K+']

# Columns whose null-rate change is interesting between datasets
KEY_FEATURES = [
    # IDs (just for null-rate sanity)
    'stock_id', 'vin', 'record_creation_date',
    # Target
    'salevalue', 'salesrevenue',
    # Vehicle basics
    'make', 'model', 'year', 'mileage', 'body_type', 'vehicle_type',
    # nav_ pair fields
    'nav_make', 'nav_model', 'nav_year',
    # Conditions
    'nav_condition', 'bodypaintcondition', 'enginecondition',
    'transmissioncondition', 'tirecondition', 'interiorcondition',
    # Damage rating
    'dsrating',
    # Geo
    'vazipcode', 'vcity', 'vstate', 'stateprovinceofregistration',
    'state_province_of_title',
]


def hr(title=""):
    bar = "=" * 90
    if title:
        print(f"\n{bar}\n{title}\n{bar}")
    else:
        print(bar)


def fmt_pct(n, total):
    if total == 0:
        return "  n/a"
    return f"{n / total * 100:5.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to training CSV")
    parser.add_argument("--train-cutoff", default="2018-01-01",
                        help="Training window starts on or after this date (default 2018-01-01)")
    parser.add_argument("--test-months", type=int, default=3,
                        help="Test window = last N months of data (default 3)")
    args = parser.parse_args()

    hr(f"DATASET DIAGNOSTIC: {os.path.basename(args.data)}")
    print(f"Full path: {os.path.abspath(args.data)}")
    if os.path.exists(args.data):
        print(f"File size: {os.path.getsize(args.data) / 1e6:.1f} MB")

    print(f"\nLoading...")
    df = pd.read_csv(args.data, low_memory=False)
    n_total = len(df)
    print(f"Loaded {n_total:,} rows × {df.shape[1]} columns")

    # ---------- 1. Shape and schema --------------------------------------------
    hr("1. SHAPE & SCHEMA")
    print(f"Rows: {n_total:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    print(f"\nColumns present from KEY_FEATURES list:")
    present = [c for c in KEY_FEATURES if c in df.columns]
    missing = [c for c in KEY_FEATURES if c not in df.columns]
    print(f"  Present ({len(present)}): {present}")
    if missing:
        print(f"  Missing ({len(missing)}): {missing}")

    # ---------- 2. Target distribution overall ---------------------------------
    hr("2. TARGET (salevalue) DISTRIBUTION — OVERALL")
    if TARGET_COL not in df.columns:
        print(f"  ERROR: {TARGET_COL} column not found")
        return
    y = pd.to_numeric(df[TARGET_COL], errors='coerce')
    n_null = y.isna().sum()
    n_valid = n_total - n_null
    print(f"  Total rows: {n_total:,}")
    print(f"  Salevalue null: {n_null:,} ({fmt_pct(n_null, n_total)})")
    print(f"  Salevalue valid: {n_valid:,} ({fmt_pct(n_valid, n_total)})")
    if n_valid > 0:
        y_valid = y.dropna()
        print(f"  Salevalue > 0: {(y_valid > 0).sum():,} ({fmt_pct((y_valid > 0).sum(), n_total)})")
        print(f"  Salevalue = 0: {(y_valid == 0).sum():,}")
        print(f"  Salevalue < 0: {(y_valid < 0).sum():,} (probably bad data if non-zero)")
        print(f"\n  Statistics (on non-null):")
        print(f"    min:    ${y_valid.min():.2f}")
        print(f"    p1:     ${y_valid.quantile(0.01):.0f}")
        print(f"    p5:     ${y_valid.quantile(0.05):.0f}")
        print(f"    p25:    ${y_valid.quantile(0.25):.0f}")
        print(f"    p50:    ${y_valid.quantile(0.50):.0f}")
        print(f"    p75:    ${y_valid.quantile(0.75):.0f}")
        print(f"    p95:    ${y_valid.quantile(0.95):.0f}")
        print(f"    p99:    ${y_valid.quantile(0.99):.0f}")
        print(f"    p99.5:  ${y_valid.quantile(0.995):.0f}")
        print(f"    p99.9:  ${y_valid.quantile(0.999):.0f}")
        print(f"    max:    ${y_valid.max():.0f}")
        print(f"    mean:   ${y_valid.mean():.0f}")

    # ---------- 3. Target by year and year-month -------------------------------
    hr("3. TARGET DISTRIBUTION BY YEAR / YEAR-MONTH")
    if TIME_COL in df.columns:
        t = pd.to_datetime(df[TIME_COL], errors='coerce')
        valid_time = t.notna()
        y_valid_mask = y.notna()
        print(f"\nRecords by year (count, count_with_target, null_rate):")
        years = t.dt.year
        for yr in sorted(years.dropna().unique()):
            mask = (years == yr)
            n_yr = mask.sum()
            n_yr_with_target = (mask & y_valid_mask).sum()
            null_rate = 1 - (n_yr_with_target / n_yr) if n_yr > 0 else 0
            mean_y = y[mask].mean() if n_yr_with_target > 0 else None
            print(f"  {int(yr)}: n={n_yr:>8,}  target_present={n_yr_with_target:>8,}  "
                  f"null_rate={null_rate*100:5.1f}%  mean_y={f'${mean_y:.0f}' if mean_y else '   n/a'}")

        # Last 6 year-months (likely covers test window)
        print(f"\nLast 6 year-months (test window candidates):")
        ym = t.dt.to_period('M')
        recent = sorted(ym.dropna().unique())[-6:]
        for p in recent:
            mask = (ym == p)
            n_p = mask.sum()
            n_p_with_target = (mask & y_valid_mask).sum()
            null_rate = 1 - (n_p_with_target / n_p) if n_p > 0 else 0
            mean_y = y[mask].mean() if n_p_with_target > 0 else None
            print(f"  {p}: n={n_p:>6,}  target_present={n_p_with_target:>6,}  "
                  f"null_rate={null_rate*100:5.1f}%  mean_y={f'${mean_y:.0f}' if mean_y else '   n/a'}")
    else:
        print(f"  WARNING: {TIME_COL} column missing — cannot do temporal analysis")

    # ---------- 4. Key feature null rates --------------------------------------
    hr("4. KEY FEATURE NULL RATES")
    print(f"\nNull rate (per feature, full dataset):")
    print(f"  {'feature':<30}  {'n_null':>10}  {'pct':>7}  {'dtype':>10}")
    for c in KEY_FEATURES:
        if c not in df.columns:
            continue
        n_null = df[c].isna().sum()
        # Treat empty strings as null for text columns
        if df[c].dtype == 'object':
            n_blank = (df[c].astype(str).str.strip() == '').sum()
            n_null += n_blank
        print(f"  {c:<30}  {n_null:>10,}  {fmt_pct(n_null, n_total):>7}  "
              f"{str(df[c].dtype):>10}")

    # ---------- 5. vazipcode special handling ---------------------------------
    hr("5. vazipcode DEEP DIVE")
    if 'vazipcode' in df.columns:
        z = df['vazipcode']
        n_null = z.isna().sum()
        z_str = z.astype(str).str.strip()
        n_blank = (z_str == '').sum()
        n_nan_str = (z_str == 'nan').sum()
        print(f"  Raw null: {n_null:,} ({fmt_pct(n_null, n_total)})")
        print(f"  Empty string: {n_blank:,}")
        print(f"  Literal 'nan' string: {n_nan_str:,}")
        # Look for the '94611.0' float-cast string artifact
        n_dotzero = z_str.str.endswith('.0').sum()
        print(f"  Ending in '.0' (float-cast artifact): {n_dotzero:,} ({fmt_pct(n_dotzero, n_total)})")
        # Length distribution
        nonblank = z_str[~z_str.isin(['', 'nan'])]
        if len(nonblank) > 0:
            len_counts = nonblank.str.len().value_counts().sort_index()
            print(f"  Length distribution (non-blank):")
            for L, cnt in len_counts.head(10).items():
                print(f"    len={L}: {cnt:,}")
        # Per-year null rate
        if TIME_COL in df.columns:
            t = pd.to_datetime(df[TIME_COL], errors='coerce')
            years = t.dt.year
            print(f"  vazipcode null rate by year:")
            null_mask = z.isna() | (z_str.isin(['', 'nan']))
            for yr in sorted(years.dropna().unique()):
                year_mask = (years == yr)
                n_yr = year_mask.sum()
                n_yr_null = (year_mask & null_mask).sum()
                if n_yr > 0:
                    print(f"    {int(yr)}: {n_yr_null:,}/{n_yr:,} ({n_yr_null/n_yr*100:.1f}%)")
    else:
        print("  vazipcode column not found.")

    # ---------- 6. nav_* / primary disagreement check -------------------------
    hr("6. nav_* / PRIMARY FIELD AGREEMENT")
    for nav_col, primary_col in [('nav_make', 'make'),
                                   ('nav_model', 'model'),
                                   ('nav_year', 'year')]:
        if nav_col in df.columns and primary_col in df.columns:
            a = df[nav_col].astype(str).str.strip().str.lower()
            b = df[primary_col].astype(str).str.strip().str.lower()
            a = a.where(~a.isin(['', 'nan', 'none']), None)
            b = b.where(~b.isin(['', 'nan', 'none']), None)
            both_present = a.notna() & b.notna()
            n_both = both_present.sum()
            n_agree = (a[both_present] == b[both_present]).sum()
            n_disagree = n_both - n_agree
            nav_only = (a.notna() & b.isna()).sum()
            prim_only = (a.isna() & b.notna()).sum()
            both_null = (a.isna() & b.isna()).sum()
            print(f"\n  ({nav_col}, {primary_col}):")
            print(f"    both present: {n_both:>8,}  ({fmt_pct(n_both, n_total)})")
            print(f"    of which agree: {n_agree:>8,} disagree: {n_disagree:>8,}")
            print(f"    {nav_col} only: {nav_only:>8,}  ({fmt_pct(nav_only, n_total)})")
            print(f"    {primary_col} only: {prim_only:>8,}  ({fmt_pct(prim_only, n_total)})")
            print(f"    both null: {both_null:>8,}  ({fmt_pct(both_null, n_total)})")
        else:
            present = [c for c in [nav_col, primary_col] if c in df.columns]
            missing = [c for c in [nav_col, primary_col] if c not in df.columns]
            print(f"\n  Skipped: have {present}, missing {missing}")

    # ---------- 7. Train/test split summary -----------------------------------
    hr("7. TRAIN / TEST SPLIT SUMMARY")
    if TIME_COL not in df.columns:
        print("  Skipping: no time column.")
    else:
        t = pd.to_datetime(df[TIME_COL], errors='coerce')
        df_t = df.copy()
        df_t['_t'] = t
        # Apply train cutoff
        train_cutoff = pd.to_datetime(args.train_cutoff)
        valid = df_t['_t'].notna() & y.notna() & (y > 0)
        in_window = valid & (df_t['_t'] >= train_cutoff)
        # Test = last N months from the window
        if in_window.any():
            max_t = df_t.loc[in_window, '_t'].max()
            test_start = max_t - pd.DateOffset(months=args.test_months)
            train_mask = in_window & (df_t['_t'] < test_start)
            test_mask  = in_window & (df_t['_t'] >= test_start)

            print(f"  Train cutoff: {train_cutoff.date()}")
            print(f"  Test window: last {args.test_months} months ({test_start.date()} to {max_t.date()})")
            for name, mask in [("Train", train_mask), ("Test", test_mask)]:
                n = mask.sum()
                y_sub = y[mask]
                if n > 0:
                    print(f"\n  {name}: {n:,} rows")
                    print(f"    target mean:   ${y_sub.mean():.0f}")
                    print(f"    target median: ${y_sub.median():.0f}")
                    print(f"    target p95:    ${y_sub.quantile(0.95):.0f}")
                    print(f"    target p99.5:  ${y_sub.quantile(0.995):.0f}")
                    print(f"    target max:    ${y_sub.max():.0f}")
                    # per-tier counts
                    print(f"    Per-tier row counts:")
                    tiers = pd.cut(y_sub, bins=TIER_EDGES, labels=TIER_LABELS,
                                    right=True, include_lowest=True)
                    tier_counts = tiers.value_counts().sort_index()
                    for tname in TIER_LABELS:
                        cnt = tier_counts.get(tname, 0)
                        if cnt > 0:
                            sub = y_sub[tiers == tname]
                            print(f"      {tname:>10}: n={cnt:>6,}  mean=${sub.mean():>7.0f}  median=${sub.median():>7.0f}")
                        else:
                            print(f"      {tname:>10}: n=0")
                else:
                    print(f"\n  {name}: 0 rows")
        else:
            print("  No valid rows in the window — cap-pct/min-salevalue may be excluding everything.")

    # ---------- 8. Cap-percentile breakdown -----------------------------------
    hr("8. CAP IMPACT AT VARIOUS PERCENTILES")
    y_pos = y[(y > 0) & y.notna()]
    if len(y_pos) > 0:
        print(f"  (computed on {len(y_pos):,} rows where salevalue > 0)")
        for p in [99.0, 99.5, 99.9, 99.99, 100.0]:
            if p < 100:
                cap = y_pos.quantile(p / 100.0)
                dropped = (y_pos > cap).sum()
                print(f"  cap = p{p:>5}  =>  ${cap:>8.0f}   "
                      f"drops {dropped:>6,} rows ({dropped/len(y_pos)*100:.2f}%)")
            else:
                print(f"  cap = p{p:>5}  =>  none      drops 0 rows (full range kept)")
    else:
        print("  No positive-target rows.")

    hr("END OF DIAGNOSTIC")
    print()


if __name__ == "__main__":
    main()
