"""
timewise_breakdown.py
======================
Ported from notebooks/timewise_result_breakdown.ipynb.

Reads a predictions file (parquet or csv, same shape as
{train,test}_predictions.{parquet,csv} produced by
_save_predictions_helper.build_predictions_frame -- i.e. it needs
record_creation_date, salevalue, predicted_sale_value, and ci_width) and
breaks accuracy down over time:

  - weekly_metrics            -- MAE/RMSE/RMSLE/CI-width per calendar week
  - monthly_metrics           -- same, per calendar month
  - monthly_sampled_metrics   -- same, per calendar month, but computed on a
                                  fixed-size random sample (default 1,000,
                                  with replacement) so months with very
                                  different record volumes are comparable
  - samples_with_metrics_{YYYY_MM}  -- one file per month: the sampled rows
                                  themselves, with that month's MAE/RMSE/RMSLE
                                  and each row's own absolute error attached
                                  (useful for spot-checking specific rows)

USAGE:
    python timewise_breakdown.py artifacts/script21/test_predictions.parquet \
        --out artifacts/script21/timewise_breakdown

OR as a library:
    from timewise_breakdown import run_timewise_breakdown
    run_timewise_breakdown(test_pred_df, out_dir="artifacts/script21/timewise_breakdown")

train_save_script21.py calls this automatically at the end of training (on
the test-set predictions, since train predictions are optimistic -- the
model saw those rows); re-run standalone whenever you want a fresh breakdown
without retraining.
"""
import os
import argparse
import numpy as np
import pandas as pd

from evaluate_predictions import mae, rmse, rmsle
from _save_predictions_helper import save_predictions

REQUIRED_COLS = ['record_creation_date', 'salevalue', 'predicted_sale_value', 'ci_width']

DEFAULT_N_SAMPLES    = 1000
DEFAULT_SAMPLE_SEED  = 42

_EXACT_DATE_COL = '__exact_date__'   # temp column; see compute_periodic_metrics


# ============================================================
# Per-period metrics (weekly / monthly, all rows)
# ============================================================
def _period_metrics(group: pd.DataFrame) -> pd.Series:
    group = group.dropna(subset=['salevalue', 'predicted_sale_value', 'ci_width'])
    if group.empty:
        return pd.Series({
            'start_date': pd.NaT, 'end_date': pd.NaT, 'record_count': 0,
            'MAE': np.nan, 'RMSE': np.nan, 'RMSLE': np.nan,
            'Median_90_CI_Width': np.nan, 'Mean_90_CI_Width': np.nan,
        })
    y_true = group['salevalue'].values
    y_pred = group['predicted_sale_value'].values
    return pd.Series({
        'start_date':         group[_EXACT_DATE_COL].min(),
        'end_date':           group[_EXACT_DATE_COL].max(),
        'record_count':       len(group),
        'MAE':                mae(y_true, y_pred),
        'RMSE':               rmse(y_true, y_pred),
        'RMSLE':              rmsle(y_true, y_pred),
        'Median_90_CI_Width': group['ci_width'].median(),
        'Mean_90_CI_Width':   group['ci_width'].mean(),
    })


def compute_periodic_metrics(df: pd.DataFrame, freq: str,
                               time_col: str = 'record_creation_date') -> pd.DataFrame:
    """Aggregate MAE/RMSE/RMSLE/CI-width per calendar period.

    freq: any pandas offset alias for pd.Grouper (e.g. 'W' weekly, 'ME'
    month-end). Returns one row per period with a 'period' column (the
    Grouper bin) plus the actual start/end timestamps and count of the rows
    that landed in it.
    """
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col])
    # Duplicate the grouping column first -- pandas' include_groups=False
    # drops the original key column from what the aggregation function sees.
    work[_EXACT_DATE_COL] = work[time_col]

    out = (work.groupby(pd.Grouper(key=time_col, freq=freq))
               .apply(_period_metrics, include_groups=False)
               .reset_index()
               .rename(columns={time_col: 'period'}))
    return out


def compute_weekly_metrics(df: pd.DataFrame,
                             time_col: str = 'record_creation_date') -> pd.DataFrame:
    return compute_periodic_metrics(df, freq='W', time_col=time_col)


def compute_monthly_metrics(df: pd.DataFrame,
                              time_col: str = 'record_creation_date') -> pd.DataFrame:
    out = compute_periodic_metrics(df, freq='ME', time_col=time_col)
    return _add_month_cols(out)


def _add_month_cols(out: pd.DataFrame) -> pd.DataFrame:
    out['year']          = out['period'].dt.year
    out['month_name']    = out['period'].dt.month_name()
    out['month_number']  = out['period'].dt.month
    front = ['period', 'year', 'month_name', 'month_number']
    return out[front + [c for c in out.columns if c not in front]]


# ============================================================
# Per-month metrics on a fixed-size random sample
# ============================================================
def _sampled_period_metrics(group: pd.DataFrame, n_samples: int, random_state: int) -> pd.Series:
    group = group.dropna(subset=['salevalue', 'predicted_sale_value', 'ci_width'])
    if group.empty:
        return pd.Series({
            'start_date': pd.NaT, 'end_date': pd.NaT, 'record_count': 0,
            'MAE': np.nan, 'RMSE': np.nan, 'RMSLE': np.nan,
            'Median_90_CI_Width': np.nan, 'Mean_90_CI_Width': np.nan,
        })
    # Always draw exactly n_samples (with replacement, so months with fewer
    # rows than n_samples still work) so every month's metrics are computed
    # on an equal sample size and are directly comparable.
    sampled = group.sample(n=n_samples, replace=True, random_state=random_state)
    y_true = sampled['salevalue'].values
    y_pred = sampled['predicted_sale_value'].values
    return pd.Series({
        'start_date':         sampled[_EXACT_DATE_COL].min(),
        'end_date':           sampled[_EXACT_DATE_COL].max(),
        'record_count':       len(sampled),
        'MAE':                mae(y_true, y_pred),
        'RMSE':               rmse(y_true, y_pred),
        'RMSLE':              rmsle(y_true, y_pred),
        'Median_90_CI_Width': sampled['ci_width'].median(),
        'Mean_90_CI_Width':   sampled['ci_width'].mean(),
    })


def compute_monthly_sampled_metrics(df: pd.DataFrame,
                                      n_samples: int = DEFAULT_N_SAMPLES,
                                      random_state: int = DEFAULT_SAMPLE_SEED,
                                      time_col: str = 'record_creation_date') -> pd.DataFrame:
    """Same shape as compute_monthly_metrics, but each month's metrics come
    from a fixed-size random sample (default 1,000, with replacement)
    instead of all rows -- gives an apples-to-apples month-over-month
    comparison when record volume varies a lot month to month.
    """
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col])
    work[_EXACT_DATE_COL] = work[time_col]

    out = (work.groupby(pd.Grouper(key=time_col, freq='ME'))
               .apply(lambda g: _sampled_period_metrics(g, n_samples, random_state),
                      include_groups=False)
               .reset_index()
               .rename(columns={time_col: 'period'}))
    return _add_month_cols(out)


def export_monthly_samples(df: pd.DataFrame, out_dir: str,
                             n_samples: int = DEFAULT_N_SAMPLES,
                             random_state: int = DEFAULT_SAMPLE_SEED,
                             time_col: str = 'record_creation_date',
                             also_csv: bool = True) -> list:
    """For each calendar month present in df, draw n_samples random rows
    (with replacement) and save them -- with that month's MAE/RMSE/RMSLE and
    each row's own absolute error attached -- as
    '{out_dir}/samples_with_metrics_{YYYY_MM}.parquet' (+ .csv).

    Returns a list of (month_str, n_rows_written) tuples.
    """
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col])

    written = []
    for period, group in work.groupby(pd.Grouper(key=time_col, freq='ME')):
        group = group.dropna(subset=['salevalue', 'predicted_sale_value', 'ci_width'])
        if group.empty:
            continue
        sampled = group.sample(n=n_samples, replace=True, random_state=random_state).copy()

        y_true = sampled['salevalue'].values
        y_pred = sampled['predicted_sale_value'].values
        sampled['month_MAE']          = mae(y_true, y_pred)
        sampled['month_RMSE']         = rmse(y_true, y_pred)
        sampled['month_RMSLE']        = rmsle(y_true, y_pred)
        sampled['row_absolute_error'] = np.abs(y_true - y_pred)

        month_str = period.strftime('%Y_%m')
        save_predictions(sampled, out_dir, f"samples_with_metrics_{month_str}", also_csv=also_csv)
        written.append((month_str, len(sampled)))
    return written


# ============================================================
# Orchestrator
# ============================================================
def run_timewise_breakdown(df: pd.DataFrame, out_dir: str,
                             n_samples: int = DEFAULT_N_SAMPLES,
                             random_state: int = DEFAULT_SAMPLE_SEED,
                             time_col: str = 'record_creation_date',
                             also_csv: bool = True) -> dict:
    """Run the full weekly/monthly/monthly-sampled breakdown on a predictions
    frame (same shape as {train,test}_predictions.{parquet,csv}) and save
    every resulting table under out_dir.

    Returns the in-memory summary tables (weekly_metrics, monthly_metrics,
    monthly_sampled_metrics) for callers that want to inspect them without
    re-reading from disk. Per-month sample files are saved but not returned
    (read them back from disk if needed -- they're one CSV/parquet per month).
    """
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"timewise_breakdown: missing required columns: {missing}")

    os.makedirs(out_dir, exist_ok=True)

    weekly = compute_weekly_metrics(df, time_col=time_col)
    save_predictions(weekly, out_dir, "weekly_metrics", also_csv=also_csv)

    monthly = compute_monthly_metrics(df, time_col=time_col)
    save_predictions(monthly, out_dir, "monthly_metrics", also_csv=also_csv)

    monthly_sampled = compute_monthly_sampled_metrics(
        df, n_samples=n_samples, random_state=random_state, time_col=time_col)
    save_predictions(monthly_sampled, out_dir, "monthly_sampled_metrics", also_csv=also_csv)

    written = export_monthly_samples(
        df, out_dir, n_samples=n_samples, random_state=random_state,
        time_col=time_col, also_csv=also_csv)
    for month_str, n in written:
        print(f"  Saved {n:,} sampled rows for {month_str} "
              f"-> {out_dir}/samples_with_metrics_{month_str}.*")

    return {
        'weekly_metrics':          weekly,
        'monthly_metrics':         monthly,
        'monthly_sampled_metrics': monthly_sampled,
    }


# ============================================================
# CLI
# ============================================================
def _load_predictions(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.parquet':
        return pd.read_parquet(path)
    if ext in ('.csv', '.tsv'):
        return pd.read_csv(path, sep=',' if ext == '.csv' else '\t')
    raise ValueError(f"Unsupported file extension: {ext} (use .parquet or .csv)")


def main():
    parser = argparse.ArgumentParser(
        description="Break a saved predictions file down by week/month.")
    parser.add_argument('path', help="Path to a predictions file (.parquet or .csv), "
                                      "e.g. artifacts/script21/test_predictions.parquet")
    parser.add_argument('--out', default=None,
                        help="Output dir (default: '<dir of path>/timewise_breakdown/')")
    parser.add_argument('--n-samples', type=int, default=DEFAULT_N_SAMPLES,
                        help=f"Rows to sample per month (default {DEFAULT_N_SAMPLES})")
    parser.add_argument('--seed', type=int, default=DEFAULT_SAMPLE_SEED,
                        help=f"Random seed for sampling (default {DEFAULT_SAMPLE_SEED})")
    parser.add_argument('--parquet-only', action='store_true',
                        help="Skip writing CSV copies (parquet only)")
    args = parser.parse_args()

    df = _load_predictions(args.path)
    out_dir = args.out or os.path.join(os.path.dirname(args.path) or '.', 'timewise_breakdown')

    run_timewise_breakdown(df, out_dir, n_samples=args.n_samples, random_state=args.seed,
                            also_csv=not args.parquet_only)


if __name__ == '__main__':
    main()
