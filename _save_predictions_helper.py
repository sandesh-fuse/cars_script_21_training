"""
_save_predictions_helper.py
============================
Shared helper for both train_save_*.py scripts. Builds a tidy DataFrame
with prediction columns + ID columns for downstream analysis.

Columns saved (in order):
    stock_id, vin, record_creation_date,
    salevalue (actual),  predicted_sale_value (p50),
    p5, p50, p95, abs_error_p50, ci_width
Plus 'is_cult' for script21 (caller adds it).

Missing ID columns (e.g., if a CSV uses different names) are filled with None
so the schema stays consistent.
"""
import os
import numpy as np
import pandas as pd

ID_COLS = ["stock_id", "vin", "record_creation_date"]


def build_predictions_frame(raw_df: pd.DataFrame,
                              actuals: np.ndarray,
                              preds_p5:  np.ndarray,
                              preds_p50: np.ndarray,
                              preds_p95: np.ndarray,
                              extra_cols: dict = None) -> pd.DataFrame:
    """Return a tidy DataFrame ready to be saved as parquet/csv."""
    n = len(raw_df)
    if not (len(actuals) == len(preds_p5) == len(preds_p50) == len(preds_p95) == n):
        raise ValueError(f"Length mismatch: rows={n}, actuals={len(actuals)}, "
                         f"p5={len(preds_p5)}, p50={len(preds_p50)}, p95={len(preds_p95)}")

    out = pd.DataFrame(index=range(n))
    for c in ID_COLS:
        out[c] = raw_df[c].values if c in raw_df.columns else None

    out["salevalue"]            = actuals
    out["predicted_sale_value"] = preds_p50
    out["p5"]                   = preds_p5
    out["p50"]                  = preds_p50
    out["p95"]                  = preds_p95
    out["abs_error_p50"]        = np.abs(preds_p50 - actuals)
    out["ci_width"]             = preds_p95 - preds_p5

    if extra_cols:
        for k, v in extra_cols.items():
            out[k] = v
    return out


def save_predictions(df: pd.DataFrame, out_dir: str, name: str, also_csv: bool = True):
    """Write parquet and (optionally) CSV next to it."""
    pq_path = os.path.join(out_dir, f"{name}.parquet")
    df.to_parquet(pq_path, index=False)
    print(f"  Saved: {pq_path}  ({len(df):,} rows)")
    if also_csv:
        csv_path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
