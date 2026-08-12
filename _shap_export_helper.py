"""
_shap_export_helper.py
=======================
Shared module for both train_save_*.py scripts. Computes SHAP for a batch of
rows against the p50 model and writes long-format CSVs (one row per
car-feature pair) showing attribution magnitudes.

Produces two views per dataset (train OR test):
  - <prefix>_shap_engineered.csv: engineered features (model internals)
  - <prefix>_shap_raw.csv:         raw user-facing groups (engineered features
                                    collapsed to their source raw input)

Each row has:
  stock_id, vin, record_creation_date, salevalue, predicted_sale_value,
  feature, feature_value (engineered only), feature_label (raw only),
  log_shap (engineered only), dollar_impact, pct_of_prediction,
  pct_of_top_feature, rank_by_abs
"""

import os
import gc
import numpy as np
import pandas as pd

from shap_dollar_helper import _patch_xgb_base_score_for_shap


ID_COLS = ["stock_id", "vin", "record_creation_date"]


def _get_explainer(model):
    """Patch the model and build a TreeExplainer."""
    import shap
    patched = _patch_xgb_base_score_for_shap(model)
    explainer = shap.TreeExplainer(patched)
    return patched, explainer


def _enrich_with_dollars(shap_values, base_log_value, log_preds):
    """Given shap values (N x F) in log space, return (dollar_impact, pct_of_pred) arrays.

    dollar_impact[i,j] = expm1(base + shap_values[i,j]) - expm1(base)
                         (marginal $ impact; broadcasting the scalar base over rows)
    pct_of_prediction[i,j] = dollar_impact[i,j] / expm1(log_preds[i]) * 100
    """
    base_dollars = float(np.expm1(base_log_value))
    # marginal dollar impact, vectorized
    dollar_impact = np.expm1(base_log_value + shap_values) - base_dollars
    final_dollars = np.clip(np.expm1(log_preds), 1.0, None)  # shape (N,)
    pct_of_pred   = (dollar_impact / final_dollars[:, None]) * 100.0
    return dollar_impact, pct_of_pred


def _format_value_for_csv(v):
    """Convert a numpy/pandas value to a CSV-friendly string scalar (or None)."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return None
        if float(v).is_integer():
            return str(int(v))
        return f"{float(v):g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    return str(v)


def compute_and_save_shap_engineered(model, X, raw_df, feature_names,
                                       actuals, preds_p50,
                                       out_dir, prefix, also_csv=True):
    """Compute engineered-feature SHAP for all rows in X, save to parquet (and CSV).

    Long-format: N rows × F features = N*F output rows, sorted by row then |attribution|.

    Parameters
    ----------
    model : the median (p50) XGBoost model
    X : DataFrame of preprocessed features (N x F)
    raw_df : raw DataFrame aligned to X (used to pull stock_id/vin/etc.)
    feature_names : list of engineered feature column names
    actuals : array of true salevalues
    preds_p50 : array of point predictions (dollars, deflated for script21)
    out_dir : directory to write into
    prefix : "train" or "test" — used in filename
    """
    print(f"\n[SHAP engineered] Computing for {len(X):,} rows × {len(feature_names)} features...")
    patched_model, explainer = _get_explainer(model)

    # Predict in log space and compute SHAP in batches to manage memory
    batch = 5000 if len(X) > 50_000 else len(X)
    all_dollar = np.empty((len(X), len(feature_names)), dtype=np.float32)
    all_pct    = np.empty((len(X), len(feature_names)), dtype=np.float32)
    all_logshap = np.empty((len(X), len(feature_names)), dtype=np.float32)

    expected = explainer.expected_value
    base_log_value = float(expected if not hasattr(expected, '__len__') else expected[0])

    log_preds_all = patched_model.predict(X)  # log space

    n_batches = (len(X) + batch - 1) // batch
    for bi in range(n_batches):
        lo, hi = bi * batch, min((bi + 1) * batch, len(X))
        X_chunk = X.iloc[lo:hi]
        shap_vals = explainer.shap_values(X_chunk)
        if hasattr(shap_vals, 'ndim') and shap_vals.ndim == 2:
            pass
        else:
            shap_vals = np.asarray(shap_vals)
        dollar_impact, pct_of_pred = _enrich_with_dollars(
            shap_vals, base_log_value, log_preds_all[lo:hi])
        all_logshap[lo:hi] = shap_vals.astype(np.float32)
        all_dollar[lo:hi]  = dollar_impact.astype(np.float32)
        all_pct[lo:hi]     = pct_of_pred.astype(np.float32)
        if n_batches > 5 and (bi + 1) % max(n_batches // 10, 1) == 0:
            print(f"  Batch {bi+1}/{n_batches} done")
        del shap_vals, dollar_impact, pct_of_pred
        gc.collect()

    # Build the long-format dataframe in chunks to avoid OOM on huge train sets
    print(f"[SHAP engineered] Assembling long-format frame ({len(X)*len(feature_names):,} rows)...")
    pq_path  = os.path.join(out_dir, f"{prefix}_shap_engineered.parquet")
    csv_path = os.path.join(out_dir, f"{prefix}_shap_engineered.csv")
    # Streamed write: append per car-chunk to parquet. We use pyarrow ParquetWriter.
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    csv_first = True
    feature_arr = np.array(feature_names, dtype=object)

    def _car_chunk_to_table(lo, hi):
        n_rows = hi - lo
        f_per_row = len(feature_names)
        total = n_rows * f_per_row
        # Repeat each ID column f_per_row times to align with the per-feature rows
        stock_ids = np.repeat(raw_df['stock_id'].iloc[lo:hi].values if 'stock_id' in raw_df.columns
                              else np.array([None]*n_rows), f_per_row)
        vins = np.repeat(raw_df['vin'].iloc[lo:hi].values if 'vin' in raw_df.columns
                         else np.array([None]*n_rows), f_per_row)
        rcds = np.repeat(
            raw_df['record_creation_date'].astype(str).iloc[lo:hi].values
            if 'record_creation_date' in raw_df.columns else np.array([None]*n_rows),
            f_per_row)
        actuals_rep = np.repeat(actuals[lo:hi], f_per_row)
        preds_rep   = np.repeat(preds_p50[lo:hi], f_per_row)
        features    = np.tile(feature_arr, n_rows)

        # Feature values: pull from X
        X_block = X.iloc[lo:hi].values  # (n_rows, F)
        feat_vals = X_block.reshape(-1)

        dol = all_dollar[lo:hi].reshape(-1)
        pct = all_pct[lo:hi].reshape(-1)
        lgs = all_logshap[lo:hi].reshape(-1)

        # Per-car: abs of dollar impact, rank, pct_of_top
        abs_dol = np.abs(all_dollar[lo:hi])  # (n_rows, F)
        # rank within each row: 1 = highest abs
        ranks = (-abs_dol).argsort(axis=1).argsort(axis=1) + 1
        max_per_row = abs_dol.max(axis=1, keepdims=True)
        max_per_row[max_per_row == 0] = 1.0  # avoid /0
        pct_of_top = (all_dollar[lo:hi] / max_per_row) * 100.0

        ranks_flat = ranks.reshape(-1)
        pct_top_flat = pct_of_top.reshape(-1)

        table = pa.table({
            'stock_id':              pa.array(stock_ids,    type=pa.string()),
            'vin':                   pa.array(vins,         type=pa.string()),
            'record_creation_date':  pa.array(rcds,         type=pa.string()),
            'salevalue':             pa.array(actuals_rep,  type=pa.float64()),
            'predicted_sale_value':  pa.array(preds_rep,    type=pa.float64()),
            'feature':               pa.array(features,     type=pa.string()),
            'feature_value':         pa.array([_format_value_for_csv(v) for v in feat_vals],
                                              type=pa.string()),
            'log_shap':              pa.array(lgs,          type=pa.float32()),
            'dollar_impact':         pa.array(dol,          type=pa.float32()),
            'pct_of_prediction':     pa.array(pct,          type=pa.float32()),
            'pct_of_top_feature':    pa.array(pct_top_flat.astype(np.float32), type=pa.float32()),
            'rank_by_abs':           pa.array(ranks_flat,   type=pa.int32()),
        })
        return table

    chunk_cars = 2000
    for lo in range(0, len(X), chunk_cars):
        hi = min(lo + chunk_cars, len(X))
        table = _car_chunk_to_table(lo, hi)
        if writer is None:
            writer = pq.ParquetWriter(pq_path, table.schema, compression='snappy')
        writer.write_table(table)
        if also_csv:
            df_chunk = table.to_pandas()
            df_chunk.to_csv(csv_path, mode='w' if csv_first else 'a',
                            header=csv_first, index=False)
            csv_first = False
        del table
        gc.collect()
    if writer is not None:
        writer.close()
    print(f"[SHAP engineered] Wrote {pq_path}")
    if also_csv:
        print(f"[SHAP engineered] Wrote {csv_path}")
    # Free buffers
    del all_dollar, all_pct, all_logshap
    gc.collect()


def compute_and_save_shap_raw(model, X, raw_df, feature_names,
                                actuals, preds_p50, request_dicts,
                                out_dir, prefix, also_csv=True):
    """Same as engineered, but collapses to raw user-facing feature groups.

    request_dicts is a list of dicts (one per row) representing the original
    raw values (used to look up the user-shown value, e.g. 'Runs & Drives'
    instead of severity-encoded integer).
    """
    from app.raw_feature_mapping import collapse_engineered_to_raw

    print(f"\n[SHAP raw] Computing for {len(X):,} rows × ~15 raw groups...")
    patched_model, explainer = _get_explainer(model)
    expected = explainer.expected_value
    base_log_value = float(expected if not hasattr(expected, '__len__') else expected[0])
    log_preds_all = patched_model.predict(X)

    pq_path  = os.path.join(out_dir, f"{prefix}_shap_raw.parquet")
    csv_path = os.path.join(out_dir, f"{prefix}_shap_raw.csv")
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    csv_first = True

    batch = 5000 if len(X) > 50_000 else len(X)
    n_batches = (len(X) + batch - 1) // batch
    rows_out_accum = []

    def _flush_to_files(rows):
        if not rows:
            return
        df_out = pd.DataFrame(rows)
        # Sanitize stringy fields
        for col in ('stock_id', 'vin', 'record_creation_date',
                    'feature_raw_key', 'feature_label', 'value', 'top_underlying'):
            if col in df_out.columns:
                df_out[col] = df_out[col].astype(object).where(df_out[col].notna(), None)
        table = pa.Table.from_pandas(df_out, preserve_index=False)
        nonlocal writer, csv_first
        if writer is None:
            writer = pq.ParquetWriter(pq_path, table.schema, compression='snappy')
        writer.write_table(table)
        if also_csv:
            df_out.to_csv(csv_path, mode='w' if csv_first else 'a',
                          header=csv_first, index=False)
            csv_first = False

    for bi in range(n_batches):
        lo, hi = bi * batch, min((bi + 1) * batch, len(X))
        X_chunk = X.iloc[lo:hi]
        shap_vals = explainer.shap_values(X_chunk)
        if not (hasattr(shap_vals, 'ndim') and shap_vals.ndim == 2):
            shap_vals = np.asarray(shap_vals)
        dollar_impact, pct_of_pred = _enrich_with_dollars(
            shap_vals, base_log_value, log_preds_all[lo:hi])

        # Build records per car and collapse to raw groups
        from shap_dollar_helper import shap_to_dollar_terms
        for i in range(hi - lo):
            row_idx = lo + i
            sv = shap_vals[i]
            log_pred_i = float(log_preds_all[row_idx])
            feat_vals_row = X_chunk.iloc[i].values
            # Compute full breakdown (k = all features, both directions)
            full = shap_to_dollar_terms(
                sv, base_log_value, log_pred_i,
                feature_names, feat_vals_row,
                k_pos=len(feature_names), k_neg=len(feature_names))
            all_recs = full['top_positive'] + full['top_negative']
            collapsed = collapse_engineered_to_raw(
                all_recs, request_dicts[row_idx],
                k_pos=len(feature_names), k_neg=len(feature_names),
                look_factor=1,
            )
            groups = collapsed['top_positive'] + collapsed['top_negative']
            # Per-row stats
            if groups:
                max_abs = max(abs(g['dollar_impact']) for g in groups) or 1.0
                # rank by abs dollar_impact
                ordered = sorted(groups, key=lambda g: abs(g['dollar_impact']), reverse=True)
                rank_map = {id(g): rnk + 1 for rnk, g in enumerate(ordered)}
            else:
                max_abs = 1.0
                rank_map = {}

            stock_id = raw_df['stock_id'].iloc[row_idx] if 'stock_id' in raw_df.columns else None
            vin      = raw_df['vin'].iloc[row_idx]      if 'vin' in raw_df.columns      else None
            rcd      = (str(raw_df['record_creation_date'].iloc[row_idx])
                        if 'record_creation_date' in raw_df.columns else None)

            for g in groups:
                rows_out_accum.append({
                    'stock_id':              stock_id,
                    'vin':                   vin,
                    'record_creation_date':  rcd,
                    'salevalue':             float(actuals[row_idx]),
                    'predicted_sale_value':  float(preds_p50[row_idx]),
                    'feature_raw_key':       g['feature_raw_key'],
                    'feature_label':         g['feature_label'],
                    'value':                 g.get('value'),
                    'dollar_impact':         float(g['dollar_impact']),
                    'pct_of_prediction':     float(g['pct_of_prediction']),
                    'pct_of_top_feature':    float(g['dollar_impact'] / max_abs * 100.0),
                    'rank_by_abs':           int(rank_map[id(g)]),
                    'n_underlying':          int(g['n_underlying']),
                    'top_underlying':        g['top_underlying'],
                })

        # Periodic flush
        if len(rows_out_accum) > 100_000 or bi == n_batches - 1:
            _flush_to_files(rows_out_accum)
            rows_out_accum = []
            gc.collect()
        if n_batches > 5 and (bi + 1) % max(n_batches // 10, 1) == 0:
            print(f"  Batch {bi+1}/{n_batches} done")

    _flush_to_files(rows_out_accum)
    if writer is not None:
        writer.close()
    print(f"[SHAP raw] Wrote {pq_path}")
    if also_csv:
        print(f"[SHAP raw] Wrote {csv_path}")


def compute_and_save_global_shap_importance(model, X, feature_names,
                                              out_dir, prefix,
                                              sample_size=None,
                                              random_state=0):
    """Compute global SHAP feature importance and write a small CSV.

    Aggregates SHAP across all rows in X for each feature:
      mean_abs_shap          = mean(|log_shap|)         — standard SHAP global importance (log space)
      mean_abs_dollar        = mean(|dollar_impact|)    — average $ magnitude per prediction
      mean_abs_pct_of_pred   = mean(|pct_of_prediction|) — average percentage impact

    Parameters
    ----------
    sample_size : int or None
        If set and X has more rows, a random sample of this size is used.
        Global importance converges quickly — 20K rows is usually within 1-2%
        of using the full set. Useful for very large datasets.
        None = use all rows.

    Writes:  {out_dir}/{prefix}_shap_global_importance.csv
    Columns: feature, mean_abs_shap, mean_abs_dollar, mean_abs_pct_of_pred,
             pct_of_top  (mean_abs_dollar / max(mean_abs_dollar) * 100),
             n_rows_used (so you know whether sampling was applied)
    Rows: one per feature, sorted descending by mean_abs_dollar.
    """
    import time
    t_start = time.time()
    n_total = len(X)

    # Subsample if requested and dataset is larger than sample_size
    if sample_size is not None and n_total > sample_size:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n_total, size=sample_size, replace=False)
        sample_idx.sort()   # preserve original order (faster pandas indexing)
        X_used = X.iloc[sample_idx] if hasattr(X, 'iloc') else X[sample_idx]
        print(f"\n[SHAP global] {prefix}: subsampled {sample_size:,} of {n_total:,} "
              f"rows × {len(feature_names)} features (random_state={random_state})")
    else:
        X_used = X
        print(f"\n[SHAP global] {prefix}: {n_total:,} rows × {len(feature_names)} features...")

    t0 = time.time()
    patched_model, explainer = _get_explainer(model)
    expected = explainer.expected_value
    base_log_value = float(expected if not hasattr(expected, '__len__') else expected[0])
    print(f"  Built explainer in {time.time() - t0:.1f}s")

    t0 = time.time()
    log_preds_all = patched_model.predict(X_used)
    final_dollars_all = np.clip(np.expm1(log_preds_all), 1.0, None)
    expm1_base = float(np.expm1(base_log_value))
    print(f"  Predicted {len(X_used):,} rows in {time.time() - t0:.1f}s")

    # Accumulate sums of absolute values; divide at the end.
    # Use float32 for intermediate to halve memory; float64 for the running sum
    # so we don't lose precision across millions of additions.
    n_feats = len(feature_names)
    sum_abs_shap   = np.zeros(n_feats, dtype=np.float64)
    sum_abs_dollar = np.zeros(n_feats, dtype=np.float64)
    sum_abs_pct    = np.zeros(n_feats, dtype=np.float64)

    # Pre-convert X_used to numpy once so SHAP doesn't redo the DataFrame -> numpy
    # conversion per batch
    X_np = X_used.values if hasattr(X_used, 'values') else np.asarray(X_used)
    # SHAP can take either a numpy array or a DataFrame; numpy is faster for
    # batch SHAP because it skips DataFrame metadata.

    # Batch sizing: SHAP TreeExplainer parallelizes well over rows but holds
    # the whole batch's shap_values in memory. Cap at 10k to keep memory bounded
    # AND to ensure we get progress prints on small datasets.
    batch = 10_000 if len(X_np) > 10_000 else len(X_np)
    n_batches = (len(X_np) + batch - 1) // batch

    t0 = time.time()
    for bi in range(n_batches):
        t_batch = time.time()
        lo, hi = bi * batch, min((bi + 1) * batch, len(X_np))
        shap_vals = explainer.shap_values(X_np[lo:hi])
        if not (hasattr(shap_vals, 'ndim') and shap_vals.ndim == 2):
            shap_vals = np.asarray(shap_vals)

        # Vectorized dollar impacts and percentages
        dollar = np.expm1(base_log_value + shap_vals) - expm1_base
        final_chunk = final_dollars_all[lo:hi][:, None]
        # Avoid the intermediate pct array — sum abs(dollar)/final*100 directly
        sum_abs_shap   += np.abs(shap_vals).sum(axis=0)
        sum_abs_dollar += np.abs(dollar).sum(axis=0)
        sum_abs_pct    += (np.abs(dollar) / final_chunk).sum(axis=0) * 100.0

        # Always print per-batch progress (with timing) — this avoids long
        # silent stretches and helps diagnose hangs.
        batch_secs = time.time() - t_batch
        elapsed    = time.time() - t0
        rate       = hi / elapsed if elapsed > 0 else 0
        eta        = (len(X_np) - hi) / rate if rate > 0 else 0
        print(f"  Batch {bi+1}/{n_batches}: rows {lo:,}-{hi:,} done in {batch_secs:.1f}s "
              f"(cumulative {elapsed:.0f}s, {rate:.0f} rows/s, ETA {eta:.0f}s)")
    print(f"  SHAP loop: {time.time() - t0:.1f}s total")

    n_rows = float(len(X_np))
    mean_abs_shap   = sum_abs_shap   / n_rows
    mean_abs_dollar = sum_abs_dollar / n_rows
    mean_abs_pct    = sum_abs_pct    / n_rows

    # pct_of_top is relative to the largest mean_abs_dollar across features
    top_dollar = float(mean_abs_dollar.max()) if mean_abs_dollar.max() > 0 else 1.0
    pct_of_top = (mean_abs_dollar / top_dollar) * 100.0

    df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap.astype(np.float64),
        'mean_abs_dollar': mean_abs_dollar.astype(np.float64),
        'mean_abs_pct_of_pred': mean_abs_pct.astype(np.float64),
        'pct_of_top': pct_of_top.astype(np.float64),
        'n_rows_used': int(len(X_np)),
        'n_rows_total': int(n_total),
    })
    df = df.sort_values('mean_abs_dollar', ascending=False).reset_index(drop=True)

    out_path = os.path.join(out_dir, f"{prefix}_shap_global_importance.csv")
    df.to_csv(out_path, index=False)
    sampled_note = ""
    if len(X_np) < n_total:
        sampled_note = f" [sampled {len(X_np):,}/{n_total:,}]"
    print(f"[SHAP global] Wrote {out_path}  ({len(df)} features, "
          f"top: {df.iloc[0]['feature']} @ ${df.iloc[0]['mean_abs_dollar']:.2f}){sampled_note}")
    print(f"[SHAP global] {prefix} total wall time: {time.time() - t_start:.1f}s")
    return df



    """Convenience: compute and save both engineered + raw SHAP for train AND test."""
    # Build request_dicts (raw column dictionaries) for the raw collapse function
    print("\n=== SHAP EXPORT ===")
    print("Building per-row raw request dicts...")
    train_req = [r._asdict() if hasattr(r, '_asdict') else r.to_dict()
                  for _, r in train_raw.iterrows()]
    test_req  = [r._asdict() if hasattr(r, '_asdict') else r.to_dict()
                  for _, r in test_raw.iterrows()]

    # Engineered first (faster — pure batched SHAP)
    compute_and_save_shap_engineered(
        model, train_X, train_raw, feature_names,
        train_actuals, train_preds_p50, out_dir, prefix="train", also_csv=also_csv)
    compute_and_save_shap_engineered(
        model, test_X, test_raw, feature_names,
        test_actuals, test_preds_p50, out_dir, prefix="test", also_csv=also_csv)

    # Raw collapse (per-row Python loop; slower)
    compute_and_save_shap_raw(
        model, train_X, train_raw, feature_names,
        train_actuals, train_preds_p50, train_req, out_dir, prefix="train",
        also_csv=also_csv)
    compute_and_save_shap_raw(
        model, test_X, test_raw, feature_names,
        test_actuals, test_preds_p50, test_req, out_dir, prefix="test",
        also_csv=also_csv)

    # Global SHAP importance per feature (small CSVs, one per set)
    compute_and_save_global_shap_importance(model, train_X, feature_names,
                                              out_dir, prefix="train")
    compute_and_save_global_shap_importance(model, test_X, feature_names,
                                              out_dir, prefix="test")
