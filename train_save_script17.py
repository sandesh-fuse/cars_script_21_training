"""
train_save_script17.py
=======================
Script 17 quick-blend with quantile prediction:
  - 5th, 50th, 95th percentile models for each of TE and no-TE bases
  - 6 models total, blended with 0.66 TE + 0.34 no-TE per quantile
  - Optional Optuna retuning per quantile
  - Optional GPU acceleration

USAGE:
    Edit DATA_PATH below, then:
        python train_save_script17.py                        # CPU, default params (fast)
        python train_save_script17.py --gpu                  # GPU, default params
        python train_save_script17.py --retune --n_trials 50 # CPU + Optuna per quantile (~6-9 hr)
        python train_save_script17.py --retune --gpu         # GPU + Optuna (~1-2 hr if GPU works)

OUTPUTS (./artifacts/script17/):
    preprocessor_te.joblib, preprocessor_no_te.joblib
    model_te_{q05,q50,q95}.json, model_no_te_{q05,q50,q95}.json
    blend_weights.json
    cult_lookup.joblib, zip_lat_map.joblib, zip_lon_map.joblib
    sample_test_rows.parquet
    training_metadata.json (best params, MAE/RMSLE, empirical coverage, SHAP sanity check)
"""

import os
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from preprocessor import (
    SaleValuePreprocessor, MONO_FEATURES,
    TARGET_COL, TIME_COL,
    build_cult_lookup, build_zip_lookup,
)
from shap_dollar_helper import sanity_check_shap
from _save_predictions_helper import build_predictions_frame, save_predictions
from evaluate_predictions import evaluate

# ============================================================
# CONFIG — edit DATA_PATH and CULT_PATH for your environment
# ============================================================
DATA_PATH        = "../../data/May29_combined_dataone_cars_data_with_specialty_item_removed_with_zip_and_salevalue_2026.csv"
CULT_PATH        = "../../data/cult_cars.xlsx"
ARTIFACTS_DIR    = "./artifacts/script17"
TRAIN_CUTOFF     = "2018-01-01"
SEED             = 42
QUANTILES        = [0.05, 0.50, 0.95]
QUANTILE_LABELS  = ["q05", "q50", "q95"]
BLEND_WEIGHTS    = {"w_te": 0.66, "w_no_te": 0.34}

# Default params per quantile (used when --retune is NOT set)
DEFAULT_PARAMS = {
    "q05": dict(learning_rate=0.03, max_depth=8, min_child_weight=20,
                subsample=0.8, colsample_bytree=0.7,
                reg_lambda=2.0, reg_alpha=1.0, gamma=0.1),
    "q50": dict(learning_rate=0.02, max_depth=10, min_child_weight=5,
                subsample=0.7, colsample_bytree=0.5,
                reg_lambda=2.0, reg_alpha=2.0, gamma=0.0),
    "q95": dict(learning_rate=0.03, max_depth=8, min_child_weight=20,
                subsample=0.8, colsample_bytree=0.7,
                reg_lambda=2.0, reg_alpha=1.0, gamma=0.1),
}

# Module-level XGB kwargs (gets overwritten by main() based on --gpu)
# Always uses tree_method='hist'. Adds device='cuda' if --gpu.
XGB_KWARGS_GLOBAL = {'tree_method': 'hist'}


# ============================================================
# METRICS
# ============================================================
def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def rmsle(y, p):
    p = np.clip(p, 1, None)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))


def fix_quantile_crossing(p5, p50, p95):
    stacked = np.stack([p5, p50, p95], axis=1)
    sorted_arr = np.sort(stacked, axis=1)
    return sorted_arr[:, 0], sorted_arr[:, 1], sorted_arr[:, 2]


def coverage(y, p_low, p_high):
    return float(np.mean((y >= p_low) & (y <= p_high)))


def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


# ============================================================
# GPU PROBE
# ============================================================
def probe_gpu():
    """Quick smoke-test that XGBoost can actually train on GPU.
       Raises a clear error if not."""
    print("Probing GPU availability for XGBoost...")
    X = np.random.rand(500, 5)
    y = np.random.rand(500)
    try:
        m = XGBRegressor(n_estimators=5, tree_method='hist', device='cuda',
                         random_state=0, verbosity=0)
        m.fit(X, y)
        _ = m.predict(X)
        print("  GPU probe OK (XGBoost can train on device='cuda').")
    except Exception as e:
        raise RuntimeError(
            "XGBoost GPU probe failed. Common causes:\n"
            "  - XGBoost not compiled with CUDA support\n"
            "  - NVIDIA driver missing or incompatible (check `nvidia-smi`)\n"
            "  - GPU is busy or out of memory\n"
            f"Underlying error: {e}\n\n"
            "Rerun without --gpu to use CPU."
        ) from e


# ============================================================
# OPTUNA — per quantile
# ============================================================
def tune_quantile(name, X_tr, y_tr, X_va, y_va, monotone, quantile_alpha, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators'    : 2000,
            'learning_rate'   : trial.suggest_categorical('learning_rate', [0.02, 0.03, 0.05]),
            'max_depth'       : trial.suggest_int('max_depth', 5, 10),
            'min_child_weight': trial.suggest_categorical('min_child_weight', [5, 10, 20, 50, 100]),
            'subsample'       : trial.suggest_categorical('subsample', [0.7, 0.8, 0.9]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.7, 0.9]),
            'reg_lambda'      : trial.suggest_categorical('reg_lambda', [1.0, 2.0, 5.0, 10.0]),
            'reg_alpha'       : trial.suggest_categorical('reg_alpha', [0.0, 0.5, 1.0, 2.0]),
            'gamma'           : trial.suggest_categorical('gamma', [0.0, 0.1, 0.5]),
        }
        m = XGBRegressor(
            objective='reg:quantileerror', quantile_alpha=quantile_alpha,
            monotone_constraints=monotone,
            early_stopping_rounds=50,
            random_state=SEED, n_jobs=-1,
            **XGB_KWARGS_GLOBAL,
            **params,
        )
        m.fit(X_tr, np.log1p(y_tr),
              eval_set=[(X_va, np.log1p(y_va))], verbose=False)
        pred_va = np.expm1(m.predict(X_va))
        return pinball_loss(y_va, pred_va, quantile_alpha)

    print(f"  [{name}] Optuna: {n_trials} trials for quantile_alpha={quantile_alpha}")
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  [{name}] Best pinball: {study.best_value:.4f}")
    print(f"  [{name}] Best params:  {study.best_params}")
    return study.best_params


# ============================================================
# TRAIN ONE BASE (TE or no-TE), 3 quantiles
# ============================================================
def train_base_with_quantiles(name, train_raw, with_target_encoding,
                                zip_lat_map, zip_lon_map, cult_lookup,
                                test_raw, args):
    print(f"\n[{name}] Building preprocessor and feature matrix...")
    pre = SaleValuePreprocessor(
        time_col=TIME_COL, seed=SEED,
        with_target_encoding=with_target_encoding,
        zip_lat_map=zip_lat_map, zip_lon_map=zip_lon_map, cult_lookup=cult_lookup,
    )
    y_train = train_raw[TARGET_COL].values
    X_train = pre.fit_transform_with_oof(train_raw, y_train)
    X_test  = pre.transform(test_raw)
    monotone = tuple(MONO_FEATURES.get(c, 0) for c in pre.feature_cols_)
    print(f"[{name}] Features: {len(pre.feature_cols_)}")

    n_es = int(len(X_train) * 0.9)
    X_es_tr, X_es_va = X_train.iloc[:n_es], X_train.iloc[n_es:]
    y_es_tr, y_es_va = y_train[:n_es], y_train[n_es:]

    models = {}
    preds_test = {}
    best_params_by_q = {}

    for q, qlab in zip(QUANTILES, QUANTILE_LABELS):
        t0 = time.time()
        print(f"\n[{name}] Training quantile {q} ({qlab})...")
        if args.retune:
            params = tune_quantile(
                f"{name}-{qlab}", X_es_tr, y_es_tr, X_es_va, y_es_va,
                monotone, q, args.n_trials)
        else:
            params = DEFAULT_PARAMS[qlab].copy()
            print(f"  Using default params: {params}")
        best_params_by_q[qlab] = params

        model = XGBRegressor(
            objective='reg:quantileerror', quantile_alpha=q,
            n_estimators=3000,
            monotone_constraints=monotone,
            early_stopping_rounds=75,
            random_state=SEED, n_jobs=-1,
            **XGB_KWARGS_GLOBAL,
            **params,
        )
        model.fit(X_es_tr, np.log1p(y_es_tr),
                  eval_set=[(X_es_va, np.log1p(y_es_va))], verbose=False)
        pred_test = np.clip(np.expm1(model.predict(X_test)), 1, None)
        models[qlab] = model
        preds_test[qlab] = pred_test
        print(f"  [{name}-{qlab}] best_iter={model.best_iteration}  "
              f"({(time.time()-t0)/60:.1f} min)")

    return pre, models, preds_test, best_params_by_q


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--cult", default=CULT_PATH)
    parser.add_argument("--out",  default=ARTIFACTS_DIR)
    parser.add_argument("--retune", action='store_true',
                        help="Run Optuna for each quantile (slow but production-grade)")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Optuna trials per quantile when --retune is set")
    parser.add_argument("--gpu", action='store_true',
                        help="Use GPU (CUDA) for training. Requires XGBoost compiled with CUDA support.")
    parser.add_argument("--cap-pct", type=float, default=99.5,
                        help="Drop rows above this salevalue percentile (default 99.5). "
                             "Use 100 to disable capping entirely. "
                             "Common values: 99.5 (drop top 0.5%%), 99.9 (drop top 0.1%%), 100 (keep all).")
    parser.add_argument("--min-salevalue", type=float, default=0.0,
                        help="Drop rows where salevalue <= this dollar amount (default 0, "
                             "which keeps current behavior: salevalue > 0). "
                             "Examples: 50 (drop rows below $50), 100 (drop scrap-priced cars).")
    parser.add_argument("--save-shap", action='store_true',
                        help="Compute and save per-row SHAP attributions for every train + "
                             "test row against the p50 model. Writes 4 large files: "
                             "train_shap_engineered, train_shap_raw, test_shap_engineered, "
                             "test_shap_raw (each as parquet AND csv). Also writes the "
                             "global importance summary (same as --save-shap-global). "
                             "Expensive: ~45-60 min CPU for full train, multi-GB output.")
    parser.add_argument("--save-shap-global", action='store_true',
                        help="Compute and save GLOBAL SHAP feature importance only (small CSV, "
                             "one row per feature, sorted by mean absolute dollar impact). "
                             "Cheap: a few minutes on CPU, ~5 KB per file. Implied by "
                             "--save-shap.")
    parser.add_argument("--shap-parquet-only", action='store_true',
                        help="When --save-shap is set, write only parquet for per-row files "
                             "(skip the multi-GB CSV). Recommended for large train sets. "
                             "Does not affect --save-shap-global (always CSV; it's tiny).")
    parser.add_argument("--shap-sample", type=int, default=20_000,
                        help="For --save-shap-global only: subsample N rows for the global "
                             "importance computation. Default 20000 — gives essentially the "
                             "same feature ranking as the full set in a fraction of the time. "
                             "Use 0 to disable sampling (compute on all rows; slow for >100k). "
                             "Ignored when computing per-row SHAP (--save-shap), which always "
                             "uses every row.")
    args = parser.parse_args()

    if not (0 < args.cap_pct <= 100):
        parser.error("--cap-pct must be in (0, 100]")
    if args.min_salevalue < 0:
        parser.error("--min-salevalue must be >= 0")

    # Configure global XGB kwargs based on --gpu
    global XGB_KWARGS_GLOBAL
    if args.gpu:
        probe_gpu()
        XGB_KWARGS_GLOBAL = {'tree_method': 'hist', 'device': 'cuda'}
        print(">>> GPU mode enabled (XGB kwargs: tree_method='hist', device='cuda')")
    else:
        XGB_KWARGS_GLOBAL = {'tree_method': 'hist'}
        print(">>> CPU mode (XGB kwargs: tree_method='hist')")

    os.makedirs(args.out, exist_ok=True)
    start_t = time.time()

    print(f"\nLoading training data from {args.data}...")
    df = pd.read_csv(args.data, low_memory=False)
    #df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format='mixed', errors='coerce')
    df = df.dropna(subset=[TARGET_COL, TIME_COL])
    pre_floor_n = len(df)
    df = df[df[TARGET_COL] > args.min_salevalue]
    floor_dropped = pre_floor_n - len(df)
    if args.min_salevalue == 0.0:
        print(f"Target-value floor: salevalue > $0 (default; dropped {floor_dropped:,} rows)")
    else:
        print(f"Target-value floor: salevalue > ${args.min_salevalue:,.2f}  "
              f"(dropped {floor_dropped:,} rows of {pre_floor_n:,})")

    pre_cap_n = len(df)
    if args.cap_pct >= 100.0:
        cap_value = None
        print(f"Target-value cap: disabled (--cap-pct {args.cap_pct})")
    else:
        cap_value = float(df[TARGET_COL].quantile(args.cap_pct / 100.0))
        df = df[df[TARGET_COL] <= cap_value]
        cap_dropped = pre_cap_n - len(df)
        print(f"Target-value cap: {args.cap_pct}th percentile = ${cap_value:,.0f}  "
              f"(dropped {cap_dropped:,} rows of {pre_cap_n:,})")

    # Specialty items (RVs, boats, heavy equipment, etc.) behave completely
    # differently price-wise than standard cars and shouldn't be mixed into
    # this model's training data. This CSV's filename already assumes
    # specialty items are excluded upstream; this makes it an explicit,
    # auditable guarantee in code rather than trusting the input file.
    # Robust string-based coercion (not a bare `== True`) since the raw
    # column may arrive as an actual bool, a 'True'/'False' string, or
    # blank. Blank/NaN is treated as NOT specialty (kept) -- only rows
    # explicitly flagged true are dropped. No-op if the column isn't
    # present in this CSV.
    if 'Specialty Item' in df.columns:
        pre_specialty_n = len(df)
        is_specialty = (df['Specialty Item'].astype(str).str.strip().str.lower()
                          .isin(['true', '1', 'yes', 't']))
        df = df[~is_specialty]
        print(f"Specialty-item filter: dropped {pre_specialty_n - len(df):,} rows "
              f"flagged as Specialty Item (of {pre_specialty_n:,})")

    print(f"After filter: {df.shape}")

    print(f"Loading cult cars from {args.cult}...")
    cult_raw = pd.read_excel(args.cult, sheet_name='Cult Vehicles')
    CULT_LOOKUP = build_cult_lookup(cult_raw)

    print("Building ZIP lookup...")
    LAT_MAP, LON_MAP = build_zip_lookup(df['vazipcode'])

    test_cutoff = df[TIME_COL].max() - pd.DateOffset(months=3)
    train_cutoff_ts = pd.Timestamp(TRAIN_CUTOFF)
    train_raw = df[(df[TIME_COL] >= train_cutoff_ts) & (df[TIME_COL] < test_cutoff)].copy().reset_index(drop=True)
    test_raw  = df[df[TIME_COL] >= test_cutoff].copy().reset_index(drop=True)
    print(f"Train: {len(train_raw):,}  Test: {len(test_raw):,}")
    y_test = test_raw[TARGET_COL].values

    # Train both bases
    print("\n" + "#" * 70 + "\n# TE BASE (3 quantiles)\n" + "#" * 70)
    pre_te, models_te, preds_te, params_te = train_base_with_quantiles(
        "TE", train_raw, with_target_encoding=True,
        zip_lat_map=LAT_MAP, zip_lon_map=LON_MAP, cult_lookup=CULT_LOOKUP,
        test_raw=test_raw, args=args)

    print("\n" + "#" * 70 + "\n# no-TE BASE (3 quantiles)\n" + "#" * 70)
    pre_no_te, models_no_te, preds_no_te, params_no_te = train_base_with_quantiles(
        "no-TE", train_raw, with_target_encoding=False,
        zip_lat_map=LAT_MAP, zip_lon_map=LON_MAP, cult_lookup=CULT_LOOKUP,
        test_raw=test_raw, args=args)

    # Blend per quantile
    w_te, w_no_te = BLEND_WEIGHTS["w_te"], BLEND_WEIGHTS["w_no_te"]
    blend = {}
    for qlab in QUANTILE_LABELS:
        blend[qlab] = w_te * preds_te[qlab] + w_no_te * preds_no_te[qlab]

    # Fix quantile crossing
    blend["q05"], blend["q50"], blend["q95"] = fix_quantile_crossing(
        blend["q05"], blend["q50"], blend["q95"])

    # ----- Evaluation -----
    print("\n" + "=" * 70)
    print("FINAL BLENDED QUANTILE PREDICTIONS")
    print("=" * 70)
    cov_90 = coverage(y_test, blend["q05"], blend["q95"])
    print(f"\nEmpirical coverage (90% CI):     {cov_90*100:.1f}%   (target: 90%)")
    p50_arr = blend["q50"]
    print(f"Point (median) MAE:               ${mae(y_test, p50_arr):.2f}")
    print(f"Point (median) RMSE:              ${rmse(y_test, p50_arr):.2f}")
    print(f"Point (median) RMSLE:             {rmsle(y_test, p50_arr):.4f}")
    print(f"\nMedian width of 90% CI:           ${np.median(blend['q95']-blend['q05']):.0f}")
    print(f"Mean width of 90% CI:             ${np.mean(blend['q95']-blend['q05']):.0f}")

    tier_edges = [0, 200, 500, 1000, 2500, np.inf]
    tier_labels = ['$0-200','$200-500','$500-1K','$1K-2.5K','$2.5K+']
    tiers = pd.cut(y_test, bins=tier_edges, labels=tier_labels)
    tier_metrics = {}
    print(f"\n{'Tier':<10} {'N':>6} {'MAE_p50':>10} {'RMSLE_p50':>10} {'Cov_90':>8} {'Width':>10}")
    for t in tier_labels:
        m = (tiers == t)
        if m.sum() == 0: continue
        yt, p5t, p50t, p95t = y_test[m], blend["q05"][m], blend["q50"][m], blend["q95"][m]
        tier_metrics[t] = {
            'N': int(m.sum()),
            'MAE_p50': mae(yt, p50t),
            'RMSE_p50': rmse(yt, p50t),
            'RMSLE_p50': rmsle(yt, p50t),
            'coverage_90': coverage(yt, p5t, p95t),
            'mean_width': float(np.mean(p95t - p5t)),
        }
        print(f"{t:<10} {m.sum():>6} ${mae(yt,p50t):>8.0f} {rmsle(yt,p50t):>10.4f} "
              f"{coverage(yt,p5t,p95t)*100:>7.1f}% ${np.mean(p95t-p5t):>8.0f}")

    # ----- SHAP sanity check (median models) -----
    print("\n--- SHAP sanity check (median models) ---")
    X_test_te    = pre_te.transform(test_raw.head(5))
    X_test_no_te = pre_no_te.transform(test_raw.head(5))
    shap_check_te    = sanity_check_shap(models_te["q50"],    X_test_te,    pre_te.feature_cols_,    label="TE-q50")
    shap_check_no_te = sanity_check_shap(models_no_te["q50"], X_test_no_te, pre_no_te.feature_cols_, label="no-TE-q50")

    # ----- Save artifacts -----
    print(f"\nSaving artifacts to {args.out}/ ...")
    joblib.dump(pre_te,    os.path.join(args.out, "preprocessor_te.joblib"))
    joblib.dump(pre_no_te, os.path.join(args.out, "preprocessor_no_te.joblib"))
    for qlab in QUANTILE_LABELS:
        models_te[qlab].save_model(os.path.join(args.out, f"model_te_{qlab}.json"))
        models_no_te[qlab].save_model(os.path.join(args.out, f"model_no_te_{qlab}.json"))
    joblib.dump(CULT_LOOKUP, os.path.join(args.out, "cult_lookup.joblib"))
    joblib.dump(LAT_MAP,     os.path.join(args.out, "zip_lat_map.joblib"))
    joblib.dump(LON_MAP,     os.path.join(args.out, "zip_lon_map.joblib"))
    with open(os.path.join(args.out, "blend_weights.json"), 'w') as f:
        json.dump(BLEND_WEIGHTS, f, indent=2)

    # ----- Save FULL train and test predictions for analysis -----
    print(f"\n--- Computing train predictions (full set) for save ---")
    # Re-transform train through both preprocessors (TE used its OOF target encoding
    # at fit time; transform here uses the holdout target maps — slight drift OK)
    X_train_te    = pre_te.transform(train_raw)
    X_train_no_te = pre_no_te.transform(train_raw)
    train_blend = {}
    for qlab in QUANTILE_LABELS:
        p_te    = np.clip(np.expm1(models_te[qlab].predict(X_train_te)),    1, None)
        p_no_te = np.clip(np.expm1(models_no_te[qlab].predict(X_train_no_te)), 1, None)
        train_blend[qlab] = w_te * p_te + w_no_te * p_no_te
    train_blend["q05"], train_blend["q50"], train_blend["q95"] = fix_quantile_crossing(
        train_blend["q05"], train_blend["q50"], train_blend["q95"])

    train_pred_df = build_predictions_frame(
        raw_df=train_raw,
        actuals=train_raw[TARGET_COL].values,
        preds_p5=train_blend["q05"],
        preds_p50=train_blend["q50"],
        preds_p95=train_blend["q95"],
    )
    test_pred_df = build_predictions_frame(
        raw_df=test_raw,
        actuals=y_test,
        preds_p5=blend["q05"],
        preds_p50=blend["q50"],
        preds_p95=blend["q95"],
    )
    print("\nSaving prediction files...")
    save_predictions(train_pred_df, args.out, "train_predictions")
    save_predictions(test_pred_df,  args.out, "test_predictions")
    # Reminder: train predictions are produced by a model that saw those rows during
    # training; they will look better than test. Use for sanity checks, not evaluation.

    # Evaluate and save per-tier metrics JSON for each
    evaluate(train_pred_df, label="script17_train",
             save_json_to=os.path.join(args.out, "train_metrics.json"), verbose=True)
    evaluate(test_pred_df,  label="script17_test",
             save_json_to=os.path.join(args.out, "test_metrics.json"),  verbose=True)

    # ----- Optional SHAP exports -----
    want_per_row = args.save_shap
    want_global  = args.save_shap or args.save_shap_global
    if want_per_row or want_global:
        from _shap_export_helper import (
            compute_and_save_shap_engineered,
            compute_and_save_shap_raw,
            compute_and_save_global_shap_importance,
        )
        print("\n" + "=" * 60)
        if want_per_row:
            print("SHAP EXPORT — per-row + global (--save-shap)")
        else:
            print("SHAP EXPORT — global only (--save-shap-global)")
        print("=" * 60)

        X_test_te_full = pre_te.transform(test_raw)

        if want_per_row:
            also_csv = not args.shap_parquet_only
            if not also_csv:
                print("Note: --shap-parquet-only set; per-row CSV files will be skipped.")
            else:
                n_rows_train = len(train_raw) * len(pre_te.feature_cols_)
                est_csv_gb = n_rows_train * 200 / 1e9
                print(f"Warning: engineered train CSV will be ~{est_csv_gb:.1f} GB "
                      f"({len(train_raw):,} rows x {len(pre_te.feature_cols_)} features). "
                      f"Use --shap-parquet-only to skip CSV.")

            # Per-row engineered
            compute_and_save_shap_engineered(
                models_te["q50"], X_train_te, train_raw, pre_te.feature_cols_,
                train_raw[TARGET_COL].values, train_blend["q50"],
                args.out, prefix="train", also_csv=also_csv)
            compute_and_save_shap_engineered(
                models_te["q50"], X_test_te_full, test_raw, pre_te.feature_cols_,
                y_test, blend["q50"],
                args.out, prefix="test", also_csv=also_csv)

            # Per-row raw (collapsed to user-facing features)
            train_req = [r.to_dict() for _, r in train_raw.iterrows()]
            test_req  = [r.to_dict() for _, r in test_raw.iterrows()]
            compute_and_save_shap_raw(
                models_te["q50"], X_train_te, train_raw, pre_te.feature_cols_,
                train_raw[TARGET_COL].values, train_blend["q50"], train_req,
                args.out, prefix="train", also_csv=also_csv)
            compute_and_save_shap_raw(
                models_te["q50"], X_test_te_full, test_raw, pre_te.feature_cols_,
                y_test, blend["q50"], test_req,
                args.out, prefix="test", also_csv=also_csv)

        # Global importance — runs for either flag (cheap)
        shap_sample = args.shap_sample if args.shap_sample > 0 else None
        compute_and_save_global_shap_importance(
            models_te["q50"], X_train_te, pre_te.feature_cols_,
            args.out, prefix="train", sample_size=shap_sample)
        compute_and_save_global_shap_importance(
            models_te["q50"], X_test_te_full, pre_te.feature_cols_,
            args.out, prefix="test", sample_size=shap_sample)

    # ----- Save sample low-MAE test rows (used by tests/test_examples.py) -----
    sample_test = test_raw.copy()
    sample_test['actual']        = y_test
    sample_test['p5']            = blend["q05"]
    sample_test['p50']           = blend["q50"]
    sample_test['p95']           = blend["q95"]
    sample_test['ci_width']      = blend["q95"] - blend["q05"]
    sample_test['abs_error_p50'] = np.abs(sample_test['p50'] - sample_test['actual'])
    sample_test = sample_test.sort_values('abs_error_p50').head(50)
    sample_test.to_parquet(os.path.join(args.out, "sample_test_rows.parquet"))
    with open(os.path.join(args.out, "training_metadata.json"), 'w') as f:
        json.dump({
            'model_type': 'script17_quickblend_quantile',
            'quantiles': QUANTILES,
            'blend_weights': BLEND_WEIGHTS,
            'best_params_te': params_te,
            'best_params_no_te': params_no_te,
            'retune_used': bool(args.retune),
            'gpu_used': bool(args.gpu),
            'cap_pct': args.cap_pct,
            'cap_value': cap_value,
            'min_salevalue': args.min_salevalue,
            'save_shap_used': bool(args.save_shap),
            'save_shap_global_used': bool(args.save_shap or args.save_shap_global),
            'n_optuna_trials_per_quantile': args.n_trials if args.retune else 0,
            'train_size': len(train_raw),
            'test_size': len(test_raw),
            'n_features_te': len(pre_te.feature_cols_),
            'n_features_no_te': len(pre_no_te.feature_cols_),
            'test_overall': {
                'MAE_p50': mae(y_test, blend["q50"]),
                'RMSE_p50': rmse(y_test, blend["q50"]),
                'RMSLE_p50': rmsle(y_test, blend["q50"]),
                'coverage_90': cov_90,
                'mean_ci_width': float(np.mean(blend["q95"] - blend["q05"])),
            },
            'test_by_tier': tier_metrics,
            'total_runtime_minutes': (time.time() - start_t) / 60,
            'shap_sanity_check': {
                'te_median':    shap_check_te,
                'no_te_median': shap_check_no_te,
            },
        }, f, indent=2)

    print(f"\nDone. Total runtime: {(time.time()-start_t)/60:.1f} min")
    print(f"Artifacts written to {args.out}/")


if __name__ == "__main__":
    main()
