"""
train_save_script21_ridge_q50.py
=================================
Experimental variant of train_save_script21.py (branched from the "515 MAE"
checkpoint, commit 2406a7a) that replaces ONLY the q50 (median) model with
an sklearn Ridge regression (wrapped in a SimpleImputer -> StandardScaler ->
Ridge pipeline, since SaleValuePreprocessor deliberately leaves NaNs in some
numeric columns for XGBoost to route natively, which sklearn's Ridge/
StandardScaler cannot tolerate). q05/q95 stay exactly as the checkpoint
(reg:quantileerror XGBoost), since they drive the 90% CI / coverage logic.
Target transform (log1p/expm1) and CPI deflation are unchanged. See
Baseline1.txt for the checkpoint's $516.73 test MAE to compare against.

Ridge is fit on 100% of each route's training rows (not the 90% early-
stopping split used by q05/q95) since it has no early-stopping mechanism
to spend a held-out slice on; no test leakage either way. See
'q50_ridge_train_fraction' in training_metadata.json.

NOTE: --save-shap / --save-shap-global are disabled in this variant, and
the SHAP sanity check is skipped for q50. _shap_export_helper.py calls
model.save_model() unconditionally (AttributeError on a Pipeline) and
shap.TreeExplainer() only supports tree ensembles, not a linear model.
q50 is the only model SHAP is ever run against in this file, so skipping
q50 SHAP is equivalent to disabling SHAP entirely for this variant.

USAGE:
    Edit DATA_PATH below, then:
        python train_save_script21_ridge_q50.py                        # CPU, defaults (fast)
        python train_save_script21_ridge_q50.py --gpu                  # GPU (only affects q05/q95)
        python train_save_script21_ridge_q50.py --ridge-alpha 5.0      # override Ridge regularization
        python train_save_script21_ridge_q50.py --retune --n_trials 50 # CPU + Optuna for q05/q95 only

OUTPUTS (./artifacts/script21_ridge_q50/):
    preprocessor_cult.joblib, preprocessor_standard.joblib
    model_cult_{q05,q95}.json, model_standard_{q05,q95}.json          (XGBoost)
    model_cult_q50_ridge.joblib, model_standard_q50_ridge.joblib       (sklearn Pipeline)
    cult_lookup.joblib, zip_lat_map.joblib, zip_lon_map.joblib
    alphas.json
    sample_test_rows.parquet
    training_metadata.json
"""

import os
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from preprocessor import (
    SaleValuePreprocessor, MONO_FEATURES,
    TARGET_COL, TIME_COL,
    build_cult_lookup, build_zip_lookup, compute_cult_flag,
    cpi_ratio_arr, adjust_target, deflate_pred,
)
from schema_adapter import map_raw_features_to_legacy, filter_to_known_columns
# NOTE: shap_dollar_helper.sanity_check_shap is intentionally NOT imported/used
# in this variant — see the SHAP-skip comment in main() for why.
from _save_predictions_helper import build_predictions_frame, save_predictions
from evaluate_predictions import evaluate

# ============================================================
# CONFIG
# ============================================================
DATA_PATH        = "../../data/June10_donation_cars_sales_nhtsa_enriched_remove_specialty_data.csv"
CULT_PATH        = "../../data/cult_cars.xlsx"
ARTIFACTS_DIR    = "./artifacts/script21_ridge_q50"
TRAIN_CUTOFF     = "2018-01-01"
RIDGE_ALPHA      = 1.0   # L2 regularization strength for the q50 Ridge model; override via --ridge-alpha
SEED             = 42
QUANTILES        = [0.05, 0.50, 0.95]
QUANTILE_LABELS  = ["q05", "q50", "q95"]

CULT_CONFIG = {
    'alpha': 0.11180936172340833,
    'use_macro': True, 'use_geo': True, 'use_cult': True,
}
STANDARD_CONFIG = {
    'alpha': 0.02390014731569895,
    'use_macro': True, 'use_geo': True, 'use_cult': False,
}

DEFAULT_PARAMS_CULT = {
    "q05": dict(learning_rate=0.03, max_depth=8, min_child_weight=20,
                subsample=0.7, colsample_bytree=0.7,
                reg_lambda=2.0, reg_alpha=1.0, gamma=0.1),
    "q50": dict(learning_rate=0.02, max_depth=10, min_child_weight=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_lambda=1.0, reg_alpha=0.5, gamma=0.0),
    "q95": dict(learning_rate=0.03, max_depth=8, min_child_weight=20,
                subsample=0.7, colsample_bytree=0.7,
                reg_lambda=2.0, reg_alpha=1.0, gamma=0.1),
}
DEFAULT_PARAMS_STANDARD = {
    "q05": dict(learning_rate=0.03, max_depth=7, min_child_weight=20,
                subsample=0.8, colsample_bytree=0.5,
                reg_lambda=5.0, reg_alpha=1.0, gamma=0.3),
    "q50": dict(learning_rate=0.02, max_depth=9, min_child_weight=10,
                subsample=0.8, colsample_bytree=0.5,
                reg_lambda=1.0, reg_alpha=0.5, gamma=0.5),
    "q95": dict(learning_rate=0.03, max_depth=7, min_child_weight=20,
                subsample=0.8, colsample_bytree=0.5,
                reg_lambda=5.0, reg_alpha=1.0, gamma=0.3),
}

# Module-level XGB kwargs (overwritten by main() based on --gpu)
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
def tune_quantile(name, X_tr, y_tr_adj, X_va, y_va_adj, monotone, quantile_alpha, n_trials):
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
        m.fit(X_tr, np.log1p(y_tr_adj),
              eval_set=[(X_va, np.log1p(y_va_adj))], verbose=False)
        pred_va = np.expm1(m.predict(X_va))
        return pinball_loss(y_va_adj, pred_va, quantile_alpha)

    print(f"  [{name}] Optuna: {n_trials} trials for quantile={quantile_alpha}")
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  [{name}] Best pinball: {study.best_value:.4f}  params: {study.best_params}")
    return study.best_params


# ============================================================
# TRAIN A SUBSET MODEL (CULT or STANDARD) WITH 3 QUANTILES
# ============================================================
def train_subset_with_quantiles(name, config, default_params, train_subset, test_subset,
                                  zip_lat_map, zip_lon_map, cult_lookup, args):
    print(f"\n[{name}] Preprocessing (train_n={len(train_subset):,}, test_n={len(test_subset):,})...")
    pre = SaleValuePreprocessor(
        time_col=TIME_COL, seed=SEED,
        use_macro=config['use_macro'], use_geo=config['use_geo'], use_cult=config['use_cult'],
        with_target_encoding=False,
        zip_lat_map=zip_lat_map, zip_lon_map=zip_lon_map, cult_lookup=cult_lookup,
    )

    R_train = cpi_ratio_arr(train_subset)
    y_train = train_subset[TARGET_COL].values

    X_train = pre.fit(train_subset).transform(train_subset)
    X_test  = pre.transform(test_subset.drop(columns=[TARGET_COL], errors='ignore'))

    monotone = tuple(MONO_FEATURES.get(c, 0) for c in pre.feature_cols_)
    y_train_adj = adjust_target(y_train, R_train, config['alpha'])

    n_es = int(len(X_train) * 0.9)
    X_es_tr, X_es_va = X_train.iloc[:n_es], X_train.iloc[n_es:]
    y_adj_tr, y_adj_va = y_train_adj[:n_es], y_train_adj[n_es:]

    R_test = cpi_ratio_arr(test_subset)
    models = {}
    preds_test = {}
    best_params_by_q = {}

    for q, qlab in zip(QUANTILES, QUANTILE_LABELS):
        t0 = time.time()
        print(f"\n[{name}] Training quantile {q} ({qlab})...")

        if qlab == "q50":
            # Experiment: Ridge regression instead of XGBoost quantile
            # regression for the median model. No Optuna search space exists
            # for Ridge here, so --retune is a no-op for q50 (use
            # --ridge-alpha instead).
            if args.retune:
                print(f"  Note: --retune has no effect on q50 in this variant "
                      f"(Ridge has no XGBoost hyperparameters to tune); "
                      f"use --ridge-alpha to tune it instead.")
            params = {'ridge_alpha': args.ridge_alpha}
            best_params_by_q[qlab] = params

            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=args.ridge_alpha, random_state=SEED)),
            ])
            # Fit on the FULL training set for this route (not the 90% es_tr
            # split): Ridge has no early-stopping mechanism to spend a held-out
            # slice on, so there's no reason to withhold data from it, and no
            # test leakage either way (X_train/y_train_adj are training-only).
            model.fit(X_train, np.log1p(y_train_adj))

            pred_adj = np.clip(np.expm1(model.predict(X_test)), 1, None)
            pred_nom = np.clip(deflate_pred(pred_adj, R_test, config['alpha']), 1, None)
            models[qlab] = model
            preds_test[qlab] = pred_nom
            print(f"  [{name}-{qlab}] best_iter=N/A (ridge)  "
                  f"({(time.time()-t0)/60:.1f} min)")
            continue

        if args.retune:
            params = tune_quantile(
                f"{name}-{qlab}", X_es_tr, y_adj_tr, X_es_va, y_adj_va,
                monotone, q, args.n_trials)
        else:
            params = default_params[qlab].copy()
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
        model.fit(X_es_tr, np.log1p(y_adj_tr),
                  eval_set=[(X_es_va, np.log1p(y_adj_va))], verbose=False)

        pred_adj = np.clip(np.expm1(model.predict(X_test)), 1, None)
        pred_nom = np.clip(deflate_pred(pred_adj, R_test, config['alpha']), 1, None)
        models[qlab] = model
        preds_test[qlab] = pred_nom
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
                        help="Run Optuna for each quantile (slow, production-grade)")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Optuna trials per quantile when --retune is set")
    parser.add_argument("--gpu", action='store_true',
                        help="Use GPU (CUDA) for training. Only affects q05/q95 (XGBoost); "
                             "q50 is a CPU-only sklearn Ridge model.")
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA,
                        help=f"L2 regularization strength for the q50 Ridge model "
                             f"(default {RIDGE_ALPHA}). Only applies to q50 — "
                             "q05/q95 remain reg:quantileerror XGBoost.")
    parser.add_argument("--cap-pct", type=float, default=99.5,
                        help="Drop rows above this salevalue percentile (default 99.5). "
                             "Use 100 to disable capping entirely. "
                             "Common values: 99.5 (drop top 0.5%%), 99.9 (drop top 0.1%%), 100 (keep all).")
    parser.add_argument("--min-salevalue", type=float, default=0.0,
                        help="Drop rows where salevalue <= this dollar amount (default 0, "
                             "which keeps current behavior: salevalue > 0). "
                             "Examples: 50 (drop rows below $50), 100 (drop scrap-priced cars).")
    parser.add_argument("--save-shap", action='store_true',
                        help="DISABLED in this variant: SHAP is only ever run against the q50 "
                             "model in this file, and q50 here is a non-tree sklearn Ridge "
                             "pipeline, incompatible with shap.TreeExplainer. Passing this "
                             "flag errors out.")
    parser.add_argument("--save-shap-global", action='store_true',
                        help="DISABLED in this variant: same non-tree-q50 reason as --save-shap. "
                             "Passing this flag errors out.")
    parser.add_argument("--shap-parquet-only", action='store_true',
                        help="When --save-shap is set, write only parquet for per-row files "
                             "(skip the multi-GB CSV). Does not affect --save-shap-global "
                             "(always CSV; it's tiny).")
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
    if args.save_shap or args.save_shap_global:
        parser.error(
            "--save-shap/--save-shap-global are disabled in this variant: SHAP is only "
            "ever run against the q50 model in this file, and q50 here is a non-tree "
            "sklearn Ridge pipeline (shap.TreeExplainer requires a tree ensemble; "
            "_shap_export_helper.py's model.save_model() call would also raise "
            "AttributeError on a Pipeline)."
        )

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

    print("Dropping columns not recognized by the schema adapter (DB noise/metadata)...")
    df = filter_to_known_columns(df)

    print("Mapping new database schema to legacy ML schema...")
    df = map_raw_features_to_legacy(df)

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

    # Partition
    train_cult_flag = compute_cult_flag(train_raw, CULT_LOOKUP)
    test_cult_flag  = compute_cult_flag(test_raw,  CULT_LOOKUP)
    print(f"Train cult: {train_cult_flag.sum():,}  Test cult: {test_cult_flag.sum()}")

    train_cult_raw = train_raw[train_cult_flag].copy().reset_index(drop=True)
    train_std_raw  = train_raw[~train_cult_flag].copy().reset_index(drop=True)
    test_cult_raw  = test_raw[test_cult_flag].copy().reset_index(drop=True)
    test_std_raw   = test_raw[~test_cult_flag].copy().reset_index(drop=True)

    # Train cult quantiles
    print("\n" + "#" * 70 + "\n# CULT MODEL (3 quantiles)\n" + "#" * 70)
    pre_cult, models_cult, preds_cult, params_cult = train_subset_with_quantiles(
        "CULT", CULT_CONFIG, DEFAULT_PARAMS_CULT,
        train_cult_raw, test_cult_raw,
        LAT_MAP, LON_MAP, CULT_LOOKUP, args)

    # Train standard quantiles
    print("\n" + "#" * 70 + "\n# STANDARD MODEL (3 quantiles)\n" + "#" * 70)
    pre_std, models_std, preds_std, params_std = train_subset_with_quantiles(
        "STANDARD", STANDARD_CONFIG, DEFAULT_PARAMS_STANDARD,
        train_std_raw, test_std_raw,
        LAT_MAP, LON_MAP, CULT_LOOKUP, args)

    # Routed predictions
    y_test = test_raw[TARGET_COL].values
    routed = {}
    for qlab in QUANTILE_LABELS:
        arr = np.zeros(len(test_raw))
        arr[test_cult_flag]  = preds_cult[qlab]
        arr[~test_cult_flag] = preds_std[qlab]
        routed[qlab] = np.clip(arr, 1, None)

    routed["q05"], routed["q50"], routed["q95"] = fix_quantile_crossing(
        routed["q05"], routed["q50"], routed["q95"])

    # Evaluation
    print("\n" + "=" * 70)
    print("FINAL ROUTED QUANTILE PREDICTIONS")
    print("=" * 70)
    cov_90 = coverage(y_test, routed["q05"], routed["q95"])
    print(f"\nEmpirical coverage (90% CI):    {cov_90*100:.1f}%   (target: 90%)")
    print(f"Point (median) MAE:              ${mae(y_test, routed['q50']):.2f}")
    print(f"Point (median) RMSE:             ${rmse(y_test, routed['q50']):.2f}")
    print(f"Point (median) RMSLE:            {rmsle(y_test, routed['q50']):.4f}")
    print(f"Median width of 90% CI:          ${np.median(routed['q95']-routed['q05']):.0f}")
    print(f"Mean width of 90% CI:            ${np.mean(routed['q95']-routed['q05']):.0f}")

    print(f"\n--- BY ROUTE ---")
    for label, mask in [('Cult', test_cult_flag), ('Non-cult', ~test_cult_flag)]:
        if mask.sum() == 0: continue
        yt = y_test[mask]
        p5t, p50t, p95t = routed["q05"][mask], routed["q50"][mask], routed["q95"][mask]
        print(f"  {label} (n={mask.sum()}): MAE_p50=${mae(yt,p50t):.0f}  "
              f"RMSLE_p50={rmsle(yt,p50t):.4f}  Cov_90={coverage(yt,p5t,p95t)*100:.1f}%  "
              f"Mean_Width=${np.mean(p95t-p5t):.0f}")

    tier_edges = [0, 200, 500, 1000, 2500, np.inf]
    tier_labels = ['$0-200','$200-500','$500-1K','$1K-2.5K','$2.5K+']
    tiers = pd.cut(y_test, bins=tier_edges, labels=tier_labels)
    tier_metrics = {}
    print(f"\n{'Tier':<10} {'N':>6} {'MAE_p50':>10} {'RMSLE_p50':>10} {'Cov_90':>8} {'Mean_Width':>12}")
    for t in tier_labels:
        m = (tiers == t)
        if m.sum() == 0: continue
        yt, p5t, p50t, p95t = y_test[m], routed["q05"][m], routed["q50"][m], routed["q95"][m]
        tier_metrics[t] = {
            'N': int(m.sum()),
            'MAE_p50': mae(yt, p50t),
            'RMSE_p50': rmse(yt, p50t),
            'RMSLE_p50': rmsle(yt, p50t),
            'coverage_90': coverage(yt, p5t, p95t),
            'mean_width': float(np.mean(p95t - p5t)),
        }
        print(f"{t:<10} {m.sum():>6} ${mae(yt,p50t):>8.0f} {rmsle(yt,p50t):>10.4f} "
              f"{coverage(yt,p5t,p95t)*100:>7.1f}% ${np.mean(p95t-p5t):>10.0f}")

    # SHAP sanity check (median models) — SKIPPED for this variant.
    # q50 is a non-tree sklearn Ridge pipeline; shap.TreeExplainer only
    # supports tree ensembles, and this file only ever runs SHAP against
    # q50, so there is nothing left to check.
    print("\n--- SHAP sanity check SKIPPED (q50 is a non-tree Ridge model, "
          "incompatible with shap.TreeExplainer) ---")
    shap_check_cult = None
    shap_check_std = None

    # Save artifacts
    print(f"\nSaving artifacts to {args.out}/ ...")
    joblib.dump(pre_cult, os.path.join(args.out, "preprocessor_cult.joblib"))
    joblib.dump(pre_std,  os.path.join(args.out, "preprocessor_standard.joblib"))
    for qlab in QUANTILE_LABELS:
        if qlab == "q50":
            # Ridge Pipeline has no .save_model() (XGBoost-only) — use joblib,
            # with a distinct filename/extension so it's unambiguous downstream.
            joblib.dump(models_cult["q50"], os.path.join(args.out, "model_cult_q50_ridge.joblib"))
            joblib.dump(models_std["q50"], os.path.join(args.out, "model_standard_q50_ridge.joblib"))
            continue
        models_cult[qlab].save_model(os.path.join(args.out, f"model_cult_{qlab}.json"))
        models_std[qlab].save_model(os.path.join(args.out, f"model_standard_{qlab}.json"))
    joblib.dump(CULT_LOOKUP, os.path.join(args.out, "cult_lookup.joblib"))
    joblib.dump(LAT_MAP, os.path.join(args.out, "zip_lat_map.joblib"))
    joblib.dump(LON_MAP, os.path.join(args.out, "zip_lon_map.joblib"))
    with open(os.path.join(args.out, "alphas.json"), 'w') as f:
        json.dump({
            'cult_alpha': CULT_CONFIG['alpha'],
            'standard_alpha': STANDARD_CONFIG['alpha'],
            'cult_use_macro': CULT_CONFIG['use_macro'],
            'cult_use_geo':   CULT_CONFIG['use_geo'],
            'cult_use_cult':  CULT_CONFIG['use_cult'],
            'standard_use_macro': STANDARD_CONFIG['use_macro'],
            'standard_use_geo':   STANDARD_CONFIG['use_geo'],
            'standard_use_cult':  STANDARD_CONFIG['use_cult'],
        }, f, indent=2)

    # ----- Save FULL train and test predictions for analysis -----
    print(f"\n--- Computing train predictions (full set) for save ---")
    # Predict on each train subset with its own preprocessor and routed quantile models,
    # then assemble into the original train_raw row order.
    n_train = len(train_raw)
    train_preds = {qlab: np.zeros(n_train, dtype=float) for qlab in QUANTILE_LABELS}

    # Cult train predictions
    if len(train_cult_raw) > 0:
        X_train_cult = pre_cult.transform(train_cult_raw)
        R_train_cult = cpi_ratio_arr(train_cult_raw)
        for qlab in QUANTILE_LABELS:
            pred_adj = np.clip(np.expm1(models_cult[qlab].predict(X_train_cult)), 1, None)
            pred_nom = np.clip(deflate_pred(pred_adj, R_train_cult, CULT_CONFIG['alpha']), 1, None)
            train_preds[qlab][train_cult_flag] = pred_nom

    # Standard train predictions
    if len(train_std_raw) > 0:
        X_train_std = pre_std.transform(train_std_raw)
        R_train_std = cpi_ratio_arr(train_std_raw)
        for qlab in QUANTILE_LABELS:
            pred_adj = np.clip(np.expm1(models_std[qlab].predict(X_train_std)), 1, None)
            pred_nom = np.clip(deflate_pred(pred_adj, R_train_std, STANDARD_CONFIG['alpha']), 1, None)
            train_preds[qlab][~train_cult_flag] = pred_nom

    # Quantile-crossing fix on the assembled train predictions
    train_preds["q05"], train_preds["q50"], train_preds["q95"] = fix_quantile_crossing(
        train_preds["q05"], train_preds["q50"], train_preds["q95"])

    train_pred_df = build_predictions_frame(
        raw_df=train_raw,
        actuals=train_raw[TARGET_COL].values,
        preds_p5=train_preds["q05"],
        preds_p50=train_preds["q50"],
        preds_p95=train_preds["q95"],
        extra_cols={'is_cult': train_cult_flag},
    )
    test_pred_df = build_predictions_frame(
        raw_df=test_raw,
        actuals=y_test,
        preds_p5=routed["q05"],
        preds_p50=routed["q50"],
        preds_p95=routed["q95"],
        extra_cols={'is_cult': test_cult_flag},
    )
    print("\nSaving prediction files...")
    save_predictions(train_pred_df, args.out, "train_predictions")
    save_predictions(test_pred_df,  args.out, "test_predictions")
    # Reminder: train predictions are produced by a model that saw those rows during
    # training; they will look better than test. Use for sanity checks, not evaluation.

    # Evaluate and save per-tier metrics JSON for each (also breaks down by route)
    evaluate(train_pred_df, label="script21_train",
             save_json_to=os.path.join(args.out, "train_metrics.json"), verbose=True)
    evaluate(test_pred_df,  label="script21_test",
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
        also_csv = not args.shap_parquet_only

        # Compute transformed feature frames per route (used by both paths)
        if len(train_cult_raw) > 0:
            X_train_cult_full = pre_cult.transform(train_cult_raw)
            X_test_cult_full  = pre_cult.transform(test_cult_raw)
        else:
            X_train_cult_full = None
            X_test_cult_full  = None
        X_train_std_full = pre_std.transform(train_std_raw)
        X_test_std_full  = pre_std.transform(test_std_raw)

        # Fast global-only path: no temp dirs, no per-row, no concat
        if want_global and not want_per_row:
            shap_sample = args.shap_sample if args.shap_sample > 0 else None
            print("\nGlobal SHAP for CULT route (train + test)...")
            if X_train_cult_full is not None:
                compute_and_save_global_shap_importance(
                    models_cult["q50"], X_train_cult_full, pre_cult.feature_cols_,
                    args.out, prefix="train_cult", sample_size=shap_sample)
                compute_and_save_global_shap_importance(
                    models_cult["q50"], X_test_cult_full, pre_cult.feature_cols_,
                    args.out, prefix="test_cult", sample_size=shap_sample)
                # Rename to match the convention (suffix-based)
                for split in ['train', 'test']:
                    src = os.path.join(args.out, f"{split}_cult_shap_global_importance.csv")
                    dst = os.path.join(args.out, f"{split}_shap_global_importance_cult.csv")
                    if os.path.exists(src):
                        os.replace(src, dst)
            else:
                print("Skipping cult global SHAP (no cult rows)")

            print("\nGlobal SHAP for STANDARD route (train + test)...")
            compute_and_save_global_shap_importance(
                models_std["q50"], X_train_std_full, pre_std.feature_cols_,
                args.out, prefix="train_standard", sample_size=shap_sample)
            compute_and_save_global_shap_importance(
                models_std["q50"], X_test_std_full, pre_std.feature_cols_,
                args.out, prefix="test_standard", sample_size=shap_sample)
            for split in ['train', 'test']:
                src = os.path.join(args.out, f"{split}_standard_shap_global_importance.csv")
                dst = os.path.join(args.out, f"{split}_shap_global_importance_standard.csv")
                if os.path.exists(src):
                    os.replace(src, dst)

        # Full per-row + global path: uses temp dirs and the existing concat dance
        elif want_per_row:
            from _shap_export_helper import export_shap_sets
            import shutil

            if not also_csv:
                print("Note: --shap-parquet-only set; per-row CSV files will be skipped.")
            else:
                n_rows_train_std = len(train_std_raw) * len(pre_std.feature_cols_)
                est_csv_gb = n_rows_train_std * 200 / 1e9
                print(f"Warning: engineered train CSV will be ~{est_csv_gb:.1f} GB "
                      f"({len(train_std_raw):,} rows x {len(pre_std.feature_cols_)} features) "
                      f"for the standard route alone. Use --shap-parquet-only to skip per-row CSV.")

            # ---- Cult route ----
            if X_train_cult_full is not None:
                print("\nSHAP for CULT route...")
                cult_tmp = os.path.join(args.out, "_shap_tmp_cult")
                os.makedirs(cult_tmp, exist_ok=True)
                train_preds_cult_p50 = train_preds["q50"][train_cult_flag]
                export_shap_sets(
                    model=models_cult["q50"],
                    train_X=X_train_cult_full, test_X=X_test_cult_full,
                    train_raw=train_cult_raw, test_raw=test_cult_raw,
                    feature_names=pre_cult.feature_cols_,
                    train_actuals=train_cult_raw[TARGET_COL].values,
                    train_preds_p50=train_preds_cult_p50,
                    test_actuals=test_raw[TARGET_COL].values[test_cult_flag],
                    test_preds_p50=preds_cult["q50"],
                    out_dir=cult_tmp,
                    also_csv=also_csv,
                )
            else:
                cult_tmp = None
                print("Skipping cult route SHAP (no cult rows)")

            # ---- Standard route ----
            print("\nSHAP for STANDARD route...")
            std_tmp = os.path.join(args.out, "_shap_tmp_std")
            os.makedirs(std_tmp, exist_ok=True)
            train_preds_std_p50 = train_preds["q50"][~train_cult_flag]
            export_shap_sets(
                model=models_std["q50"],
                train_X=X_train_std_full, test_X=X_test_std_full,
                train_raw=train_std_raw, test_raw=test_std_raw,
                feature_names=pre_std.feature_cols_,
                train_actuals=train_std_raw[TARGET_COL].values,
                train_preds_p50=train_preds_std_p50,
                test_actuals=test_raw[TARGET_COL].values[~test_cult_flag],
                test_preds_p50=preds_std["q50"],
                out_dir=std_tmp,
                also_csv=also_csv,
            )

            # ---- Move global importance per-route files (no concat; different feature spaces) ----
            print("\nMoving per-route global importance files...")
            for set_name in ['train', 'test']:
                for route_dir, route_label in [(cult_tmp, 'cult'), (std_tmp, 'standard')]:
                    if route_dir is None:
                        continue
                    src = os.path.join(route_dir, f'{set_name}_shap_global_importance.csv')
                    dst = os.path.join(args.out, f'{set_name}_shap_global_importance_{route_label}.csv')
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                        print(f"  Wrote {dst}")
                        os.remove(src)

            # ---- Concatenate per-row files across routes ----
            print("\nConcatenating per-route per-row SHAP files...")
            for fname in ['train_shap_engineered.parquet', 'train_shap_engineered.csv',
                          'test_shap_engineered.parquet',  'test_shap_engineered.csv',
                          'train_shap_raw.parquet',         'train_shap_raw.csv',
                          'test_shap_raw.parquet',          'test_shap_raw.csv']:
                if not also_csv and fname.endswith('.csv'):
                    continue
                target = os.path.join(args.out, fname)
                parts = []
                if cult_tmp is not None and os.path.exists(os.path.join(cult_tmp, fname)):
                    parts.append(os.path.join(cult_tmp, fname))
                if os.path.exists(os.path.join(std_tmp, fname)):
                    parts.append(os.path.join(std_tmp, fname))
                if not parts:
                    continue
                if fname.endswith('.parquet'):
                    import pyarrow.parquet as pq
                    import pyarrow as pa
                    tables = [pq.read_table(p) for p in parts]
                    tables = [t.append_column('route',
                              pa.array([('cult' if 'cult' in p else 'standard')] * t.num_rows))
                              for t, p in zip(tables, parts)]
                    combined = pa.concat_tables(tables, promote=True)
                    pq.write_table(combined, target, compression='snappy')
                    print(f"  Wrote {target}  ({combined.num_rows:,} rows)")
                else:
                    first = True
                    total = 0
                    with open(target, 'w', newline='') as out_f:
                        for p in parts:
                            route_label = 'cult' if 'cult' in p else 'standard'
                            for chunk in pd.read_csv(p, chunksize=200_000):
                                chunk['route'] = route_label
                                chunk.to_csv(out_f, mode='a', header=first, index=False)
                                first = False
                                total += len(chunk)
                    print(f"  Wrote {target}  ({total:,} rows)")
            # Clean up temp dirs
            for d in (cult_tmp, std_tmp):
                if d is not None and os.path.exists(d):
                    shutil.rmtree(d)

    # ----- Save sample low-MAE rows (used by tests/test_examples.py) -----
    sample_cult = test_cult_raw.copy().reset_index(drop=True)
    sample_cult['actual']        = test_raw[TARGET_COL].values[test_cult_flag]
    sample_cult['p5']            = preds_cult["q05"]
    sample_cult['p50']           = preds_cult["q50"]
    sample_cult['p95']           = preds_cult["q95"]
    sample_cult['ci_width']      = sample_cult['p95'] - sample_cult['p5']
    sample_cult['abs_error_p50'] = np.abs(sample_cult['p50'] - sample_cult['actual'])
    sample_cult['is_cult']       = True
    sample_cult_low = sample_cult.sort_values('abs_error_p50').head(15)

    sample_std = test_std_raw.copy().reset_index(drop=True)
    sample_std['actual']        = test_raw[TARGET_COL].values[~test_cult_flag]
    sample_std['p5']            = preds_std["q05"]
    sample_std['p50']           = preds_std["q50"]
    sample_std['p95']           = preds_std["q95"]
    sample_std['ci_width']      = sample_std['p95'] - sample_std['p5']
    sample_std['abs_error_p50'] = np.abs(sample_std['p50'] - sample_std['actual'])
    sample_std['is_cult']       = False
    sample_std_low = sample_std[
        (sample_std['actual'] >= 500) & (sample_std['actual'] <= 2500)
    ].sort_values('abs_error_p50').head(15)

    pd.concat([sample_cult_low, sample_std_low], ignore_index=True).to_parquet(
        os.path.join(args.out, "sample_test_rows.parquet"))

    with open(os.path.join(args.out, "training_metadata.json"), 'w') as f:
        json.dump({
            'model_type': 'script21_routed_quantile',
            'q50_model_type': 'ridge',
            'ridge_alpha': args.ridge_alpha,
            'q50_ridge_train_fraction': 1.0,  # ridge fit on 100% of train; q05/q95 use 90% (es_tr)
            'quantiles': QUANTILES,
            'cult_alpha': CULT_CONFIG['alpha'],
            'standard_alpha': STANDARD_CONFIG['alpha'],
            # NOTE: cult_xgb_params_by_q/standard_xgb_params_by_q's "q50" entry is
            # {'ridge_alpha': ...}, not XGBoost hyperparameters, in this variant.
            'cult_xgb_params_by_q': params_cult,
            'standard_xgb_params_by_q': params_std,
            'retune_used': bool(args.retune),
            'gpu_used': bool(args.gpu),
            'cap_pct': args.cap_pct,
            'cap_value': cap_value,
            'min_salevalue': args.min_salevalue,
            'save_shap_used': bool(args.save_shap),
            'save_shap_global_used': bool(args.save_shap or args.save_shap_global),
            'n_optuna_trials_per_quantile': args.n_trials if args.retune else 0,
            'train_size': len(train_raw),
            'train_cult_size': int(train_cult_flag.sum()),
            'train_standard_size': int((~train_cult_flag).sum()),
            'test_size': len(test_raw),
            'test_cult_size': int(test_cult_flag.sum()),
            'test_standard_size': int((~test_cult_flag).sum()),
            'test_overall': {
                'MAE_p50':  mae(y_test, routed["q50"]),
                'RMSE_p50': rmse(y_test, routed["q50"]),
                'RMSLE_p50': rmsle(y_test, routed["q50"]),
                'coverage_90': cov_90,
                'mean_ci_width': float(np.mean(routed["q95"] - routed["q05"])),
            },
            'test_by_tier': tier_metrics,
            'n_features_cult': len(pre_cult.feature_cols_),
            'n_features_standard': len(pre_std.feature_cols_),
            'total_runtime_minutes': (time.time() - start_t) / 60,
            'shap_sanity_check': {
                'cult_median':     shap_check_cult,
                'standard_median': shap_check_std,
            },
        }, f, indent=2)

    print(f"\nDone. Total runtime: {(time.time()-start_t)/60:.1f} min")
    print(f"Artifacts written to {args.out}/")


if __name__ == "__main__":
    main()
