"""
inference_script21.py
======================
Loads Script 21 routed quantile artifacts (cult x3 + standard x3 = 6 models)
and serves predictions routed by cult flag.

Prediction:
    Detect cult by (make, model, year) matching the cult_lookup.
    If cult:    use cult preprocessor + cult quantile models.
    Otherwise:  use standard preprocessor + standard quantile models.

SHAP:
    Explains the routed quantile model (cult or standard) for the requested
    quantile (default p50).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from preprocessor import compute_cult_flag, cpi_ratio_arr, deflate_pred
from app.shap_utils import compute_user_shap_payload

QUANTILE_LABELS = ["q05", "q50", "q95"]


class Script21Pipeline:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        self._load()

    def _load(self):
        ad = self.artifacts_dir
        self.pre_cult = joblib.load(os.path.join(ad, "preprocessor_cult.joblib"))
        self.pre_standard = joblib.load(
            os.path.join(ad, "preprocessor_standard.joblib")
        )

        self.models_cult = {}
        self.models_standard = {}
        for qlab in QUANTILE_LABELS:
            m_cult = XGBRegressor()
            m_cult.load_model(os.path.join(ad, f"model_cult_{qlab}.json"))
            self.models_cult[qlab] = m_cult

            m_std = XGBRegressor()
            m_std.load_model(os.path.join(ad, f"model_standard_{qlab}.json"))
            self.models_standard[qlab] = m_std

        with open(os.path.join(ad, "alphas.json")) as f:
            alphas = json.load(f)
        self.cult_alpha = float(alphas["cult_alpha"])
        self.standard_alpha = float(alphas["standard_alpha"])

        self.cult_lookup = joblib.load(os.path.join(ad, "cult_lookup.joblib"))

    def predict(
        self,
        request_dict: dict,
        k_pos: int = 5,
        k_neg: int = 5,
        shap_quantile: str = "q50",
        explain: bool = False,
    ):
        df = pd.DataFrame([request_dict])
        # app/main.py now sends exclude_none=False (every declared field
        # present, None when the caller omitted it) specifically so every
        # raw column preprocessor.py expects exists here -- but a Python
        # None stored in an object-dtype column is NOT the same thing as a
        # real NaN to _normalize_text()/_basic_clean(): `.astype(str)` on
        # None gives the literal string "None" -> "none" after lowercasing,
        # which is NOT caught by their `s != 'nan'` NaN-recovery check, so
        # it would silently become a bogus real category. Coerce explicitly
        # so every "field wasn't provided" case behaves exactly like a
        # genuinely-missing value in the training data (which is what
        # `.fillna(-1)`/`.fillna('na')` downstream are built to expect).
        df = df.where(df.notna(), np.nan)
        if (
            "record_creation_date" not in df.columns
            or df["record_creation_date"].isna().all()
        ):
            df["record_creation_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

        # Determine route via cult lookup
        if "make" not in df.columns:
            df["make"] = None
        if "model" not in df.columns:
            df["model"] = None
        if "year" not in df.columns:
            df["year"] = None

        is_cult = bool(compute_cult_flag(df, self.cult_lookup)[0])
        route = "cult" if is_cult else "standard"

        # Pick preprocessor + models + alpha for route
        if is_cult:
            pre = self.pre_cult
            models = self.models_cult
            alpha = self.cult_alpha
            model_key = "script21_cult"
        else:
            pre = self.pre_standard
            models = self.models_standard
            alpha = self.standard_alpha
            model_key = "script21_standard"

        X = pre.transform(df)
        X_csv = (
            X.copy()
            if hasattr(X, "copy")
            else pd.DataFrame(X, columns=pre.feature_cols_)
        )
        X_csv["stock_id"] = request_dict.get("stock_id") or "unknown"

        csv_file = "api_engineered_features.csv"
        X_csv.to_csv(
            csv_file, mode="a", index=False, header=not os.path.exists(csv_file)
        )

        R = cpi_ratio_arr(df)

        preds = {}
        for qlab in QUANTILE_LABELS:
            pred_adj = float(np.expm1(models[qlab].predict(X))[0])
            pred_nom = float(max(deflate_pred(pred_adj, R, alpha)[0], 1.0))
            preds[qlab] = pred_nom

        ordered = sorted([preds["q05"], preds["q50"], preds["q95"]])
        preds = {
            "q05": round(ordered[0], 2),
            "q50": round(ordered[1], 2),
            "q95": round(ordered[2], 2),
        }   

        result = {
            "predictions": {
                "low": preds["q05"],
                "predicted_price": preds["q50"],
                "high": preds["q95"],
            },
            "is_cult": is_cult,
            "route": route,
        }

        if explain or (k_pos > 0 or k_neg > 0):
            shap_payload = compute_user_shap_payload(
                model=models[shap_quantile],
                X_row=X,
                feature_names=pre.feature_cols_,
                model_key=model_key,
                qlabel=shap_quantile,
                request_dict=request_dict,
                k_pos=k_pos,
                k_neg=k_neg,
            )
            shap_payload["quantile_explained"] = shap_quantile
            result["shap"] = shap_payload

        return result
