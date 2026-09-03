"""
inference_script17.py
======================
Loads Script 17 quick-blend artifacts (6 quantile models: TE x3 + no-TE x3)
and serves quantile predictions + SHAP attributions.

Prediction:
    p_q = w_te * te_q.predict(X_te) + w_no_te * no_te_q.predict(X_no_te)
    Then quantile-crossing fix per row.

SHAP:
    Explains the requested quantile (p50 default). Because the prediction is
    a 0.66/0.34 blend of TE and no-TE bases, SHAP is computed on the
    blended TE-base attributions (TE has the slight majority weight).
    This is a deliberate simplification — we explain "what drives the prediction"
    using the dominant base, rather than try to blend SHAP across two different
    feature spaces with different feature names.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from preprocessor import map_raw_features_to_legacy
from app.shap_utils import compute_user_shap_payload

QUANTILE_LABELS = ["q05", "q50", "q95"]


class Script17Pipeline:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        self._load()

    def _load(self):
        ad = self.artifacts_dir
        self.pre_te = joblib.load(os.path.join(ad, "preprocessor_te.joblib"))
        self.pre_no_te = joblib.load(os.path.join(ad, "preprocessor_no_te.joblib"))

        self.models_te = {}
        self.models_no_te = {}
        for qlab in QUANTILE_LABELS:
            m_te = XGBRegressor()
            m_te.load_model(os.path.join(ad, f"model_te_{qlab}.json"))
            self.models_te[qlab] = m_te

            m_no_te = XGBRegressor()
            m_no_te.load_model(os.path.join(ad, f"model_no_te_{qlab}.json"))
            self.models_no_te[qlab] = m_no_te

        with open(os.path.join(ad, "blend_weights.json")) as f:
            bw = json.load(f)
        self.w_te = float(bw["w_te"])
        self.w_no_te = float(bw["w_no_te"])

    def predict(
        self,
        request_dict: dict,
        k_pos: int = 5,
        k_neg: int = 5,
        shap_quantile: str = "q50",
        explain: bool = False,
    ):
        """
        Returns dict with: predictions {p5, p50, p95}, shap (if explain=True or k>0).
        """
        # Single-row DataFrame from request
        df = pd.DataFrame([request_dict])
        # request_dict uses the CURRENT PredictRequest field names (new DB
        # schema); rename onto the legacy names the date-injection check
        # right below is written against. pre_te/pre_no_te.transform() would
        # apply this same rename internally anyway (preprocessor.py's
        # _basic_clean() calls it automatically), but on an internal copy
        # AFTER this point -- this method reads `df` directly here too.
        df = map_raw_features_to_legacy(df)
        if (
            "record_creation_date" not in df.columns
            or df["record_creation_date"].isna().all()
        ):
            df["record_creation_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

        X_te = self.pre_te.transform(df)
        X_no_te = self.pre_no_te.transform(df)

        preds = {}
        for qlab in QUANTILE_LABELS:
            p_te = float(np.expm1(self.models_te[qlab].predict(X_te))[0])
            p_no_te = float(np.expm1(self.models_no_te[qlab].predict(X_no_te))[0])
            preds[qlab] = max(self.w_te * p_te + self.w_no_te * p_no_te, 1.0)

        # Enforce quantile ordering per row (single row -> just sort)
        ordered = sorted([preds["q05"], preds["q50"], preds["q95"]])
        preds = {"q05": ordered[0], "q50": ordered[1], "q95": ordered[2]}

        result = {
            "predictions": {
                "low": preds["q05"],
                "predicted_price": preds["q50"],
                "high": preds["q95"],
            },
            "is_cult": None,
            "route": "quick_blend",
        }

        # k != 0 rather than k > 0: a negative k means "all features"
        # (see collapse_engineered_to_raw), so only an explicit 0 on BOTH
        # sides means the caller wants no SHAP work done at all.
        #
        # Deliberately NOT `explain or (...)`: with k_pos=k_neg=0 there are no
        # attributions for the explanation to be about, and the LLM would be
        # handed an empty feature list and invent generic filler. Skipping the
        # whole block leaves no "shap" key on the result, which is what makes
        # main.py's `if explain_flag and pred_result.get("shap")` fall through
        # -- so explain=true with k=0 returns explanation=null AND avoids the
        # expensive TreeExplainer pass rather than computing SHAP to discard it.
        if k_pos != 0 or k_neg != 0:
            # 1. Extract calculated values from the preprocessor output
            if hasattr(X_te, "iloc"):
                calc_vals = X_te.iloc[0].values
            else:
                calc_vals = X_te[0]

            # 2. Map them to their column names
            calc_dict = dict(zip(self.pre_te.feature_cols_, calc_vals))

            # 3. Merge: request_dict overwrites calc_dict so raw strings are kept
            enriched_request = {**calc_dict, **request_dict}

            shap_payload = compute_user_shap_payload(
                model=self.models_te[shap_quantile],
                X_row=X_te,
                feature_names=self.pre_te.feature_cols_,
                model_key=f"script17_te",
                qlabel=shap_quantile,
                request_dict=enriched_request,  # <-- Passed here
                k_pos=k_pos,
                k_neg=k_neg,
            )
            shap_payload["quantile_explained"] = shap_quantile
            result["shap"] = shap_payload

        return result
