"""
shap_utils.py
==============
Wraps the shared shap_dollar_helper to compute SHAP values for a single
preprocessed row, attaching human-readable names.

Caches one TreeExplainer per (model_id, quantile) so we don't rebuild
the explainer on every request.

Two payloads are available:
  - compute_shap_payload (engineered features, full detail) for debugging
    and internal use.
  - compute_user_shap_payload (collapsed to raw user-facing features) used by
    the LLM prompt and the public API response.
"""

import threading
import shap
import numpy as np
from typing import Any, Dict, List

from shap_dollar_helper import shap_to_dollar_terms, _patch_xgb_base_score_for_shap
from app.feature_descriptions import humanize_feature
from app.raw_feature_mapping import collapse_engineered_to_raw

# Cache: {(model_key, qlabel): (patched_model, explainer)}
_EXPLAINER_CACHE: Dict[tuple, Any] = {}
_CACHE_LOCK = threading.Lock()


def get_explainer(model, model_key: str, qlabel: str):
    """Get or build a cached TreeExplainer for this (model_key, qlabel) pair."""
    cache_id = (model_key, qlabel)
    with _CACHE_LOCK:
        if cache_id in _EXPLAINER_CACHE:
            patched, explainer = _EXPLAINER_CACHE[cache_id]
            return patched, explainer

        patched = _patch_xgb_base_score_for_shap(model)
        explainer = shap.TreeExplainer(patched)
        _EXPLAINER_CACHE[cache_id] = (patched, explainer)
        return patched, explainer


def _compute_raw_shap_arrays(model, X_row, feature_names, model_key, qlabel):
    """Internal: get shap values, base log, log pred, and feature values for one row."""
    patched_model, explainer = get_explainer(model, model_key, qlabel)

    shap_values_arr = explainer.shap_values(X_row)
    if hasattr(shap_values_arr, 'shape') and shap_values_arr.ndim == 2:
        shap_vals = shap_values_arr[0]
    else:
        shap_vals = shap_values_arr

    expected = explainer.expected_value
    base_log_value = float(expected if not hasattr(expected, '__len__') else expected[0])
    log_pred = float(patched_model.predict(X_row)[0])
    feature_values_row = X_row.iloc[0].values
    return shap_vals, base_log_value, log_pred, feature_values_row


def compute_shap_payload(model, X_row, feature_names, model_key, qlabel,
                          k_pos=5, k_neg=5):
    """Engineered-feature SHAP payload (full detail). For debugging / internal use.

    Returns top-K positive and top-K negative engineered features with full
    log_shap, marginal/proportional dollar impacts, and percentages.
    """
    shap_vals, base_log_value, log_pred, feature_values_row = \
        _compute_raw_shap_arrays(model, X_row, feature_names, model_key, qlabel)

    breakdown = shap_to_dollar_terms(
        shap_vals, base_log_value, log_pred,
        feature_names, feature_values_row, k_pos=k_pos, k_neg=k_neg)

    for rec_list in [breakdown['top_positive'], breakdown['top_negative']]:
        for r in rec_list:
            r['human_readable'] = humanize_feature(r['feature'])

    return {
        'baseline_dollars':   breakdown['baseline_dollars'],
        'final_pred_dollars': breakdown['final_pred_dollars'],
        'top_positive':       breakdown['top_positive'],
        'top_negative':       breakdown['top_negative'],
    }


def compute_user_shap_payload(model, X_row, feature_names, model_key, qlabel,
                                request_dict, k_pos=5, k_neg=5, is_cult=None):
    """User-facing SHAP payload: collapses engineered features back to raw inputs.

    Computes SHAP for ALL engineered features, then groups them by their raw
    source feature (e.g., 'make_freq' + 'make_tgt_enc' -> 'make'), sums dollar
    impacts per group, and returns the top-K positive/negative raw groups.

    Raw values come from request_dict (so the user sees "Runs & Drives" rather
    than the encoded integer the model uses internally). `is_cult` is a rare
    exception: it's the caller's already-computed cult/collectible flag
    (Script21Pipeline only; script17 passes None), threaded through purely so
    the '__collectible' group gets a real value instead of always being null
    -- see collapse_engineered_to_raw's docstring for why request_dict itself
    can't supply it.
    """
    shap_vals, base_log_value, log_pred, feature_values_row = \
        _compute_raw_shap_arrays(model, X_row, feature_names, model_key, qlabel)

    # Compute the FULL breakdown of all features (not just top K). We need
    # everything in order to collapse correctly across many engineered features.
    # Pass huge K to get all features ranked.
    full_breakdown = shap_to_dollar_terms(
        shap_vals, base_log_value, log_pred,
        feature_names, feature_values_row,
        k_pos=len(feature_names), k_neg=len(feature_names))

    # Combine positive + negative engineered records and let collapse decide
    all_records = full_breakdown['top_positive'] + full_breakdown['top_negative']

    collapsed = collapse_engineered_to_raw(
        feature_records=all_records,
        request_dict=request_dict,
        k_pos=k_pos, k_neg=k_neg,
        look_factor=2,
        is_cult=is_cult,
    )

    return {
        'baseline_dollars':   full_breakdown['baseline_dollars'],
        'final_pred_dollars': full_breakdown['final_pred_dollars'],
        'top_positive':       collapsed['top_positive'],
        'top_negative':       collapsed['top_negative'],
    }
