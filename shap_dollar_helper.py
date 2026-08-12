"""
shap_dollar_helper.py
======================
Converts log-space SHAP values to dollar and percentage representations.

Two dollar conversion methods are computed for every feature:
  - "marginal":    expm1(base + shap_i) - expm1(base)
                   "if you removed only this feature, dollars would change by this much."
                   Most faithful to the user mental model.
                   Sums approximately to (final - baseline); residual reported separately.
  - "proportional": shap_i / sum(shap) * (final - baseline)
                   Sums exactly to (final - baseline) by construction.
                   Distorts magnitudes when shap values are large.

Percentage is "% of final prediction" only (Option A from our design discussion).
"""

import numpy as np


def shap_to_dollar_terms(shap_values, base_log_value, final_log_pred,
                          feature_names, feature_values, k_pos=5, k_neg=5):
    """
    Parameters
    ----------
    shap_values : np.ndarray, shape (n_features,)
        SHAP values for ONE prediction in log space. Sums to (final_log_pred - base_log_value).
    base_log_value : float
        Expected log prediction (explainer.expected_value).
    final_log_pred : float
        Model's final log-space prediction for this row.
    feature_names : list of str
    feature_values : array-like, shape (n_features,)
    k_pos : int, top-K positive contributions to return
    k_neg : int, bottom-K negative contributions to return

    Returns
    -------
    dict with:
        baseline_dollars
        final_pred_dollars
        marginal_dollar_sum               # sum of marginal dollar impacts
        marginal_interaction_residual     # (final - baseline) - marginal_dollar_sum
        proportional_dollar_sum           # always equals (final - baseline)
        top_positive, top_negative        # ranked by marginal dollar impact
        all_features_count
    """
    shap_arr      = np.asarray(shap_values, dtype=float)
    base_dollars  = float(np.expm1(base_log_value))
    final_dollars = float(np.expm1(final_log_pred))
    gap_dollars   = final_dollars - base_dollars

    # ---- Method 1: marginal dollar impact ----
    # For feature i: dollar change if we replace that feature's contribution with zero
    marginal_dollars = np.expm1(base_log_value + shap_arr) - base_dollars

    # ---- Method 2: proportional allocation ----
    # Distribute (final - baseline) by each shap_i's share of the total log contribution
    total_log_contrib = shap_arr.sum()
    if abs(total_log_contrib) > 1e-9:
        proportional_dollars = (shap_arr / total_log_contrib) * gap_dollars
    else:
        proportional_dollars = np.zeros_like(shap_arr)

    # ---- Percentages (% of final prediction; Option A only) ----
    if final_dollars > 0:
        pct_marginal     = (marginal_dollars     / final_dollars) * 100
        pct_proportional = (proportional_dollars / final_dollars) * 100
    else:
        pct_marginal     = np.zeros_like(shap_arr)
        pct_proportional = np.zeros_like(shap_arr)

    # ---- Build feature records ----
    records = []
    for i, fname in enumerate(feature_names):
        fv = feature_values[i] if i < len(feature_values) else None
        # Pretty-print feature value
        if fv is None:
            fv_pretty = None
        elif isinstance(fv, float) and np.isnan(fv):
            fv_pretty = None
        elif isinstance(fv, float) and fv.is_integer():
            fv_pretty = int(fv)
        elif isinstance(fv, (int, np.integer)):
            fv_pretty = int(fv)
        elif isinstance(fv, (float, np.floating)):
            fv_pretty = round(float(fv), 4)
        else:
            fv_pretty = str(fv)
        records.append({
            'feature': fname,
            'value': fv_pretty,
            'log_shap': float(shap_arr[i]),
            'dollar_impact_marginal':     float(marginal_dollars[i]),
            'dollar_impact_proportional': float(proportional_dollars[i]),
            'pct_of_prediction_marginal':     float(pct_marginal[i]),
            'pct_of_prediction_proportional': float(pct_proportional[i]),
        })

    # Rank by marginal dollar impact (defensible: that's the most interpretable signal)
    positives = sorted(
        [r for r in records if r['dollar_impact_marginal'] > 0],
        key=lambda r: r['dollar_impact_marginal'], reverse=True
    )
    negatives = sorted(
        [r for r in records if r['dollar_impact_marginal'] < 0],
        key=lambda r: r['dollar_impact_marginal']
    )

    return {
        'baseline_dollars': base_dollars,
        'final_pred_dollars': final_dollars,
        'marginal_dollar_sum': float(marginal_dollars.sum()),
        'marginal_interaction_residual': float(gap_dollars - marginal_dollars.sum()),
        'proportional_dollar_sum': float(proportional_dollars.sum()),  # equals gap_dollars
        'top_positive': positives[:k_pos],
        'top_negative': negatives[:k_neg],
        'all_features_count': len(feature_names),
    }


def _install_shap_xgb_loader_patch():
    """Monkey-patch SHAP's XGBTreeModelLoader so it tolerates the array-form
    base_score string that XGBoost 2.x+ writes (e.g. '[6.534637E0]').

    This is idempotent — safe to call many times. It only patches once.

    Older SHAP versions (< 0.50) call `float(learner_model_param["base_score"])`
    directly, which crashes when the value is "[6.534637E0]". This patch wraps
    `__init__` so that, if the original raises that exact ValueError, we
    temporarily swap in a parser that handles array-form base_score (the same
    approach SHAP 0.50+ uses internally via ast.literal_eval).

    Returns True if the patch was applied; False if SHAP isn't importable or
    is already patched.
    """
    try:
        from shap.explainers import _tree as shap_tree
    except ImportError:
        return False

    loader_cls = getattr(shap_tree, 'XGBTreeModelLoader', None)
    if loader_cls is None or getattr(loader_cls, '_anthropic_base_score_patched', False):
        return False

    original_init = loader_cls.__init__

    def patched_init(self, xgb_model, *args, **kwargs):
        try:
            return original_init(self, xgb_model, *args, **kwargs)
        except ValueError as e:
            msg = str(e)
            if 'could not convert string to float' not in msg or '[' not in msg:
                raise
            # SHAP's float() failed on an array-form base_score. Run __init__
            # again, but temporarily replace the global `float` builtin in the
            # shap_tree module with a tolerant version that strips brackets.
            import builtins, ast
            tolerant_float = lambda v: (
                float(v) if not (isinstance(v, str) and v.lstrip().startswith('['))
                else (lambda parsed: float(parsed[0]) if isinstance(parsed, (list, tuple)) else float(parsed))(
                    ast.literal_eval(v.strip())
                )
            )
            original_float = getattr(shap_tree, 'float', builtins.float)
            shap_tree.float = tolerant_float
            try:
                return original_init(self, xgb_model, *args, **kwargs)
            finally:
                shap_tree.float = original_float

    loader_cls.__init__ = patched_init
    loader_cls._anthropic_base_score_patched = True
    return True


def _patch_xgb_base_score_for_shap(model):
    """
    Workaround for SHAP/XGBoost incompatibility.

    XGBoost 2.0+ stores `base_score` as a JSON array string like "[6.3117347E0]"
    instead of a plain float. Older SHAP versions try `float("[6.3117347E0]")`
    and crash with ValueError.

    Two-pronged fix:
      1. Install a monkey-patch on SHAP's XGBTreeModelLoader so even if the
         array-form string leaks through, we recover gracefully (handles cases
         where XGBoost 3.x re-serializes the value after our JSON edit).
      2. Round-trip the model through JSON, normalizing base_score in the file
         (handles cases where SHAP can read the file directly).

    Returns a NEW model object with the patch applied. The original is unchanged.
    """
    # Step 1: install loader patch (idempotent)
    _install_shap_xgb_loader_patch()

    # Step 2: round-trip the JSON to clean up base_score
    import json
    import tempfile
    import os
    import xgboost as xgb

    # Save the current model to a temp JSON
    tmpf = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    tmp_path = tmpf.name
    tmpf.close()
    try:
        if hasattr(model, 'save_model'):
            model.save_model(tmp_path)
        else:  # raw Booster
            model.save_model(tmp_path)

        with open(tmp_path, 'r') as f:
            model_json = json.load(f)

        def _walk_and_fix(obj):
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if k == 'base_score' and isinstance(v, str):
                        cleaned = v.strip().lstrip('[').rstrip(']').strip()
                        try:
                            obj[k] = str(float(cleaned))
                        except ValueError:
                            pass
                    else:
                        _walk_and_fix(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk_and_fix(item)

        _walk_and_fix(model_json)

        with open(tmp_path, 'w') as f:
            json.dump(model_json, f)

        # Reload into a fresh model object with the same type as the input
        if isinstance(model, xgb.Booster):
            new_model = xgb.Booster()
            new_model.load_model(tmp_path)
        else:
            new_model = type(model)()  # XGBRegressor() or XGBClassifier()
            new_model.load_model(tmp_path)
        return new_model
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def sanity_check_shap(model, X_sample, feature_names, label="model", n_rows=3):
    """Quick sanity check called at end of training to catch SHAP wiring issues."""
    try:
        import shap
    except ImportError:
        print(f"  [{label}] shap not installed; skipping sanity check.")
        return None

    print(f"  [{label}] Running SHAP sanity check on {n_rows} samples...")

    # Patch model for SHAP compatibility (handles XGBoost 2.0+ base_score format)
    try:
        model = _patch_xgb_base_score_for_shap(model)
    except Exception as e:
        print(f"  [{label}] WARNING: base_score patch failed ({e}); proceeding anyway")

    try:
        explainer = shap.TreeExplainer(model)
    except Exception as e:
        print(f"  [{label}] SHAP TreeExplainer init failed: {e}")
        print(f"  [{label}] Skipping SHAP sanity check (artifacts will still be saved)")
        return {'error': str(e)}

    sample = X_sample.iloc[:n_rows] if hasattr(X_sample, 'iloc') else X_sample[:n_rows]
    try:
        shap_values_arr = explainer.shap_values(sample)
    except Exception as e:
        print(f"  [{label}] SHAP shap_values() failed: {e}")
        return {'error': str(e)}

    expected = explainer.expected_value
    base_log_value = float(expected if not hasattr(expected, '__len__') else expected[0])

    log_preds = model.predict(sample)
    results = []
    for i in range(n_rows):
        shap_vals = shap_values_arr[i]
        log_pred  = float(log_preds[i])
        log_additivity_diff = float(abs((base_log_value + shap_vals.sum()) - log_pred))

        feature_values_row = sample.iloc[i].values if hasattr(sample, 'iloc') else sample[i]

        breakdown = shap_to_dollar_terms(
            shap_vals, base_log_value, log_pred,
            feature_names, feature_values_row, k_pos=3, k_neg=3)

        results.append({
            'row': i,
            'log_additivity_diff': log_additivity_diff,
            'baseline_dollars':  breakdown['baseline_dollars'],
            'final_dollars':     breakdown['final_pred_dollars'],
            'marginal_sum':      breakdown['marginal_dollar_sum'],
            'marginal_residual': breakdown['marginal_interaction_residual'],
            'proportional_sum':  breakdown['proportional_dollar_sum'],
        })
        print(f"    Row {i}: base=${breakdown['baseline_dollars']:.0f}  "
              f"final=${breakdown['final_pred_dollars']:.0f}  "
              f"marginal_sum=${breakdown['marginal_dollar_sum']:.0f}  "
              f"residual=${breakdown['marginal_interaction_residual']:.0f}  "
              f"log_diff={log_additivity_diff:.5f}")

    return results
