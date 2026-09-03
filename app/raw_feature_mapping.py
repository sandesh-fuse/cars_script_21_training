"""
raw_feature_mapping.py
=======================
Maps engineered features (used by the model) back to the closest raw feature
that the end-user understands.

This serves two purposes:
  1. Hides model internals from the user (e.g., they shouldn't see
     'nav_cond_x_age_bkt' or 'make_freq').
  2. Consolidates multiple engineered features that derive from the same
     raw input — e.g., make_freq + make_tgt_enc both attribute to 'make'.

When called on a list of SHAP feature records, returns a new list where each
record represents a raw feature (or plain-English bucket), with summed
dollar impact and the most-impactful underlying engineered feature kept as
provenance for debugging.
"""

from typing import List, Dict, Any, Optional

# Raw keys never shown to users, however large their SHAP attribution.
#
# 'Specialty Item': specialty vehicles (RV/boat/heavy equipment) are dropped
# from training entirely (see the Specialty-item filter in
# train_save_script21.py / train_save_script17.py), so this column carries no
# vehicle signal by construction. What survived training was a MIX of rows
# where the source export wrote an explicit "false" (encoded 0) and rows where
# it left the field blank (encoded -1, the unknown sentinel) -- roughly 6% vs
# 94%. The models therefore split on it ~278 times (standard) / ~95 (cult)
# purely to separate "field was populated" from "field was blank", i.e. on
# data provenance, not on anything about the car.
#
# It is also unreachable at serving time: `speciality_item` is not a
# PredictRequest field and the schema is extra="forbid", so the column is
# always absent and encodes to NaN -- a third value that appears nowhere in
# training, which every tree routes to the ~6% branch. SHAP consequently
# hands it a real, non-zero dollar impact and it surfaces as a user-facing
# "reason" that is pure train/serve skew.
#
# Hiding it here is a DISPLAY fix only -- the model still uses the feature to
# produce the number. The real fix is retraining with the column excluded.
HIDDEN_RAW_KEYS = frozenset({"Specialty Item"})

from preprocessor import describe_picklist_value, describe_other_damages_value


# ============================================================
# MAPPING TABLE
# Maps engineered feature name -> (raw_feature_key, user_facing_label)
# Exact matches first, then suffix-based fallbacks.
# ============================================================

# Plain-English bucket labels for groups of engineered features that don't
# map back to a single raw input the user controls (macro/cult/geo).
BUCKET_MARKET   = ("__market_trend",      "Used-vehicle market trend")
BUCKET_CULT     = ("__collectible",       "Collectible/cult vehicle status")
BUCKET_LOCATION = ("__location",          "Vehicle location (ZIP)")
BUCKET_UNKNOWNS = ("__unknowns",          "How many condition fields are unknown")
BUCKET_MECH     = ("__mechanical",        "Overall mechanical condition")
BUCKET_TIME     = ("__time_of_sale",      "Time of sale (day/month/season)")
BUCKET_ENGINE   = ("__engine_specs",      "Engine specifications")
# Label, not key: the key is part of the /predict response contract
# (feature_raw_key), so renaming it would break consumers. "Vehicle profile
# group" told users nothing -- this is a K-means grouping of (make, model)
# pairs by shared attributes (age, mileage, condition, body type) and
# explicitly NOT by price, so what it really means to a reader is "cars like
# this one". See _fit_vehicle_profile_clusters() in preprocessor.py.
BUCKET_CLUSTER  = ("__vehicle_profile",   "Similar vehicles (make/model group)")
# Add alongside other BUCKET_ definitions:
BUCKET_DAMAGE = ("__other_damages", "Reported damages")

# Exact matches: engineered_name -> (raw_key, label)
EXACT_MAP: Dict[str, tuple] = {
    # --- Macro (all derived from record_creation_date) ---
    'cpi_at_sale':           BUCKET_MARKET,
    'manheim_at_sale':       BUCKET_MARKET,
    'loan_at_sale':           BUCKET_MARKET,

    # --- Cult (derived from make/model/year lookup) ---
    'is_cult':                BUCKET_CULT,
    'cult_tier_num':          BUCKET_CULT,
    'cult_enthusiast_score':  BUCKET_CULT,
    'cult_uplift_low':        BUCKET_CULT,
    'cult_uplift_mid':        BUCKET_CULT,
    'cult_uplift_high':       BUCKET_CULT,
    'cult_uplift_range':      BUCKET_CULT,
    'cult_last_of_kind':      BUCKET_CULT,
    'cult_liquidity_num':     BUCKET_CULT,
    'cult_volatility_num':    BUCKET_CULT,
    'cult_origsens_num':      BUCKET_CULT,
    'cult_match_strictness':  BUCKET_CULT,
    'cult_x_age':             BUCKET_CULT,
    'cult_x_mileage_bkt':     BUCKET_CULT,
    'culttier_x_age':         BUCKET_CULT,
    'origsens_x_runs':        BUCKET_CULT,

    # --- Geo ---
    'zip_region':              BUCKET_LOCATION,
    'zip_first2':              BUCKET_LOCATION,
    'zip_first3':              BUCKET_LOCATION,
    'zip_first2_freq':         BUCKET_LOCATION,
    'zip_first3_freq':         BUCKET_LOCATION,
    'zip_full_freq':           BUCKET_LOCATION,
    'zip_lat':                 BUCKET_LOCATION,
    'zip_lon':                 BUCKET_LOCATION,
    'lat_x_age':               BUCKET_LOCATION,
    'lon_x_age':               BUCKET_LOCATION,
    'zip_region_x_vehicle_type':  BUCKET_LOCATION,
    'zip_region_x_body_type':     BUCKET_LOCATION,
    'zip_region_x_nav_condition': BUCKET_LOCATION,

    # --- Unknown counters ---
    'n_unknowns':             BUCKET_UNKNOWNS,
    'all_unknown':            BUCKET_UNKNOWNS,
    'any_unknown':            BUCKET_UNKNOWNS,
    'old_and_unknown':        ('age',     'Vehicle age (years)'),

    # --- Mechanical aggregates ---
    'mechanical_severity_sum':   BUCKET_MECH,
    'mechanical_severity_mean':  BUCKET_MECH,
    'mechanical_severity_max':   BUCKET_MECH,

    # --- Severity encodings (numeric form of condition categorical) ---
    'nav_severity':           ('nav_condition',         'Driveability condition'),
    'body_severity':          ('bodypaintcondition',    'Body/paint condition'),
    'engine_severity':        ('enginecondition',       'Engine condition'),
    'trans_severity':         ('transmissioncondition', 'Transmission condition'),
    'tire_severity':          ('tirecondition',         'Tire/wheel condition'),
    'interior_severity':      ('interiorcondition',     'Interior condition'),
    'runs_flag':              ('nav_condition',         'Driveability condition'),

    # --- Bucketing/transformation of single raw feature ---
    'age':                    ('age',          'Vehicle age (years)'),
    'age_sq':                 ('age',          'Vehicle age (years)'),
    'age_bucket':             ('age',          'Vehicle age (years)'),
    'mileage':                ('mileage',      'Mileage (miles driven)'),
    'mileage_bucket':         ('mileage',      'Mileage (miles driven)'),
    'miles_per_year':         ('mileage',      'Mileage (miles driven)'),

    # --- Temporal features (from record_creation_date) ---
    'dow':                    BUCKET_TIME,
    'month':                  BUCKET_TIME,
    'quarter':                BUCKET_TIME,
    'day_of_month':           BUCKET_TIME,
    'day_of_year':            BUCKET_TIME,
    'dow_sin':                BUCKET_TIME,
    'dow_cos':                BUCKET_TIME,
    'month_sin':              BUCKET_TIME,
    'month_cos':              BUCKET_TIME,
    'day_of_month_sin':       BUCKET_TIME,
    'day_of_month_cos':       BUCKET_TIME,
    'day_of_year_sin':        BUCKET_TIME,
    'day_of_year_cos':        BUCKET_TIME,

    # --- Interaction features: pick the DOMINANT raw input (option a) ---
    # First-listed feature in the combo is considered dominant.
    'make_x_body_type':       ('make',         'Make'),
    'make_x_vehicle_type':    ('make',         'Make'),
    'vtype_x_nav_condition':  ('nav_condition','Driveability condition'),
    'body_x_drive':           ('body_type',    'Body type'),
    'condition_combo':        ('nav_condition','Driveability condition'),
    'engine_x_trans_cond':    ('enginecondition','Engine condition'),
    'nav_cond_x_age_bkt':     ('nav_condition','Driveability condition'),
    'runs_x_mileage_bkt':     ('mileage',      'Mileage (miles driven)'),
    'all_cond_combo':         ('nav_condition','Driveability condition'),
    # User-added interactions (dominant input first in name):
    'make_x_age':             ('make',         'Make'),
    'month_x_age':            BUCKET_TIME,
    'mileage_x_age':          ('mileage',      'Mileage (miles driven)'),
    'year_x_dow':             ('year',         'Model year'),
    'zip_region_x_mileage_bkt': BUCKET_LOCATION,
    'quarter_x_make':         ('make',         'Make'),

    # --- Engine specs (parsed from engine_name; grouped under one bucket
    # because they all derive from the same upstream string) ---
    'displacementl':              BUCKET_ENGINE,
    'enginecylinders':            BUCKET_ENGINE,
    'engineconfiguration':        BUCKET_ENGINE,
    'engineconfiguration_freq':   BUCKET_ENGINE,
    'enginehp':                   BUCKET_ENGINE,
    'enginehp_bucket':            BUCKET_ENGINE,
    'valvetraindesign':           BUCKET_ENGINE,
    'valvetraindesign_freq':      BUCKET_ENGINE,
    'displacementl_x_age':        BUCKET_ENGINE,
    'enginecylinders_x_make':     BUCKET_ENGINE,
    'enginecylinders_x_make_freq':BUCKET_ENGINE,
    'enginehp_bkt_x_age_bkt':     BUCKET_ENGINE,
    'enginehp_bkt_x_age_bkt_freq':BUCKET_ENGINE,
    'engineconfig_x_make':        BUCKET_ENGINE,
    'engineconfig_x_make_freq':   BUCKET_ENGINE,

    # --- Vehicle-profile cluster (target-free k-means on vehicle attributes) ---
    'vehicle_profile_cluster_id': BUCKET_CLUSTER,
    'cluster_x_age_bkt':          BUCKET_CLUSTER,
    'cluster_x_age_bkt_freq':     BUCKET_CLUSTER,
    'cluster_x_mileage_bkt':      BUCKET_CLUSTER,
    'cluster_x_mileage_bkt_freq': BUCKET_CLUSTER,

    # Other_damage:
    'has_other_damage':            BUCKET_DAMAGE,
    'n_other_damages':             BUCKET_DAMAGE,
    'other_damages_normalized':    BUCKET_DAMAGE,
    'other_damages_normalized_freq': BUCKET_DAMAGE,
    'has_mold':                    BUCKET_DAMAGE,
    'has_undercarriage_rust':      BUCKET_DAMAGE,
    'has_smog_fail':               BUCKET_DAMAGE,
    'damage_x_age':                BUCKET_DAMAGE,
    'damage_x_mileage_bkt':        BUCKET_DAMAGE,
    'damage_x_mileage_bkt_freq':   BUCKET_DAMAGE,

    # --- Worst-dollar-error-tier interactions (WORST_TIER_FEATURE_COLS) ---
    'unknowns_x_mileage_bkt':       BUCKET_UNKNOWNS,
    'unknowns_x_age_bkt':           BUCKET_UNKNOWNS,
    'mech_severity_x_mileage_bkt':  BUCKET_MECH,
    'cult_x_n_unknowns':            BUCKET_CULT,
    'vtype_x_mileage_bkt':          ('vehicle_type', 'Vehicle type'),

    # --- Condition-x-make interactions (CONDITION_MAKE_FEATURE_COLS) ---
    'runs_x_make':                  ('nav_condition', 'Driveability condition'),
    'mech_severity_x_make':         BUCKET_MECH,
    'all_cond_combo_x_make':        ('nav_condition', 'Driveability condition'),

    # --- Mileage-trust interactions (true_mileage_unknown; NEW_FEATURE_COLS) ---
    'mileage_unknown_x_age':            ('true_mileage_unknown', 'Odometer reading may be inaccurate'),
    'mileage_unknown_x_mileage_bucket': ('true_mileage_unknown', 'Odometer reading may be inaccurate'),
    'mileage_unknown_x_make':           ('true_mileage_unknown', 'Odometer reading may be inaccurate'),
    'mileage_unknown_x_n_unknowns':     ('true_mileage_unknown', 'Odometer reading may be inaccurate'),

    # --- Clean-title interactions (NEW_FEATURE_COLS) ---
    'clean_title_x_age':            ('clean_title', 'Clean title'),
    'clean_title_x_mileage_bucket': ('clean_title', 'Clean title'),
}

# Suffix rules applied if no exact match.
# Engineered feature suffix -> stripped base feature.
# After stripping, we re-lookup the base in HUMAN_READABLE / EXACT_MAP.
SUFFIX_RULES = [
    ('_tgt_enc', ''),     # make_tgt_enc -> make
    ('_freq', ''),        # make_freq -> make
]

# User-facing label for direct raw features. Reuses the same mapping as the
# existing feature_descriptions humanizer so we don't drift.
from app.feature_descriptions import HUMAN_READABLE

# Internal/engineered raw-feature name (as used above, and in preprocessor.py's
# feature_cols_) -> the PredictRequest field name it's actually sourced from
# at request time. preprocessor.py's _basic_clean() renames these new-schema
# request fields onto the legacy internal names before feature engineering
# (see NEW_TO_OLD_SCHEMA_MAP in preprocessor.py), so the raw VALUE to show
# the user (below) must be looked up under the REQUEST's field name, not the
# internal one, wherever they now differ.
INTERNAL_TO_REQUEST_FIELD: Dict[str, str] = {
    'nav_condition':          'vehicle_cond_picklist_id',
    'bodypaintcondition':     'body_paint_cond_picklist_id',
    'enginecondition':        'engine_cond_picklist_id',
    'transmissioncondition':  'transmission_cond_picklist_id',
    'tirecondition':          'tire_cond_picklist_id',
    'interiorcondition':      'interior_cond_picklist_id',
    'other_damages':          'other_damage_pklist_id',
    'nav_color':              'color',
    'body_type':              'vehicle_category',
    'vazipcode':              'zip',
    'vstate_name':            'state_picklist_id',
    'state_province_of_title': 'state_title_picklist',
    'all_clean_notes':        'comment',
    'accessiblefortwotruck':  'accessible_for_tow_truck',
    'locatedatdonationca':    'located_at_donation_c_a',
}


def _resolve_raw(engineered_name: str) -> tuple:
    """Resolve an engineered feature name to (raw_key, user_label).

    Returns ('', engineered_name) if no mapping found — caller can decide
    whether to drop or pass through.
    """
    # 1. Exact match
    if engineered_name in EXACT_MAP:
        return EXACT_MAP[engineered_name]

    # 2. Suffix rules
    for suffix, _replacement in SUFFIX_RULES:
        if engineered_name.endswith(suffix):
            base = engineered_name[: -len(suffix)]
            if base in EXACT_MAP:
                return EXACT_MAP[base]
            # Use HUMAN_READABLE if available, else titleize the base
            label = HUMAN_READABLE.get(base, base.replace('_', ' ').title())
            return (base, label)

    # 3. Already a raw feature?
    if engineered_name in HUMAN_READABLE:
        return (engineered_name, HUMAN_READABLE[engineered_name])

    # 4. Unknown — pass through (don't hide; user might want to know)
    return ('', engineered_name.replace('_', ' '))


def _format_value(val: Any) -> Optional[str]:
    """Pretty-format a raw value for display. None for missing."""
    if val is None:
        return None
    if isinstance(val, float):
        if val != val:  # NaN
            return None
        if val.is_integer():
            return str(int(val))
        return f"{val:g}"
    if isinstance(val, bool):
        return "yes" if val else "no"
    s = str(val).strip()
    return s if s else None


def collapse_engineered_to_raw(
    feature_records: List[Dict],
    request_dict: Dict,
    k_pos: int = 5,
    k_neg: int = 5,
    look_factor: int = 2,
    is_cult: Optional[bool] = None,
) -> Dict[str, List[Dict]]:
    """Collapse engineered SHAP attributions to raw features for user-facing display.

    Parameters
    ----------
    feature_records : full list of SHAP feature records (positives and negatives
                       interleaved is fine; we will partition by sign).
    request_dict : original API request body. Used to pull raw values for display
                   (so we show "Runs & Drives" instead of severity-encoded "1").
    k_pos, k_neg : how many raw groups to return for each side. A negative
                   value means "all groups on this side" (see the
                   normalization at the top of the body).
    look_factor : how many K-multiples of engineered features to consider before
                  collapsing. look_factor=2 means we examine top 2K engineered
                  features per side, then collapse, then return top K raw groups.
    is_cult : the pipeline's already-computed cult/collectible flag for this
              request (Script21Pipeline.predict()'s `is_cult`, derived from
              preprocessor.compute_cult_flag() -- see app/inference_script21.py).
              request_dict can NEVER supply this on its own: it's a value the
              pipeline computes from make/model/year, not a field the caller
              sent, so the BUCKET_CULT group's normal request_dict lookup below
              always misses. Passed in separately purely so the '__collectible'
              group can carry a real value (e.g. "yes"/"no") instead of always
              being null. Script17 has no cult routing and passes None here,
              which keeps its '__collectible' value null as before.

    Returns
    -------
    {'top_positive': [...], 'top_negative': [...]} where each record has:
       feature_raw_key   (e.g. 'make')
       feature_label     (e.g. 'Make' or 'Collectible/cult vehicle status')
       value             (raw value from request_dict, formatted; or None)
       dollar_impact     (summed marginal dollar impact across underlying eng features)
       pct_of_prediction (summed pct_of_prediction_marginal across same)
       n_underlying      (count of engineered features that contributed to this group)
       top_underlying    (name of the single highest-magnitude engineered contributor)
    """
    # Sort all records by signed dollar impact: most positive first
    sorted_recs = sorted(feature_records,
                         key=lambda r: r.get('dollar_impact_marginal', 0.0),
                         reverse=True)

    # A negative k means "return every group on this side" -- the documented
    # meaning of /predict's k_pos=-1 / k_neg=-1. This MUST be normalized
    # before either slice below: both `[:look_factor * k]` and `[:k]` are
    # plain Python slices, and a negative k there silently means "all but
    # the last N" -- the opposite of "all". Left unnormalized, k=-1 trimmed
    # the pool to [:-2] and the result to [:-1], which collapsed to an empty
    # list, so -1 behaved identically to 0 and returned nothing at all.
    #
    # len(sorted_recs) is the count of ENGINEERED records, which is always
    # >= the number of collapsed raw groups, so it disables both slices
    # without needing to know the post-collapse count up front.
    if k_pos < 0:
        k_pos = len(sorted_recs)
    if k_neg < 0:
        k_neg = len(sorted_recs)

    # Drop hidden keys BEFORE the pool slice below, not after the top-K cut:
    # filtering afterwards would silently return K-1 groups whenever a hidden
    # feature landed in the top K, and would also let it consume pool budget.
    sorted_recs = [r for r in sorted_recs
                   if _resolve_raw(r['feature'])[0] not in HIDDEN_RAW_KEYS]

    # raw_key -> the value the model actually saw, for engineered features
    # named EXACTLY like their raw key. Such a feature IS the raw input, so
    # its recorded value is the one to display when request_dict has nothing
    # (e.g. 'age' = record_year - year, computed inside the preprocessor and
    # therefore never present in the request body).
    #
    # Built here, over ALL records, rather than inside _collapse: the positive
    # and negative pools are collapsed independently, so a group appearing on
    # both sides would otherwise only get its value on whichever side happened
    # to contain the self-named record.
    #
    # Restricted to self-named features on purpose: the 'make' group also
    # holds make_freq / make_tgt_enc, whose values are encoded frequencies,
    # NOT the make -- displaying those would be actively wrong.
    self_named_values = {
        r['feature']: r.get('value')
        for r in sorted_recs
        if _resolve_raw(r['feature'])[0] == r['feature']
    }

    # Take top look_factor*K from each end
    pos_pool = [r for r in sorted_recs if r.get('dollar_impact_marginal', 0.0) > 0][:look_factor * k_pos]
    neg_pool = list(reversed([r for r in sorted_recs if r.get('dollar_impact_marginal', 0.0) < 0]))[:look_factor * k_neg]

    def _collapse(pool: List[Dict]) -> List[Dict]:
        # group by raw key
        groups: Dict[str, Dict] = {}
        for r in pool:
            raw_key, label = _resolve_raw(r['feature'])
            # If raw_key is empty, we have no mapping. Show under engineered name
            # but as a "miscellaneous" group keyed by the engineered name itself.
            group_key = raw_key if raw_key else f"__misc__{r['feature']}"

            if group_key not in groups:
                # Try to pull the raw value from the original request body.
                # request_dict uses the CURRENT PredictRequest field names,
                # which differ from raw_key (the internal/engineered name)
                # for fields renamed by the new-schema migration -- fall back
                # to INTERNAL_TO_REQUEST_FIELD's alias when the internal name
                # itself isn't a key in request_dict.
                raw_val = None
                if raw_key:
                    val = None
                    if raw_key in request_dict:
                        val = request_dict[raw_key]
                    else:
                        request_field = INTERNAL_TO_REQUEST_FIELD.get(raw_key)
                        if request_field and request_field in request_dict:
                            val = request_dict[request_field]
                    if val is not None:
                        # Decode a numeric picklist ID to its display name
                        # (e.g. 22968 -> "Runs & Drives") for condition/damage
                        # raw keys -- safe no-op for every other raw_key or
                        # an already-text value (legacy caller).
                        if raw_key == 'other_damages':
                            val = describe_other_damages_value(val)
                        else:
                            val = describe_picklist_value(raw_key, val)
                        raw_val = _format_value(val)
                    elif raw_key == BUCKET_CULT[0] and is_cult is not None:
                        # '__collectible' never maps to a raw request field --
                        # it's a computed flag, not something the caller sent
                        # -- so the lookup above always misses. Surface the
                        # pipeline's actual computed value here instead.
                        raw_val = _format_value(is_cult)
                    elif raw_key in self_named_values:
                        # No request value, but the model computed this input
                        # itself -- show what it actually used. See
                        # self_named_values above.
                        raw_val = _format_value(self_named_values[raw_key])

                groups[group_key] = {
                    'feature_raw_key':  raw_key or r['feature'],
                    'feature_label':    label,
                    'value':            raw_val,
                    'dollar_impact':    0.0,
                    'pct_of_prediction': 0.0,
                    'n_underlying':     0,
                    'top_underlying':   r['feature'],
                    '_top_abs':         0.0,  # internal: track largest contributor
                }
            g = groups[group_key]
            g['dollar_impact']    += r.get('dollar_impact_marginal', 0.0)
            g['pct_of_prediction'] += r.get('pct_of_prediction_marginal', 0.0)
            g['n_underlying']     += 1
            abs_imp = abs(r.get('dollar_impact_marginal', 0.0))
            if abs_imp > g['_top_abs']:
                g['_top_abs']      = abs_imp
                g['top_underlying'] = r['feature']
        # remove internal field
        for g in groups.values():
            g.pop('_top_abs', None)
        return list(groups.values())

    pos_groups = _collapse(pos_pool)
    neg_groups = _collapse(neg_pool)

    # After summing, a group might cross zero — keep only those still on the right side
    pos_groups = [g for g in pos_groups if g['dollar_impact'] > 0]
    neg_groups = [g for g in neg_groups if g['dollar_impact'] < 0]

    pos_groups.sort(key=lambda g: g['dollar_impact'], reverse=True)
    neg_groups.sort(key=lambda g: g['dollar_impact'])

    # Round to cents / hundredths-of-a-percent for the response -- these are
    # sums of several engineered features' marginal contributions, so they
    # otherwise come out with double-precision noise (e.g. 385.6654971455845)
    # that's meaningless past 2 decimal places for a dollar amount or a
    # percentage.
    for g in pos_groups + neg_groups:
        g['dollar_impact'] = round(g['dollar_impact'], 2)
        g['pct_of_prediction'] = round(g['pct_of_prediction'], 2)

    return {
        'top_positive': pos_groups[:k_pos],
        'top_negative': neg_groups[:k_neg],
    }
