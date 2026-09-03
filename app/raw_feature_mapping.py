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

import re
from typing import List, Dict, Any, NamedTuple, Optional

# Shown instead of null wherever a group has no value to report. The /predict
# response is read by non-technical users, and a bare null reads as a bug
# rather than as an explanation.
#
# TWO strings, not one, because the two situations are different facts and
# only one of them is actionable: a reader who sees "Not provided" can go and
# supply the missing field and get a better prediction, whereas
# "Internally calculated value" means the model derived this itself and there
# is nothing for them to send.
VALUE_NOT_PROVIDED = "Not provided"             # the caller didn't send the input
VALUE_INTERNAL = "Internally calculated value"  # derived by the model; no caller input exists

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
# Label, not key (same rule as BUCKET_CLUSTER below): "How many condition
# fields are unknown" reads as "how many were left blank", which is the one
# thing n_unknowns does NOT count -- _is_unknown_condition() in
# preprocessor.py returns False for a missing value, so this counts only
# fields the submitter explicitly answered with that field's "Unknown"
# picklist option.
BUCKET_UNKNOWNS = ("__unknowns",          "Condition fields marked Unknown")
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
    'old_and_unknown':        ('year',         'Model year / vehicle age'),

    # --- Mechanical aggregates ---
    'mechanical_severity_sum':   BUCKET_MECH,
    'mechanical_severity_mean':  BUCKET_MECH,
    'mechanical_severity_max':   BUCKET_MECH,

    # --- Severity encodings (numeric form of condition categorical) ---
    # engine/transmission/tire fold into __mechanical rather than standing
    # alone: mechanical_severity_{sum,mean,max} ARE the aggregate of exactly
    # these three, so a group per component counted the same fault twice --
    # a bad engine showed as BOTH "Overall mechanical condition -$44" AND
    # "Engine condition -$48", inviting a reader to add them. __mechanical's
    # value already spells out each component ("Engine: Major Malfunction,
    # Transmission: Operational, Tires: ..."), so merging de-duplicates
    # without hiding anything. nav/body/interior severity are NOT in
    # mech_parts and keep their own groups.
    #
    # age likewise folds into 'year': age is record_year - year, the SAME
    # fact, and as separate groups they read as two independent reasons
    # (+$458 for the model year AND +$148 for the age of that same car).
    # 'year' wins the key because it is the field the caller actually sends.
    'nav_severity':           ('nav_condition',         'Driveability condition'),
    'body_severity':          ('bodypaintcondition',    'Body/paint condition'),
    'engine_severity':        BUCKET_MECH,
    'trans_severity':         BUCKET_MECH,
    'tire_severity':          BUCKET_MECH,
    # The bare columns too, not just their severity encodings -- otherwise
    # 'enginecondition' still forms its own group via HUMAN_READABLE.
    'enginecondition':        BUCKET_MECH,
    'transmissioncondition':  BUCKET_MECH,
    'tirecondition':          BUCKET_MECH,
    # An explicit 'year' entry so HUMAN_READABLE's plain 'Model year' label
    # doesn't win over the merged label above.
    'year':                   ('year',         'Model year / vehicle age'),
    'interior_severity':      ('interiorcondition',     'Interior condition'),
    'runs_flag':              ('nav_condition',         'Driveability condition'),

    # --- Bucketing/transformation of single raw feature ---
    'age':                    ('year',         'Model year / vehicle age'),
    'age_sq':                 ('year',         'Model year / vehicle age'),
    'age_bucket':             ('year',         'Model year / vehicle age'),
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
    'engine_x_trans_cond':    BUCKET_MECH,
    'nav_cond_x_age_bkt':     ('nav_condition','Driveability condition'),
    'runs_x_mileage_bkt':     ('mileage',      'Mileage (miles driven)'),
    'all_cond_combo':         ('nav_condition','Driveability condition'),
    # User-added interactions (dominant input first in name):
    'make_x_age':             ('make',         'Make'),
    'month_x_age':            BUCKET_TIME,
    'mileage_x_age':          ('mileage',      'Mileage (miles driven)'),
    'year_x_dow':             ('year',         'Model year / vehicle age'),
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


def _request_value(request_dict: Dict, internal_key: str) -> Any:
    """The value the caller sent for an INTERNAL raw-feature name, or None.

    request_dict uses the CURRENT PredictRequest field names, which differ
    from the internal/engineered names for every field the new-schema
    migration renamed -- so look under the internal name first, then fall
    back to INTERNAL_TO_REQUEST_FIELD's alias.

    A key that IS present but holds None short-circuits to None without
    consulting the alias. That is deliberate and matches /predict: main.py
    dumps the body with exclude_none=False, so a field the caller omitted is
    present-and-None rather than absent.
    """
    if internal_key in request_dict:
        return request_dict[internal_key]
    request_field = INTERNAL_TO_REQUEST_FIELD.get(internal_key)
    return request_dict.get(request_field) if request_field else None


def _is_caller_field(request_dict: Dict, internal_key: str) -> bool:
    """True when internal_key names a field the CALLER can supply.

    Presence of the key is what matters, not its value: /predict dumps the
    body with exclude_none=False, so every declared PredictRequest field is
    present even when the caller left it out -- present-and-None means "the
    caller could have sent this and didn't".

    Used to keep the self-named fallback in _collapse() away from fields the
    caller owns. 'age' is computed inside the preprocessor and appears in no
    request body, so it legitimately falls back to the value the model saw;
    'trim'/'body_subtype'/'body_type' are request fields, and falling back
    for those surfaced the int-map's internal "unknown" sentinel (-1) as if
    it were the caller's answer.
    """
    if internal_key in request_dict:
        return True
    alias = INTERNAL_TO_REQUEST_FIELD.get(internal_key)
    return bool(alias) and alias in request_dict


# ============================================================
# BUCKET VALUE RESOLVERS
# ============================================================
# A BUCKET_* group is keyed by a synthetic sentinel ('__mechanical', ...),
# never by a real request field, so the request_dict lookup in _collapse()
# below can only ever MISS for one -- which is why every bucket rendered
# `value: null` no matter what the caller sent. Each resolver here rebuilds a
# readable value from whatever its bucket is actually made of: the request
# fields that feed it, or the engineered value the model saw.
#
# A table, not a chain of `elif raw_key == ...` branches in _collapse():
# '__collectible' was originally done that way, and there are five now, each
# with a different source.
#
# Resolvers return anything _format_value() understands; the CALL SITE
# formats. Returning None means "no honest value", and _collapse() then
# substitutes VALUE_NOT_PROVIDED.


class _BucketContext(NamedTuple):
    """Everything a bucket resolver is allowed to read.

    Bundled rather than passed as three positional args so adding a fourth
    source later doesn't touch every resolver's signature.
    """
    request_dict:   Dict            # the original API request body
    feature_values: Dict[str, Any]  # engineered feature name -> value the model saw
    is_cult:        Optional[bool]  # pipeline-computed cult flag (script21 only)


# (internal raw-feature name, short label) for the three condition fields the
# '__mechanical' bucket is ACTUALLY built from. Not the six condition fields:
# mechanical_severity_{sum,mean,max} aggregate engine_severity +
# trans_severity + tire_severity only (see preprocessor.py's _engineer), so
# interior/body/driveability are deliberately absent -- each already has its
# own raw group.
_MECH_PARTS = (
    ('enginecondition',       'Engine'),
    ('transmissioncondition', 'Transmission'),
    ('tirecondition',         'Tires'),
)

# The six condition fields n_unknowns counts over (UNKNOWN_FLAG_COLS in
# preprocessor.py), as PredictRequest field names. Used only to tell "caller
# answered all six, none Unknown" apart from "caller sent none of them" --
# n_unknowns is 0 in both cases.
_CONDITION_REQUEST_FIELDS = (
    'vehicle_cond_picklist_id',
    'body_paint_cond_picklist_id',
    'engine_cond_picklist_id',
    'transmission_cond_picklist_id',
    'tire_cond_picklist_id',
    'interior_cond_picklist_id',
)
_N_CONDITION_FIELDS = len(_CONDITION_REQUEST_FIELDS)

# The same digit-extract the preprocessor applies to vazipcode before it
# buckets the ZIP.
_ZIP_DIGITS_RE = re.compile(r'(\d{1,5})')


def _value_collectible(ctx: _BucketContext) -> Any:
    """'__collectible' -> the pipeline's computed cult flag ("yes"/"no").

    Never a request field: it's derived from make/model/year by
    compute_cult_flag(), so only the pipeline can supply it. script17 has no
    cult routing and passes is_cult=None -- reported as VALUE_INTERNAL rather
    than "Not provided", which would wrongly imply the caller could send it.
    """
    return ctx.is_cult if ctx.is_cult is not None else VALUE_INTERNAL


def _value_mechanical(ctx: _BucketContext) -> Optional[str]:
    """'__mechanical' -> "Engine: Operational, Transmission: Operational,
    Tires: 1 or More Tires are Flat*".

    Rebuilt from the REQUEST's picklist IDs, never from the severity numbers
    the model saw: the *_SEV maps are many-to-one (ENGINE_SEV sends both
    23055 "Rebuilt/Replaced" and 23056 "Minor Issues / Still Functional" to
    1), so a severity integer cannot be turned back into the words the caller
    picked. The picklist ID is the only lossless route back to text.

    Only the parts the caller actually sent are listed -- a request carrying
    just an engine condition reads "Engine: Operational", not
    "Engine: Operational, Transmission: None, Tires: None".

    A part whose ID doesn't decode is DROPPED rather than shown:
    describe_picklist_value() is speculative-safe and returns its input
    unchanged when it can't decode, so an unrecognised ID would otherwise
    surface to the reader as the bare number ("Engine: 99999").
    """
    parts = []
    for internal_key, short_label in _MECH_PARTS:
        raw = _request_value(ctx.request_dict, internal_key)
        if raw is None:
            continue
        decoded = describe_picklist_value(internal_key, raw)
        # Only a str means it actually decoded -- an undecodable ID comes
        # back as the original number.
        if not isinstance(decoded, str):
            continue
        text = _format_value(decoded)
        if text is not None:
            parts.append(f"{short_label}: {text}")
    return ', '.join(parts) if parts else None


def _value_unknowns(ctx: _BucketContext) -> Optional[str]:
    """'__unknowns' -> "2 of 6 condition fields marked Unknown".

    Read from the n_unknowns SHAP record -- the number the model actually
    used -- rather than recounted from request_dict. Recounting would mean
    reimplementing preprocessor._is_unknown_condition()'s exact predicate
    (a different "Unknown" picklist ID per column, plus the legacy text form)
    in a second place, free to drift from it.

    Note what that predicate does NOT count: _is_unknown_condition() returns
    False for a MISSING value, so this is the number of condition fields the
    submitter explicitly ANSWERED "Unknown" -- fields left blank are not
    included. Hence "marked Unknown", not "are unknown".

    n_unknowns == 0 is therefore ambiguous on its own: it means either "all
    six answered, none Unknown" or "the caller sent none of them". Only the
    first deserves the "0 of 6" phrasing -- claiming it for a request that
    answered nothing would imply questions were asked and cleared.
    """
    if not any(ctx.request_dict.get(f) is not None
               for f in _CONDITION_REQUEST_FIELDS):
        return None  # -> "Not provided": the caller answered none of them

    n = ctx.feature_values.get('n_unknowns')
    try:
        n = int(n)
    except (TypeError, ValueError):
        # Only reachable when n_unknowns has exactly 0.0 SHAP impact and so
        # isn't among the records at all (shap_to_dollar_terms keeps strictly
        # >0 / <0), and the group was formed by all_unknown / any_unknown /
        # unknowns_x_* instead.
        return None
    if n <= 0:
        return f"No condition fields marked Unknown (0 of {_N_CONDITION_FIELDS})"
    return f"{n} of {_N_CONDITION_FIELDS} condition fields marked Unknown"


def _value_other_damages(ctx: _BucketContext) -> Optional[str]:
    """'__other_damages' -> "Mold, Flood Damage", or "None reported".

    other_damage_pklist_id is a multi-select, so the value can be a LIST of
    picklist IDs; describe_other_damages_value() handles that as well as the
    scalar and legacy-text forms.

    "None reported" -- NOT "Not provided" -- for an absent or empty field,
    because that is exactly what the model scored: a missing field reads as
    zero damage tokens (has_other_damage=0). "We know there is no damage" is
    a real fact and a different claim from "we weren't told"; this group most
    often surfaces with a POSITIVE impact on a clean car, where it is the
    entire explanation.
    """
    raw = _request_value(ctx.request_dict, 'other_damages')
    # An empty list is "the multi-select was opened and nothing was picked".
    # describe_other_damages_value() passes it through untouched (no labels
    # to join) and _format_value() would then str() it into a literal "[]".
    if isinstance(raw, list) and not raw:
        raw = None
    if raw is None:
        return "None reported"
    decoded = describe_other_damages_value(raw)
    # As in _value_mechanical: a non-str means nothing decoded, and the bare
    # ID must not reach the reader.
    return _format_value(decoded) if isinstance(decoded, str) else None


def _value_location(ctx: _BucketContext) -> Optional[str]:
    """'__location' -> "ZIP 01234".

    Normalized with the SAME digit-extract-then-zero-pad the preprocessor
    applies before bucketing the ZIP, so we show the ZIP the model actually
    used. The padding is not cosmetic: PredictRequest.zip is coerced to a
    float, so a Massachusetts "01234" arrives as 1234.0 and would otherwise
    display as "ZIP 1234" -- a real but DIFFERENT ZIP -- while
    zip_first2/zip_first3/zip_full were all built from the padded "01234".

    Zero is treated as absent: it is a placeholder, and zero-padding would
    turn it into the plausible-looking "ZIP 00000".
    """
    raw = _request_value(ctx.request_dict, 'vazipcode')
    if raw is None:
        return None
    try:
        if float(raw) == 0:
            return None
    except (TypeError, ValueError):
        pass  # non-numeric text -- the regex below decides
    match = _ZIP_DIGITS_RE.search(str(raw))  # NaN -> "nan" -> no match
    if match is None:
        return None
    return f"ZIP {match.group(1).zfill(5)}"


# Bucket sentinel -> resolver.
BUCKET_VALUE_RESOLVERS = {
    BUCKET_CULT[0]:     _value_collectible,
    BUCKET_MECH[0]:     _value_mechanical,
    BUCKET_UNKNOWNS[0]: _value_unknowns,
    BUCKET_DAMAGE[0]:   _value_other_damages,
    BUCKET_LOCATION[0]: _value_location,
}

# Buckets with no per-vehicle value BY NATURE -- deliberately absent from the
# table above, and reported as VALUE_INTERNAL rather than "Not provided"
# because nothing was ever expected from the caller:
#   __market_trend, __time_of_sale -- derived only from the submission date
#       (CPI / Manheim / auto-loan index; day/month/season). Identical for
#       every vehicle submitted that day, so there is no per-vehicle value.
#   __vehicle_profile -- an opaque k-means cluster id (0-11, and -1 for most
#       vehicles); the number means nothing to a reader, and the label
#       already says "cars like this one".
#   __engine_specs -- dead at serving time: use_dataone is False in the
#       deployed artifacts, so none of its features exist.
INTERNAL_ONLY_BUCKETS = frozenset({
    BUCKET_MARKET[0], BUCKET_TIME[0], BUCKET_CLUSTER[0], BUCKET_ENGINE[0],
})


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

    # Engineered feature name -> the value the model actually saw, for EVERY
    # record. Read by BUCKET_VALUE_RESOLVERS for inputs the caller never sent
    # and that aren't self-named either -- 'n_unknowns' is the only one today.
    #
    # Built here, over ALL records, for the same reason self_named_values is:
    # the positive and negative pools are collapsed INDEPENDENTLY, so a map
    # built per-pool would give a group appearing on both sides its value on
    # only one of them.
    #
    # This does NOT subsume self_named_values. That map is looked up BY
    # raw_key, which is why it must stay restricted to self-named features (a
    # group's other members carry encoded values). This one is only ever read
    # by an explicit feature name.
    feature_values = {r['feature']: r.get('value') for r in sorted_recs}
    bucket_ctx = _BucketContext(request_dict=request_dict,
                                feature_values=feature_values,
                                is_cult=is_cult)

    # ALL records collapse together into ONE set of groups, rather than the
    # positive and negative halves being pooled and collapsed separately.
    #
    # Two bugs died with that split. First, a raw feature whose engineered
    # members pulled in both directions appeared in BOTH lists -- 'make' came
    # back as +$4.51 AND -$152.54 for the same car, two contradictory answers
    # to "is this make good or bad?" when the honest answer is the net,
    # -$148. Second, `[:look_factor * k]` truncated the pool BEFORE the
    # collapse, and since collapsing SUMS a group's members, a group's dollar
    # amount changed with k -- a top-5 result was not the first five rows of
    # a top-20 result. Summing every member exactly once fixes both: the
    # numbers are now stable whatever k is, and k only decides how many rows
    # come back.
    #
    # look_factor is kept in the signature for call-site compatibility but is
    # no longer used; there is no pool to widen when nothing is truncated.

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
                    val = _request_value(request_dict, raw_key)
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
                    elif raw_key in BUCKET_VALUE_RESOLVERS:
                        # A BUCKET_* sentinel is never a request field, so the
                        # lookup above ALWAYS misses for one. Rebuild a
                        # readable value from what the bucket is really made
                        # of -- see BUCKET_VALUE_RESOLVERS.
                        raw_val = _format_value(
                            BUCKET_VALUE_RESOLVERS[raw_key](bucket_ctx))
                    elif (raw_key in self_named_values
                          and not _is_caller_field(request_dict, raw_key)):
                        # No request value AND not a field the caller could
                        # have sent, so the model must have computed this
                        # input itself -- show what it actually used ('age').
                        #
                        # The _is_caller_field guard is essential: without it
                        # a declared request field left null fell through to
                        # here and rendered the int-map's internal "unknown"
                        # sentinel, so `"trim": null` came back as
                        # `"value": "-1"`. A field the caller owns and left
                        # blank is "Not provided", never an encoding.
                        raw_val = _format_value(self_named_values[raw_key])

                # Never emit null: a bare null reads as a bug to a
                # non-technical reader. Which sentinel depends on WHY there is
                # no value -- see VALUE_NOT_PROVIDED / VALUE_INTERNAL.
                if raw_val is None:
                    raw_val = (VALUE_INTERNAL if raw_key in INTERNAL_ONLY_BUCKETS
                               else VALUE_NOT_PROVIDED)

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

    all_groups = _collapse(sorted_recs)

    # Split by the sign of the NET impact -- a group lands in exactly one list.
    pos_groups = [g for g in all_groups if g['dollar_impact'] > 0]
    neg_groups = [g for g in all_groups if g['dollar_impact'] < 0]

    # Sort on the FULL-PRECISION sums, before the rounding below: rounding
    # first would collapse near-ties to the same 2dp value and order them
    # arbitrarily.
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

    # Drop anything not on the right side of zero, AFTER rounding. Two things
    # get caught here, and the order matters:
    #   - a group whose members' contributions cancelled out and crossed zero
    #     while being summed;
    #   - a group whose impact is real but smaller than a cent, e.g.
    #     0.0004 -- it passes an unrounded `> 0` test and then renders as
    #     "dollar_impact": 0.0, which reads to a user as "this changed
    #     nothing, so why is it listed?".
    # Rounding before the test folds the second case into the first: -0.0 < 0
    # is False in Python, so a negative sub-cent value is dropped too.
    pos_groups = [g for g in pos_groups if g['dollar_impact'] > 0]
    neg_groups = [g for g in neg_groups if g['dollar_impact'] < 0]

    return {
        'top_positive': pos_groups[:k_pos],
        'top_negative': neg_groups[:k_neg],
    }
