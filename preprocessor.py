# ============================================================
# Shared preprocessor module (Script 17 & Script 21 compatible)
# Used by:
#   - train_save_script17.py
#   - train_save_script21.py
#   - app/inference_script17.py
#   - app/inference_script21.py
# ============================================================
import re
import warnings
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Suppress pandas DataFrame-fragmentation warnings. These fire because we add
# many columns one-by-one in _basic_clean/_engineer/_apply_freq/etc. Refactoring
# to pd.concat would touch ~5 functions for a sub-second saving per fit; the
# bottleneck is elsewhere (SHAP, target encoding, zip lookup). Silenced at
# import time so callers don't get spammed.
warnings.filterwarnings("ignore", category=PerformanceWarning)

# ---- Constants ----
TARGET_COL = "salevalue"
TIME_COL   = "record_creation_date"
BASE_YEAR  = 2026
BASE_MONTH = 4

# Raw feature set extracted from commit 20c8c17 ("added new features
# true_mileage_unknown, clean_title (bool), gvm_range, tonnage, engine_type").
# Single source of truth so train_save_script21.py's --enable-new-features
# flag can gate inclusion via SaleValuePreprocessor(extra_drop_cols=...) —
# same mechanism, same intent as script21's existing --use-dataone/
# DATAONE_FEATURES pattern, just for a different feature set. Default
# behavior (none enabled) is byte-identical to the 2406a7a checkpoint.
NEW_FEATURE_COLS = ['true_mileage_unknown', 'clean_title', 'gvm_range', 'tonnage', 'engine_type']

# Interaction features added to target the $2.5K-10K worst-dollar-error tier
# (see worst_case_analysis_2500_10000/tier_band_feature_correlation.md):
# systematic underprediction there correlates with condition-unknown counts,
# mechanical severity, and cult routing not being interacted with
# mileage/age, none of which existed before. Gated by
# SaleValuePreprocessor(enable_worst_tier_features=...) / script21's
# --enable-worst-tier-features so they can be ablated the same way
# NEW_FEATURE_COLS/--enable-new-features is. Unlike NEW_FEATURE_COLS these
# aren't raw input columns to drop — they're derived from always-on
# engineered columns (n_unknowns, mileage_bucket, is_cult, ...) — so the
# gate is a plain boolean/name-list, not an extra_drop_cols entry.
WORST_TIER_FEATURE_COLS = [
    'unknowns_x_mileage_bkt', 'unknowns_x_age_bkt', 'mech_severity_x_mileage_bkt',
    'cult_x_n_unknowns', 'vtype_x_mileage_bkt', 'mileage_unknown_x_n_unknowns',
]

# Interaction features targeting the $100-2000 tier instead (most vehicle
# volume): the correlation analysis there (worst_case_analysis_100_2000/)
# showed a DIFFERENT pattern than $2.5K-10K -- worst-overpredicted rows
# have BETTER-looking condition (more "Operational"/"Runs & Drives", less
# damage reported) than the rest of the band, not missing data. The model
# applies one global "good condition -> higher price" rule regardless of
# make; these features let it learn a make-specific version of that rule
# instead. Gated the same way as WORST_TIER_FEATURE_COLS (see _cmf()).
CONDITION_MAKE_FEATURE_COLS = [
    'runs_x_make', 'mech_severity_x_make', 'all_cond_combo_x_make',
]

# ============================================================
# MACRO DATA
# ============================================================
CPI_ANNUAL_CONF = {
    2005: 195.3, 2006: 201.6, 2007: 207.342, 2008: 215.303, 2009: 214.537,
    2010: 218.056, 2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736,
    2015: 237.017, 2016: 240.007, 2017: 245.120, 2018: 251.107, 2019: 255.657,
    2020: 258.811, 2021: 270.970, 2022: 292.655, 2023: 304.702, 2024: 313.689,
    2025: 322.000, 2026: 329.000,
}
MANHEIM_MONTHLY = {
    (2026, 4): 211.9, (2026, 3): 215.3, (2026, 2): 212.0, (2026, 1): 209.2,
    (2025, 12): 206.0, (2025, 6): 208.0, (2024, 6): 197.0, (2023, 6): 217.0,
    (2022, 6): 219.0, (2022, 1): 236.3, (2021, 12): 224.0, (2021, 6): 196.0,
    (2020, 6): 145.0, (2019, 6): 138.0, (2018, 6): 133.0, (2017, 6): 129.0,
    (2016, 6): 124.0, (2015, 6): 122.0, (2014, 6): 122.0, (2013, 6): 122.0,
    (2012, 6): 122.0, (2011, 6): 124.0, (2010, 6): 118.0, (2009, 6): 102.0,
    (2008, 6): 110.0, (2007, 6): 115.0, (2006, 6): 115.0, (2005, 6): 110.0,
}
AUTO_LOAN_QUARTERLY_CONF = {
    (2026, 4): 7.44, (2026, 2): 7.30, (2025, 11): 7.22, (2025, 8): 7.64,
    (2024, 11): 7.71, (2024, 8): 7.91, (2024, 5): 8.20, (2024, 2): 8.40,
    (2023, 11): 7.71, (2023, 8): 7.61, (2023, 5): 7.30, (2023, 2): 6.84,
    (2022, 11): 6.61, (2022, 8): 5.50, (2022, 5): 4.55, (2022, 2): 4.52,
    (2021, 11): 4.52, (2021, 8): 4.31, (2021, 5): 4.50, (2021, 2): 4.66,
    (2020, 11): 4.98, (2020, 8): 4.98, (2020, 5): 5.21, (2020, 2): 5.18,
    (2019, 11): 5.29, (2019, 8): 5.31, (2019, 5): 5.45, (2019, 2): 5.27,
    (2018, 11): 5.51, (2018, 8): 5.10, (2018, 5): 4.74, (2018, 2): 4.61,
}

def build_monthly_series(annual_dict=None, monthly_dict=None, year_range=(2005, 2026)):
    points = []
    if annual_dict:
        for y, v in annual_dict.items(): points.append((y * 12 + 6, v))
    if monthly_dict:
        for (y, m), v in monthly_dict.items(): points.append((y * 12 + m, v))
    months = [(y, m) for y in range(year_range[0], year_range[1] + 1) for m in range(1, 13)]
    idx = [y * 12 + m for y, m in months]
    s = pd.Series(np.nan, index=idx, dtype=float)
    for k, v in sorted(set(points)): s[k] = v
    s = s.interpolate(method='linear', limit_direction='both')
    return dict(zip(months, s.values))

CPI_MONTHLY          = build_monthly_series(annual_dict=CPI_ANNUAL_CONF)
MANHEIM_MONTHLY_FULL = build_monthly_series(monthly_dict=MANHEIM_MONTHLY)
LOAN_MONTHLY         = build_monthly_series(monthly_dict=AUTO_LOAN_QUARTERLY_CONF)
CPI_BASE     = CPI_MONTHLY[(BASE_YEAR, BASE_MONTH)]
MANHEIM_BASE = MANHEIM_MONTHLY_FULL[(BASE_YEAR, BASE_MONTH)]
LOAN_BASE    = LOAN_MONTHLY[(BASE_YEAR, BASE_MONTH)]

# ============================================================
# CULT LOOKUP
# ============================================================
TIER_NUM     = {'tier a': 3, 'tier b': 2, 'tier c': 1}
LIQUIDITY_N  = {'low': 0, 'medium': 1, 'high': 2, 'very high': 3}
VOLATILITY_N = {'low': 0, 'medium': 1, 'high': 2}
ORIGSENS_N   = {'low': 0, 'medium': 1, 'high': 2, 'very high': 3}
YESNO        = {'yes': 1, 'no': 0}

def parse_year_range(s):
    s = str(s).strip()
    m = re.match(r'(\d{4})\s*-\s*(\d{4})', s)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.match(r'(\d{4})', s)
    if m: y = int(m.group(1)); return y, y
    return None, None

def model_variants(model_str):
    s = str(model_str).strip().lower()
    variants = [(s, 1)]
    if '/' in s:
        for part in s.split('/'):
            p = part.strip()
            if p and p != s: variants.append((p, 2))
    first_word = s.split()[0] if s.split() else s
    if first_word != s and len(first_word) >= 3:
        variants.append((first_word, 3))
    seen = {}
    for v, st in variants:
        if v not in seen or st < seen[v]: seen[v] = st
    return list(seen.items())

def build_cult_lookup(cult_df):
    lookup = {}
    for _, r in cult_df.iterrows():
        make_n = str(r['Make']).strip().lower()
        ymin, ymax = parse_year_range(r['Years'])
        if ymin is None: continue
        feat = {
            'cult_tier_num'         : TIER_NUM.get(str(r['Tier']).lower(), 0),
            'cult_enthusiast_score' : float(r['Enthusiast Score']) if pd.notna(r['Enthusiast Score']) else np.nan,
            'cult_uplift_low'       : float(r['Uplift Low %'])  if pd.notna(r['Uplift Low %'])  else np.nan,
            'cult_uplift_high'      : float(r['Uplift High %']) if pd.notna(r['Uplift High %']) else np.nan,
            'cult_last_of_kind'     : YESNO.get(str(r['Last of Kind Flag']).lower(), 0),
            'cult_liquidity_num'    : LIQUIDITY_N.get(str(r['Liquidity']).lower(), -1),
            'cult_volatility_num'   : VOLATILITY_N.get(str(r['Volatility']).lower(), -1),
            'cult_origsens_num'     : ORIGSENS_N.get(str(r['Originality Sensitivity']).lower(), -1),
        }
        for variant, strictness in model_variants(r['Model']):
            lookup.setdefault((make_n, variant), []).append((ymin, ymax, strictness, feat))
    return lookup

def lookup_cult(make_n, model_n, year, cult_lookup):
    if pd.isna(year): return None
    year = int(year)
    candidates = cult_lookup.get((make_n, model_n), [])
    if not candidates and model_n:
        fw = model_n.split()[0] if model_n.split() else model_n
        candidates = cult_lookup.get((make_n, fw), [])
    if not candidates: return None
    matches = [(ymin, ymax, st, feat) for (ymin, ymax, st, feat) in candidates
               if ymin <= year <= ymax]
    if not matches: return None
    best = min(matches, key=lambda x: x[2])
    return {'strictness': best[2], **best[3]}

def compute_cult_flag(df_subset, cult_lookup):
    flags = []
    for mk, md, yr in zip(df_subset['make'], df_subset['model'], df_subset['year']):
        mk_s = str(mk).strip().lower() if pd.notna(mk) else ''
        md_s = str(md).strip().lower() if pd.notna(md) else ''
        flags.append(lookup_cult(mk_s, md_s, yr, cult_lookup) is not None)
    return np.array(flags, dtype=bool)

# ============================================================
# ZIP LOOKUP (lazy import of pgeocode)
# ============================================================
def build_zip_lookup(zip_series):
    import pgeocode
    nomi = pgeocode.Nominatim('us')
    z = (zip_series.dropna().astype(str)
                  .str.extract(r'(\d{1,5})')[0].str.zfill(5))
    uniq = z.dropna().unique().tolist()
    print(f"Looking up {len(uniq):,} unique ZIPs via pgeocode...")
    out = nomi.query_postal_code(uniq)
    out = out[['postal_code','latitude','longitude']].set_index('postal_code')
    return out['latitude'].to_dict(), out['longitude'].to_dict()

# ============================================================
# NEW (upstream DB) SCHEMA -> LEGACY (this file's) SCHEMA
# ============================================================
# Owned here (not in a separate schema_adapter.py) so this module is
# self-sufficient: any caller -- train_save_script21.py, app/inference_*.py,
# a live request dict -- can hand this class a DataFrame using EITHER the
# new upstream column names OR these legacy names directly, with no
# translation-layer dependency of its own. Applied automatically as the
# first step of _basic_clean() (see below), so it's transparent to every
# caller; also exported for callers (train_save_script21.py) that need the
# rename applied BEFORE their own pre-preprocessing logic runs (row
# filtering on salevalue/record_creation_date, zip lookup, etc.).
#
# Condition/damage/state columns are keyed off the numeric picklist ID
# (source of truth), not the resolved *_name text column -- see the
# NAV_CONDITION_SEV/BODY_SEV/etc. dicts below, which carry both text-label
# AND numeric-ID keys, so a value under either representation resolves to
# the same severity.
NEW_TO_OLD_SCHEMA_MAP = {
    # --- Core target & time ---
    "sale_value": "salevalue",
    "creation_datetime": "record_creation_date",

    # --- Condition & visuals ---
    "vehicle_cond_picklist_id": "nav_condition",
    "engine_cond_picklist_id": "enginecondition",
    "transmission_cond_picklist_id": "transmissioncondition",
    "body_paint_cond_picklist_id": "bodypaintcondition",
    "interior_cond_picklist_id": "interiorcondition",
    "tire_cond_picklist_id": "tirecondition",
    "color": "nav_color",

    # --- Engine & drivetrain (DataOne-sourced; see train_save_script21.py's
    # DATAONE_FEATURES / --use-dataone) ---
    "engines_name": "engine_name",
    "transmissions_name": "transmission_name",
    "ice_displacement": "displacementl",
    "ice_cylinders": "enginecylinders",
    "ice_block_type": "engineconfiguration",
    "ice_max_hp": "enginehp",

    # --- Core attributes & categoricals ---
    "vehicle_category": "body_type",
    "accessible_for_tow_truck": "accessiblefortwotruck",
    "located_at_donation_c_a": "locatedatdonationca",
    "speciality_item": "Specialty Item",
    "state_title_picklist": "state_province_of_title",
    "us_styles": "us_style_name",
    "state_picklist_id": "vstate_name",
    "zip": "vazipcode",
    "vin_hin_no": "vin",                    # currently inert: dropped outright by DROP_COLS_NO_ZIP
    "comment": "all_clean_notes",              # currently inert: not read anywhere below

    # --- Damage ---
    # Decoded (scalar ID, or numeric IDs inside a live-request JSON list) via
    # OTHER_DAMAGE_ID_TO_LABEL inside _parse_other_damages()/_extract_tokens()
    # below, so this reduces to the exact same normalized tokens the *_name
    # text used to produce.
    "other_damage_pklist_id": "other_damages",
}

# `vehicle_category` -> `body_type` and `body_type` -> `oem_body_style` together mean the
# raw incoming `body_type` column is repurposed as `oem_body_style`, NOT dropped — order of
# operations in .rename() doesn't matter here since pandas renames off the ORIGINAL column
# names simultaneously, not sequentially, so no chaining collision occurs.


def _normalize_date_value(value):
    """Parse one timestamp (ISO8601 or legacy format, tz-aware or naive) and
    strip any timezone so it's safe to feed into columns that get bulk-parsed
    downstream (pd.to_datetime is called on the whole column later).

    Stripping tz here matters specifically because a training CSV spanning
    the schema cutover will likely mix legacy tz-naive timestamps with new
    tz-aware ISO8601 ones in the same column — pandas can outright fail to
    parse a genuinely mixed-tz column even with errors='coerce'.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return value  # leave unparseable values alone; downstream errors='coerce' will drop them
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.isoformat()


def map_raw_features_to_legacy(df):
    """Rename a DataFrame's columns from the new DB schema to the legacy
    schema this module's feature engineering is written against.

    Safe to call on data that's already legacy-schema (or a mix of columns
    from both, or already-renamed) — only columns present in
    NEW_TO_OLD_SCHEMA_MAP are touched, everything else passes through
    untouched, and it's a no-op the second time it's called on the same
    frame. Called automatically inside _basic_clean() so any caller can
    hand either schema straight to fit()/transform(); also safe to call
    explicitly and early (train_save_script21.py does, since its own
    pre-preprocessing row-filtering logic reads salevalue/
    record_creation_date/vazipcode/Specialty Item before fit()/transform()
    ever run).
    """
    # Detect genuine collisions: two DIFFERENT incoming columns that would land
    # on the same final name after renaming. Chained renames (e.g. incoming
    # 'vehicle_category' -> 'body_type' AND incoming 'body_type' -> 'oem_body_style')
    # are NOT collisions — pandas .rename() maps off the original column names
    # simultaneously, so both resolve to distinct final names. Only flag it when
    # the resulting column list would actually have a duplicate.
    final_names = [NEW_TO_OLD_SCHEMA_MAP.get(c, c) for c in df.columns]
    seen, dupes = set(), set()
    for final in final_names:
        if final in seen:
            dupes.add(final)
        seen.add(final)
    if dupes:
        offenders = {
            final: [orig for orig, f in zip(df.columns, final_names) if f == final]
            for final in dupes
        }
        raise ValueError(
            f"map_raw_features_to_legacy: multiple incoming columns would collide onto "
            f"the same legacy column name after renaming — refusing to guess which is "
            f"authoritative: {offenders}. Drop or rename one of each group first."
        )

    df = df.rename(columns=NEW_TO_OLD_SCHEMA_MAP)

    if "record_creation_date" in df.columns:
        df["record_creation_date"] = df["record_creation_date"].apply(_normalize_date_value)

    return df


def map_raw_features_to_legacy_record(record: dict) -> dict:
    """Rename a single request dict's keys from the new schema to the legacy
    schema. Same mapping as map_raw_features_to_legacy, for a single-row
    dict rather than a DataFrame (not currently needed by app/inference_*.py,
    which builds a DataFrame directly and relies on _basic_clean()'s
    automatic call to map_raw_features_to_legacy() instead — kept for
    parity/for any future single-dict caller)."""
    renamed = {NEW_TO_OLD_SCHEMA_MAP.get(k, k): v for k, v in record.items()}
    if "record_creation_date" in renamed:
        renamed["record_creation_date"] = _normalize_date_value(renamed["record_creation_date"])
    return renamed


# ============================================================
# SEVERITY MAPS
# ============================================================
def _norm_key(s): return s.strip().lower() if isinstance(s, str) else s
def _norm_dict(d): return {_norm_key(k): v for k, v in d.items()}


# Each dict below carries BOTH the legacy text-label keys (script17 / current
# live inference, which still send resolved *_name text) AND the matching
# new-schema numeric picklist-ID keys (script21 training, post schema_adapter
# rename) -- same severity value either way, since the ID keys were built by
# combining the confirmed ID<->name correspondence with the text severities
# already below, NOT from raw ID magnitude (the IDs are not ordinal).
# _norm_dict/_norm_key leave non-string keys untouched, so int keys coexist
# safely with the lowercased string keys in the same dict.
NAV_CONDITION_SEV = _norm_dict({
    'Runs & Drives': 0, 'Runs & Moves / Don\u2019t Drive': 1,
    'Runs / Doesn\u2019t Move': 2, 'Cranks, won\u2019t start': 3,
    'Doesn\u2019t Run / Can be Moved': 4, 'Doesn\u2019t Run / Doesn\u2019t Move': 5, 'Unknown': 6,
    # vehicle_cond_picklist_id (new schema)
    22968: 0, 22971: 1, 22969: 2, 22974: 3, 22973: 4, 22972: 5, 22970: 6,
})
BODY_SEV = _norm_dict({
    'Normal Wear & Tear (all body panels intact & attached)': 0,
    'Some Mirrors, Glass, or Lights are Cracked/Missing': 1,
    'Loose or Missing Panels*': 2, 'Baseball-sized or Larger Damage*': 3,
    'Major Damage*': 4, 'Unknown': 5,
    # body_paint_cond_picklist_id (new schema)
    23044: 0, 23046: 1, 23049: 2, 23047: 3, 23048: 4, 23045: 5,
})
ENGINE_SEV = _norm_dict({
    'Operational': 0, 'Minor Issues / Still Functional': 1, 'Rebuilt/Replaced': 1,
    'Major Malfunction / Still Installed': 3, 'Missing': 4, 'Removed': 4, 'Unknown': 5,
    # engine_cond_picklist_id (new schema)
    23053: 0, 23055: 1, 23056: 1, 23057: 3, 23054: 4, 23058: 4, 23059: 5,
})
# Same 7 text labels as ENGINE_SEV, but its own dict now -- the transmission
# picklist-ID block (23060-23066) is disjoint from engine's (23053-23059),
# and engine/transmission flip which label is "Removed" vs "Missing" at the
# ID level, so the two can no longer share one dict object once ID keys are
# involved (they could when only text labels were shared).
TRANS_SEV = _norm_dict({
    'Operational': 0, 'Minor Issues / Still Functional': 1, 'Rebuilt/Replaced': 1,
    'Major Malfunction / Still Installed': 3, 'Missing': 4, 'Removed': 4, 'Unknown': 5,
    # transmission_cond_picklist_id (new schema)
    23060: 0, 23061: 1, 23062: 1, 23063: 3, 23064: 4, 23065: 4, 23066: 5,
})
TIRE_SEV = _norm_dict({
    'All Wheels Mounted & Tires Inflated': 0, '1 or More Tires are Flat*': 2,
    '1 or More Wheels are Removed*': 3, 'Major Malfunction / Still Installed': 3,
    'Missing': 4, 'Removed': 4, 'Unknown': 5,
    # tire_cond_picklist_id (new schema) -- note the "good" value (0) is the
    # HIGHEST id (23073) here, reversed relative to every other condition col.
    23073: 0, 23072: 2, 23071: 3, 23070: 3, 23069: 4, 23068: 4, 23067: 5,
})
INTERIOR_SEV = _norm_dict({
    'Normal Wear & Tear (all interior intact & attached)': 0,
    'Damaged or Removed Parts (notes required)': 2, 'Unknown': 4,
    # interior_cond_picklist_id (new schema)
    23050: 0, 23051: 2, 23052: 4,
})
DSRATING_NUM = _norm_dict({'DS1-1': 1, 'DS1-2': 2, 'DS1-3': 3, 'DS1-4': 4, 'DS1-5': 5, 'DS3': 6})
SEV_COLS = [
    ('nav_condition',         NAV_CONDITION_SEV, 'nav_severity'),
    ('bodypaintcondition',    BODY_SEV,          'body_severity'),
    ('enginecondition',       ENGINE_SEV,        'engine_severity'),
    ('transmissioncondition', TRANS_SEV,         'trans_severity'),
    ('tirecondition',         TIRE_SEV,          'tire_severity'),
    ('interiorcondition',     INTERIOR_SEV,      'interior_severity'),
]
UNKNOWN_FLAG_COLS = ['nav_condition','bodypaintcondition','enginecondition',
                     'transmissioncondition','tirecondition','interiorcondition']

# IDs whose resolved name contains "Runs" (vehicle_cond_picklist_id) -- used
# by _is_runs_condition() below as the numeric-ID counterpart of the legacy
# text substring check on nav_condition.
RUNS_CONDITION_IDS = {22968, 22969, 22971}

# Per-column "Unknown" picklist ID -- each condition column has its own ID
# space, so this can't be a single shared value like the text 'unknown' was.
UNKNOWN_CONDITION_IDS = {
    'nav_condition':         22970,
    'bodypaintcondition':    23045,
    'enginecondition':       23059,
    'transmissioncondition': 23066,
    'tirecondition':         23067,
    'interiorcondition':     23052,
}

# ============================================================
# PICKLIST ID -> DISPLAY NAME (human-readable contexts only: the Granite
# LLM prompt's car-summary line and the SHAP "value" shown to the user --
# NOT used by feature engineering above, which only needs severity numbers).
# Same confirmed ID<->name correspondence as the *_SEV dicts' ID keys, just
# inverted for text output instead of a severity number. Kept as separate,
# explicit dicts rather than derived from *_SEV (multiple IDs can share one
# severity, e.g. ENGINE_SEV's 23055/23056 both -> 1, so severity alone can't
# be inverted back to a unique name).
#
# State fields (state_title_picklist/state_picklist_id) have no equivalent
# dict here: their ID<->name mapping is only partially known (the ~59 middle
# IDs were never confirmed -- see the original mapping notes), so a raw
# numeric ID displays as-is for those two.
NAV_CONDITION_ID_TO_NAME = {
    22968: 'Runs & Drives', 22971: 'Runs & Moves / Don’t Drive',
    22969: 'Runs / Doesn’t Move', 22974: 'Cranks, won’t start',
    22973: 'Doesn’t Run / Can be Moved', 22972: 'Doesn’t Run / Doesn’t Move',
    22970: 'Unknown',
}
BODY_PAINT_COND_ID_TO_NAME = {
    23044: 'Normal Wear & Tear (all body panels intact & attached)',
    23046: 'Some Mirrors, Glass, or Lights are Cracked/Missing',
    23049: 'Loose or Missing Panels*', 23047: 'Baseball-sized or Larger Damage*',
    23048: 'Major Damage*', 23045: 'Unknown',
}
ENGINE_COND_ID_TO_NAME = {
    23053: 'Operational', 23055: 'Rebuilt/Replaced',
    23056: 'Minor Issues / Still Functional',
    23057: 'Major Malfunction / Still Installed',
    23054: 'Removed', 23058: 'Missing', 23059: 'Unknown',
}
TRANSMISSION_COND_ID_TO_NAME = {
    23060: 'Operational', 23061: 'Rebuilt/Replaced',
    23062: 'Minor Issues / Still Functional',
    23063: 'Major Malfunction / Still Installed',
    23064: 'Removed', 23065: 'Missing', 23066: 'Unknown',
}
TIRE_COND_ID_TO_NAME = {
    23073: 'All Wheels Mounted & Tires Inflated', 23072: '1 or More Tires are Flat*',
    23071: '1 or More Wheels are Removed*', 23070: 'Major Malfunction / Still Installed',
    23069: 'Removed', 23068: 'Missing', 23067: 'Unknown',
}
INTERIOR_COND_ID_TO_NAME = {
    23050: 'Normal Wear & Tear (all interior intact & attached)',
    23051: 'Damaged or Removed Parts (notes required)', 23052: 'Unknown',
}

# raw_key (as used by app/raw_feature_mapping.py's EXACT_MAP / INTERNAL_TO_REQUEST_FIELD)
# -> its ID_TO_NAME dict. other_damages is deliberately absent here -- it's
# decoded via SaleValuePreprocessor.OTHER_DAMAGE_ID_TO_LABEL instead, since
# that field can hold a list of IDs, not just one.
CONDITION_ID_TO_NAME_BY_RAW_KEY = {
    'nav_condition':         NAV_CONDITION_ID_TO_NAME,
    'bodypaintcondition':    BODY_PAINT_COND_ID_TO_NAME,
    'enginecondition':       ENGINE_COND_ID_TO_NAME,
    'transmissioncondition': TRANSMISSION_COND_ID_TO_NAME,
    'tirecondition':         TIRE_COND_ID_TO_NAME,
    'interiorcondition':     INTERIOR_COND_ID_TO_NAME,
}


def describe_picklist_value(raw_key, value):
    """Decode a numeric picklist ID to its human-readable name for display
    (LLM prompts, SHAP "value" fields) -- NOT used by feature engineering,
    which works directly off the ID via the *_SEV dicts above.

    Returns `value` unchanged when raw_key isn't a condition column, the
    value isn't numeric (e.g. it's already legacy text), or the ID isn't
    recognized -- so this is always safe to call speculatively.
    """
    id_to_name = CONDITION_ID_TO_NAME_BY_RAW_KEY.get(raw_key)
    if id_to_name is None or value is None:
        return value
    if isinstance(value, str):
        return value  # already text (legacy caller) -- nothing to decode
    try:
        return id_to_name.get(int(value), value)
    except (TypeError, ValueError):
        return value


def describe_other_damages_value(value):
    """Decode other_damages' display value: a scalar numeric picklist ID, or
    a list of them (live multi-select), to comma-joined label text via
    SaleValuePreprocessor.OTHER_DAMAGE_ID_TO_LABEL. Text/JSON-string legacy
    values and anything unrecognized pass through unchanged -- safe to call
    speculatively, same convention as describe_picklist_value above.
    """
    id_to_label = SaleValuePreprocessor.OTHER_DAMAGE_ID_TO_LABEL
    if value is None:
        return value
    if isinstance(value, (int, float)):
        try:
            return id_to_label.get(int(value), value)
        except (TypeError, ValueError):
            return value
    if isinstance(value, list):
        labels = []
        for item in value:
            if isinstance(item, (int, float)):
                labels.append(id_to_label.get(int(item), str(item)))
            elif isinstance(item, dict):
                labels.append(str(item.get('name', item)))
            else:
                labels.append(str(item))
        return ', '.join(labels) if labels else value
    return value  # already text/JSON string -- leave as-is


def _is_runs_condition(v):
    """True for the legacy text label (post-_normalize_text lowercasing, so
    a plain substring check) and for the new-schema numeric picklist ID
    (vehicle_cond_picklist_id) whose resolved name contains 'Runs'."""
    if pd.isna(v):
        return False
    if isinstance(v, str):
        return 'runs' in v
    try:
        return int(v) in RUNS_CONDITION_IDS
    except (TypeError, ValueError):
        return False

def _is_unknown_condition(v, unknown_id):
    """True for the legacy text label 'unknown' and for the new-schema
    numeric picklist ID equal to this column's own Unknown ID
    (see UNKNOWN_CONDITION_IDS -- each condition column has a different one)."""
    if pd.isna(v):
        return False
    if isinstance(v, str):
        return v == 'unknown'
    try:
        return int(v) == unknown_id
    except (TypeError, ValueError):
        return False

# ============================================================
# PREPROCESSOR — same as Script 20/21
# ============================================================
class SaleValuePreprocessor(BaseEstimator, TransformerMixin):
    # ---- Hard-drop columns ----------------------------------------------------
    # These are dropped at the very start of _basic_clean so nothing downstream
    # sees them. The two sublists exist for legacy reasons; the union is used.

    # User-requested drop list: columns that are not available at inference time
    # (or are leaky), plus their previously-derived engineered features.
    # Keep this list in alphabetical order for readability/audit.
    USER_DROP_COLS = [
        'actualpickupdate',
        'category',                  # not available at inference
        'current_fiscal_year_start_month',
        'data_inj_date',
        'data_source',
        'days_in_status',
        'donationsource',
        'dsrating',
        'dsrating_num',              # was a derived feature; drop too
        'dsratingshortdescription',
        'fiscal_year_start_month',
        'has_dsrating',              # was a derived feature; drop too
        'last_year_previous_np_quarter_fiscal_yr',
        'marketer',
        'marketercommission',
        'mechanical',
        'name_on_title',
        'non_profit_grouping',
        'non_profit_grouping_count',
        'non_profit_grouping_id',
        'nonprofit',
        'nonprofit_code',
        'nonprofit_id',
        'np_quarter_fiscal_yr',
        'num_vehicle_tags',
        'other',
        'paintnbody',
        'paymentdate',
        'previous_np_quarter_fiscal_yr',
        'previous_quarter_cal_yr',
        'previous_quarter_end_date_cal_yr',
        'primarycategory',
        'quarter_cal_yr',            # not available at inference
        'referrer',
        'saledate',
        'salesrevenue',              # would leak target signal
        'saletype',
        'schedulepickupdate',
        'screenpop_id',
        'status',                    # not available at inference
        'submitted',
        'submittedby_email',
        'submittedby_firstname',
        'submittedby_lastname',
        'titleenddate',
        'titlestartdate',
        'totalexpenses',
        'towfees',
        'unconfirmedsaledate',
        'updated',
        'vehicle_tag',
        'vendor',
        'vendor_assigned_by',
        'vendor_assignment_method',
        'vendor_id',
        'vendor_sale_type',
        'vendorassignmentdate',
        'vendorcheckdate',
        'vengrouping',
        'year_match',
    ]

    DROP_COLS_NO_ZIP = [
        'vehicle_id','vehicle_uuid','stock_id','vin_hin_no','api_log_id','api_log_uuid',
        'vin','vinhinno','bodypaint_id',
        'api_type','chassis_type','brake_system','vehiclecountryname',
        'body_segment','size_segment','luxury','hybrid','electric','sport','crossover','exotic',
        'creation_datetime','last_update',
        'restraint_type','engines_json','transmissions_json','questions',
        'vehicletype','stateprovinceofregistration',
        'vcity','industry',
        'nav_category',
    ]
    DROP_COLS_WITH_ZIP_DROP = DROP_COLS_NO_ZIP + ['vazipcode']
    FREQ_COLS_BASE = [
        'make','model','trim','model_number','oem_body_style','us_style_name',
        'engine_name','transmission_name','state_province_of_title',
        'make_x_body_type','make_x_vehicle_type','vtype_x_nav_condition',
        'body_x_drive','condition_combo','engine_x_trans_cond',
        'nav_cond_x_age_bkt','runs_x_mileage_bkt','all_cond_combo',
        # User-added interactions (string-concat → freq-encoded)
        'make_x_age', 'month_x_age', 'quarter_x_make',
        # Vehicle-profile cluster interactions (zero target leakage)
        'cluster_x_age_bkt', 'cluster_x_mileage_bkt',
        # Parsed engine attributes (categorical → freq-encoded; numerics flow through)
        'engineconfiguration', 'valvetraindesign',
        # Engine interactions (string-concat → freq-encoded)
        'enginecylinders_x_make', 'enginehp_bkt_x_age_bkt', 'engineconfig_x_make',
        # Other-damages normalized string (sorted, pipe-joined tokens)
        'other_damages_normalized',
        # Other-damages interactions (string-concat → freq-encoded)
        'damage_x_mileage_bkt',
        # New raw categoricals (mileage-trust/title/weight-class/body/engine)
        'gvm_range', 'body_subtype', 'engine_type',
        # Mileage-trust interaction (string-concat → freq-encoded)
        'mileage_unknown_x_make',
        # Worst-case-tier interaction (string-concat → freq-encoded)
        'vtype_x_mileage_bkt',
        # Condition-x-make interactions ($100-2000 tier, string-concat → freq-encoded)
        'runs_x_make', 'mech_severity_x_make', 'all_cond_combo_x_make',
    ]
    GEO_FREQ_COLS = [
        'zip_region_x_vehicle_type','zip_region_x_body_type','zip_region_x_nav_condition',
        'zip_first2', 'zip_first3', 'zip_full',
        'zip_region_x_mileage_bkt',
    ]
    TARGET_ENC_COLS_ALL = [
        'make','model','trim','model_number','us_style_name',
        'engine_name','transmission_name','state_province_of_title',
        'condition_combo','all_cond_combo','nav_cond_x_age_bkt',
    ]
    MILEAGE_EDGES = [-np.inf, 60_000, 110_000, 155_000, 200_000, 245_000, np.inf]
    AGE_EDGES     = [-1, 5, 10, 15, 20, 60]

    # Raw fields that arrive as inconsistently-encoded booleans (Python bool,
    # 'True'/'False'/'t'/'f'/'yes'/'no' strings, 0/1, or blank) and need
    # coercing to a clean 0/1 flag before they're useful as model features.
    # Without this, e.g. 't' and 'true' would int-encode as two DIFFERENT
    # categories instead of collapsing to the same flag. Blank/unrecognized
    # values become NaN (not 0) so the model can tell "confirmed false" apart
    # from "unknown" — see _coerce_bool_flag.
    BOOL_FLAG_COLS = ['true_mileage_unknown', 'clean_title']
    _BOOL_TRUE_TOKENS  = {'true', 't', 'yes', 'y', '1'}
    _BOOL_FALSE_TOKENS = {'false', 'f', 'no', 'n', '0'}

    # Raw picklist-ID columns that are fed straight through into the final
    # feature matrix as their own raw feature (in addition to whatever
    # derived features get built from them, e.g. nav_condition -> both
    # nav_severity AND the raw nav_condition column itself survive). A live
    # caller can legitimately send the ID as a JSON STRING rather than a
    # JSON number (these fields are typed Union[str, int, float] on
    # PredictRequest) -- pydantic does not coerce that to a number. Two
    # distinct problems follow if it's left as a string:
    #   1. SEV_COLS' `.map(sev_map)` below does an exact-key lookup with NO
    #      normalization of the value being looked up (only the dict's own
    #      keys were lowercased/left-as-is when built) -- a string like
    #      "22968" does not match the int key 22968, so the severity/
    #      runs_flag/unknown-flag extraction silently falls back to
    #      "unknown" instead of raising, which is a much quieter bug than #2.
    #   2. The raw column itself reaches XGBoost as an unencoded 'object'
    #      dtype (a lone string in a single-row column can't be anything
    #      else), which XGBoost's inplace_predict hard-rejects outright
    #      (ValueError: DataFrame.dtypes for data must be int, float, bool
    #      or category).
    # _coerce_numeric_id_like() below fixes both by converting a numeric-
    # looking value to a real float BEFORE any of that runs. This is a
    # no-op for real training data on both pipelines -- script21's
    # numeric-schema data is already float-dtype at fit time (nothing here
    # is ever a string to begin with), and script17's legacy text data
    # never happens to be a numeric-looking string either -- so it only
    # ever changes behavior for a live request that stringified an ID.
    # Genuine non-numeric text (e.g. script17's real condition labels) is
    # left completely untouched.
    RAW_PICKLIST_ID_COLS = [
        'vehicle_type', 'nav_color',
        'nav_condition', 'enginecondition', 'transmissioncondition',
        'bodypaintcondition', 'interiorcondition', 'tirecondition',
        'state_province_of_title', 'vstate_name',
    ]

    @staticmethod
    def _coerce_numeric_id_like(series):
        """Convert values that look like a number (already int/float, or a
        numeric-looking string e.g. '22968'/'22968.0') to a real float.
        Anything else (genuine non-numeric text, None/NaN, unparseable
        strings) is returned completely unchanged."""
        def _try_num(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float, np.integer, np.floating)):
                return v
            if isinstance(v, str):
                try:
                    return float(v.strip())
                except ValueError:
                    return v
            return v
        return series.map(_try_num)

    @classmethod
    def _coerce_bool_flag(cls, series):
        """Map messy boolean-ish values to a clean 0/1 float.

        Handles Python/numpy bool, numeric 0/1, and string variants
        ('true'/'t'/'yes'/'y'/'1', 'false'/'f'/'no'/'n'/'0'), case-
        insensitively. Missing or unrecognized values become NaN rather
        than being silently treated as False.
        """
        def _one(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, (bool, np.bool_)):
                return float(v)
            if isinstance(v, (int, float, np.integer, np.floating)):
                if v == 1: return 1.0
                if v == 0: return 0.0
                return np.nan
            s = str(v).strip().lower()
            if s in cls._BOOL_TRUE_TOKENS:  return 1.0
            if s in cls._BOOL_FALSE_TOKENS: return 0.0
            return np.nan
        return series.map(_one)

    def __init__(self, time_col=TIME_COL, seed=42,
                 use_macro=False, use_geo=False, use_cult=False,
                 with_target_encoding=False, smoothing=20, n_folds=5,
                 zip_lat_map=None, zip_lon_map=None, cult_lookup=None,
                 n_clusters=12, cluster_min_samples=50,
                 extra_drop_cols=None, enable_worst_tier_features=True,
                 enable_condition_make_features=True):
        self.time_col = time_col
        self.seed     = seed
        self.use_macro = use_macro
        self.use_geo   = use_geo
        self.use_cult  = use_cult
        # See WORST_TIER_FEATURE_COLS. Accepts True (all 6 on, default),
        # False (all off — reproduces the pre-worst-tier baseline), or an
        # iterable of specific column names to enable individually (e.g.
        # {'unknowns_x_mileage_bkt'}) so each can be ablation-tested one at
        # a time. See _wtf().
        self.enable_worst_tier_features = enable_worst_tier_features
        # See CONDITION_MAKE_FEATURE_COLS. Same True/False/iterable
        # convention as enable_worst_tier_features above. See _cmf().
        self.enable_condition_make_features = enable_condition_make_features
        self.with_target_encoding = with_target_encoding
        self.smoothing = smoothing
        self.n_folds   = n_folds
        self.zip_lat_map = zip_lat_map or {}
        self.zip_lon_map = zip_lon_map or {}
        self.cult_lookup = cult_lookup or {}
        # Caller-supplied extra columns to hard-drop, on top of USER_DROP_COLS /
        # DROP_COLS_NO_ZIP. Lets one pipeline (e.g. script21) toggle a feature
        # subset on/off without changing the shared class defaults other
        # callers (script17) rely on. See NEW_FEATURE_COLS / --enable-new-features.
        self.extra_drop_cols = extra_drop_cols or []
        self.TARGET_ENC_COLS = self.TARGET_ENC_COLS_ALL if with_target_encoding else []
        # Vehicle-profile clustering: K-means on (make, model) attribute vectors.
        # NOTE: clusters are NOT fit on salevalue — zero target leakage. They group
        # vehicles by what kind of car they are (size, condition profile, body type),
        # NOT by what they sell for. The model still has to learn depreciation
        # from price data, but with the cluster as an extra grouping signal.
        self.n_clusters = n_clusters
        self.cluster_min_samples = cluster_min_samples

    def _wtf(self, name):
        """Whether one WORST_TIER_FEATURE_COLS column is enabled. Handles
        all three forms enable_worst_tier_features can take: True (all on),
        False (all off), or an iterable of specific names to enable one at
        a time (see --enable-worst-tier-features in train_save_script21.py)."""
        v = self.enable_worst_tier_features
        if isinstance(v, bool):
            return v
        return name in v

    def _cmf(self, name):
        """Same as _wtf() but for CONDITION_MAKE_FEATURE_COLS /
        enable_condition_make_features (see --enable-condition-make-features
        in train_save_script21.py)."""
        v = self.enable_condition_make_features
        if isinstance(v, bool):
            return v
        return name in v

    def _normalize_text(self, X):
        for col in X.select_dtypes(include='object').columns:
            if col == self.time_col: continue
            s = X[col].astype(str).str.strip().str.lower()
            s = s.str.replace(r'\s+', ' ', regex=True)
            s = s.where(s != 'nan', other=np.nan)
            X[col] = s
        return X

    # Layout code -> human-readable engine configuration. Used by
    # _parse_engine_features below. Letter is the layout prefix; digit is
    # the cylinder count (kept as-is for fidelity).
    _ENGINE_LAYOUT_MAPPING = {
        "I3": "In-Line",  "I4": "In-Line",  "I5": "In-Line",  "I6": "In-Line",
        "V6": "V-Shaped", "V8": "V-Shaped", "V10": "V-Shaped", "V12": "V-Shaped",
        "H4": "Horizontally opposed (boxer)",
        "H6": "Horizontally opposed (boxer)",
        "W12": "W Shaped",
        "ROTARY": "Rotary",
    }

    def _parse_engine_features(self, X):
        """Parse structured engine features from the raw engine_name string.

        Adds five new columns:
          displacementl       -- float, engine displacement in liters
          enginecylinders     -- float, number of cylinders (kept as float
                                 because NaN is a valid "couldn't parse" value)
          engineconfiguration -- str, layout family (In-Line / V-Shaped / etc.)
          enginehp            -- float, horsepower
          valvetraindesign    -- str, valvetrain design (DOHC / SOHC / OHV)

        Coverage is partial: not every engine_name carries all five signals.
        Unparseable values become NaN; XGBoost handles that fine.

        Original engine_name column is PRESERVED — it stays in the freq-encoded
        and target-encoded feature set as before.
        """
        if 'engine_name' not in X.columns:
            return X

        # Work on the raw string. The series may contain NaN; the .str
        # accessor handles them by returning NaN-valued series.
        en = X['engine_name'].astype('string')   # nullable string type

        # 1. Displacement in liters: "2.0L" -> 2.0, "6L" -> 6.0
        X['displacementl'] = (
            en.str.extract(r'(\d+(?:\.\d+)?)\s*[lL]', expand=False)
              .astype(float)
        )

        # 2. Cylinder count: "V8" -> 8, "I6" -> 6, "W12" -> 12
        en_upper = en.str.upper()
        X['enginecylinders'] = (
            en_upper.str.extract(r'\b[IVWH](\d{1,2})\b', expand=False).astype(float)
        )

        # 3. Engine configuration via layout code + mapping
        layout_raw = en_upper.str.extract(
            r'\b([IVHW]\d{1,2}|ROTARY)\b', expand=False
        )
        X['engineconfiguration'] = layout_raw.map(self._ENGINE_LAYOUT_MAPPING)

        # 4. Horsepower: "350hp", "247 hp", "1200.5hp"
        X['enginehp'] = (
            en.str.extract(r'(\d+(?:\.\d+)?)\s*hp', flags=re.IGNORECASE,
                            expand=False).astype(float)
        )

        # 5. Valvetrain design: DOHC / SOHC / OHV (priority in that order)
        X['valvetraindesign'] = np.select(
            [
                en.str.contains(r'\bDOHC\b|double overhead cam', case=False,
                                 na=False, regex=True),
                en.str.contains(r'\bSOHC\b|single overhead cam', case=False,
                                 na=False, regex=True),
                en.str.contains(r'\bOHV\b|overhead valves', case=False,
                                 na=False, regex=True),
            ],
            ["DOHC", "SOHC", "OHV"],
            default=None,
        )
        # np.select returns np.nan as 'None' object; convert
        X['valvetraindesign'] = X['valvetraindesign'].replace({'': np.nan, None: np.nan})
        return X

    # Aliases for normalized damage type tokens. Single source of truth: every
    # member of `aliases` must already be in normalized form (lowercase, ASCII
    # apostrophe, no trailing punctuation, single spaces).
    _OTHER_DAMAGES_INDICATOR_RULES = [
        ('has_mold',                ['mold']),
        ('has_undercarriage_rust',  ['severe undercarriage rust',
                                      'undercarriage rust']),
        ('has_smog_fail',           ["won't pass smog/state inspection",
                                      "won't pass smog",
                                      'wont pass smog/state inspection',
                                      'wont pass smog']),
    ]

    # other_damage_pklist_id (new schema) -> canonical label. Decoded by
    # _extract_tokens() below (scalar ID, or numeric IDs inside a live-request
    # list) into the same label text _normalize_damage_token() already knows
    # how to reduce to the tokens above -- e.g. 23074 -> 'Mold' -> 'mold'.
    OTHER_DAMAGE_ID_TO_LABEL = {
        23074: 'Mold',
        23075: 'Flood Damage',
        23076: 'Fire Damage',
        23077: 'Air Bag(s) Deployed',
        23078: "Won't Pass Smog/State Inspection",
        23079: 'Severe Undercarriage Rust',
        23080: 'Other*',
    }

    @staticmethod
    def _normalize_damage_token(t):
        """Lowercase, ASCII-fy quotes, collapse whitespace, strip trailing punct.

        Notes:
          - Asterisk is NOT stripped because the source data uses 'other*' as the
            actual category label (there is no plain 'other'). Stripping it would
            be over-normalization.
          - Punctuation strip is conservative: only spaces and a few sentence-end
            chars. Internal punctuation (parens, slashes, apostrophes) is kept.
        """
        if t is None:
            return ''
        s = str(t).strip().lower()
        # Smart quotes -> ASCII (\u2019 = right single quote, common in 'won't')
        s = (s.replace('\u2019', "'").replace('\u2018', "'")
              .replace('\u201c', '"').replace('\u201d', '"'))
        # Collapse all whitespace runs
        s = re.sub(r'\s+', ' ', s).strip()
        # Conservative trailing-punctuation strip
        s = s.strip(' .,;:').strip()
        return s

    def _parse_other_damages(self, X):
        """Parse + normalize the 'other_damages' field into model-usable features.

        Handles three messes in the source data:
          1. Two coexisting formats: plain text ('mold, other*') AND JSON-encoded
             ('[{"id": 23074, "name": "Mold"}]'). Both reduce to the same tokens.
          2. Case + smart-quote variation ('won\u2019t' vs "won't", 'Other*' vs 'other*').
          3. Multi-label values: 'mold, other*' -> two tokens, sorted, deduplicated.

        Adds 6 features (5 indicators + 1 categorical for freq encoding):
          has_other_damage          -- 1 if any damage reported, else 0
          n_other_damages           -- count of normalized damage tokens (0 if missing)
          other_damages_normalized  -- sorted '|'-joined tokens, e.g. 'mold|other*'
                                       (freq-encoded downstream by FREQ_COLS_BASE)
          has_mold                  -- 1 if mold detected
          has_undercarriage_rust    -- 1 if any undercarriage-rust variant
          has_smog_fail             -- 1 if smog/state inspection failure

        The raw 'other_damages' column is DROPPED at the end since the
        normalized representation captures everything the model needs and the
        raw string has 110 noisy distinct values.

        Note: this field is ~99% null in production data. Don't expect
        large SHAP attributions; the value is mostly for the ~1% of rows that
        do carry damage flags.
        """
        if 'other_damages' not in X.columns:
            # Fill the engineered columns with safe defaults so downstream code
            # doesn't have to do `if 'has_other_damage' in X.columns` everywhere.
            X['has_other_damage'] = 0
            X['n_other_damages'] = 0
            X['other_damages_normalized'] = 'none'
            for feat_name, _ in self._OTHER_DAMAGES_INDICATOR_RULES:
                X[feat_name] = 0
            return X

        import json as _json
        raw = X['other_damages']

        def _decode_damage_id(it):
            """Resolve one other_damage_pklist_id (new schema, numeric) to its
            canonical label via OTHER_DAMAGE_ID_TO_LABEL, or '' if unrecognized."""
            try:
                return self.OTHER_DAMAGE_ID_TO_LABEL.get(int(it), '')
            except (TypeError, ValueError):
                return ''

        def _extract_tokens(v):
            """Return list of normalized damage type strings from one cell."""
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return []
            # New schema: a bare numeric picklist ID (other_damage_pklist_id)
            # instead of the resolved *_name text -- decode it to the same
            # label text the *_name column used to carry.
            if isinstance(v, (int, float, np.integer, np.floating)):
                label = _decode_damage_id(v)
                return [self._normalize_damage_token(label)] if label else []
            # A live API request can send other_damages as an actual JSON list
            # (e.g. ["Mold", "Rust"]) rather than the training data's string
            # formats. Handle that BEFORE the str(v) conversion below, since
            # str(["Mold"]) -> "['Mold']" (Python repr, not JSON) would fail
            # the JSON branch and get mangled by the plain-text comma-split
            # fallback instead. Accepts a list of plain strings, a list of
            # {'name': ...} dicts (existing JSON-object format), or a list of
            # numeric picklist IDs (new schema multi-select).
            if isinstance(v, list):
                names = []
                for it in v:
                    if isinstance(it, dict):
                        names.append(it.get('name', ''))
                    elif isinstance(it, (int, float, np.integer, np.floating)):
                        names.append(_decode_damage_id(it))
                    else:
                        names.append(str(it))
                return [self._normalize_damage_token(n) for n in names if n]
            s = str(v).strip()
            if not s or s.lower() in ('nan', 'none', 'null'):
                return []
            # New schema: a bare numeric picklist ID sent as a JSON STRING
            # (e.g. "23080") rather than a number -- other_damage_pklist_id's
            # declared type (Union[str, int, float, List[Any]], see
            # app/schemas.py) explicitly allows str, and a JSON client can
            # legitimately serialize a numeric ID as a string. Decode it the
            # same way as the numeric branch above, BEFORE falling into the
            # JSON/plain-text parsing below -- otherwise a stringified ID
            # gets treated as freeform text (a bogus damage-type token
            # instead of its real label), silently diverging from what the
            # identical value would produce as an actual int/float.
            if s.lstrip('-').isdigit():
                label = _decode_damage_id(s)
                return [self._normalize_damage_token(label)] if label else []
            # JSON-encoded list of {'id': ..., 'name': ...} dicts
            if s.startswith('['):
                try:
                    items = _json.loads(s)
                    if isinstance(items, list):
                        names = [str(it.get('name', '')) for it in items
                                 if isinstance(it, dict)]
                        return [self._normalize_damage_token(n) for n in names if n]
                except (ValueError, AttributeError):
                    pass
                # Fallback: regex extraction of "name" : "..." pairs when JSON
                # is malformed (which does happen in this data).
                names = re.findall(r'"name"\s*:\s*"([^"]+)"', s)
                if names:
                    return [self._normalize_damage_token(n) for n in names]
            # Plain text — split on ; or , only (NOT /). The slash appears
            # inside damage type names like "won't pass smog/state inspection"
            # and is NOT a multi-label separator in this data.
            parts = re.split(r'[;,]', s)
            return [self._normalize_damage_token(p) for p in parts
                     if self._normalize_damage_token(p)]

        token_lists = raw.apply(_extract_tokens)
        # Sort + dedupe so 'mold|other*' == 'other*|mold'
        token_lists = token_lists.apply(lambda lst: sorted(set(t for t in lst if t)))

        X['has_other_damage'] = (token_lists.str.len() > 0).astype(int)
        X['n_other_damages']  = token_lists.str.len().astype(int)
        X['other_damages_normalized'] = token_lists.apply(
            lambda lst: '|'.join(lst) if lst else 'none'
        )

        for feat_name, alias_list in self._OTHER_DAMAGES_INDICATOR_RULES:
            X[feat_name] = token_lists.apply(
                lambda toks: int(any(a in toks for a in alias_list))
            )

        # Drop the raw string column — keeping it would create a noisy 110-value
        # categorical that adds nothing on top of `other_damages_normalized`.
        X = X.drop(columns=['other_damages'])
        return X

    def _add_cult_features(self, X):
        if not self.use_cult: return X
        if 'make' not in X.columns or 'model' not in X.columns or 'year' not in X.columns:
            return X
        cult_results = []
        for mk, md, yr in zip(X['make'].values, X['model'].values, X['year'].values):
            mk_s = str(mk).strip().lower() if pd.notna(mk) else ''
            md_s = str(md).strip().lower() if pd.notna(md) else ''
            cult_results.append(lookup_cult(mk_s, md_s, yr, self.cult_lookup))
        X['is_cult'] = np.array([1 if r else 0 for r in cult_results], dtype=int)
        X['cult_match_strictness'] = np.array(
            [r['strictness'] if r else 0 for r in cult_results], dtype=int)
        for feat in ['cult_tier_num','cult_enthusiast_score','cult_uplift_low',
                     'cult_uplift_high','cult_last_of_kind',
                     'cult_liquidity_num','cult_volatility_num','cult_origsens_num']:
            X[feat] = np.array([r[feat] if r else np.nan for r in cult_results], dtype=float)
        X['cult_uplift_mid']   = (X['cult_uplift_low'] + X['cult_uplift_high']) / 2.0
        X['cult_uplift_range'] = (X['cult_uplift_high'] - X['cult_uplift_low'])
        return X

    def _basic_clean(self, X):
        # NOTE: renaming new-DB-schema column names onto the legacy names
        # this method is written against is the CALLER's job (via
        # map_raw_features_to_legacy()), not done automatically here.
        # train_save_script21.py's main(), and app/inference_script21.py /
        # app/inference_script17.py's predict(), all call it explicitly
        # before fit()/transform() ever runs (they need to -- they read
        # columns like salevalue/record_creation_date/vazipcode directly,
        # before this method would get a chance to). An automatic call used
        # to live here too, as a safety net for a caller that forgot to
        # rename -- removed because it's unsafe to call twice: renaming is
        # NOT idempotent in one specific case (vehicle_category -> body_type
        # is a legacy TARGET that collides with the unrelated DataOne raw
        # 'body_type' -> oem_body_style RENAME SOURCE), so a caller that
        # already renamed once and then hit this automatic second call would
        # see 'vehicle_category' silently renamed twice into 'oem_body_style'
        # and dropped (since oem_body_style is excluded when --use-dataone is
        # off) -- exactly what was happening here before this fix.
        #
        # Step 1: coalesce paired (nav_*, primary) columns. Prefer nav_*, fall
        # back to the primary field. The unified value lives in the primary
        # column. After coalesce both source columns become equivalent so we
        # drop the nav_* version next to avoid double-counting.
        for nav_col, primary_col in [('nav_make', 'make'),
                                       ('nav_model', 'model'),
                                       ('nav_year', 'year')]:
            if nav_col in X.columns and primary_col in X.columns:
                # nav_* wins when non-null; primary fills the gap
                X[primary_col] = X[nav_col].where(X[nav_col].notna(), X[primary_col])
                X = X.drop(columns=[nav_col])
            elif nav_col in X.columns:
                # Only the nav variant exists — rename to the primary
                X = X.rename(columns={nav_col: primary_col})

        # Step 2: drop the user-specified columns + the legacy drop list.
        # USER_DROP_COLS includes engineered features (dsrating_num,
        # has_dsrating) that no longer make sense without the source column.
        all_drops = list(self.USER_DROP_COLS) + list(
            self.DROP_COLS_NO_ZIP if self.use_geo else self.DROP_COLS_WITH_ZIP_DROP
        ) + list(self.extra_drop_cols)
        X = X.drop(columns=[c for c in all_drops if c in X.columns])

        # Step 3: parse structured features from engine_name BEFORE text
        # normalization (which lowercases and would break case-sensitive regex
        # for DOHC/SOHC/V8/etc.). The original engine_name column is preserved
        # for downstream freq/target encoding.
        X = self._parse_engine_features(X)

        # Step 4: parse other_damages (also case-sensitive — JSON values contain
        # capitalized strings like '"Mold"' that the text-normalizer would mangle).
        X = self._parse_other_damages(X)

        # Step 4.5: coerce messy boolean-ish raw fields to a clean 0/1 float
        # BEFORE text normalization. Once coerced these are numeric, so
        # _normalize_text's dtype-based column selection skips them below.
        for col in self.BOOL_FLAG_COLS:
            if col in X.columns:
                X[col] = self._coerce_bool_flag(X[col])

        # Step 4.6: coerce a numeric-looking string (e.g. a live caller
        # sending a picklist ID as a JSON string) back to a real number on
        # the raw picklist-ID columns, BEFORE severity/runs-flag/unknown-flag
        # extraction (_engineer()'s SEV_COLS loop, below) or the final
        # feature matrix ever see it -- see RAW_PICKLIST_ID_COLS above for
        # why. Also before _normalize_text() so a coerced column is numeric
        # dtype and skipped by its object-dtype column selection, same as
        # the BOOL_FLAG_COLS coercion just above.
        for col in self.RAW_PICKLIST_ID_COLS:
            if col in X.columns:
                X[col] = self._coerce_numeric_id_like(X[col])

        X = self._normalize_text(X)
        if self.use_cult:
            X = self._add_cult_features(X)
        if 'mileage' in X.columns:
            X['mileage'] = pd.to_numeric(X['mileage'], errors='coerce')
            bad = (X['mileage'] == -1) | (X['mileage'] > 400_000) | (X['mileage'] < 0)
            X.loc[bad, 'mileage'] = np.nan
        if self.time_col in X.columns:
            t = pd.to_datetime(X[self.time_col], errors='coerce')
            X['_record_year']  = t.dt.year.fillna(BASE_YEAR).astype(int)
            X['_record_month'] = t.dt.month.fillna(BASE_MONTH).astype(int)
            # Calendar features kept in output (these are useful as both
            # ordinal features and as sin/cos pairs below).
            X['dow']          = t.dt.dayofweek.fillna(0).astype(int)       # 0=Mon ... 6=Sun
            X['month']        = t.dt.month.fillna(BASE_MONTH).astype(int)  # 1-12
            X['quarter']      = t.dt.quarter.fillna(((BASE_MONTH - 1) // 3) + 1).astype(int)  # 1-4
            X['day_of_month'] = t.dt.day.fillna(15).astype(int)            # 1-31
            X['day_of_year']  = t.dt.dayofyear.fillna(180).astype(int)     # 1-366
        else:
            X['_record_year']  = BASE_YEAR
            X['_record_month'] = BASE_MONTH
            X['dow']          = 0
            X['month']        = BASE_MONTH
            X['quarter']      = ((BASE_MONTH - 1) // 3) + 1
            X['day_of_month'] = 15
            X['day_of_year']  = 180
        # Sin/cos cyclical encodings — let trees learn that month=12 is close
        # to month=1, dow=6 is close to dow=0, etc.
        X['dow_sin']          = np.sin(2 * np.pi *  X['dow']          / 7)
        X['dow_cos']          = np.cos(2 * np.pi *  X['dow']          / 7)
        X['month_sin']        = np.sin(2 * np.pi * (X['month']        - 1) / 12)
        X['month_cos']        = np.cos(2 * np.pi * (X['month']        - 1) / 12)
        X['day_of_month_sin'] = np.sin(2 * np.pi * (X['day_of_month'] - 1) / 31)
        X['day_of_month_cos'] = np.cos(2 * np.pi * (X['day_of_month'] - 1) / 31)
        X['day_of_year_sin']  = np.sin(2 * np.pi * (X['day_of_year']  - 1) / 366)
        X['day_of_year_cos']  = np.cos(2 * np.pi * (X['day_of_year']  - 1) / 366)
        if 'year' in X.columns:
            # 'year' may now be a string (post-coalesce, since nav_year may be a string)
            X['year'] = pd.to_numeric(X['year'], errors='coerce')
            X['age'] = X['_record_year'] - X['year']
            X.loc[(X['age'] < 0) | (X['age'] > 60), 'age'] = np.nan
            X['age_sq'] = X['age'] ** 2
        if self.use_macro:
            ym = list(zip(X['_record_year'].values, X['_record_month'].values))
            X['cpi_at_sale']     = [CPI_MONTHLY.get(k, CPI_BASE)              for k in ym]
            X['manheim_at_sale'] = [MANHEIM_MONTHLY_FULL.get(k, MANHEIM_BASE) for k in ym]
            X['loan_at_sale']    = [LOAN_MONTHLY.get(k, LOAN_BASE)            for k in ym]
        if self.use_geo and 'vazipcode' in X.columns:
            z = (X['vazipcode'].astype(str)
                              .str.extract(r'(\d{1,5})')[0]
                              .str.zfill(5))
            X['zip_region']  = z.str[0].fillna('NA')      # 1-digit; 10 buckets
            X['zip_first2']  = z.str[:2].fillna('NA')      # 2-digit regional; ~100 buckets
            X['zip_first3']  = z.str[:3].fillna('NA')      # 3-digit state-area; ~900 buckets
            # zip_full as a string (will be frequency-encoded later by FREQ_COLS_BASE).
            # Cardinality is high (~15K-40K), so we cap the freq table to the
            # top 5000 most common ZIPs at fit time (see _apply_freq override).
            X['zip_full']    = z.fillna('NA')
            X['zip_lat']     = z.map(self.zip_lat_map)
            X['zip_lon']     = z.map(self.zip_lon_map)
            X = X.drop(columns=['vazipcode'])
        X = X.drop(columns=['_record_year','_record_month'], errors='ignore')
        if self.time_col in X.columns:
            X = X.drop(columns=[self.time_col])
        return X

    def _engineer(self, X):
        if 'age' in X.columns:
            X['age_bucket'] = pd.cut(X['age'], bins=self.AGE_EDGES, labels=False).fillna(-1).astype(int)
        if 'mileage' in X.columns:
            X['mileage_bucket'] = pd.cut(X['mileage'], bins=self.MILEAGE_EDGES, labels=False).fillna(-1).astype(int)
            age_safe = X['age'].replace(0, np.nan) if 'age' in X.columns else np.nan
            X['miles_per_year'] = X['mileage'] / age_safe
        if 'nav_condition' in X.columns:
            X['runs_flag'] = X['nav_condition'].map(_is_runs_condition).astype(int)
        else:
            X['runs_flag'] = 0
        for src_col, sev_map, new_col in SEV_COLS:
            if src_col in X.columns:
                X[new_col] = X[src_col].map(sev_map).fillna(-1).astype(int)
        unk_cols_present = [c for c in UNKNOWN_FLAG_COLS if c in X.columns]
        if unk_cols_present:
            unk = pd.DataFrame({
                c: X[c].map(lambda v, _uid=UNKNOWN_CONDITION_IDS[c]: _is_unknown_condition(v, _uid)).astype(int)
                for c in unk_cols_present
            })
            X['n_unknowns']  = unk.sum(axis=1)
            X['all_unknown'] = (unk.sum(axis=1) == len(unk_cols_present)).astype(int)
            X['any_unknown'] = (unk.sum(axis=1) > 0).astype(int)
        mech_parts = [c for c in ['engine_severity','trans_severity','tire_severity'] if c in X.columns]
        if mech_parts:
            mech = X[mech_parts].replace(-1, np.nan)
            X['mechanical_severity_sum']  = mech.sum(axis=1, min_count=1)
            X['mechanical_severity_mean'] = mech.mean(axis=1)
            X['mechanical_severity_max']  = mech.max(axis=1)
        # NOTE: dsrating_num and has_dsrating used to be engineered here from
        # the 'dsrating' column. Per user request, the dsrating column and its
        # derivatives are dropped (not available at inference time). See
        # USER_DROP_COLS in the class header.
        if {'age','n_unknowns'}.issubset(X.columns):
            X['old_and_unknown'] = ((X['age'] >= 15) & (X['n_unknowns'] >= 3)).astype(int)

        # Worst-case-tier interactions ($2.5K-10K band): the tier/field
        # correlation analysis (worst_case_analysis_2500_10000/) showed
        # condition-unknown counts and mechanical severity correlate with
        # systematic underprediction in this band far more than in the
        # $100-2.5K band, but neither is interacted with mileage/age today.
        # Numeric x numeric/bucket interactions, matching the cult_x_age /
        # mileage_unknown_x_age pattern (plain multiplication, trees handle
        # the rest).
        if self._wtf('unknowns_x_mileage_bkt') and {'n_unknowns','mileage_bucket'}.issubset(X.columns):
            X['unknowns_x_mileage_bkt'] = X['n_unknowns'] * X['mileage_bucket']
        if self._wtf('unknowns_x_age_bkt') and {'n_unknowns','age_bucket'}.issubset(X.columns):
            X['unknowns_x_age_bkt'] = X['n_unknowns'] * X['age_bucket']
        if self._wtf('mech_severity_x_mileage_bkt') and {'mechanical_severity_mean','mileage_bucket'}.issubset(X.columns):
            X['mech_severity_x_mileage_bkt'] = (
                X['mechanical_severity_mean'].fillna(-1) * X['mileage_bucket']
            )
        def cc(*cols):
            out = cols[0].fillna('na').astype(str)
            for c in cols[1:]: out = out + '__' + c.fillna('na').astype(str)
            return out
        pairs = [
            ('make_x_body_type',     ['make','body_type']),
            ('make_x_vehicle_type',  ['make','vehicle_type']),
            ('vtype_x_nav_condition',['vehicle_type','nav_condition']),
            ('body_x_drive',         ['body_type','drive_type']),
            ('condition_combo',      ['nav_condition','enginecondition']),
            ('engine_x_trans_cond',  ['enginecondition','transmissioncondition']),
        ]
        for new, src in pairs:
            if set(src).issubset(X.columns):
                X[new] = cc(*[X[c] for c in src])
        if {'nav_condition','age_bucket'}.issubset(X.columns):
            X['nav_cond_x_age_bkt'] = cc(X['nav_condition'], X['age_bucket'].astype(str))
        if {'runs_flag','mileage_bucket'}.issubset(X.columns):
            X['runs_x_mileage_bkt'] = X['runs_flag'].astype(str) + '__' + X['mileage_bucket'].astype(str)
        if {'nav_condition','enginecondition','transmissioncondition','bodypaintcondition'}.issubset(X.columns):
            X['all_cond_combo'] = cc(X['nav_condition'], X['enginecondition'],
                                     X['transmissioncondition'], X['bodypaintcondition'])
        if self.use_geo:
            if {'zip_region','vehicle_type'}.issubset(X.columns):
                X['zip_region_x_vehicle_type'] = cc(X['zip_region'], X['vehicle_type'])
            if {'zip_region','body_type'}.issubset(X.columns):
                X['zip_region_x_body_type'] = cc(X['zip_region'], X['body_type'])
            if {'zip_region','nav_condition'}.issubset(X.columns):
                X['zip_region_x_nav_condition'] = cc(X['zip_region'], X['nav_condition'])
            if {'zip_lat','age'}.issubset(X.columns):
                X['lat_x_age'] = X['zip_lat'] * X['age']
            if {'zip_lon','age'}.issubset(X.columns):
                X['lon_x_age'] = X['zip_lon'] * X['age']
        if self.use_cult and 'is_cult' in X.columns:
            if 'age' in X.columns:
                X['cult_x_age'] = X['is_cult'] * X['age'].fillna(-1)
            if 'mileage_bucket' in X.columns:
                X['cult_x_mileage_bkt'] = X['is_cult'] * X['mileage_bucket']
            if 'cult_tier_num' in X.columns and 'age' in X.columns:
                X['culttier_x_age'] = X['cult_tier_num'].fillna(0) * X['age'].fillna(-1)
            if 'cult_origsens_num' in X.columns and 'runs_flag' in X.columns:
                X['origsens_x_runs'] = X['cult_origsens_num'].fillna(-1) * X['runs_flag']
            if self._wtf('cult_x_n_unknowns') and 'n_unknowns' in X.columns:
                # Cult route shows its own elevated MAE (by_route breakdown);
                # missing condition data plausibly discounts value less on a
                # cult car (make/model desirability dominates) than on a
                # standard one. Never interacted with completeness before.
                X['cult_x_n_unknowns'] = X['is_cult'] * X['n_unknowns']

        # Additional interaction features (user-requested).
        #   - string-concat interactions become frequency-encoded later
        #   - numeric interactions are used as-is by trees
        if {'make','age_bucket'}.issubset(X.columns):
            X['make_x_age'] = cc(X['make'], X['age_bucket'].astype(str))
        if {'month','age_bucket'}.issubset(X.columns):
            X['month_x_age'] = cc(X['month'].astype(str), X['age_bucket'].astype(str))
        if {'mileage','age'}.issubset(X.columns):
            X['mileage_x_age'] = X['mileage'].fillna(0) * X['age'].fillna(-1)
        if {'year','dow'}.issubset(X.columns):
            X['year_x_dow'] = X['year'].fillna(-1) * X['dow'].astype(int)
        if self.use_geo and {'zip_region','mileage_bucket'}.issubset(X.columns):
            X['zip_region_x_mileage_bkt'] = cc(X['zip_region'], X['mileage_bucket'].astype(str))
        if {'quarter','make'}.issubset(X.columns):
            X['quarter_x_make'] = cc(X['quarter'].astype(str), X['make'])
        if self._wtf('vtype_x_mileage_bkt') and {'vehicle_type','mileage_bucket'}.issubset(X.columns):
            # vehicle_type is already combined with make/nav_condition/zip
            # region but never with mileage — trucks/vans/SUVs plausibly
            # hold value on a different mileage curve than sedans, which is
            # exactly the kind of body-type premium the $2.5K-10K tier's
            # systematic underprediction is missing.
            X['vtype_x_mileage_bkt'] = cc(X['vehicle_type'], X['mileage_bucket'].astype(str))

        # Condition-x-make interactions ($100-2000 tier): the correlation
        # analysis there (worst_case_analysis_100_2000/) found the worst-
        # overpredicted rows have BETTER condition signals (more
        # "Operational"/"Runs & Drives", less damage reported) than the
        # rest of the band, not worse -- the model applies one global
        # "good condition -> higher price" rule regardless of make. These
        # let it learn a make-specific version of that rule instead. See
        # CONDITION_MAKE_FEATURE_COLS / _cmf().
        if self._cmf('runs_x_make') and {'runs_flag', 'make'}.issubset(X.columns):
            X['runs_x_make'] = cc(X['runs_flag'].astype(str), X['make'])
        if self._cmf('mech_severity_x_make') and {'mechanical_severity_mean', 'make'}.issubset(X.columns):
            # Bucketed (not the raw decimal) so each make x severity-band
            # combo has enough rows to give a meaningful frequency count --
            # matches the enginehp_bucket/age_bucket precedent elsewhere in
            # this file. Bands: 0 = perfect, 1 = minor issues, 2 = major/
            # heavily-unknown; -1 = no severity signal at all (all three
            # source condition fields missing).
            sev_bkt = pd.cut(X['mechanical_severity_mean'], bins=[-np.inf, 0, 2, np.inf], labels=False)
            sev_bkt = sev_bkt.fillna(-1).astype(int)
            X['mech_severity_x_make'] = cc(sev_bkt.astype(str), X['make'])
        if self._cmf('all_cond_combo_x_make') and {'all_cond_combo', 'make'}.issubset(X.columns):
            X['all_cond_combo_x_make'] = cc(X['all_cond_combo'], X['make'])

        # Engine-derived interactions (use parsed engine features from _basic_clean)
        # All four are added if their source columns are present; otherwise skipped silently.
        if {'displacementl', 'age'}.issubset(X.columns):
            # Numeric × numeric interaction. NaN displacement -> 0 (no signal);
            # NaN age -> -1 (matches existing pattern for missing age).
            X['displacementl_x_age'] = X['displacementl'].fillna(0) * X['age'].fillna(-1)
        if {'enginecylinders', 'make'}.issubset(X.columns):
            # NaN cylinders -> '<NA>' string (pandas Int64 nullable int rendering),
            # which becomes a distinct freq-encoded category.
            cyl_str = X['enginecylinders'].astype('Int64').astype(str)
            X['enginecylinders_x_make'] = cc(cyl_str, X['make'])
        if 'enginehp' in X.columns:
            # Bucket horsepower into 6 ranges. Buckets match common HP tiers:
            # economy (<120) / standard (120-180) / sport (180-250) / performance
            # (250-350) / muscle (350-500) / supercar (500+). -1 means unknown.
            hp_bkt = pd.cut(
                X['enginehp'],
                bins=[-np.inf, 120, 180, 250, 350, 500, np.inf],
                labels=False,
            )
            X['enginehp_bucket'] = hp_bkt.fillna(-1).astype('int32')
            if 'age_bucket' in X.columns:
                X['enginehp_bkt_x_age_bkt'] = cc(
                    hp_bkt.astype('Int64').astype(str),  # NaN -> '<NA>'
                    X['age_bucket'].astype(str),
                )
        if {'engineconfiguration', 'make'}.issubset(X.columns):
            X['engineconfig_x_make'] = cc(
                X['engineconfiguration'].fillna('na'),
                X['make'],
            )

        # Other-damages interactions. Note: damage is sparse (~99% zero) so
        # both interactions are sparse too. The numeric one is mostly zero;
        # the string-concat one mostly hits "0__<bucket>" categories. We add
        # them anyway since the cost is small and they help when damage IS
        # reported, but expect modest SHAP attribution.
        if {'has_other_damage', 'age'}.issubset(X.columns):
            X['damage_x_age'] = X['has_other_damage'] * X['age'].fillna(-1)
        if {'has_other_damage', 'mileage_bucket'}.issubset(X.columns):
            X['damage_x_mileage_bkt'] = cc(
                X['has_other_damage'].astype(str),
                X['mileage_bucket'].astype(str),
            )

        # Mileage-trust flag interactions (true_mileage_unknown: 1 = odometer
        # reading isn't trustworthy). Numeric x numeric/bucket use plain
        # multiplication (matches cult_x_age/cult_x_mileage_bkt pattern); the
        # make interaction is string-concat, freq-encoded downstream.
        if {'true_mileage_unknown', 'age'}.issubset(X.columns):
            X['mileage_unknown_x_age'] = (
                X['true_mileage_unknown'].fillna(0) * X['age'].fillna(-1)
            )
        if {'true_mileage_unknown', 'mileage_bucket'}.issubset(X.columns):
            X['mileage_unknown_x_mileage_bucket'] = (
                X['true_mileage_unknown'].fillna(0) * X['mileage_bucket']
            )
        if {'true_mileage_unknown', 'make'}.issubset(X.columns):
            X['mileage_unknown_x_make'] = cc(
                X['true_mileage_unknown'].fillna(-1).astype(str), X['make']
            )
        if self._wtf('mileage_unknown_x_n_unknowns') and {'true_mileage_unknown', 'n_unknowns'}.issubset(X.columns):
            # Compounding data-quality signal: true_mileage_unknown and
            # condition-unknown count move together in the $2.5K-10K worst-
            # underpredicted rows (both show ~+22pp lift there) but were
            # never interacted with each other, only separately with
            # age/mileage_bucket/make.
            X['mileage_unknown_x_n_unknowns'] = (
                X['true_mileage_unknown'].fillna(0) * X['n_unknowns']
            )

        # Clean-title interactions (clean_title: 1 = clean title, vs.
        # branded/salvage/rebuilt).
        if {'clean_title', 'age'}.issubset(X.columns):
            X['clean_title_x_age'] = X['clean_title'].fillna(0) * X['age'].fillna(-1)
        if {'clean_title', 'mileage_bucket'}.issubset(X.columns):
            X['clean_title_x_mileage_bucket'] = (
                X['clean_title'].fillna(0) * X['mileage_bucket']
            )
        return X

    # =================================================================
    # Vehicle-profile clustering — NO TARGET LEAKAGE
    # =================================================================
    # Cluster (make, model) pairs by their vehicle profile (size, condition,
    # body type, etc.) — NOT by their sale prices. The model will still learn
    # depreciation from price data, but with the cluster providing an extra
    # grouping signal. Honest naming: this is a "vehicle profile cluster,"
    # not a "depreciation cluster," because we never use salevalue.

    # Numeric attributes aggregated per (make, model) pair
    _CLUSTER_NUMERIC_ATTRS = [
        'mileage',                                              # typical mileage
        'age',                                                  # typical age at sale
        'miles_per_year',                                       # usage intensity
        'nav_severity', 'body_severity', 'engine_severity',     # condition profile
        'trans_severity', 'tire_severity', 'interior_severity',
        'runs_flag',                                            # mostly driveable?
        'n_unknowns',                                           # data quality proxy
    ]
    # Categorical attributes — we'll use the modal value per pair and one-hot encode
    _CLUSTER_CATEGORICAL_ATTRS = ['vehicle_type', 'body_type', 'drive_type']

    def _build_cluster_vectors(self, X_engineered):
        """Build per-(make, model) attribute vectors WITHOUT using salevalue.

        Returns
        -------
        pairs : list of (make, model) tuples that meet min_samples
        vectors : numpy array (n_pairs, n_dims) of standardized attribute vectors
        """
        df = X_engineered.copy()
        df['_make_n']  = df['make'].fillna('na').astype(str)
        df['_model_n'] = df['model'].fillna('na').astype(str)

        counts = df.groupby(['_make_n', '_model_n']).size()
        eligible = counts[counts >= self.cluster_min_samples].index.tolist()
        if len(eligible) < self.n_clusters:
            return [], np.empty((0, 0))

        # Build numeric aggregates per pair
        numeric_present = [c for c in self._CLUSTER_NUMERIC_ATTRS if c in df.columns]
        if not numeric_present:
            return [], np.empty((0, 0))
        agg_numeric = (df.groupby(['_make_n', '_model_n'])[numeric_present]
                          .mean()
                          .loc[eligible])

        # Build categorical mode aggregates per pair, then one-hot the global top-N
        cat_present = [c for c in self._CLUSTER_CATEGORICAL_ATTRS if c in df.columns]
        cat_frames = []
        for col in cat_present:
            modes = (df.groupby(['_make_n', '_model_n'])[col]
                        .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else 'na')
                        .loc[eligible])
            # One-hot the top 8 values to keep dimensionality bounded
            top_vals = modes.value_counts().head(8).index.tolist()
            for v in top_vals:
                cat_frames.append(pd.DataFrame({f'{col}__{v}': (modes == v).astype(int)},
                                                index=modes.index))
        if cat_frames:
            agg_cat = pd.concat(cat_frames, axis=1)
            agg = pd.concat([agg_numeric, agg_cat], axis=1)
        else:
            agg = agg_numeric

        # Impute missing numeric values with column median (still target-free)
        agg = agg.fillna(agg.median(numeric_only=True)).fillna(0.0)

        # Standardize — k-means is scale-sensitive, and our features span many orders
        # of magnitude (mileage in tens of thousands vs runs_flag in [0, 1]).
        scaler = StandardScaler()
        vectors = scaler.fit_transform(agg.values)
        return list(agg.index), vectors

    def _fit_vehicle_profile_clusters(self, X_engineered):
        """Fit k-means on (make, model) vehicle-profile vectors.

        TARGET-FREE: never touches salevalue. Cluster IDs reflect vehicle attributes
        and condition profile only. Stores self.cluster_map_ mapping
        (make, model) -> int cluster_id (or skip if too few eligible pairs).
        """
        if self.n_clusters <= 0:
            self.cluster_map_, self.cluster_k_ = {}, 0
            return

        pairs, vectors = self._build_cluster_vectors(X_engineered)
        if len(pairs) < self.n_clusters:
            print(f"[clusters] only {len(pairs)} (make,model) pairs meet "
                  f"min_samples={self.cluster_min_samples}; need >= n_clusters="
                  f"{self.n_clusters}. Skipping clustering.")
            self.cluster_map_, self.cluster_k_ = {}, 0
            return

        km = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        cluster_ids = km.fit_predict(vectors)
        self.cluster_map_ = {pair: int(cid) for pair, cid in zip(pairs, cluster_ids)}
        self.cluster_centers_ = km.cluster_centers_
        self.cluster_k_ = int(self.n_clusters)
        print(f"[clusters] fit {self.cluster_k_} vehicle-profile clusters on "
              f"{len(pairs):,} (make, model) pairs "
              f"(target-free; clustered on attributes/condition profile only)")

    def _apply_clusters(self, X):
        """Look up vehicle_profile_cluster_id and compute its interactions.

        Adds:
          vehicle_profile_cluster_id  -- int; -1 if (make, model) wasn't clustered
          cluster_x_age_bkt           -- string concat, freq-encoded downstream
          cluster_x_mileage_bkt       -- string concat, freq-encoded downstream

        Requires self.cluster_map_ to exist (set by _fit_vehicle_profile_clusters
        during fit). If empty, this is a no-op.
        """
        if not getattr(self, 'cluster_map_', None):
            return X
        mk = X['make'].fillna('na').astype(str)
        md = X['model'].fillna('na').astype(str)
        keys = list(zip(mk, md))
        cluster_ids = np.array([self.cluster_map_.get(k, -1) for k in keys], dtype=int)
        X['vehicle_profile_cluster_id'] = cluster_ids
        if 'age_bucket' in X.columns:
            X['cluster_x_age_bkt'] = (X['vehicle_profile_cluster_id'].astype(str)
                                       + '__' + X['age_bucket'].astype(str))
        if 'mileage_bucket' in X.columns:
            X['cluster_x_mileage_bkt'] = (X['vehicle_profile_cluster_id'].astype(str)
                                           + '__' + X['mileage_bucket'].astype(str))
        return X
    # =================================================================

    @property
    def FREQ_COLS(self):
        cols = list(self.FREQ_COLS_BASE)
        if self.use_geo: cols += self.GEO_FREQ_COLS
        return cols

    def _smooth_mean(self, series, target):
        agg = pd.DataFrame({'x': series.values, 'y': np.asarray(target)}).groupby('x')['y'].agg(['mean','count'])
        return (agg['count'] * agg['mean'] + self.smoothing * self.global_mean_) / (agg['count'] + self.smoothing)

    def _apply_freq(self, X):
        for col, fmap in self.freq_maps_.items():
            if col in X.columns:
                X[f'{col}_freq'] = X[col].map(fmap).fillna(0)
        # zip_full's raw form has very high cardinality (5000+ after capping);
        # the int-encoded version isn't useful for tree splits. Keep only the
        # frequency-encoded version (zip_full_freq), drop the raw string.
        # zip_first2 and zip_first3 are kept in raw form (lower cardinality)
        # so they can also become int-encoded features below.
        if 'zip_full' in X.columns:
            X = X.drop(columns=['zip_full'])
        return X

    def _apply_tgt_enc_full(self, X):
        for col, tmap in self.target_enc_maps_.items():
            if col in X.columns:
                X[f'{col}_tgt_enc'] = X[col].map(tmap).fillna(self.global_mean_)
        return X

    def _apply_int_enc(self, X):
        for col, m in self.int_maps_.items():
            if col in X.columns:
                X[col] = X[col].map(m).fillna(-1).astype('int32')
        return X

    def _finalize(self, X):
        X = X.drop(columns=[TARGET_COL], errors='ignore')
        if hasattr(self, 'feature_cols_'):
            for c in self.feature_cols_:
                if c not in X.columns: X[c] = np.nan
            X = X[self.feature_cols_]
        return X

    def fit(self, X, y=None):
        if self.with_target_encoding:
            self.global_mean_ = float(np.asarray(y).mean())
        Xc = self._basic_clean(X.copy())
        Xc = self._engineer(Xc)
        # Fit vehicle-profile clusters BEFORE freq maps so the cluster
        # interaction columns get freq-encoded along with everything else.
        # This is target-free: no salevalue / y access in the cluster fit.
        self._fit_vehicle_profile_clusters(Xc)
        Xc = self._apply_clusters(Xc)
        # Build frequency maps. Cap zip_full at the top-5000 most common ZIPs
        # since its cardinality (~15-40K) would otherwise produce a huge map
        # and overfit. Less-common ZIPs map to 0 at transform time, which
        # XGBoost handles fine.
        ZIP_FULL_CAP = 5000
        self.freq_maps_ = {}
        for col in self.FREQ_COLS:
            if col not in Xc.columns:
                continue
            counts = Xc[col].value_counts(dropna=False)
            if col == 'zip_full' and len(counts) > ZIP_FULL_CAP:
                counts = counts.head(ZIP_FULL_CAP)
            self.freq_maps_[col] = counts.to_dict()
        if self.with_target_encoding:
            self.target_enc_maps_ = {col: self._smooth_mean(Xc[col], y).to_dict()
                                     for col in self.TARGET_ENC_COLS if col in Xc.columns}
        else:
            self.target_enc_maps_ = {}
        Xc2 = self._apply_freq(Xc.copy())
        if self.with_target_encoding:
            Xc2 = self._apply_tgt_enc_full(Xc2)
        self.cat_cols_ = Xc2.select_dtypes(include='object').columns.tolist()
        self.int_maps_ = {col: {v: i for i, v in enumerate(Xc2[col].dropna().unique())}
                          for col in self.cat_cols_}
        Xc3 = self._apply_int_enc(Xc2)
        Xc3 = Xc3.drop(columns=[TARGET_COL], errors='ignore')
        self.feature_cols_ = list(Xc3.columns)
        return self

    def transform(self, X):
        Xc = self._basic_clean(X.copy())
        Xc = self._engineer(Xc)
        Xc = self._apply_clusters(Xc)
        Xc = self._apply_freq(Xc)
        if self.with_target_encoding:
            Xc = self._apply_tgt_enc_full(Xc)
        Xc = self._apply_int_enc(Xc)
        return self._finalize(Xc)

    def fit_transform_with_oof(self, X, y):
        """OOF target encoding for training (avoids leakage)."""
        self.fit(X, y)
        Xc = self._basic_clean(X.copy())
        Xc = self._engineer(Xc)
        Xc = self._apply_clusters(Xc)
        Xc = self._apply_freq(Xc)
        if self.with_target_encoding:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
            y_arr = np.asarray(y)
            for col in self.TARGET_ENC_COLS:
                if col not in Xc.columns: continue
                oof = np.full(len(Xc), self.global_mean_, dtype=float)
                vals = Xc[col].reset_index(drop=True)
                for fold_in, fold_out in kf.split(vals):
                    fmap = self._smooth_mean(vals.iloc[fold_in], pd.Series(y_arr[fold_in])).to_dict()
                    oof[fold_out] = vals.iloc[fold_out].map(fmap).fillna(self.global_mean_).values
                Xc[f'{col}_tgt_enc'] = oof
        Xc = self._apply_int_enc(Xc)
        return self._finalize(Xc)


# Monotonic feature constraints (used in training)
MONO_FEATURES = {
    'age': -1, 'age_sq': -1,
    'mileage': -1, 'mileage_bucket': -1, 'miles_per_year': -1,
    'nav_severity': -1, 'body_severity': -1, 'engine_severity': -1,
    'trans_severity': -1, 'tire_severity': -1, 'interior_severity': -1,
    'mechanical_severity_sum': -1, 'mechanical_severity_mean': -1, 'mechanical_severity_max': -1,
    'n_unknowns': -1, 'all_unknown': -1, 'any_unknown': -1, 'old_and_unknown': -1,
    'loan_at_sale': -1, 'manheim_at_sale': +1,
    'is_cult': +1, 'cult_tier_num': +1, 'cult_enthusiast_score': +1,
    'cult_uplift_mid': +1, 'cult_uplift_high': +1, 'cult_last_of_kind': +1,
    'true_mileage_unknown': -1, 'clean_title': +1,
}


def cpi_ratio_arr(df_subset, time_col=TIME_COL):
    years = pd.to_datetime(df_subset[time_col]).dt.year.fillna(BASE_YEAR).astype(int)
    return (CPI_BASE / years.map(CPI_ANNUAL_CONF).fillna(CPI_BASE)).values


def adjust_target(y, R, alpha):     return y * (1.0 + alpha * R)
def deflate_pred(pred, R, alpha):  return pred / (1.0 + alpha * R)
