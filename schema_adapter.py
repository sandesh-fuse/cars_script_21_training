"""
schema_adapter.py
==================
Translation layer between the upstream (new) database schema and the legacy
column names `preprocessor.py` and the pre-trained models expect.

Why this exists: the upstream DB schema changed (renamed columns, renamed
target, ISO8601 timestamps, consolidated damage flags). Rather than rewrite
`preprocessor.py` or retrain the production models, incoming data is renamed
back to the legacy schema immediately after it's read (CSV at training time,
JSON payload at inference time) — see `map_raw_features_to_legacy` (DataFrame,
bulk/training use) and `map_raw_features_to_legacy_record` (single dict,
API use, wired in separately when the inference side is upgraded).

Centralized on purpose: if the DB team renames a column again, only the
dict below needs to change.
"""

import pandas as pd

# ============================================================
# NEW (incoming) schema column -> legacy (ML pipeline) column
# ============================================================
NEW_TO_OLD_SCHEMA_MAP = {
    # --- Core target & time ---
    "sale_value": "salevalue",
    "creation_datetime": "record_creation_date",

    # --- Condition & visuals ---
    "vehicle_cond_picklist_id_name": "nav_condition",
    "engine_cond_picklist_id_name": "enginecondition",
    "transmission_cond_picklist_id_name": "transmissioncondition",
    "body_paint_cond_picklist_id_name": "bodypaintcondition",
    "interior_cond_picklist_id_name": "interiorcondition",
    "tire_cond_picklist_id_name": "tirecondition",
    "color": "nav_color",
    "zip" : "vazipcode",

    # --- Engine & drivetrain ---
    "engines_name": "engine_name",             # regex-parsed for HP/displacement/etc. downstream
    "transmissions_name": "transmission_name",
    "ice_displacement": "displacementl",
    "ice_cylinders": "enginecylinders",
    "ice_block_type": "engineconfiguration",
    "ice_max_hp": "enginehp",

    # --- Core attributes & categoricals ---
    "vehicle_category": "body_type",
    "body_type": "oem_body_style",             # NOTE: incoming 'body_type' != legacy 'body_type' — see below
    "accessible_for_tow_truck": "accessiblefortwotruck",
    "located_at_donation_c_a": "locatedatdonationca",
    "speciality_item": "Specialty Item",
    "state_title_picklist_name": "state_province_of_title",
    "us_styles": "us_style_name",
    "state_picklist_id_name": "vstate_name",
    "vin_hin_no": "vin",                    # currently inert: preprocessor.py drops the VIN outright
                                                # and no is_valid_vin logic exists anywhere in the codebase
    "comment": "all_clean_notes",              # currently inert: not read anywhere in preprocessor.py

    # --- Damage (new schema: single value per row, not JSON/multi-select) ---
    # Maps onto preprocessor.py's _parse_other_damages(), which already treats
    # 'other_damages' as free text split on ';'/',' (never '/', so values like
    # "Won't Pass Smog/State Inspection" survive as one token). A lone value
    # per row (e.g. "Mold", "Fire Damage", or NaN) is valid input to that same
    # code path with zero extra parsing needed.
    "other_damage_pklist_id_name": "other_damages",
}

# `vehicle_category` -> `body_type` and `body_type` -> `oem_body_style` together mean the
# raw incoming `body_type` column is repurposed as `oem_body_style`, NOT dropped — order of
# operations in .rename() doesn't matter here since pandas renames off the ORIGINAL column
# names simultaneously, not sequentially, so no chaining collision occurs.


def _normalize_date_value(value):
    """Parse one timestamp (ISO8601 or legacy format, tz-aware or naive) and
    strip any timezone so it's safe to feed into columns that get bulk-parsed
    downstream (preprocessor.py calls pd.to_datetime on the whole column).

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


def map_raw_features_to_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a full DataFrame's columns from the new DB schema to the legacy
    ML schema. Used at training time, right after `pd.read_csv`.

    Safe to call on data that's already legacy-schema (or a mix of columns
    from both) — only columns present in NEW_TO_OLD_SCHEMA_MAP are touched,
    everything else passes through untouched.
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
            f"schema_adapter: multiple incoming columns would collide onto the same "
            f"legacy column name after renaming — refusing to guess which is "
            f"authoritative: {offenders}. Drop or rename one of each group before "
            f"calling map_raw_features_to_legacy()."
        )

    df = df.rename(columns=NEW_TO_OLD_SCHEMA_MAP)

    if "record_creation_date" in df.columns:
        df["record_creation_date"] = df["record_creation_date"].apply(_normalize_date_value)

    return df


def known_raw_columns() -> set:
    """All column names the pipeline recognizes in a raw row, before renaming.

    Union of:
      - NEW_TO_OLD_SCHEMA_MAP keys   — new-schema names that get renamed.
      - NEW_TO_OLD_SCHEMA_MAP values — legacy names, kept in case a file
        already uses them directly (so old-format CSVs keep working
        unchanged, not just new-format ones).
      - app.schemas.PredictRequest's declared fields — the documented raw
        input contract ("Field names must match the column names in your
        training CSV").

    Anything NOT in this set is DB noise/metadata the pipeline never reads
    (e.g. vehicle_uuid, api_log_id, engines_json — see preprocessor.py's
    DROP_COLS_NO_ZIP for examples of columns that exist in the raw export
    but were never features to begin with).
    """
    from app.schemas import PredictRequest

    try:
        pydantic_fields = set(PredictRequest.model_fields.keys())  # pydantic v2
    except AttributeError:
        pydantic_fields = set(PredictRequest.__fields__.keys())  # pydantic v1 fallback

    return (
        set(NEW_TO_OLD_SCHEMA_MAP.keys())
        | set(NEW_TO_OLD_SCHEMA_MAP.values())
        | pydantic_fields
    )


def filter_to_known_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop any column not in known_raw_columns() — i.e. anything that's
    neither a mapped new-schema field, a legacy field the pipeline already
    expects, nor a documented PredictRequest field.

    Call this right after reading the raw CSV, BEFORE map_raw_features_to_legacy(),
    to strip DB noise/junk columns before they ride along through training.
    """
    keep = known_raw_columns()
    kept_cols = [c for c in df.columns if c in keep]
    dropped = [c for c in df.columns if c not in keep]
    if verbose:
        if dropped:
            print(f"schema_adapter: dropping {len(dropped)} unrecognized column(s): {dropped}")
        else:
            print("schema_adapter: no unrecognized columns found — nothing dropped.")
    return df[kept_cols]


def map_raw_features_to_legacy_record(record: dict) -> dict:
    """Rename a single request dict's keys from the new schema to the legacy
    schema. Same mapping as map_raw_features_to_legacy, for the single-row
    API path (not yet wired into app/main.py — add there when the inference
    side of this migration is tackled).
    """
    renamed = {NEW_TO_OLD_SCHEMA_MAP.get(k, k): v for k, v in record.items()}
    if "record_creation_date" in renamed:
        renamed["record_creation_date"] = _normalize_date_value(renamed["record_creation_date"])
    return renamed
