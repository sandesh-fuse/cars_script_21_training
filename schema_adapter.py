"""
schema_adapter.py
==================
Historical home of the new-DB-schema -> legacy-ML-schema translation layer.

The actual mapping/rename logic now lives in preprocessor.py (see
NEW_TO_OLD_SCHEMA_MAP / map_raw_features_to_legacy there) — preprocessor.py
applies it automatically as the first step of its own _basic_clean(), so
neither train_save_script21.py nor app/inference_script21.py import this
module anymore; they hand either schema straight to SaleValuePreprocessor.

This module re-exports the same names purely for backward compatibility with
the standalone scripts that still import from here (diagnose_schema_mapping.py,
validate_live_predictions.py) — there's a single source of truth
(preprocessor.py) so the two can't drift out of sync.

`known_raw_columns()`/`filter_to_known_columns()` stay defined here rather
than in preprocessor.py since they cross-reference app.schemas.PredictRequest
— pulling that into preprocessor.py (imported by both training scripts and
both inference pipelines) would be an awkward reverse dependency on the app
package for something only these two diagnostic helpers need.
"""

from preprocessor import (  # noqa: F401 -- re-exported for backward compatibility
    NEW_TO_OLD_SCHEMA_MAP,
    map_raw_features_to_legacy,
    map_raw_features_to_legacy_record,
)


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


def filter_to_known_columns(df, verbose: bool = True):
    """Drop any column not in known_raw_columns() — i.e. anything that's
    neither a mapped new-schema field, a legacy field the pipeline already
    expects, nor a documented PredictRequest field.

    Call this right after reading a raw CSV, BEFORE map_raw_features_to_legacy(),
    to strip DB noise/junk columns before they ride along through training.

    train_save_script21.py no longer calls this — it restricts the CSV read
    itself to SCRIPT21_RAW_COLUMNS (+ the DataOne raw columns when
    --use-dataone is set) via `usecols=`, which is a stricter, cheaper
    filter applied at read time instead of after. Kept here for
    diagnose_schema_mapping.py / validate_live_predictions.py.
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
