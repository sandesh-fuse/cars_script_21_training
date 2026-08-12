"""
diagnose_schema_mapping.py
===========================
Sanity-checks schema_adapter.map_raw_features_to_legacy() against a SAMPLE of
a real (possibly huge) CSV, without loading the whole file into memory.

Checks performed:
  1. Loads only the first N rows (pd.read_csv(..., nrows=N) stops early —
     safe on a multi-GB file).
  2. Shows which columns got renamed, and flags any NEW_TO_OLD_SCHEMA_MAP
     source column that ISN'T present in your file (typo / schema drift).
  3. Confirms every column preprocessor.py actually reads (TARGET_COL,
     TIME_COL, engine_name, other_damages, etc.) is present after mapping.
  4. Runs the REAL preprocessor.py damage-parsing and engine-parsing methods
     (SaleValuePreprocessor()._parse_other_damages / _parse_engine_features)
     on the mapped sample — not a reimplementation — and prints the result
     so you can eyeball it against the original raw values.
  5. Checks record_creation_date parses cleanly (no unexpected NaT blowup).

USAGE:
    python diagnose_schema_mapping.py --csv your_file.csv
    python diagnose_schema_mapping.py --csv your_file.csv --n 2000
"""

import argparse
import sys

import pandas as pd

from schema_adapter import (
    NEW_TO_OLD_SCHEMA_MAP,
    map_raw_features_to_legacy,
    filter_to_known_columns,
)
from preprocessor import SaleValuePreprocessor, TARGET_COL, TIME_COL

# Columns preprocessor.py reads directly by name (beyond TARGET_COL/TIME_COL).
# Keep in sync with schema_adapter's mapping targets if you add more.
EXPECTED_LEGACY_COLS = [
    "make", "model", "year", "engine_name", "transmission_name",
    "nav_condition", "enginecondition", "transmissioncondition",
    "bodypaintcondition", "interiorcondition", "tirecondition",
    "other_damages", "vazipcode",
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the (possibly large) CSV")
    ap.add_argument("--n", type=int, default=500, help="Number of rows to sample (default 500)")
    args = ap.parse_args()

    section(f"Loading first {args.n} rows of {args.csv}")
    df_raw = pd.read_csv(args.csv, nrows=args.n, low_memory=False)
    print(f"Sample shape: {df_raw.shape}")

    # --- 1. Which mapped source columns are actually present? ---
    section("New-schema columns expected by schema_adapter vs. found in file")
    missing_sources = [c for c in NEW_TO_OLD_SCHEMA_MAP if c not in df_raw.columns]
    present_sources = [c for c in NEW_TO_OLD_SCHEMA_MAP if c in df_raw.columns]
    print(f"Present  ({len(present_sources)}): {present_sources}")
    print(f"MISSING  ({len(missing_sources)}): {missing_sources}")
    if missing_sources:
        print(
            "  -> Not necessarily a problem (file may already be legacy-schema, "
            "or that field is simply absent), but double-check for typos if you "
            "expected these to be there."
        )

    # --- 2. Apply the noise filter (mirrors what train_save_script21.py now does) ---
    section("Applying filter_to_known_columns() — drops DB noise/metadata")
    df_filtered = filter_to_known_columns(df_raw.copy())
    print(f"Shape after filtering: {df_filtered.shape} (was {df_raw.shape})")

    # --- 3. Apply the mapping ---
    section("Applying map_raw_features_to_legacy()")
    try:
        df = map_raw_features_to_legacy(df_filtered)
    except ValueError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    renamed = {k: v for k, v in NEW_TO_OLD_SCHEMA_MAP.items() if k in present_sources}
    print(f"Renamed {len(renamed)} columns: {renamed}")

    # --- 4. Does everything preprocessor.py needs exist now? ---
    section("Legacy columns preprocessor.py expects — present after mapping?")
    need = [TARGET_COL, TIME_COL] + EXPECTED_LEGACY_COLS
    for col in need:
        status = "OK" if col in df.columns else "MISSING"
        print(f"  [{status:7}] {col}")

    # --- 5. record_creation_date parses cleanly? ---
    if TIME_COL in df.columns:
        section(f"Checking {TIME_COL} parses cleanly")
        parsed = pd.to_datetime(df[TIME_COL], errors="coerce")
        n_bad = parsed.isna().sum() - df[TIME_COL].isna().sum()
        print(f"Dtype after mapping: {df[TIME_COL].dtype}")
        print(f"Range: {parsed.min()} .. {parsed.max()}")
        print(f"Rows that became NaT due to unparseable dates: {n_bad} / {len(df)}")

    # --- 6. Real preprocessor.py damage parsing on the mapped sample ---
    if "other_damages" in df.columns:
        section("Running the REAL preprocessor._parse_other_damages() on the sample")
        pre = SaleValuePreprocessor()
        damage_out = pre._parse_other_damages(df.copy())
        cols = ["has_other_damage", "n_other_damages", "other_damages_normalized",
                "has_mold", "has_undercarriage_rust", "has_smog_fail"]
        print(damage_out[[c for c in cols if c in damage_out.columns]].value_counts().head(20))

    # --- 7. Real preprocessor.py engine parsing on the mapped sample ---
    if "engine_name" in df.columns:
        section("Running the REAL preprocessor._parse_engine_features() on the sample")
        pre = SaleValuePreprocessor()
        engine_out = pre._parse_engine_features(df.copy())
        cols = ["engine_name", "enginehp", "displacementl", "enginecylinders", "valvetraindesign"]
        print(engine_out[[c for c in cols if c in engine_out.columns]].head(15).to_string())

    section("Done")
    print("Eyeball the sections above. If everything under section 4 says OK, "
          "the date range looks sane, and the damage/engine tables look right, "
          "the mapping is working correctly on this sample.")


if __name__ == "__main__":
    main()
