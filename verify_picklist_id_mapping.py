"""
verify_picklist_id_mapping.py
==============================
Correctness gate for the script21 picklist-ID migration (schema_adapter.py's
NEW_TO_OLD_SCHEMA_MAP + preprocessor.py's merged text+ID severity/runs/
unknown/damage dicts). Hand-transcribing ~40 numeric IDs is easy to get
wrong in ways that are invisible in aggregate training metrics (a silently
mis-ordered severity value just looks like normal model noise) -- this
script asserts the exact expected engineered-feature values for a handful of
synthetic rows covering every column's edge cases, and fails loudly and
specifically if any of them are off.

Two things are checked:
  1. End-to-end: raw NEW-SCHEMA column names (vehicle_cond_picklist_id, etc.)
     -> schema_adapter.map_raw_features_to_legacy() -> SaleValuePreprocessor
     -> expected engineered severity/flag columns. This is exactly what
     train_save_script21.py now does.
  2. Backward-compat: the SAME preprocessor fed the OLD legacy TEXT columns
     directly (bypassing schema_adapter, as script17 / current live script21
     inference still do) produces IDENTICAL engineered values -- i.e. this
     migration changed nothing for existing callers.

No pytest suite exists in this repo (see CLAUDE.md) -- this follows the
existing diagnose_*.py/validate_*.py convention: a standalone script with
plain assert statements, run manually.

USAGE:
    python verify_picklist_id_mapping.py
"""

import sys

import pandas as pd

from schema_adapter import map_raw_features_to_legacy
from preprocessor import SaleValuePreprocessor, TARGET_COL, TIME_COL

FAILURES = []


def check(label, actual, expected):
    ok = (actual == expected) or (pd.isna(actual) and pd.isna(expected))
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}: got {actual!r}, expected {expected!r}")
    if not ok:
        FAILURES.append(label)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Synthetic rows, NEW SCHEMA (raw column names, as they appear in the
# taegram export) -- covers every condition column's edge cases:
#   row A: everything "best" (severity 0), runs_flag=1, no damage/unknowns
#   row B: everything "Unknown", other_damage as a scalar ID (Mold)
#   row C: mixed severities incl. transmission's "Missing" (id differs from
#          engine's "Missing" id -- must not cross-contaminate), tire's
#          reversed-ID ordering, other_damage as a LIST of IDs (live
#          multi-select forward-compat)
#   row D: engine "Removed" vs transmission "Removed" (different IDs, same
#          severity) -- the flip the mapping notes warn about
# ============================================================
NEW_SCHEMA_ROWS = [
    dict(  # row A
        vin_hin_no="VINA", make="Toyota", model="Camry", year=2015, trim="LE",
        vehicle_category="Sedan", body_subtype="4dr", color=22947.0, mileage=50000,
        vehicle_cond_picklist_id=22968.0,        # Runs & Drives -> sev 0, runs_flag=1
        engine_cond_picklist_id=23053.0,         # Operational -> sev 0
        transmission_cond_picklist_id=23060.0,   # Operational -> sev 0
        body_paint_cond_picklist_id=23044.0,     # Normal Wear & Tear -> sev 0
        interior_cond_picklist_id=23050.0,       # Normal Wear & Tear -> sev 0
        tire_cond_picklist_id=23073.0,           # All Wheels Mounted... -> sev 0 (highest ID!)
        other_damage_pklist_id=float("nan"),
        state_title_picklist=13318.0, state_picklist_id=13318.0,
        zip="94040", located_at_donation_c_a="True", accessible_for_tow_truck="True",
        speciality_item="False", sale_value=5000.0,
        creation_datetime="2023-01-15", comment=None,
    ),
    dict(  # row B
        vin_hin_no="VINB", make="Honda", model="Civic", year=2010, trim="EX",
        vehicle_category="Sedan", body_subtype="4dr", color=22947.0, mileage=150000,
        vehicle_cond_picklist_id=22970.0,        # Unknown -> sev 6
        engine_cond_picklist_id=23059.0,         # Unknown -> sev 5
        transmission_cond_picklist_id=23066.0,   # Unknown -> sev 5
        body_paint_cond_picklist_id=23045.0,     # Unknown -> sev 5
        interior_cond_picklist_id=23052.0,       # Unknown -> sev 4
        tire_cond_picklist_id=23067.0,           # Unknown -> sev 5 (lowest ID in tire's block!)
        other_damage_pklist_id=23074.0,          # Mold (scalar ID)
        state_title_picklist=13333.0, state_picklist_id=13333.0,
        zip="21201", located_at_donation_c_a="False", accessible_for_tow_truck="False",
        speciality_item="False", sale_value=300.0,
        creation_datetime="2023-02-20", comment=None,
    ),
    dict(  # row C
        vin_hin_no="VINC", make="Ford", model="Focus", year=2012, trim="SE",
        vehicle_category="Sedan", body_subtype="4dr", color=22947.0, mileage=120000,
        vehicle_cond_picklist_id=22969.0,        # Runs / Doesn't Move -> sev 2, runs_flag=1
        engine_cond_picklist_id=23054.0,         # Removed -> sev 4
        transmission_cond_picklist_id=23065.0,   # Missing -> sev 4 (own ID, not engine's Missing=23058)
        body_paint_cond_picklist_id=23047.0,     # Baseball-sized or Larger Damage* -> sev 3
        interior_cond_picklist_id=23051.0,       # Damaged or Removed Parts -> sev 2
        tire_cond_picklist_id=23072.0,           # 1 or More Tires are Flat* -> sev 2
        other_damage_pklist_id=[23074.0, 23079.0],  # list: Mold + Severe Undercarriage Rust
        state_title_picklist=13355.0, state_picklist_id=13355.0,
        zip="73301", located_at_donation_c_a="True", accessible_for_tow_truck="True",
        speciality_item="False", sale_value=800.0,
        creation_datetime="2023-03-10", comment=None,
    ),
    dict(  # row D -- engine "Missing" (23058) vs transmission "Removed" (23064): different
           # IDs, both severity 4, must not cross-contaminate between the two dicts.
        vin_hin_no="VIND", make="Nissan", model="Altima", year=2018, trim="S",
        vehicle_category="Sedan", body_subtype="4dr", color=22947.0, mileage=80000,
        vehicle_cond_picklist_id=22974.0,        # Cranks, won't start -> sev 3, runs_flag=0
        engine_cond_picklist_id=23058.0,         # Missing -> sev 4
        transmission_cond_picklist_id=23064.0,   # Removed -> sev 4
        body_paint_cond_picklist_id=23048.0,     # Major Damage* -> sev 4
        interior_cond_picklist_id=23050.0,       # Normal Wear & Tear -> sev 0
        tire_cond_picklist_id=23071.0,           # 1 or More Wheels are Removed* -> sev 3
        other_damage_pklist_id=23078.0,          # Won't Pass Smog/State Inspection
        state_title_picklist=13357.0, state_picklist_id=13357.0,
        zip="20301", located_at_donation_c_a="True", accessible_for_tow_truck="False",
        speciality_item="False", sale_value=450.0,
        creation_datetime="2023-04-05", comment=None,
    ),
]

EXPECTED = [
    # (nav_severity, engine_severity, trans_severity, body_severity,
    #  interior_severity, tire_severity, runs_flag, n_unknowns,
    #  has_mold, has_undercarriage_rust, has_smog_fail, n_other_damages)
    dict(nav=0, eng=0, trans=0, body=0, interior=0, tire=0,
         runs=1, unk=0, mold=0, rust=0, smog=0, ndmg=0),
    dict(nav=6, eng=5, trans=5, body=5, interior=4, tire=5,
         runs=0, unk=6, mold=1, rust=0, smog=0, ndmg=1),
    dict(nav=2, eng=4, trans=4, body=3, interior=2, tire=2,
         runs=1, unk=0, mold=1, rust=1, smog=0, ndmg=2),
    dict(nav=3, eng=4, trans=4, body=4, interior=0, tire=3,
         runs=0, unk=0, mold=0, rust=0, smog=1, ndmg=1),
]


def run_new_schema_check():
    section("1. End-to-end: new-schema raw columns -> schema_adapter -> preprocessor")
    df_new = pd.DataFrame(NEW_SCHEMA_ROWS)
    df_legacy = map_raw_features_to_legacy(df_new)

    # Confirm the rename actually happened onto the expected legacy names.
    for legacy_col in ["nav_condition", "enginecondition", "transmissioncondition",
                        "bodypaintcondition", "interiorcondition", "tirecondition",
                        "other_damages", "state_province_of_title", "vstate_name",
                        "salevalue", "record_creation_date"]:
        assert legacy_col in df_legacy.columns, f"expected legacy column {legacy_col!r} missing after rename"
    print("  Rename OK: all expected legacy columns present.")

    pre = SaleValuePreprocessor(time_col=TIME_COL, use_macro=False, use_geo=False, use_cult=False)
    X = pre.fit(df_legacy).transform(df_legacy)

    for i, exp in enumerate(EXPECTED):
        print(f"\n  -- row {chr(65+i)} --")
        check(f"row{chr(65+i)} nav_severity",       X.loc[i, "nav_severity"],       exp["nav"])
        check(f"row{chr(65+i)} engine_severity",     X.loc[i, "engine_severity"],    exp["eng"])
        check(f"row{chr(65+i)} trans_severity",      X.loc[i, "trans_severity"],     exp["trans"])
        check(f"row{chr(65+i)} body_severity",       X.loc[i, "body_severity"],      exp["body"])
        check(f"row{chr(65+i)} interior_severity",   X.loc[i, "interior_severity"],  exp["interior"])
        check(f"row{chr(65+i)} tire_severity",       X.loc[i, "tire_severity"],      exp["tire"])
        check(f"row{chr(65+i)} runs_flag",           X.loc[i, "runs_flag"],          exp["runs"])
        check(f"row{chr(65+i)} n_unknowns",          X.loc[i, "n_unknowns"],         exp["unk"])
        check(f"row{chr(65+i)} has_mold",            X.loc[i, "has_mold"],           exp["mold"])
        check(f"row{chr(65+i)} has_undercarriage_rust", X.loc[i, "has_undercarriage_rust"], exp["rust"])
        check(f"row{chr(65+i)} has_smog_fail",       X.loc[i, "has_smog_fail"],      exp["smog"])
        check(f"row{chr(65+i)} n_other_damages",     X.loc[i, "n_other_damages"],    exp["ndmg"])


# ============================================================
# Backward-compat: same rows, but hand-written with the OLD legacy TEXT
# values directly (bypassing schema_adapter entirely, as script17 / current
# live script21 inference still do) -- must produce IDENTICAL engineered
# values to the new-schema numeric-ID rows above.
# ============================================================
LEGACY_TEXT_ROWS = [
    dict(nav_condition="Runs & Drives", enginecondition="Operational",
         transmissioncondition="Operational", bodypaintcondition="Normal Wear & Tear (all body panels intact & attached)",
         interiorcondition="Normal Wear & Tear (all interior intact & attached)",
         tirecondition="All Wheels Mounted & Tires Inflated", other_damages=None),
    dict(nav_condition="Unknown", enginecondition="Unknown",
         transmissioncondition="Unknown", bodypaintcondition="Unknown",
         interiorcondition="Unknown", tirecondition="Unknown", other_damages="Mold"),
    dict(nav_condition="Runs / Doesn’t Move", enginecondition="Removed",
         transmissioncondition="Missing", bodypaintcondition="Baseball-sized or Larger Damage*",
         interiorcondition="Damaged or Removed Parts (notes required)",
         tirecondition="1 or More Tires are Flat*", other_damages="Mold, Severe Undercarriage Rust"),
    dict(nav_condition="Cranks, won’t start", enginecondition="Missing",
         transmissioncondition="Removed", bodypaintcondition="Major Damage*",
         interiorcondition="Normal Wear & Tear (all interior intact & attached)",
         tirecondition="1 or More Wheels are Removed*", other_damages="Won't Pass Smog/State Inspection"),
]

COMMON_COLS = dict(
    vin_hin_no="VIN", make="Toyota", model="Camry", year=2015, trim="LE",
    body_type="Sedan", mileage=50000,
    salevalue=1000.0, record_creation_date="2023-01-15",
)


def run_legacy_text_check():
    section("2. Backward-compat: legacy TEXT columns (script17 / current live inference path)")
    rows = [dict(**COMMON_COLS, **r) for r in LEGACY_TEXT_ROWS]
    df_legacy = pd.DataFrame(rows)

    pre = SaleValuePreprocessor(time_col=TIME_COL, use_macro=False, use_geo=False, use_cult=False)
    X = pre.fit(df_legacy).transform(df_legacy)

    for i, exp in enumerate(EXPECTED):
        print(f"\n  -- row {chr(65+i)} (text) --")
        check(f"text-row{chr(65+i)} nav_severity",     X.loc[i, "nav_severity"],       exp["nav"])
        check(f"text-row{chr(65+i)} engine_severity",  X.loc[i, "engine_severity"],    exp["eng"])
        check(f"text-row{chr(65+i)} trans_severity",   X.loc[i, "trans_severity"],     exp["trans"])
        check(f"text-row{chr(65+i)} body_severity",    X.loc[i, "body_severity"],      exp["body"])
        check(f"text-row{chr(65+i)} interior_severity",X.loc[i, "interior_severity"],  exp["interior"])
        check(f"text-row{chr(65+i)} tire_severity",    X.loc[i, "tire_severity"],      exp["tire"])
        check(f"text-row{chr(65+i)} runs_flag",        X.loc[i, "runs_flag"],          exp["runs"])
        check(f"text-row{chr(65+i)} n_unknowns",       X.loc[i, "n_unknowns"],         exp["unk"])
        check(f"text-row{chr(65+i)} has_mold",         X.loc[i, "has_mold"],           exp["mold"])
        check(f"text-row{chr(65+i)} has_undercarriage_rust", X.loc[i, "has_undercarriage_rust"], exp["rust"])
        check(f"text-row{chr(65+i)} has_smog_fail",    X.loc[i, "has_smog_fail"],      exp["smog"])
        check(f"text-row{chr(65+i)} n_other_damages",  X.loc[i, "n_other_damages"],    exp["ndmg"])


if __name__ == "__main__":
    run_new_schema_check()
    run_legacy_text_check()

    section("RESULT")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)
