"""
validate_live_predictions.py
=============================
End-to-end regression check: replay real rows through the LIVE /predict
HTTP endpoint and see whether it reproduces the offline predictions
recorded alongside them. Two modes, selected via --source:

--source sample (small, curated, exact):
    Replays train_save_script21.py's own curated test-set sample
    (artifacts/script21/sample_test_rows.parquet -- the 15 lowest-abs-error
    CULT rows + 15 lowest-abs-error STANDARD rows priced $500-2500). That
    parquet already carries the FULL raw car-attribute columns -- it IS
    test_cult_raw/test_std_raw, the exact raw DataFrame train_save_script21.py's
    own pre.transform()/model.predict() call evaluated -- so this mode needs
    no reconstruction and no join: the row used here IS the row training's
    own evaluation saw, eliminating an entire class of "did I reconstruct
    this correctly" bugs. --n-samples is capped at 30; a targeted
    regression check, not a representative benchmark.

--source full (large, reconstructed, representative -- the default):
    Replays EVERY row of artifacts/script21/test_predictions.csv (12.7k+
    rows, duplicate stock_ids kept, not deduped -- see build below). Unlike
    sample_test_rows.parquet, test_predictions.csv carries no raw
    car-attribute columns at all -- just
    stock_id/vin/record_creation_date/salevalue/predicted_sale_value/
    p5/p50/p95/abs_error_p50/ci_width/is_cult (see _save_predictions_helper.py's
    build_predictions_frame -- it deliberately keeps only ID + prediction
    output columns). So this mode reconstructs the raw feature row for each
    stock_id by looking it up in taegram_all_table_merged_2018_2026.csv (a
    ~930k-row raw DB export in a DIFFERENT column-naming schema than
    app/schemas.py's PredictRequest uses -- e.g. 'zip'/'color'/'sale_value'
    instead of 'vazipcode'/'nav_color'/'salevalue'), then translates it via
    schema_adapter.py's NEW_TO_OLD_SCHEMA_MAP (the exact same
    filter_to_known_columns()/map_raw_features_to_legacy() calls
    train_save_script21.py itself makes at training-data-ingestion time).

    CAVEAT specific to this mode: unlike sample mode's exact-echo
    guarantee, a "full" row is a *reconstruction* -- taegram's raw export
    is not guaranteed to be byte-identical to whatever raw_df
    train_save_script21.py originally loaded (from DATA_PATH, a derived
    file not present in this checkout), and taegram itself has ~11.6k rows
    sharing a duplicate stock_id, requiring disambiguation (see
    resolve_raw_row()). So expect live_mae to deviate from stored_mae by
    more than sample mode's near-zero delta -- that's expected, not
    necessarily a regression; read the per-row failure_reason/resolve_status
    columns in the output CSV to see why any given row differs.

app/schemas.py's PredictRequest is imported directly and used as the actual
schema for the outgoing request in BOTH modes -- not just as a reference
this file's authors read once and hand-copied types from. Every mapped
value is run through PredictRequest(**mapped) before being sent, so: (a)
the exact same class app/main.py validates incoming requests against also
validates this script's requests before they're sent, catching a bad
mapping/type client-side with a clear pydantic error instead of a generic
422; (b) the field types used for numeric-vs-string cleanup (FIELD_TYPES
below) are read off PredictRequest.model_fields itself, so this file can't
silently drift out of sync if schemas.py's field list or types change
again later; and (c) the JSON body actually sent is PredictRequest.model_dump()
-- the same method app/main.py itself calls on every incoming request --
so what leaves this script is, byte-for-byte, "a PredictRequest instance
serialized," matching the live server's own contract exactly.

PredictRequest.Config.extra is "forbid" (not "allow"): any raw column NOT
explicitly declared as a field would 422 the WHOLE request, not just get
silently ignored. build_request() below filters each row down to only
PredictRequest.model_fields before constructing it, for exactly that
reason. Some of the raw columns this drops are genuinely in
preprocessor_*.joblib's feature_cols_ (e.g. 'Specialty Item') but still
harmless to drop: train_save_script21.py filters out every Specialty-Item=
True row BEFORE fitting, so int_maps_['Specialty Item'] only ever saw
'false' at fit time -- a zero-variance constant feature carries no real
signal regardless of what (or whether) a live caller sends for it.

USAGE:
    # 1. start the API somewhere reachable (needs API_KEY_PREDICTION set,
    #    already present in .env):
    uvicorn app.main:app --port 8000

    # 2. smoke test the small curated regression check:
    python validate_live_predictions.py --source sample --n-samples 5

    # 3. the full-scale run (all of test_predictions.csv, dupes kept):
    python validate_live_predictions.py --source full --batch-size 100
"""

import argparse
import os
import sys
import typing
import warnings

import numpy as np
import pandas as pd
import requests
from pydantic import ValidationError

from app.schemas import PredictRequest
from evaluate_predictions import mae
from preprocessor import TARGET_COL
from schema_adapter import (
    filter_to_known_columns,
    known_raw_columns,
    map_raw_features_to_legacy,
)

try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    pass

# Field type, read directly off PredictRequest.model_fields rather than
# hand-copied -- if schemas.py's field list/types change again, this adapts
# automatically instead of silently going stale.
# 'int'/'float'/'str': the field's sole declared type.
# 'numeric_or_str': a str|int|float|None union (vehicle_type/nav_color --
#   unresolved taegram picklist IDs the model was trained on as raw numbers,
#   but the field still accepts real text too). Numeric input must pass
#   through as a genuine number, NOT get stringified: the fitted
#   preprocessor never learned an encoding for these two columns as
#   strings, so e.g. "23101" (str) makes XGBoost reject the column as
#   non-numeric dtype at predict time, while 23101 (number) predicts fine.
# 'other': anything else multi-type (e.g. other_damages: str | list[str] |
#   None) -- a list is passed through untouched, since the live parser
#   handles it natively; everything else stringifies (original behavior).
def _field_kind(annotation):
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if not args:
        args = [annotation]  # a bare (non-Optional) annotation, no Union at all
    if len(args) == 1:
        base = args[0]
        if base is int:
            return 'int'
        if base is float:
            return 'float'
        if base is str:
            return 'str'
        return 'other'
    if set(args) == {str, int, float}:
        return 'numeric_or_str'
    return 'other'


FIELD_TYPES = {name: _field_kind(finfo.annotation)
                for name, finfo in PredictRequest.model_fields.items()}

# sample_test_rows.parquet columns (and, harmlessly, some taegram-derived
# legacy columns in full mode) that are evaluation output, not a raw car
# attribute -- see the "Save sample low-MAE rows" block in
# train_save_script21.py's main() for exactly how each gets appended onto
# the raw test_cult_raw/test_std_raw row. Never sent as part of the request.
EXCLUDE_FROM_REQUEST = {
    TARGET_COL,  # 'salevalue' -- duplicate of 'actual'; the answer, never a real input field
    'actual', 'p5', 'p50', 'p95', 'ci_width', 'abs_error_p50', 'is_cult',
}

# Model fields actually declared on PredictRequest right now -- with
# Config.extra = "forbid", anything NOT in this set would 422 the whole
# request if sent, so build_request() filters to this set rather than
# relying on the server to ignore extras.
DECLARED_FIELDS = set(PredictRequest.model_fields.keys())


def _clean_value(field, value):
    """None-out NaN/NaT; otherwise cast to the type PredictRequest expects
    for this field (per FIELD_TYPES, i.e. per PredictRequest.model_fields
    itself -- not a hand-maintained guess at it)."""
    kind = FIELD_TYPES.get(field, 'str')
    # A list (e.g. other_damages sent as ["Mold", "Rust"]) must be checked
    # BEFORE pd.isna(value) -- pd.isna() on a list/array returns an
    # elementwise array, not a scalar bool, and `if is_na:` on that raises
    # ValueError ("truth value of an array... is ambiguous"), not the
    # TypeError/ValueError this function already guards against.
    if isinstance(value, list):
        return value if kind == 'other' else str(value)
    try:
        is_na = pd.isna(value)
    except (TypeError, ValueError):
        is_na = False
    if is_na:
        return None
    if kind == 'int':
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
    if kind == 'float':
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if kind == 'numeric_or_str':
        # Pass a genuine number through as a number -- NOT stringified, see
        # the 'numeric_or_str' note on _field_kind above. Always a FLOAT,
        # never an int: in the taegram-derived data these two columns are
        # float64 (e.g. 23101.0), and preprocessor.py's combo-builder does
        # `.astype(str)` on the raw column when building interaction
        # features (make_x_vehicle_type, vtype_x_nav_condition, etc.) --
        # str(23101) == "23101" but str(23101.0) == "23101.0", a DIFFERENT
        # string. Sending an int here makes every vehicle_type/body_type-
        # derived combo/freq feature look like an unseen category. A real
        # string value (e.g. if some other data source ever provides actual
        # text for these two fields) still passes through unchanged.
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        return str(value).strip()
    # str-typed field. Verified directly against PredictRequest (pydantic
    # v2, lax mode): sending raw taegram types as-is raises `string_type`
    # errors -- pydantic does NOT coerce float/bool -> str (e.g.
    # vazipcode=90620.0 or accessiblefortwotruck=True fail validation
    # outright), so every non-numeric/non-other field is cleaned to a
    # proper string first (no trailing ".0" on whole-number floats).
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return str(int(v)) if v.is_integer() else str(v)
    return str(value).strip()


def build_request(row):
    """Map one legacy-schema row (a pandas Series -- either a
    sample_test_rows.parquet row, or a taegram row already translated by
    schema_adapter.map_raw_features_to_legacy(), see load_taegram_subset()/
    apply_schema_adapter() below) to a /predict JSON body.

    PredictRequest (imported from app.schemas -- the actual live schema, not
    a hand-copied mirror of it) is the authority here: the mapped+cleaned
    values are run through PredictRequest(**mapped) before anything is sent,
    so a bad mapping/type surfaces as a clear pydantic ValidationError
    client-side, and the JSON body sent is exactly req.model_dump() -- the
    same serialization app/main.py itself applies to every request it
    receives (exclude_none=False here so a genuinely-missing value arrives
    as an explicit JSON `null` rather than an omitted key -- this matters
    for correctness, not just style: app/main.py itself now also sends
    exclude_none=False downstream, specifically so preprocessor.py sees
    every raw column it expects, present but NaN, exactly like training
    saw it, rather than a column silently absent).

    Filters to DECLARED_FIELDS first -- Config.extra = "forbid" means
    PredictRequest(**mapped) would raise (not silently ignore) for any key
    not declared as a field, so an undeclared raw column has to be dropped
    HERE rather than left for pydantic to reject the whole request over.
    """
    mapped = {field: _clean_value(field, row[field]) for field in row.index
              if field not in EXCLUDE_FROM_REQUEST and field in DECLARED_FIELDS}
    req = PredictRequest(**mapped)
    return req.model_dump(exclude_none=False)


def load_sample_rows(path, n, seed):
    """Load sample_test_rows.parquet -- train_save_script21.py's own curated
    sample of the script21 test set (15 lowest-abs-error CULT rows + 15
    lowest-abs-error STANDARD rows priced $500-2500). See the module
    docstring above for why this file needs no taegram CSV scan: it already
    IS the raw row train_save_script21.py's own evaluation saw, plus the
    offline prediction it already computed (actual/p5/p50/p95/
    abs_error_p50/is_cult)."""
    df = pd.read_parquet(path)
    n = min(n, len(df))
    sample = (df.sample(n=n, random_state=seed) if n < len(df) else df).reset_index(drop=True)
    n_cult = int(df['is_cult'].sum())
    print(f"Loaded {len(df)} rows from {path} "
          f"({n_cult} cult + {len(df) - n_cult} standard, "
          f"train_save_script21.py's own low-abs-error test sample); "
          f"sampled {len(sample)}.")
    return sample


def call_predict(base_url, model, api_key, body, timeout=30, k_pos=None, k_neg=None):
    headers = {"x-api-key": api_key}
    params = {"model": model, "explain": "false"}
    if k_pos is not None:
        params["k_pos"] = k_pos
    if k_neg is not None:
        params["k_neg"] = k_neg
    r = requests.post(f"{base_url}/predict", params=params, json=body,
                       headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def call_predict_batch(base_url, model, api_key, bodies, timeout=60):
    """POST a JSON LIST of request bodies to /predict in one call (list
    mode -- see app/main.py's Union[PredictRequest, List[PredictRequest]]
    payload handling). Returns the parsed list of response dicts,
    order-preserving 1:1 with `bodies` (app/main.py appends to
    final_responses once per loop iteration over the input list, no
    reordering).

    List mode is ALL-OR-NOTHING server-side: main.py wraps the whole loop
    in one try/except, so a single bad row raises HTTPException(500) for
    the ENTIRE batch with zero partial results -- callers must catch and
    fall back to per-row call_predict() rather than retry this as-is.
    k_pos=0&k_neg=0 skips SHAP computation (the TreeExplainer step still
    runs even with explain=false if k_pos/k_neg are left at their
    server-side default of 5 each) -- not needed for an MAE check and
    meaningfully slower at this scale.
    """
    headers = {"x-api-key": api_key}
    params = {"model": model, "explain": "false", "k_pos": 0, "k_neg": 0}
    r = requests.post(f"{base_url}/predict", params=params, json=bodies,
                       headers=headers, timeout=timeout)
    r.raise_for_status()
    resp = r.json()
    if not isinstance(resp, list) or len(resp) != len(bodies):
        got = len(resp) if isinstance(resp, list) else type(resp).__name__
        raise ValueError(f"batch response shape mismatch: sent {len(bodies)} bodies, got {got}")
    return resp


# ======================================================================
# --source sample: today's curated, exact, no-reconstruction-needed check
# ======================================================================

def run_sample_mode(args):
    api_key = args.api_key or os.getenv('API_KEY_PREDICTION')
    if not api_key:
        raise SystemExit("No API key available -- pass --api-key or set API_KEY_PREDICTION "
                          "(env var or .env).")

    sample = load_sample_rows(args.sample_parquet, args.n_samples, args.seed)

    results = []
    n_build_failed = 0
    n_call_failed = 0
    n_route_mismatch = 0
    for _, row in sample.iterrows():
        sid = row['stock_id']

        try:
            body = build_request(row)
        except ValidationError as e:
            n_build_failed += 1
            print(f"  [{sid}] request failed PredictRequest validation: {e}")
            continue

        try:
            resp = call_predict(args.base_url, args.model, api_key, body)
        except Exception as e:
            n_call_failed += 1
            print(f"  [{sid}] API call failed: {e}")
            continue

        live_price = resp.get('predictions', {}).get('predicted_price')
        if live_price is None:
            n_call_failed += 1
            print(f"  [{sid}] no predicted_price in response: {resp}")
            continue

        actual = row['actual']
        stored_is_cult = bool(row['is_cult'])
        live_is_cult = resp.get('is_cult')
        route_match = stored_is_cult == live_is_cult
        if not route_match:
            n_route_mismatch += 1
        results.append({
            'stock_id': sid,
            'vin': row['vin'],
            'actual': actual,
            'offline_predicted': row['p50'],
            'live_predicted': live_price,
            'stored_abs_error': row['abs_error_p50'],
            'live_abs_error': abs(actual - live_price),
            'match': abs(live_price - row['p50']) <= args.tolerance,
            'stored_is_cult': stored_is_cult,
            'live_is_cult': live_is_cult,
            'route_match': route_match,
        })

    if not results:
        print("No rows were successfully scored -- nothing to compare.")
        sys.exit(1)

    res_df = pd.DataFrame(results)
    print()
    print(res_df.to_string(index=False))

    stored_mae = mae(res_df['actual'].values, res_df['offline_predicted'].values)
    live_mae = mae(res_df['actual'].values, res_df['live_predicted'].values)
    delta = abs(stored_mae - live_mae)
    n_match = int(res_df['match'].sum())
    n_total = len(res_df)
    passed = delta <= args.tolerance and n_route_mismatch == 0

    print("\n" + "=" * 60)
    print(f"Sampled:              {len(sample)}")
    print(f"Request build fail:   {n_build_failed}")
    print(f"API call fail:        {n_call_failed}")
    print(f"Scored rows:          {n_total}")
    print(f"Per-row match:        {n_match}/{n_total}")
    print(f"Cult-route mismatch:  {n_route_mismatch}/{n_total}")
    print(f"Stored (offline) MAE: {stored_mae:.4f}")
    print(f"Live MAE:             {live_mae:.4f}")
    print(f"Delta:                {delta:.4f}  (tolerance {args.tolerance})")
    verdict = "PASSED -- live API reproduces the offline predictions" if passed \
        else "FAILED -- live API differs from the offline predictions"
    print(f"RESULT: {verdict}")
    print("=" * 60)

    res_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved per-row results to {args.output_csv}")

    sys.exit(0 if passed else 1)


# ======================================================================
# --source full: all of test_predictions.csv, reconstructed via taegram +
# schema_adapter.py
# ======================================================================

def load_test_predictions_csv(path):
    """Load artifacts/script21/test_predictions.csv WHOLE, in its original
    row order, duplicate stock_ids kept as-is -- this is the row list the
    final output must match 1:1 in count and order (per
    _save_predictions_helper.build_predictions_frame, it's an ID +
    prediction-output table only: stock_id/vin/record_creation_date +
    salevalue/predicted_sale_value/p5/p50/p95/abs_error_p50/ci_width/is_cult
    -- no raw car-attribute columns, hence the taegram reconstruction
    below)."""
    df = pd.read_csv(path)
    required = {'stock_id', 'vin', 'salevalue', 'p50', 'abs_error_p50', 'is_cult'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected column(s): {sorted(missing)}")
    n_dup = len(df) - df['stock_id'].nunique()
    print(f"Loaded {len(df)} rows from {path} "
          f"({df['stock_id'].nunique()} unique stock_ids, {n_dup} duplicate rows kept).")
    return df


def load_taegram_subset(path, stock_ids, chunksize=100_000):
    """Read taegram_all_table_merged_2018_2026.csv (~930k rows, ~250 raw
    DB-export columns -- far too much to load whole) and return only the
    rows whose 'stock_id' is one we actually need, restricted to the
    columns schema_adapter.py knows how to translate
    (schema_adapter.known_raw_columns() -- the union of
    NEW_TO_OLD_SCHEMA_MAP keys/values + PredictRequest's declared fields).
    Reading in chunks and filtering each one keeps memory bounded to the
    matched subset rather than the full 930k-row file."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    known = known_raw_columns()
    usecols = [c for c in header if c in known]
    if 'stock_id' not in usecols:
        raise ValueError(f"'stock_id' column not found in {path} (or not recognized by "
                          f"schema_adapter.known_raw_columns()) -- cannot join.")
    print(f"Reading {path}: keeping {len(usecols)}/{len(header)} columns schema_adapter "
          f"recognizes, filtering to {len(stock_ids)} target stock_ids...")

    chunks = []
    n_seen = 0
    with warnings.catch_warnings():
        # Mixed dtypes within a chunk are expected/harmless here: the only
        # column this function's own matching logic relies on ('stock_id')
        # has its dtype forced above; every other column's per-value type
        # coercion happens later in _clean_value(), which works off
        # individual Python values, not a column's inferred dtype.
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype={'stock_id': str}):
            n_seen += len(chunk)
            match = chunk[chunk['stock_id'].isin(stock_ids)]
            if len(match):
                chunks.append(match)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    n_found = df['stock_id'].nunique() if len(df) else 0
    print(f"  scanned {n_seen} taegram rows, matched {len(df)} rows for "
          f"{n_found}/{len(stock_ids)} target stock_ids "
          f"({len(stock_ids) - n_found} stock_ids have no taegram row at all).")
    return df


def apply_schema_adapter(df):
    """Translate a taegram-schema DataFrame to the legacy schema
    build_request() expects -- the same two calls train_save_script21.py
    itself makes at training-data-ingestion time (renames e.g.
    zip->vazipcode, color->nav_color, sale_value->salevalue,
    vehicle_cond_picklist_id_name->nav_condition, ...)."""
    df = filter_to_known_columns(df, verbose=False)
    df = map_raw_features_to_legacy(df)
    return df


def build_taegram_lookup(df):
    """Group the (already schema-adapted) taegram subset by stock_id for
    O(1) lookup. Values are lists of pandas Series (not dict records) --
    build_request()/_clean_value() index into row.index, matching a
    sample_test_rows.parquet row's shape exactly."""
    if 'stock_id' in df.columns:
        df = df.copy()
        df['stock_id'] = df['stock_id'].astype(str).str.strip()
    lookup = {}
    for sid, group in df.groupby('stock_id', sort=False):
        lookup[sid] = [row for _, row in group.iterrows()]
    return lookup


def resolve_raw_row(sid, vin, salevalue, taegram_by_stock_id):
    """Resolve exactly one taegram raw row for one test_predictions.csv
    row. taegram itself has ~11.6k rows sharing a duplicate stock_id (out
    of ~930k), so a stock_id match alone isn't always unique -- disambiguate
    by vin first (taegram's vin_hin_no -> legacy 'vin'), then by closest
    'salevalue' (taegram's sale_value -> legacy 'salevalue') if vin doesn't
    narrow it to exactly one. Every disambiguated/ambiguous case is logged
    with its stock_id and candidate count -- auditable, not silent.

    Returns (row_or_None, status_str) where status_str is one of:
    'no_taegram_match', 'ok', 'ok_vin_disambiguated', 'ok_salevalue_disambiguated'.
    """
    candidates = taegram_by_stock_id.get(sid)
    if not candidates:
        return None, 'no_taegram_match'
    if len(candidates) == 1:
        return candidates[0], 'ok'

    def _vin_of(row):
        v = row.get('vin')
        return str(v).strip() if pd.notna(v) else None

    vin_key = str(vin).strip() if pd.notna(vin) else None
    vin_matches = [r for r in candidates if vin_key is not None and _vin_of(r) == vin_key]
    if len(vin_matches) == 1:
        return vin_matches[0], 'ok_vin_disambiguated'

    pool = vin_matches if vin_matches else candidates
    if len(pool) == 1:
        return pool[0], 'ok_vin_disambiguated'

    target_sv = float(salevalue) if pd.notna(salevalue) else None

    def _dist(row):
        v = row.get('salevalue')
        if target_sv is None or pd.isna(v):
            return float('inf')
        return abs(float(v) - target_sv)

    best = min(pool, key=_dist)
    print(f"  [{sid}] ambiguous taegram match ({len(candidates)} candidates, "
          f"{len(vin_matches)} vin-matched) -- picked closest salevalue.")
    return best, 'ok_salevalue_disambiguated'


def run_full_mode(args):
    api_key = args.api_key or os.getenv('API_KEY_PREDICTION')
    if not api_key:
        raise SystemExit("No API key available -- pass --api-key or set API_KEY_PREDICTION "
                          "(env var or .env).")

    test_df = load_test_predictions_csv(args.test_predictions_csv)
    n_total = len(test_df)

    target_ids = set(test_df['stock_id'].astype(str).str.strip())
    taegram_raw = load_taegram_subset(args.taegram_csv, target_ids)
    taegram_legacy = apply_schema_adapter(taegram_raw)
    taegram_by_stock_id = build_taegram_lookup(taegram_legacy)

    # Pre-populate one result dict per test_predictions.csv row, in order,
    # so the final output always has exactly n_total rows -- whether or not
    # that row ever reaches a live call. `pending` tracks only the rows
    # that got a valid request body, for the batching step below.
    results = []
    pending = []  # (result_index, stock_id, body)
    n_no_match = 0
    n_build_failed = 0

    for _, row in test_df.iterrows():
        sid = str(row['stock_id']).strip()
        vin = row['vin']
        actual = row['salevalue']

        raw_row, status = resolve_raw_row(sid, vin, actual, taegram_by_stock_id)
        result = {
            'stock_id': sid,
            'vin': vin,
            'actual': actual,
            'offline_predicted': row['p50'],
            'live_predicted': None,
            'stored_abs_error': row['abs_error_p50'],
            'live_abs_error': None,
            'match': False,
            'stored_is_cult': bool(row['is_cult']) if pd.notna(row['is_cult']) else None,
            'live_is_cult': None,
            'route_match': None,
            'resolve_status': status,
            'failure_reason': None,
        }

        if raw_row is None:
            n_no_match += 1
            result['failure_reason'] = 'no_taegram_match'
            results.append(result)
            continue

        try:
            body = build_request(raw_row)
        except ValidationError as e:
            n_build_failed += 1
            result['failure_reason'] = f'validation_error: {e}'
            results.append(result)
            continue

        pending.append((len(results), sid, body))
        results.append(result)

    print(f"Resolved {len(pending)}/{n_total} rows to a valid request body "
          f"({n_no_match} no taegram match, {n_build_failed} failed PredictRequest validation).")

    # Batch through /predict in list mode, falling back to per-row calls on
    # a batch failure -- list mode is all-or-nothing server-side (one bad
    # row 500s the whole batch), so a fallback keeps one bad row from
    # sinking the rest of that batch's already-good results.
    n_call_failed = 0
    n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
    for b_idx, b in enumerate(range(0, len(pending), args.batch_size), start=1):
        chunk = pending[b:b + args.batch_size]
        bodies = [c[2] for c in chunk]
        try:
            responses = call_predict_batch(args.base_url, args.model, api_key, bodies)
        except Exception as e:
            print(f"  batch {b_idx}/{n_batches} failed ({e}) -- "
                  f"retrying its {len(chunk)} rows individually...")
            responses = []
            for _, sid, body in chunk:
                try:
                    responses.append(
                        call_predict(args.base_url, args.model, api_key, body, k_pos=0, k_neg=0))
                except Exception as e2:
                    responses.append(e2)

        n_ok = 0
        for (result_index, sid, _), resp in zip(chunk, responses):
            r = results[result_index]
            if isinstance(resp, Exception):
                n_call_failed += 1
                r['failure_reason'] = f'api_call_failed: {resp}'
                continue
            live_price = resp.get('predictions', {}).get('predicted_price')
            if live_price is None:
                n_call_failed += 1
                r['failure_reason'] = f'no_predicted_price: {resp}'
                continue
            r['live_predicted'] = live_price
            r['live_abs_error'] = abs(r['actual'] - live_price)
            r['match'] = abs(live_price - r['offline_predicted']) <= args.tolerance
            r['live_is_cult'] = resp.get('is_cult')
            r['route_match'] = (r['stored_is_cult'] == r['live_is_cult'])
            n_ok += 1
        print(f"  batch {b_idx}/{n_batches}: {n_ok}/{len(chunk)} ok "
              f"({n_call_failed} total call failures so far)")

    res_df = pd.DataFrame(results)
    assert len(res_df) == n_total, (
        f"internal error: result count {len(res_df)} != input row count {n_total}")

    scored = res_df[res_df['live_predicted'].notna()]
    n_scored = len(scored)
    n_route_mismatch = int((~scored['route_match'].astype(bool)).sum())
    n_match = int(scored['match'].sum())

    stored_mae = mae(res_df['actual'].values, res_df['offline_predicted'].values)
    live_mae = mae(scored['actual'].values, scored['live_predicted'].values) if n_scored else float('nan')
    delta = abs(stored_mae - live_mae) if n_scored else float('nan')
    passed = n_scored > 0 and delta <= args.tolerance and n_route_mismatch == 0

    print("\n" + "=" * 60)
    print(f"Total rows (== {os.path.basename(args.test_predictions_csv)}): {n_total}")
    print(f"No taegram match:        {n_no_match}")
    print(f"Build/validation fail:   {n_build_failed}")
    print(f"API call fail:           {n_call_failed}")
    print(f"Scored rows:             {n_scored}")
    print(f"Per-row match (<= tol):  {n_match}/{n_scored}")
    print(f"Cult-route mismatch:     {n_route_mismatch}/{n_scored}")
    print(f"Stored (offline) MAE:    {stored_mae:.4f}  (over all {n_total} rows)")
    print(f"Live MAE:                {live_mae:.4f}  (over {n_scored} scored rows)")
    print(f"Delta:                   {delta:.4f}  (tolerance {args.tolerance})")
    verdict = "PASSED" if passed else "FAILED (or nothing scored)"
    print(f"RESULT: {verdict}")
    print("=" * 60)

    res_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(res_df)} rows to {args.output_csv} "
          f"(matches {os.path.basename(args.test_predictions_csv)} row count exactly, dupes kept).")

    sys.exit(0 if passed else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', choices=['sample', 'full'], default='full',
                         help="'sample': today's curated 30-row regression check via "
                              "sample_test_rows.parquet (exact, no reconstruction). "
                              "'full': replay ALL of test_predictions.csv (dupes kept), "
                              "reconstructing raw features from the taegram CSV via "
                              "schema_adapter.py. (default: full)")

    # --source sample only
    parser.add_argument('--sample-parquet', default='artifacts/script21/sample_test_rows.parquet',
                         help="train_save_script21.py's curated low-abs-error test "
                              "sample (30 rows max: 15 cult + 15 standard). Sample mode only.")
    parser.add_argument('--n-samples', type=int, default=30,
                         help="How many of the (at most 30) available sample-mode rows to use.")
    parser.add_argument('--seed', type=int, default=42)

    # --source full only
    parser.add_argument('--test-predictions-csv', default='artifacts/script21/test_predictions.csv',
                         help="Full mode only -- every row is replayed, duplicate "
                              "stock_ids kept.")
    parser.add_argument('--taegram-csv', default='taegram_all_table_merged_2018_2026.csv',
                         help="Full mode only -- raw DB-export CSV used to reconstruct "
                              "car-attribute columns for each stock_id via schema_adapter.py.")
    parser.add_argument('--batch-size', type=int, default=100,
                         help="Full mode only -- rows per /predict list-mode HTTP call.")

    # shared
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--model', default='script21')
    parser.add_argument('--api-key', default=None,
                         help="Overrides API_KEY_PREDICTION from env/.env if given.")
    parser.add_argument('--tolerance', type=float, default=0.01,
                         help="Dollar tolerance for 'same prediction' (default: $0.01).")
    parser.add_argument('--output-csv', default=None,
                         help="Default: artifacts/script21/live_validation_sample.csv "
                              "(sample mode) or artifacts/script21/live_validation_full.csv "
                              "(full mode).")
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = ('artifacts/script21/live_validation_sample.csv' if args.source == 'sample'
                            else 'artifacts/script21/live_validation_full.csv')

    if args.source == 'sample':
        run_sample_mode(args)
    else:
        run_full_mode(args)


if __name__ == '__main__':
    main()
