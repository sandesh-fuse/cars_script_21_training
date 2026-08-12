"""
diagnose_vehicle_notes.py
=========================
Inspect the 'Vehicle Notes' column to understand its JSON structure before
designing features. Reports:
  1. Missing vs present counts
  2. Top-level type distribution (string, dict, list of dicts, ...)
  3. If list-of-dicts: list length distribution
  4. Key frequency (which keys appear and in how many rows)
  5. Per-key value diagnostics (type, sample values, cardinality, length)
  6. Timestamp detection (which keys look like dates/datetimes)
  7. Random samples of full notes for eyeballing

Pipe to a file so you can paste it back:
    python diagnose_vehicle_notes.py <path_to_data.csv> > vehicle_notes_report.txt

Usage:
    python diagnose_vehicle_notes.py path/to/data.csv
    python diagnose_vehicle_notes.py path/to/data.csv --column-name "Vehicle Notes"
    python diagnose_vehicle_notes.py path/to/data.csv --sample-limit 20
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict

import pandas as pd
import numpy as np


def _try_parse_json(s):
    """Try to interpret a cell as JSON. Return (parsed_or_None, parse_status)."""
    if s is None:
        return None, 'null'
    if not isinstance(s, str):
        return None, f'non-string ({type(s).__name__})'
    s_stripped = s.strip()
    if not s_stripped or s_stripped.lower() in ('nan', 'none', 'null'):
        return None, 'blank/nan'
    try:
        return json.loads(s_stripped), 'ok'
    except json.JSONDecodeError:
        # Try fixing common issues: single quotes instead of double
        try:
            fixed = s_stripped.replace("'", '"')
            return json.loads(fixed), 'ok_with_quote_fix'
        except json.JSONDecodeError:
            return None, 'json_error'


def _shape_of(val):
    """Describe the JSON shape of a parsed value."""
    if val is None: return 'null'
    if isinstance(val, bool): return 'bool'
    if isinstance(val, int): return 'int'
    if isinstance(val, float): return 'float'
    if isinstance(val, str): return 'string'
    if isinstance(val, list):
        if not val:
            return 'list[empty]'
        inner = ', '.join(sorted({_shape_of(x) for x in val[:5]}))
        return f'list[{inner}]'
    if isinstance(val, dict):
        return 'dict'
    return type(val).__name__


def _looks_like_timestamp(val):
    """Heuristic: does a string look like a date/datetime?"""
    if not isinstance(val, str):
        return False
    if len(val) < 8 or len(val) > 40:
        return False
    # Common date/datetime patterns
    patterns = [
        r'^\d{4}-\d{1,2}-\d{1,2}',           # 2024-01-15
        r'^\d{1,2}/\d{1,2}/\d{2,4}',         # 1/15/2024 or 01/15/24
        r'^\d{4}/\d{1,2}/\d{1,2}',           # 2024/01/15
        r'^\d{1,2}-[A-Za-z]{3}-\d{2,4}',     # 15-Jan-2024
    ]
    return any(re.match(p, val.strip()) for p in patterns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Path to training CSV')
    parser.add_argument('--column-name', default='Vehicle Notes',
                        help='Exact column name (default: "Vehicle Notes")')
    parser.add_argument('--sample-limit', type=int, default=5,
                        help='How many random non-null full samples to print (default 5)')
    args = parser.parse_args()

    print(f"Loading column '{args.column_name}' from {args.data}...")
    try:
        df = pd.read_csv(args.data, usecols=[args.column_name], low_memory=False)
    except (ValueError, KeyError) as e:
        # Column might have different name or capitalization
        all_cols = pd.read_csv(args.data, nrows=0).columns.tolist()
        candidates = [c for c in all_cols if 'note' in c.lower() or 'comment' in c.lower()]
        print(f"\nERROR: column '{args.column_name}' not found.")
        if candidates:
            print(f"Similar columns in the file: {candidates}")
        else:
            print(f"First 30 columns in file: {all_cols[:30]}")
        sys.exit(1)

    n_total = len(df)
    s = df[args.column_name]
    s_str = s.astype(str).str.strip()
    missing_mask = (
        s.isna()
        | (s_str == '')
        | s_str.str.lower().isin(['nan', 'none', 'null', '{}', '[]'])
    )
    n_missing = int(missing_mask.sum())
    n_present = n_total - n_missing

    # ----- 1. Missing vs present -----
    print()
    print("=" * 70)
    print(f"  '{args.column_name}' — MISSING vs PRESENT")
    print("=" * 70)
    print(f"  Total rows:           {n_total:>10,}")
    print(f"  Missing values:       {n_missing:>10,}  ({n_missing/n_total*100:5.1f}%)")
    print(f"  Samples with value:   {n_present:>10,}  ({n_present/n_total*100:5.1f}%)")
    print("=" * 70)

    if n_present == 0:
        print("\nNo values to analyze. Exiting.")
        sys.exit(0)

    # Subsample for speed if huge
    SAMPLE_CAP = 20_000
    s_present = s[~missing_mask]
    if len(s_present) > SAMPLE_CAP:
        print(f"\nNote: sampling {SAMPLE_CAP:,} of {len(s_present):,} present rows for analysis.")
        s_present = s_present.sample(SAMPLE_CAP, random_state=0)

    # ----- 2. Parse status & top-level shape -----
    print("\n--- Top-level JSON shape ---")
    parse_counts = Counter()
    shape_counts = Counter()
    parsed_values = []
    for v in s_present:
        parsed, status = _try_parse_json(v)
        parse_counts[status] += 1
        if parsed is not None:
            shape_counts[_shape_of(parsed)] += 1
            parsed_values.append(parsed)
    print(f"\nParse status:")
    for status, cnt in parse_counts.most_common():
        print(f"  {status:25}: {cnt:>8,} ({cnt/len(s_present)*100:5.1f}%)")
    print(f"\nTop-level shape of parsed values:")
    for shape, cnt in shape_counts.most_common(15):
        print(f"  {shape:30}: {cnt:>8,} ({cnt/len(parsed_values)*100:5.1f}%)" if parsed_values else "")

    # ----- 3. If list of dicts, length distribution -----
    list_lengths = [len(v) for v in parsed_values if isinstance(v, list)]
    if list_lengths:
        ll = pd.Series(list_lengths)
        print(f"\n--- List length distribution (when top-level is a list) ---")
        print(f"  count: {len(list_lengths):,}")
        print(f"  min/median/p95/max: {ll.min()} / {int(ll.median())} / {int(ll.quantile(0.95))} / {ll.max()}")
        print(f"  histogram (top 10):")
        len_counts = pd.Series(list_lengths).value_counts().sort_index().head(10)
        for length, cnt in len_counts.items():
            print(f"    length={length}: {cnt:,}")

    # ----- 4. Key frequency (flattened: works for both dict-of and list-of-dicts) -----
    print("\n--- Key frequency across all notes ---")
    key_appearances = Counter()         # how many rows have this key at least once
    key_total_appearances = Counter()   # how many times the key is seen (incl. multiple per row)
    for v in parsed_values:
        rows_with_key = set()
        if isinstance(v, dict):
            for k in v.keys():
                rows_with_key.add(str(k))
                key_total_appearances[str(k)] += 1
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for k in item.keys():
                        rows_with_key.add(str(k))
                        key_total_appearances[str(k)] += 1
        for k in rows_with_key:
            key_appearances[k] += 1

    print(f"\n{'Key':<35} {'Rows w/ key':>13} {'% of present':>14} {'Total occurrences':>20}")
    print('-' * 84)
    for k, n_rows in key_appearances.most_common(30):
        pct = n_rows / len(parsed_values) * 100
        print(f"  {k[:33]:<33} {n_rows:>13,} {pct:>13.1f}% {key_total_appearances[k]:>20,}")

    # ----- 5. Per-key value diagnostics for top 10 keys -----
    print("\n--- Per-key value diagnostics (top 10 keys by row coverage) ---")
    top_keys = [k for k, _ in key_appearances.most_common(10)]
    per_key_values = defaultdict(list)
    for v in parsed_values:
        if isinstance(v, dict):
            for k, val in v.items():
                if k in top_keys:
                    per_key_values[k].append(val)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for k, val in item.items():
                        if k in top_keys:
                            per_key_values[k].append(val)

    for k in top_keys:
        vals = per_key_values.get(k, [])
        if not vals:
            continue
        print(f"\nKey: '{k}'  ({len(vals):,} values)")
        # Type distribution
        type_counts = Counter(_shape_of(v) for v in vals)
        print(f"  Value types: {dict(type_counts.most_common(5))}")
        # Cardinality (only meaningful for strings/scalars)
        scalar_vals = [v for v in vals if isinstance(v, (str, int, float, bool))]
        if scalar_vals:
            uniq = set(str(v).strip().lower() for v in scalar_vals)
            print(f"  Distinct values (case-insensitive, on {len(scalar_vals):,} scalar values): {len(uniq):,}")
            # Top 5 values
            top_vals = Counter(str(v).strip().lower()[:80] for v in scalar_vals).most_common(5)
            print(f"  Top 5 values:")
            for v, cnt in top_vals:
                print(f"    {cnt:>6,}: {v}")
            # Length distribution (for strings)
            str_vals = [v for v in scalar_vals if isinstance(v, str)]
            if str_vals:
                lens = pd.Series([len(v) for v in str_vals])
                print(f"  String length: min={lens.min()} median={int(lens.median())} p95={int(lens.quantile(0.95))} max={lens.max()}")
        # Timestamp-likeness
        if scalar_vals:
            n_ts_like = sum(1 for v in scalar_vals if _looks_like_timestamp(str(v)))
            if n_ts_like > 0:
                print(f"  TIMESTAMP-LIKE: {n_ts_like:,}/{len(scalar_vals):,} values match date/datetime patterns")

    # ----- 6. Random full samples -----
    print(f"\n--- Random sample of {args.sample_limit} full notes (raw JSON) ---")
    sample = s_present.sample(min(args.sample_limit, len(s_present)), random_state=0)
    for i, val in enumerate(sample, 1):
        v_str = str(val)
        v_short = v_str if len(v_str) <= 500 else v_str[:497] + '...'
        print(f"\n[{i}] {v_short}")


if __name__ == "__main__":
    main()
