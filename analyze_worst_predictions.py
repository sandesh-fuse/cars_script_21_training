"""
analyze_worst_predictions.py
=============================
Finds the worst-predicted cars in a given price band from a saved predictions
file (e.g. `test_predictions.csv`), joins them back onto the raw source
export (`taegram_all_table_merged_2018_2026.csv`) via vin/stock_id, and
reports which raw fields — many of them NOT currently used as model
features — correlate with the biggest misses. Intended as a discovery tool
to surface concrete feature candidates; it does not modify preprocessor.py
or retrain anything.

Mirrors the join pattern in `diagnose_test_split.py` (predictions -> raw CSV
by stock_id/vin) and reuses the tier definitions from `evaluate_predictions.py`.

USAGE:
    python analyze_worst_predictions.py
    python analyze_worst_predictions.py --min-price 100 --max-price 2500 --top-n 100
    python analyze_worst_predictions.py --predictions test_predictions.csv \\
        --raw taegram_all_table_merged_2018_2026.csv --out-dir worst_case_analysis
"""
import argparse
import os
import numpy as np
import pandas as pd


TIER_EDGES  = [0, 200, 500, 1000, 2500, 4000, 6000, 10000, 15000, 25000, np.inf]
TIER_LABELS = ['$0-200', '$200-500', '$500-1K', '$1K-2.5K', '$2.5K-4K',
               '$4K-6K',  '$6K-10K',  '$10K-15K', '$15K-25K', '$25K+']

# Columns actually present in taegram_all_table_merged_2018_2026.csv that we
# want for diagnosis: join keys, currently-used condition/damage/title
# signal, and untapped title/keys/mileage-trust/spec fields worth checking.
RAW_USECOLS = [
    # join keys
    'vin_hin_no', 'stock_id',
    # sanity-check / context
    'sale_value', 'creation_datetime', 'make', 'model', 'year', 'trim',
    'vehicle_type', 'body_subtype', 'vehicle_category', 'mileage', 'doors',
    # condition (currently used by preprocessor.py)
    'vehicle_cond_picklist_id_name', 'body_paint_cond_picklist_id_name',
    'engine_cond_picklist_id_name', 'transmission_cond_picklist_id_name',
    'tire_cond_picklist_id_name', 'interior_cond_picklist_id_name',
    'other_damage_pklist_id_name',
    # title / mileage-trust — NOT currently used by preprocessor.py
    'clean_title', 'is_title_clear', 'true_mileage_unknown',
    'days_in_title_issues', 'title_start_date', 'title_end_date',
    'title_keys_poc', 'title_and_keyspoc', 'ids_on_title',
    'is_confirm_nameontitle', 'name_on_title', 'state_title_picklist_name',
    # spec fields main-branch added on top of 2406a7a — not present here
    'gvm_range', 'tonnage', 'engine_type',
    # geo
    'zip',
]

# Readability rename for condition/title columns -> legacy names used
# elsewhere in the codebase (matches schema_adapter.NEW_TO_OLD_SCHEMA_MAP).
# sale_value/creation_datetime are deliberately prefixed `raw_` instead of
# renamed to salevalue/record_creation_date so they stay distinguishable
# from the prediction file's own columns after the join (used as a sanity
# check that the join matched the right vehicle).
RAW_RENAME = {
    'sale_value': 'raw_salevalue',
    'creation_datetime': 'raw_record_creation_date',
    'vehicle_cond_picklist_id_name': 'nav_condition',
    'body_paint_cond_picklist_id_name': 'bodypaintcondition',
    'engine_cond_picklist_id_name': 'enginecondition',
    'transmission_cond_picklist_id_name': 'transmissioncondition',
    'tire_cond_picklist_id_name': 'tirecondition',
    'interior_cond_picklist_id_name': 'interiorcondition',
    'other_damage_pklist_id_name': 'other_damages',
    'state_title_picklist_name': 'state_province_of_title',
    'vehicle_category': 'body_type',
    'zip': 'vazipcode',
}

# Fields to run through the aggregate worst-vs-rest correlation report, and
# how to summarize each: 'bool' (rate of truthy value), 'missing' (null
# rate only), 'numeric' (mean/median), 'multi_label' (';'/',' split, top
# token frequencies, same convention preprocessor.py's other_damages parser
# uses).
REPORT_FIELDS = [
    ('clean_title', 'bool'),
    ('is_title_clear', 'bool'),
    ('true_mileage_unknown', 'bool'),
    ('is_confirm_nameontitle', 'bool'),
    ('days_in_title_issues', 'numeric'),
    ('title_start_date', 'missing'),
    ('title_end_date', 'missing'),
    ('title_keys_poc', 'missing'),
    ('title_and_keyspoc', 'missing'),
    ('ids_on_title', 'missing'),
    ('name_on_title', 'missing'),
    ('nav_condition', 'categorical'),
    ('bodypaintcondition', 'categorical'),
    ('enginecondition', 'categorical'),
    ('transmissioncondition', 'categorical'),
    ('tirecondition', 'categorical'),
    ('interiorcondition', 'categorical'),
    ('other_damages', 'multi_label'),
    ('gvm_range', 'missing'),
    ('tonnage', 'numeric'),
    ('engine_type', 'missing'),
]


def _normalize_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().replace({'NAN': np.nan, 'NONE': np.nan})


def load_predictions(path: str, min_price: float, max_price: float) -> pd.DataFrame:
    print(f"Loading predictions ({os.path.basename(path)})...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df):,} total rows")
    df['record_creation_date'] = pd.to_datetime(df['record_creation_date'], errors='coerce')
    band = df[(df['salevalue'] >= min_price) & (df['salevalue'] <= max_price)].copy()
    print(f"  {len(band):,} rows in ${min_price:.0f}-${max_price:.0f} band")

    band['signed_error_p50'] = band['p50'] - band['salevalue']
    band['abs_error_p50'] = band['abs_error_p50'] if 'abs_error_p50' in band.columns \
        else band['signed_error_p50'].abs()
    band['pct_error_p50'] = band['abs_error_p50'] / band['salevalue'].replace(0, np.nan)

    band['_vin_norm'] = _normalize_key(band['vin'])
    band['_stock_id_norm'] = _normalize_key(band['stock_id'])
    band = band.reset_index(drop=True)
    band['_pred_row_id'] = band.index
    return band


def stream_join_raw(band: pd.DataFrame, raw_path: str, chunksize: int) -> pd.DataFrame:
    """Stream the (large) raw CSV in chunks, keeping only rows whose vin or
    stock_id appears in the band, so we never hold the full file in memory.
    """
    target_vins = set(band['_vin_norm'].dropna())
    target_stock_ids = set(band['_stock_id_norm'].dropna())

    print(f"\nStreaming raw source ({os.path.basename(raw_path)}) in chunks of {chunksize:,}...")
    matches = []
    n_scanned = 0
    for chunk in pd.read_csv(raw_path, usecols=RAW_USECOLS, chunksize=chunksize, low_memory=False):
        n_scanned += len(chunk)
        chunk['_vin_norm'] = _normalize_key(chunk['vin_hin_no'])
        chunk['_stock_id_norm'] = _normalize_key(chunk['stock_id'])
        mask = chunk['_vin_norm'].isin(target_vins) | chunk['_stock_id_norm'].isin(target_stock_ids)
        if mask.any():
            matches.append(chunk[mask])
        print(f"  scanned {n_scanned:,} rows, {sum(len(m) for m in matches):,} candidate matches so far", end='\r')
    print()

    if not matches:
        raise RuntimeError("No candidate rows matched on vin or stock_id — check join key normalization.")

    raw = pd.concat(matches, ignore_index=True)
    raw['creation_datetime'] = pd.to_datetime(raw['creation_datetime'], errors='coerce')
    raw = raw.rename(columns=RAW_RENAME)
    print(f"  {len(raw):,} raw rows collected as join candidates")
    return raw


def resolve_best_match(band: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """For each prediction row, pick the single best-matching raw row:
    prefer a vin match; if several raw rows share a vin, take the one whose
    raw_record_creation_date is closest to the prediction's
    record_creation_date. Fall back to stock_id for rows vin didn't match.
    """
    raw_cols = [c for c in raw.columns if c not in ('_vin_norm', '_stock_id_norm')]

    def best_by_key(sub_band, key_col, raw_key_col):
        if len(sub_band) == 0:
            return pd.DataFrame(columns=['_pred_row_id'] + raw_cols)
        merged = sub_band[['_pred_row_id', key_col, 'record_creation_date']].merge(
            raw, left_on=key_col, right_on=raw_key_col, how='inner')
        if len(merged) == 0:
            return pd.DataFrame(columns=['_pred_row_id'] + raw_cols)
        merged['_time_delta'] = (
            merged['record_creation_date'] - merged['raw_record_creation_date']
        ).abs()
        merged['_time_delta'] = merged['_time_delta'].fillna(pd.Timedelta.max)
        merged = merged.sort_values('_time_delta').drop_duplicates('_pred_row_id', keep='first')
        return merged[['_pred_row_id'] + raw_cols]

    vin_matched = best_by_key(band, '_vin_norm', '_vin_norm')
    matched_ids = set(vin_matched['_pred_row_id'])
    remaining = band[~band['_pred_row_id'].isin(matched_ids)]
    stock_matched = best_by_key(remaining, '_stock_id_norm', '_stock_id_norm')

    all_matched = pd.concat([vin_matched, stock_matched], ignore_index=True)
    joined = band.merge(all_matched, on='_pred_row_id', how='left')
    return joined


def field_lift(df: pd.DataFrame, worst_mask: pd.Series, field: str, kind: str) -> dict:
    if field not in df.columns:
        return {'field': field, 'kind': kind, 'note': 'not present in joined data'}
    worst = df.loc[worst_mask, field]
    rest = df.loc[~worst_mask, field]
    out = {'field': field, 'kind': kind, 'n_worst': int(worst_mask.sum()), 'n_rest': int((~worst_mask).sum())}

    if kind == 'missing':
        out['worst_missing_pct'] = float(worst.isna().mean() * 100)
        out['rest_missing_pct'] = float(rest.isna().mean() * 100)
        out['lift_pp'] = out['worst_missing_pct'] - out['rest_missing_pct']
    elif kind == 'bool':
        w = worst.astype(str).str.lower().isin(['true', '1', 'yes'])
        r = rest.astype(str).str.lower().isin(['true', '1', 'yes'])
        out['worst_true_pct'] = float(w.mean() * 100)
        out['rest_true_pct'] = float(r.mean() * 100)
        out['lift_pp'] = out['worst_true_pct'] - out['rest_true_pct']
    elif kind == 'numeric':
        wn = pd.to_numeric(worst, errors='coerce')
        rn = pd.to_numeric(rest, errors='coerce')
        out['worst_mean'] = float(wn.mean()) if wn.notna().any() else None
        out['rest_mean'] = float(rn.mean()) if rn.notna().any() else None
        out['worst_missing_pct'] = float(wn.isna().mean() * 100)
        out['rest_missing_pct'] = float(rn.isna().mean() * 100)
    elif kind == 'categorical':
        w_top = worst.value_counts(normalize=True, dropna=True)
        r_top = rest.value_counts(normalize=True, dropna=True)
        delta = (w_top.reindex(w_top.index.union(r_top.index), fill_value=0)
                 - r_top.reindex(w_top.index.union(r_top.index), fill_value=0))
        top_delta = delta.abs().sort_values(ascending=False).head(3)
        out['top_categories_by_lift'] = [
            {'value': str(v), 'worst_pct': float(w_top.get(v, 0) * 100),
             'rest_pct': float(r_top.get(v, 0) * 100)}
            for v in top_delta.index
        ]
        out['worst_unknown_pct'] = float(worst.astype(str).str.lower().eq('unknown').mean() * 100)
        out['rest_unknown_pct'] = float(rest.astype(str).str.lower().eq('unknown').mean() * 100)
    elif kind == 'multi_label':
        def token_counts(series):
            tokens = (series.dropna().astype(str)
                      .str.split(r'[;,]')
                      .explode().str.strip())
            tokens = tokens[tokens != '']
            return tokens.value_counts(normalize=True)
        w_tok = token_counts(worst)
        r_tok = token_counts(rest)
        delta = (w_tok.reindex(w_tok.index.union(r_tok.index), fill_value=0)
                 - r_tok.reindex(w_tok.index.union(r_tok.index), fill_value=0))
        top_delta = delta.abs().sort_values(ascending=False).head(5)
        out['top_tokens_by_lift'] = [
            {'value': str(v), 'worst_pct': float(w_tok.get(v, 0) * 100),
             'rest_pct': float(r_tok.get(v, 0) * 100)}
            for v in top_delta.index
        ]
        out['worst_has_any_pct'] = float(worst.notna().mean() * 100)
        out['rest_has_any_pct'] = float(rest.notna().mean() * 100)
    return out


def format_report(sections: dict) -> str:
    lines = ["# Worst-case feature correlation report ($100-2.5K band)\n"]
    for section_name, (worst_label, results) in sections.items():
        lines.append(f"\n## {section_name} ({worst_label})\n")
        lines.append("| field | signal |")
        lines.append("|---|---|")
        for r in results:
            if 'note' in r:
                lines.append(f"| {r['field']} | {r['note']} |")
                continue
            if r['kind'] == 'missing':
                lines.append(f"| {r['field']} | missing: worst={r['worst_missing_pct']:.1f}% "
                              f"vs rest={r['rest_missing_pct']:.1f}% (lift {r['lift_pp']:+.1f}pp) |")
            elif r['kind'] == 'bool':
                lines.append(f"| {r['field']} | true: worst={r['worst_true_pct']:.1f}% "
                              f"vs rest={r['rest_true_pct']:.1f}% (lift {r['lift_pp']:+.1f}pp) |")
            elif r['kind'] == 'numeric':
                wm = f"{r['worst_mean']:.1f}" if r['worst_mean'] is not None else 'n/a'
                rm = f"{r['rest_mean']:.1f}" if r['rest_mean'] is not None else 'n/a'
                lines.append(f"| {r['field']} | mean: worst={wm} vs rest={rm}; "
                              f"missing: worst={r['worst_missing_pct']:.1f}% vs rest={r['rest_missing_pct']:.1f}% |")
            elif r['kind'] == 'categorical':
                cats = "; ".join(f"{c['value']}: worst={c['worst_pct']:.1f}% vs rest={c['rest_pct']:.1f}%"
                                  for c in r['top_categories_by_lift'])
                lines.append(f"| {r['field']} | unknown: worst={r['worst_unknown_pct']:.1f}% "
                              f"vs rest={r['rest_unknown_pct']:.1f}%; top-lift categories: {cats} |")
            elif r['kind'] == 'multi_label':
                toks = "; ".join(f"{t['value']}: worst={t['worst_pct']:.1f}% vs rest={t['rest_pct']:.1f}%"
                                  for t in r['top_tokens_by_lift'])
                lines.append(f"| {r['field']} | has-any: worst={r['worst_has_any_pct']:.1f}% "
                              f"vs rest={r['rest_has_any_pct']:.1f}%; top-lift tokens: {toks} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', default='test_predictions.csv')
    parser.add_argument('--raw', default='taegram_all_table_merged_2018_2026.csv')
    parser.add_argument('--min-price', type=float, default=100)
    parser.add_argument('--max-price', type=float, default=2500)
    parser.add_argument('--top-n', type=int, default=100)
    parser.add_argument('--out-dir', default='worst_case_analysis')
    parser.add_argument('--chunksize', type=int, default=100_000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    band = load_predictions(args.predictions, args.min_price, args.max_price)
    raw = stream_join_raw(band, args.raw, args.chunksize)
    joined = resolve_best_match(band, raw)

    hit_rate = joined['raw_salevalue'].notna().mean() * 100
    print(f"\nJoin hit-rate: {hit_rate:.1f}% of band rows matched a raw record")
    mismatch = joined[joined['raw_salevalue'].notna()]
    if len(mismatch):
        sanity_diff = (mismatch['raw_salevalue'] - mismatch['salevalue']).abs()
        bad_sanity = (sanity_diff > 1).mean() * 100
        print(f"Sanity check: {bad_sanity:.1f}% of matched rows have "
              f"raw sale_value != predictions salevalue by more than $1 "
              f"(should be ~0% for a correct join)")

    tiers = pd.cut(joined['salevalue'], bins=TIER_EDGES, labels=TIER_LABELS, right=True, include_lowest=True)
    print("\nBand composition by tier:")
    print(tiers.value_counts().reindex(TIER_LABELS).dropna())

    worst_dollar = joined.nlargest(args.top_n, 'abs_error_p50')
    worst_pct = joined.nlargest(args.top_n, 'pct_error_p50')

    worst_dollar.to_csv(os.path.join(args.out_dir, 'worst_by_dollar_error.csv'), index=False)
    worst_pct.to_csv(os.path.join(args.out_dir, 'worst_by_pct_error.csv'), index=False)
    print(f"\nSaved worst_by_dollar_error.csv ({len(worst_dollar)} rows) "
          f"and worst_by_pct_error.csv ({len(worst_pct)} rows) to {args.out_dir}/")

    notably_bad_ids = set(worst_dollar['_pred_row_id']) | set(worst_pct['_pred_row_id'])
    worst_mask = joined['_pred_row_id'].isin(notably_bad_ids)
    over_mask = worst_mask & (joined['signed_error_p50'] > 0)   # model overpredicted (predicted > actual)
    under_mask = worst_mask & (joined['signed_error_p50'] < 0)  # model underpredicted

    sections = {
        'ALL notably-bad rows': ('worst (dollar-top-N union pct-top-N) vs rest of band',
                                  [field_lift(joined, worst_mask, f, k) for f, k in REPORT_FIELDS]),
        'OVER-predicted (model too high)': ('overpredicted worst rows vs rest of band',
                                             [field_lift(joined, over_mask, f, k) for f, k in REPORT_FIELDS]),
        'UNDER-predicted (model too low)': ('underpredicted worst rows vs rest of band',
                                             [field_lift(joined, under_mask, f, k) for f, k in REPORT_FIELDS]),
    }
    report = format_report(sections)
    report_path = os.path.join(args.out_dir, 'tier_band_feature_correlation.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved feature correlation report to {report_path}")
    print(f"\n{len(notably_bad_ids):,} distinct notably-bad rows "
          f"({over_mask.sum():,} overpredicted / {under_mask.sum():,} underpredicted)")


if __name__ == '__main__':
    main()
