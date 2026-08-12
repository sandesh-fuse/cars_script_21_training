"""
diagnose_other_damages.py
=========================
Quick check of the other_damages column to understand its cardinality,
typical content, and whether values are single-label or multi-label.

Pipe to a file so you can paste it back:
    python diagnose_other_damages.py path/to/data.csv > other_damages_report.txt
"""
import sys
import pandas as pd
import re
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: python diagnose_other_damages.py <path_to_data.csv>")
    sys.exit(1)

path = sys.argv[1]
print(f"Loading {path} (other_damages column only)...")
# Load just the one column to keep this fast even on big files
df = pd.read_csv(path, usecols=['other_damages'], low_memory=False)
n_total = len(df)

# Null / blank handling — define "missing" as null OR empty/whitespace-only
# OR the literal string "nan"/"None"/"NULL". These all mean "no real value."
s = df['other_damages']
s_str = s.astype(str).str.strip()
missing_mask = (
    s.isna()
    | (s_str == '')
    | s_str.str.lower().isin(['nan', 'none', 'null'])
)
n_missing = int(missing_mask.sum())
n_present = n_total - n_missing

# ========== TOP-LINE SUMMARY ==========
print()
print("=" * 60)
print("  other_damages — MISSING vs PRESENT")
print("=" * 60)
print(f"  Total rows:           {n_total:>10,}")
print(f"  Missing values:       {n_missing:>10,}  ({n_missing/n_total*100:5.1f}%)")
print(f"  Samples with value:   {n_present:>10,}  ({n_present/n_total*100:5.1f}%)")
print("=" * 60)

if n_present == 0:
    print("\nNo non-missing values to analyze further. Exiting.")
    sys.exit(0)

# ========== DETAILED BREAKDOWN ==========
# Show the same numbers broken into sub-categories so you can see exactly
# what kind of "missing" we're dealing with (true NaN vs blank vs 'nan' string).
n_null = int(s.isna().sum())
n_blank = int((s_str == '').sum())
n_nan_str = int(s_str.str.lower().isin(['nan', 'none', 'null']).sum())
print(f"\nDetailed missingness breakdown:")
print(f"  Raw null (NaN):       {n_null:>10,} ({n_null/n_total*100:5.1f}%)")
print(f"  Empty string:         {n_blank:>10,} ({n_blank/n_total*100:5.1f}%)")
print(f"  Literal 'nan'/'none'/'null' string: {n_nan_str:>10,} ({n_nan_str/n_total*100:5.1f}%)")
print(f"  (sum of above can exceed `Missing values` if categories overlap)")

# Keep only meaningful values for downstream analysis
s_clean = s[~missing_mask].astype(str).str.strip()

# Cardinality
n_unique = s_clean.nunique()
print(f"\nCardinality: {n_unique:,} distinct values")

# Length distribution
lens = s_clean.str.len()
print(f"\nString length distribution (on non-null values):")
print(f"  min:    {lens.min()}")
print(f"  p25:    {lens.quantile(0.25):.0f}")
print(f"  median: {lens.median():.0f}")
print(f"  p75:    {lens.quantile(0.75):.0f}")
print(f"  p95:    {lens.quantile(0.95):.0f}")
print(f"  max:    {lens.max()}")

# Multi-label detection: look for common separators
print(f"\nMulti-label detection (fraction of non-null containing each separator):")
for sep, label in [(';', 'semicolon'), (',', 'comma'),
                    ('|', 'pipe'), ('/', 'forward slash'),
                    (r'\bAND\b', 'word "AND"')]:
    pattern = sep if sep in (';', ',', '|', '/') else sep
    if pattern in (';', ',', '|', '/'):
        n_with = s_clean.str.contains(re.escape(pattern), regex=True, na=False).sum()
    else:
        n_with = s_clean.str.contains(pattern, case=False, regex=True, na=False).sum()
    pct = n_with / n_present * 100
    print(f"  {label:>18}: {n_with:>8,} ({pct:5.1f}%)")

# Top values
print(f"\nTop 20 most common values:")
top = s_clean.value_counts().head(20)
for val, cnt in top.items():
    pct = cnt / n_present * 100
    val_short = val if len(val) <= 80 else val[:77] + '...'
    print(f"  {cnt:>8,} ({pct:5.1f}%)  {val_short}")

# If multi-label seems likely, also show the top 'tokens' after splitting
sample_has_semicolons = s_clean.str.contains(';', na=False).mean()
sample_has_commas = s_clean.str.contains(',', na=False).mean()
likely_multi = sample_has_semicolons > 0.05 or sample_has_commas > 0.10
if likely_multi:
    print(f"\nLooks possibly multi-label. Top 20 tokens after splitting on ';' and ',':")
    tokens = []
    for v in s_clean:
        for tok in re.split(r'[;,]', v):
            tok = tok.strip().lower()
            if tok:
                tokens.append(tok)
    tok_counts = Counter(tokens).most_common(20)
    for tok, cnt in tok_counts:
        tok_short = tok if len(tok) <= 80 else tok[:77] + '...'
        print(f"  {cnt:>8,}  {tok_short}")

# Sample a few values at random for visual inspection
print(f"\nRandom sample of 10 values (to eyeball the format):")
sample = s_clean.sample(min(10, n_present), random_state=0)
for i, v in enumerate(sample, 1):
    v_short = v if len(v) <= 200 else v[:197] + '...'
    print(f"  [{i}] {v_short}")
