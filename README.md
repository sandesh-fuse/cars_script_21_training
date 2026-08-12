# Donated Car Sale-Value Predictor

A FastAPI service that predicts the donation sale value of a vehicle, with a 90% confidence interval, SHAP-based feature attributions in dollars and percentages, and optional natural-language explanations via IBM watsonx.ai Granite.

Two models can be served side-by-side for A/B comparison:

- **`script17`** — quick-blend (TE + no-TE ensemble). Best overall MAE/RMSLE.
- **`script21`** — routed cult/standard split. Better on $500-1K tier specifically.

The service serves both via a `?model=` query parameter on a single `/predict` endpoint.

---

## Project layout

```
script-21-prod/
├── preprocessor.py             # Shared preprocessor + cult/zip lookups
├── shap_dollar_helper.py       # SHAP -> dollar/% conversion
├── train_save_script17.py      # Trains Script 17 quantile models (run once)
├── train_save_script21.py      # Trains Script 21 quantile models (run once)
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── inference_script17.py   # Script 17 prediction + SHAP
│   ├── inference_script21.py   # Script 21 routed prediction + SHAP
│   ├── shap_utils.py           # Per-row SHAP with caching
│   ├── explainer.py            # watsonx.ai Granite client
│   ├── feature_descriptions.py # Machine name -> human label
│   └── schemas.py              # Pydantic request/response models
├── tests/
│   └── test_examples.py        # Example requests using low-MAE samples
├── artifacts/                  # Created by training scripts (gitignored)
│   ├── script17/...
│   └── script21/...
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example                # Template for IBM Cloud credentials
└── README.md
```

---

## Quick start

### Step 1: Train the models locally (once)

The training scripts produce model artifacts (preprocessors, XGBoost models, lookups, sample test rows) that the service loads at runtime. Artifacts are mounted into the container, not baked into the image.

Edit `DATA_PATH` at the top of each training script to point to your CSV, then:

```bash
# Install training deps (one-time)
pip install -r requirements.txt
pip install optuna   # only needed when --retune is passed

# Fast smoke run (default params, ~30-60 min each on CPU; ~5-15 min on GPU)
python train_save_script17.py
python train_save_script21.py

# Production-grade with Optuna per quantile
python train_save_script17.py --retune --n_trials 50 --gpu
python train_save_script21.py --retune --n_trials 50 --gpu
```

#### Training options

| Flag                | Default | Purpose |
|---------------------|---------|---------|
| `--data PATH`       | (CONFIG) | CSV path. Edits the `DATA_PATH` constant at top of file if omitted. |
| `--cult PATH`       | (CONFIG) | Path to `cult_cars.xlsx`. |
| `--out DIR`         | `./artifacts/scriptXX` | Where to write all artifacts. |
| `--retune`          | off     | Run Optuna for every quantile model (slow but production-grade). |
| `--n_trials N`      | `50`    | Optuna trials per quantile when `--retune` is set. |
| `--gpu`             | off     | Use CUDA GPU. Requires XGBoost compiled with CUDA support and a working `nvidia-smi`. |
| `--cap-pct N`       | `99.5`  | Drop rows above this salevalue percentile. Use `100` to disable capping entirely. |
| `--min-salevalue N` | `0`     | Drop rows where `salevalue <= N` dollars. Default `0` matches existing behavior (`salevalue > 0`). |
| `--save-shap`       | off     | Compute and save **per-row** SHAP for every train and test row against the p50 model. Multi-GB output. Implies `--save-shap-global`. |
| `--save-shap-global` | off    | Compute and save **only global** SHAP feature importance (small CSV per dataset). Cheap; recommended over `--save-shap` unless you specifically need per-row attributions. |
| `--shap-parquet-only` | off   | When `--save-shap` is set, write only parquet for per-row files (skip CSV — recommended for large train sets). Does not affect `--save-shap-global` (always CSV; tiny). |
| `--shap-sample N`   | `20000` | For `--save-shap-global` only: subsample N rows for global importance. `0` = use all rows. Default of 20,000 gives essentially the same ranking as the full set, much faster. |

#### About `--cap-pct`

By default the training scripts cap the target variable at the **99.5th percentile** (so the top 0.5% of highest-priced cars are dropped from training). This prevents a small number of extreme high-value outliers (rare collector cars sold at auction prices, data-entry errors, etc.) from skewing the model — they're unpredictable from the available features anyway, and including them tends to inflate predictions for normal cars.

You may want to override this:

```bash
# Disable cap entirely — keep ALL rows in training
python train_save_script21.py --cap-pct 100

# Softer cap — drop only the top 0.1% (most extreme outliers)
python train_save_script21.py --cap-pct 99.9

# Tighter cap — drop top 1% (only train on the most typical 99%)
python train_save_script21.py --cap-pct 99.0

# Combine with other flags
python train_save_script21.py --cap-pct 99.9 --retune --n_trials 50 --gpu
```

When to keep the default (99.5):
- You're training on the full historical dataset and want to predict typical donations
- You see a handful of $50k+ sale values in your data that look like outliers
- You don't have features (like a "cult/collector flag") that would let the model learn what makes those cars different

When to use `--cap-pct 100` (disable):
- You're confident every high-priced row is legitimate AND you want the model to learn from them
- You've added features that explain the high-value cases (cult markers, auction flags, etc.)
- You're doing exploratory analysis and want raw, unfiltered metrics

The actual cap dollar value applied for a given run is recorded in `training_metadata.json` under `cap_pct` and `cap_value` so you can audit which threshold was used.

#### About `--min-salevalue`

By default the training scripts keep all rows where `salevalue > 0` (i.e., `--min-salevalue 0`). You can raise this floor to exclude very low-value sales — for example, when scrap-priced cars (titles sold for $20-$50 because the vehicle was effectively unsellable) add more noise than signal at the low end:

```bash
# Drop rows with salevalue at or below $50 (i.e., keep salevalue > $50)
python train_save_script21.py --min-salevalue 50

# Drop scrap-priced cars (anything at or below $100)
python train_save_script17.py --min-salevalue 100

# Combine with other flags
python train_save_script21.py --min-salevalue 50 --cap-pct 99.9 --retune --gpu
```

The filter is **strict inequality** (`salevalue > N`), so `--min-salevalue 50` keeps rows where the sale value is strictly greater than $50. With the default of `0`, you get the same `salevalue > 0` filter the script has always used.

When to keep the default (0):
- You want the model to learn from low-value sales as legitimate signal
- You don't have evidence that low-value rows are systematically noisy or recorded wrong
- You're using the full data distribution to estimate uncertainty (90% CI coverage)

When to raise the floor (e.g., 50 or 100):
- Visual inspection of low-value rows shows they're mostly title-only / scrap transactions you don't care to predict
- The `$0-200` tier diagnostics show the model has a structural floor near $300 that low-value rows can't be predicted under anyway (this is the case in our existing data — see prior diagnostic work)
- You want to focus the model's capacity on the price range that actually matters for your business

Honest caveat: raising the floor improves training signal but **also reduces test set coverage for the low tier**. If 30% of your true distribution sells for under $200 and you train without those rows, your model is no longer calibrated for that segment — it may predict $400 when the actual is $50, with no warning. Use this together with the per-tier metrics in `test_metrics.json` to audit the tradeoff.

The actual floor used for a given run is recorded in `training_metadata.json` under `min_salevalue`.

#### About `--save-shap` and `--save-shap-global`

Two flags control SHAP exports, chosen based on whether you want per-row attributions or just a feature-ranking summary.

**`--save-shap-global` — cheap, recommended starting point.** Computes only the global SHAP feature importance: one row per feature, aggregated across the whole train (or test) set. Output is a small CSV (~5 KB) per dataset, sorted by average dollar impact. **By default subsamples 20,000 rows** (controllable via `--shap-sample N`) which gives essentially the same feature ranking as the full set in a fraction of the time.

```bash
# Just the small global importance CSVs (uses 20K-row sample by default)
python train_save_script21.py --save-shap-global

# Use a different sample size — 50K is more thorough but slower
python train_save_script21.py --save-shap-global --shap-sample 50000

# Use ALL rows (most accurate but slow for large datasets)
python train_save_script21.py --save-shap-global --shap-sample 0

# Combined with other flags
python train_save_script17.py --retune --gpu --save-shap-global
```

**Why subsample?** Global SHAP importance is a statistical aggregate — `mean(|shap_value|)` per feature. For a typical XGBoost model on this data, the top-10 feature ranking from a 20K random sample matches the ranking from the full set, and the dollar magnitudes differ by less than 1%. On a 600K-row standard set, this is the difference between **~8 minutes** (full) and **~17 seconds** (20K sample). For exploratory work, sampling is the right default; for an audited final report, set `--shap-sample 0` to use all rows.

The output CSV records `n_rows_used` and `n_rows_total` so anyone reading the file can tell whether sampling was applied.

**`--save-shap` — expensive, only if you need per-row.** Computes per-row SHAP for every train and test row against the p50 model, then writes long-format files (one row per `(car, feature)` pair). Also writes the global importance summary, so you don't need both flags. **Off by default** because of the disk and compute cost.

```bash
# Per-row + global (parquet AND csv)
python train_save_script21.py --save-shap

# Parquet-only (much smaller; recommended whenever using --save-shap on a large train set)
python train_save_script21.py --save-shap --shap-parquet-only

# Combined with other flags
python train_save_script21.py --retune --gpu --save-shap --shap-parquet-only
```

**Files written:**

With `--save-shap-global` (small):
- `train_shap_global_importance.csv` — global importance summary, one row per feature
- `test_shap_global_importance.csv` — same for test set

With `--save-shap` (large per-row PLUS the small global):
- `train_shap_engineered.parquet` / `.csv` — every train row, every model feature
- `test_shap_engineered.parquet` / `.csv` — every test row, every model feature
- `train_shap_raw.parquet` / `.csv` — engineered features collapsed to user-facing raw groups (same labels as the `/predict` response)
- `test_shap_raw.parquet` / `.csv` — same for test set
- `train_shap_global_importance.csv` and `test_shap_global_importance.csv` (also produced)

For `script21`, each row's SHAP is computed against its routed model (cult or standard). For the per-row files this adds a `route` column. **For the global importance files**, the per-route files stay separate (`train_shap_global_importance_cult.csv`, `train_shap_global_importance_standard.csv`) because cult and standard use different feature spaces and aggregating them together would be misleading.

**Columns in `*_shap_global_importance.csv`** (small file, one row per feature, sorted descending by `mean_abs_dollar`):

| Column | Description |
|---|---|
| `feature` | Engineered feature name (model internal) |
| `mean_abs_shap` | Mean absolute SHAP value across rows (log space). The standard SHAP global-importance metric. |
| `mean_abs_dollar` | Mean absolute marginal dollar impact across rows. Most interpretable: "this feature shifts the prediction by $X on average." |
| `mean_abs_pct_of_pred` | Mean absolute percentage of final prediction (signed magnitudes ignored). Useful for comparing features across price tiers. |
| `pct_of_top` | `mean_abs_dollar` of this feature as a percentage of the highest-ranked feature's `mean_abs_dollar`. Top row = 100%. |
| `n_rows_used` | How many rows actually fed into the aggregation (may be smaller than `n_rows_total` if `--shap-sample` was active). |
| `n_rows_total` | Total rows available in the source dataset, before any sampling. |

**Columns in `*_shap_engineered.csv`:**

| Column | Description |
|---|---|
| `stock_id`, `vin`, `record_creation_date` | Joinable IDs |
| `salevalue` | True target value |
| `predicted_sale_value` | Model's p50 prediction (dollars) |
| `feature` | Engineered feature name (model internal — e.g. `make_freq`, `nav_cond_x_age_bkt`) |
| `feature_value` | Value the model saw for this feature on this row |
| `log_shap` | Raw SHAP value in log space |
| `dollar_impact` | Marginal dollar impact: `expm1(base + shap) − expm1(base)` |
| `pct_of_prediction` | Dollar impact as % of the final prediction (signed) |
| `pct_of_top_feature` | Dollar impact as % of the **largest absolute attribution** for this car (signed; rank-1 row = ±100%) |
| `rank_by_abs` | This feature's rank for this car (1 = largest absolute attribution) |

**Columns in `*_shap_raw.csv`** — same metadata + ID columns, plus:

| Column | Description |
|---|---|
| `feature_raw_key` | Internal key (e.g. `make`, `__collectible`, `__market_trend`) |
| `feature_label` | User-facing label (e.g. `Make`, `Collectible/cult vehicle status`) |
| `value` | Raw input value (e.g. `Toyota`, `Runs & Drives`), pulled from the original raw row |
| `dollar_impact`, `pct_of_prediction`, `pct_of_top_feature`, `rank_by_abs` | Same meaning as engineered |
| `n_underlying` | How many engineered features collapsed into this raw group |
| `top_underlying` | Name of the highest-magnitude engineered feature in this group (audit info) |

**Honest realism — compute and disk:**

- **Compute time on CPU**: roughly 45-60 min for a 470K train set, plus 1-3 min for test. The raw collapse adds another 30-60 min because it iterates per row in Python (the engineered version is fully vectorized). With `--gpu` it's noticeably faster but not zero.
- **Disk size**: The train engineered CSV for a typical 470K-row training set with ~80 features is around **5 GB**. Parquet compresses to ~600 MB. **Use `--shap-parquet-only` if disk space is tight.**
- **Memory**: Builds SHAP in batches (5000 rows at a time) and streams writes via pyarrow. Peak memory is usually under 4 GB but can climb if your features are wide.

**Why this is useful:**

- Find systematically wrong predictions: `SELECT * FROM test_shap_raw WHERE abs(salevalue - predicted_sale_value) > 1000`, group by top features, see which features keep showing up. Tells you where the model is failing.
- Audit specific cars for fairness/correctness: filter by `stock_id` and read top 5 rows by `rank_by_abs`.
- Find unstable features: features whose `pct_of_top_feature` swings wildly across cars are leading the model into noisy patterns.
- Build dashboards that explain individual predictions without re-running the model.

The actual flag values used for a given run are recorded in `training_metadata.json` under `save_shap_used`.

After each script finishes, you should see an `artifacts/script17/` or `artifacts/script21/` directory containing:

- `preprocessor_*.joblib` — fitted preprocessors
- `model_*_q05.json`, `model_*_q50.json`, `model_*_q95.json` — XGBoost models
- `cult_lookup.joblib`, `zip_lat_map.joblib`, `zip_lon_map.joblib` — lookups
- `sample_test_rows.parquet` — low-MAE samples for testing
- `train_predictions.parquet` / `.csv` — full train-set predictions per row
- `test_predictions.parquet` / `.csv` — full test-set predictions per row
- `train_metrics.json` / `test_metrics.json` — per-tier MAE/RMSLE/coverage/bias
- `training_metadata.json` — metrics, params, coverage, SHAP sanity check, cap settings used

### Step 2: Configure IBM Cloud credentials (optional)

If you want natural-language explanations via Granite, copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
# Edit .env with your real WATSONX_API_KEY and WATSONX_PROJECT_ID
```

If you skip this step, the service still works — explanations fall back to a templated paragraph built from the SHAP data.

### Step 3: Run the service

**Option A — Docker Compose (recommended):**

```bash
docker compose up --build
```

The service will be on `http://localhost:8000`. Artifacts are mounted read-only from `./artifacts`.

**Option B — Local Python:**

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000 --reload
```

Interactive Swagger UI is at `http://localhost:8000/docs`.

### Step 4: Test with sample rows

The training scripts saved low-MAE example rows. The test script reads those and demonstrates the API end-to-end:

```bash
# Sanity check the request shapes (no service needed)
python tests/test_examples.py --dry-run

# Live test against running service (no LLM call)
python tests/test_examples.py --model script21

# Live test with LLM explanation (slower)
python tests/test_examples.py --model script21 --explain

# Different model
python tests/test_examples.py --model script17 --explain
```

You should see actual vs predicted side-by-side, top SHAP features with dollar impacts, and (if `--explain`) a natural-language paragraph.

---

## API reference

### `POST /predict`

**Query parameters:**

| Parameter           | Type    | Default     | Description                                   |
|---------------------|---------|-------------|-----------------------------------------------|
| `model`             | string  | `script17`  | `script17` or `script21`                      |
| `explain`           | bool    | `false`     | Call Granite for natural-language explanation |
| `explanation_units` | string  | `dollars`   | `dollars` or `percentage` — units used in the natural-language explanation (only matters when `explain=true`) |
| `shap_quantile`     | string  | `p50`       | `p5`, `p50`, or `p95` — which quantile's SHAP |
| `k_pos`             | int     | `5`         | Top-K positive SHAP features (max 20)         |
| `k_neg`             | int     | `5`         | Top-K negative SHAP features (max 20)         |

**Request body (JSON)** — all fields optional, unknown fields ignored:

```json
{
  "make": "Toyota",
  "model": "Camry",
  "year": 2008,
  "vehicle_type": "Sedan",
  "body_type": "Sedan",
  "mileage": 145000,
  "nav_condition": "Runs & Drives",
  "bodypaintcondition": "Normal Wear & Tear (all body panels intact & attached)",
  "enginecondition": "Operational",
  "transmissioncondition": "Operational",
  "tirecondition": "All Wheels Mounted & Tires Inflated",
  "vazipcode": "94087",
  "record_creation_date": "2026-04-15"
}
```

**Response:**

```json
{
  "model_used": "script21",
  "is_cult": false,
  "route": "standard",
  "predictions": {
    "p5": 412.50,
    "p50": 750.20,
    "p95": 1320.80
  },
  "shap": {
    "quantile_explained": "q50",
    "baseline_dollars": 880.00,
    "final_pred_dollars": 750.20,
    "top_positive": [
      {
        "feature_raw_key": "make",
        "feature_label": "Make",
        "value": "Toyota",
        "dollar_impact": 130.40,
        "pct_of_prediction": 17.4,
        "n_underlying": 2,
        "top_underlying": "make_freq"
      },
      {
        "feature_raw_key": "__collectible",
        "feature_label": "Collectible/cult vehicle status",
        "value": null,
        "dollar_impact": 95.20,
        "pct_of_prediction": 12.7,
        "n_underlying": 3,
        "top_underlying": "cult_uplift_mid"
      }
    ],
    "top_negative": [
      {
        "feature_raw_key": "age",
        "feature_label": "Vehicle age (years)",
        "value": "18",
        "dollar_impact": -210.80,
        "pct_of_prediction": -28.1,
        "n_underlying": 3,
        "top_underlying": "age"
      }
    ]
  },
  "explanation": "This 2008 Toyota Camry has an estimated donation sale value of about $750...",
  "elapsed_ms": {
    "predict": 47.2,
    "explain": 2840.5,
    "total": 2887.7
  }
}
```

### `GET /healthz`

Returns whether each pipeline is loaded. Used by Docker healthcheck.

### `GET /models`

Returns metadata for each loaded pipeline (best params, test metrics, coverage, SHAP sanity check). Useful for verifying which artifact set is in use.

### `GET /docs`

FastAPI auto-generated Swagger UI.

---

## SHAP details

The SHAP attribution in the response is **collapsed from engineered features back to raw user-facing inputs**, for two reasons:

1. Hides model internals — users don't see implementation details like `nav_cond_x_age_bkt`, `make_freq`, or `cult_uplift_mid`.
2. Consolidates split contributions — `make_freq` and `make_tgt_enc` both encode the make of the car, so their dollar impacts are summed and reported once under the raw key `make`.

Each record in `top_positive` / `top_negative` represents one raw feature group:

- **`feature_raw_key`** — internal key (e.g. `make`, `mileage`, `__collectible` for the cult bucket)
- **`feature_label`** — user-facing label (e.g. `"Make"`, `"Collectible/cult vehicle status"`)
- **`value`** — the actual value submitted in the request (e.g. `"Toyota"`, `"145000"`), or `null` for derived buckets like market trend
- **`dollar_impact`** — summed marginal dollar impact across underlying engineered features (`expm1(base + shap_i) - expm1(base)`, summed within group)
- **`pct_of_prediction`** — summed percentage of the final prediction
- **`n_underlying`** — how many engineered features collapsed into this group (audit field)
- **`top_underlying`** — the single highest-magnitude engineered contributor in this group (audit field; safe to ignore)

**Collapsing rules:**

- Frequency / target encodings (`make_freq`, `model_tgt_enc`) → their base feature (`make`, `model`)
- Severity numerics (`nav_severity`, `engine_severity`) → original condition column (`nav_condition`, `enginecondition`)
- Interactions (e.g. `nav_cond_x_age_bkt`) → the **dominant** raw input listed first in the combo
- Macro features (`cpi_at_sale`, `manheim_at_sale`, `loan_at_sale`) → bucket `"Used-vehicle market trend"`
- Cult features (any `cult_*`) → bucket `"Collectible/cult vehicle status"`
- Geo features (`zip_*`, `lat_x_age`) → bucket `"Vehicle location (ZIP)"`
- Unknown counters (`n_unknowns`, `all_unknown`) → bucket `"How many condition fields are unknown"`
- Mechanical aggregates (`mechanical_severity_*`) → bucket `"Overall mechanical condition"`

**Top-K behavior**: the service examines the top 2K engineered features per side, collapses them to raw groups, then returns the top K raw groups. This ensures we usually fill K slots even when several engineered features collapse to the same raw group.

**SHAP for Script 17** explains the TE base (66% weight in the blend), not the full blended prediction. Blending SHAP across two different feature spaces is non-trivial and TE attributions are representative of the blended output.

---

## Performance notes

- **Cold start**: ~5 s to load artifacts at container startup
- **`/predict` without explain**: 50-150 ms
- **`/predict` with explain**: 2-8 s (depends on Granite latency)
- **SHAP first call per `(model, quantile)`**: ~300-500 ms (builds TreeExplainer); cached for subsequent calls (~20-50 ms)

For higher throughput, increase `--workers` in the Dockerfile CMD. Each worker loads its own copy of the artifacts, so RAM scales linearly (~1-1.5 GB per worker).

---

## Honest known limitations

- **$0-200 and $2.5K+ tiers** are structurally harder to predict and the 90% CI does not cover them as well as the middle tiers. See `training_metadata.json` for per-tier empirical coverage.
- **Geographic features** in 2026+ data are mostly null in our test set. The model still works but loses some predictive signal for cars without ZIP info.
- **Single-row API only**. No batch endpoint. If you need to score thousands of cars at once, batch at the client level for now.
