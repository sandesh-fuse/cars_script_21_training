# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

FastAPI service ("Donated Car Sale-Value Predictor") that predicts the donation sale value of a vehicle with a 90% confidence interval (p5/p50/p95 via XGBoost quantile regression), SHAP-based feature attributions in dollars/percentages, and optional natural-language explanations via IBM watsonx.ai Granite. Two interchangeable pipelines are served side-by-side and selected per-request via `?model=`:

- `script17` — TE + no-TE blend ensemble. Best overall MAE/RMSLE.
- `script21` — routed cult/standard split. Better on the $500-1K tier specifically.

This is not a git repository, and there is no linter, formatter, or CI configuration anywhere in the repo — don't invent lint commands or assume any are enforced.

## Commands

Install dependencies:
```bash
pip install -r requirements.txt
pip install optuna   # only needed when passing --retune to a training script
```

Run the service:
```bash
# Docker (recommended) — artifacts mounted read-only from ./artifacts
docker compose up --build

# Local
uvicorn app.main:app --port 8000 --reload
```

Train models (produces the `artifacts/script17/` or `artifacts/script21/` directory the service loads at runtime — edit `DATA_PATH` at the top of each script first, or pass `--data`):
```bash
python train_save_script17.py
python train_save_script21.py

# Production-grade, with Optuna tuning per quantile
python train_save_script17.py --retune --n_trials 50 --gpu
python train_save_script21.py --retune --n_trials 50 --gpu
```
Other flags both scripts accept: `--out DIR` (default `./artifacts/scriptXX`), `--cap-pct N` (default `99.5`, drop rows above this salevalue percentile; `100` disables), `--min-salevalue N` (default `0`, strict `salevalue > N` filter), `--save-shap` (per-row SHAP, expensive, multi-GB), `--save-shap-global` (cheap global feature-importance CSV, recommended default), `--shap-parquet-only`, `--shap-sample N` (default `20000`).

`train_save_script21.py`-only: `--use-dataone` (default off) — DataOne-sourced vehicle spec features (`oem_body_style`, `drive_type`, `engine_name`, `engineconfiguration`, `enginecylinders`, `displacementl`, `enginehp`, `msrp`, `transmission_name`, `us_style_name` — see `DATAONE_FEATURES` in the script) are excluded from training by default via `SaleValuePreprocessor(extra_drop_cols=...)`; pass this flag to include them. Recorded per-run in `training_metadata.json` (`use_dataone`, `dataone_features_excluded`). `train_save_script17.py` has no equivalent flag — the shared `SaleValuePreprocessor` defaults are untouched for it.

Smoke-test the API (there is no pytest suite — `tests/test_examples.py` is a manual, live-integration script using `argparse`/`requests` against `sample_test_rows.parquet`):
```bash
python tests/test_examples.py --dry-run                        # validate request shapes, no service needed
python tests/test_examples.py --model script21                 # live call, no LLM
python tests/test_examples.py --model script21 --explain        # live call + Granite explanation
python tests/test_examples.py --model script17 --explain
```

## Architecture

Request flow for `POST /predict`: [app/main.py](app/main.py) validates the body against `PredictRequest`/`List[PredictRequest]` ([app/schemas.py](app/schemas.py)) → requires an `x-api-key` header checked against the relevant `API_KEY_*` env var (`_make_auth` dependency, one key per route: `prediction`, `get_logs`, `killswitch`) → dispatches by `?model=` to the pre-loaded `Script17Pipeline` or `Script21Pipeline` (both instantiated once at startup in `load_pipelines()` from `ARTIFACTS_DIR`, default `./artifacts`, into module-level globals) → the pipeline runs the shared [preprocessor.py](preprocessor.py) transform, then XGBoost quantile models predict p5/p50/p95 → [app/shap_utils.py](app/shap_utils.py) computes SHAP via a cached `shap.TreeExplainer` (one per `(model, quantile)`, guarded by a lock) → [shap_dollar_helper.py](shap_dollar_helper.py) converts log-space SHAP into dollar/% impacts → [app/raw_feature_mapping.py](app/raw_feature_mapping.py) collapses engineered features back to raw user-facing groups → [app/feature_descriptions.py](app/feature_descriptions.py) humanizes labels → optionally [app/explainer.py](app/explainer.py) calls Granite for natural-language text → structured JSON response, plus structured JSON logging ([app/log_config.json](app/log_config.json), via `python-json-logger`) with PII (zip code) masked by `_mask_sensitive_data`. `/predict` accepts either a single request object or a list of them and normalizes internally.

Key files:
- [preprocessor.py](preprocessor.py) (top-level, ~1100 lines) — the single most shared module: feature engineering, macro data tables (CPI/Manheim/auto-loan), `build_monthly_series()` interpolation, `compute_cult_flag()`, `cpi_ratio_arr()`/`deflate_pred()`. Imported by both inference pipelines and both training scripts. `SaleValuePreprocessor` also takes an opt-in `extra_drop_cols` param (default `None`/no-op) letting one caller hard-drop extra features without changing the shared class defaults — currently only used by `train_save_script21.py`'s `--use-dataone` flag.
- [app/inference_script17.py](app/inference_script17.py) (`Script17Pipeline`) — loads 6 XGBoost models (`model_te_{q05,q50,q95}.json`, `model_no_te_{q05,q50,q95}.json`) + 2 preprocessors; predicts via a weighted blend (`p_q = w_te * te_q + w_no_te * no_te_q`) with a quantile-crossing fix; SHAP explains only the TE base (66% of the blend).
- [app/inference_script21.py](app/inference_script21.py) (`Script21Pipeline`) — loads 6 XGBoost models (cult/standard × q05/q50/q95) + 2 preprocessors + `cult_lookup.joblib` + `alphas.json`; routes each request to the cult or standard model via `compute_cult_flag()` (make/model/year match); CPI-deflates predictions. **Side effect**: appends every request's engineered features to `api_engineered_features.csv` in the working directory on every call — unbounded growth, script21-only, not a bug to silently "fix".
- [app/schemas.py](app/schemas.py) — Pydantic I/O models. `PredictRequest` has all-optional fields and `extra = "allow"` (unknown fields are ignored, not rejected).
- [app/explainer.py](app/explainer.py) (`WatsonxGraniteClient`) — lazy singleton; `_get_token()` exchanges `WATSONX_API_KEY` for a cached IAM bearer token; `generate()` POSTs to watsonx.ai; falls back to a templated explanation if unconfigured or the call fails.
- [app/ibm_logs_client.py](app/ibm_logs_client.py) — used by `POST /logs` to query IBM Cloud Logs (DataPrime); uses `IBM_CLOUD_API_KEY`, falling back to `WATSONX_API_KEY` if unset.
- No centralized settings/config class exists — env vars are read ad hoc via `os.environ.get`/`os.getenv` scattered across `main.py`, `explainer.py`, and the inference modules. Key vars: `ARTIFACTS_DIR`, `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, `WATSONX_MODEL_ID`, `IBM_LOGS_INSTANCE_ID`, `IBM_LOGS_REGION`, `IBM_CLOUD_API_KEY`, `API_KEY_PREDICTION`, `API_KEY_GET_LOGS`, `API_KEY_KILLSWITCH`.
- Training scripts ([train_save_script17.py](train_save_script17.py), [train_save_script21.py](train_save_script21.py)) and the one-off `diagnose_*.py` / `evaluate_predictions.py` / `read_praquet.py` scripts are standalone CLIs, not imported by the running service. `artifacts/` is produced by these scripts and is not checked in (only `.gitkeep`).
- [timewise_breakdown.py](timewise_breakdown.py) — same standalone-CLI-or-library shape as `evaluate_predictions.py`, but breaks accuracy down over time (weekly/monthly MAE/RMSE/RMSLE/CI-width, plus a monthly fixed-size random-sample summary and per-month sample export). `train_save_script21.py` calls it automatically (`run_timewise_breakdown`) on `test_pred_df` at the end of training, writing to `artifacts/script21/timewise_breakdown/`; not wired into `train_save_script17.py`.

## Known discrepancies between README.md and current code

The README is otherwise detailed and reliable, but trust the code over it in these spots:
- README says Swagger UI is live at `/docs`; [app/main.py](app/main.py) explicitly sets `docs_url=None, redoc_url=None, openapi_url=None` — the docs UI is currently disabled.
- README's default `?model=` is `script17`; the actual `Query(...)` default in `main.py`'s `/predict` is `script21`.
- README's `explanation_units` options are `dollars`/`percentage` (default `dollars`); the code's pattern is `^(both|percentage|dollar)$` with default `both`.
- `/predict` accepts a single object **or a list** (batch), contradicting the README's "Single-row API only" limitation note.

## Secrets

The `.env` file at the repo root currently holds real-looking credential values (IBM Cloud API key, watsonx project ID, and the `API_KEY_*` values). Never print or quote its contents.
