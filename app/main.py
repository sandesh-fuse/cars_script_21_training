"""
main.py
========
FastAPI service exposing /predict for both Script 17 and Script 21 pipelines.

Endpoint:
    POST /predict?model={script17|script21}&explain={true|false}&shap_quantile={p5|p50|p95}&k_pos=N&k_neg=N
         &explanation_units={both|percentage|dollar}&prompt_version={v1|v2|v3}

Request body (JSON): car attributes — see schemas.PredictRequest.

Response (JSON): see schemas.PredictResponse with predictions {p5, p50, p95},
optional SHAP payload, optional natural-language explanation.

Env vars:
    ARTIFACTS_DIR  — base directory containing script17/ and script21/ subdirs
                     (default: ./artifacts)
    WATSONX_* — optional, for natural-language explanations
    API_KEY_PREDICTION — API key for the predict endpoint
    API_KEY_GET_LOGS — API key for the logs endpoint
"""

import os
import time
import logging
import uuid
import json
from typing import Optional, List, Union

from fastapi import FastAPI, HTTPException, Query, Request, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas import PredictRequest, PredictResponse, LogsQueryRequest, LogsResponse
from app.inference_script17 import Script17Pipeline
from app.inference_script21 import Script21Pipeline
from app.explainer import explain
from app.ibm_logs_client import fetch_and_format_logs

# ==========================================
# Setup Logging
# ==========================================
logger = logging.getLogger("car-sale-value-predictor")

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./artifacts")

# ==========================================
# Authentication Setup
# ==========================================
API_KEYS = {
    "prediction": os.getenv("API_KEY_PREDICTION"),
    "get_logs": os.getenv("API_KEY_GET_LOGS"),
    "killswitch": os.getenv("API_KEY_KILLSWITCH"),
}


def _make_auth(route_key: str):
    def _dep(
        request: Request,
        x_api_key: Optional[str] = Header(None),
        x_user_id: Optional[str] = Header(None),
    ) -> dict:
        client_ip = request.client.host if request.client else "unknown"

        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing x-api-key header.")
        if x_api_key != API_KEYS.get(route_key):
            raise HTTPException(
                status_code=403, detail="Invalid API key for this endpoint."
            )

        user_id = x_user_id.strip().lower() if x_user_id else "anonymous"
        return {"api_key": x_api_key, "user_id": user_id, "client_ip": client_ip}

    return _dep


app = FastAPI(
    title="Donated-car sale-value predictor",
    description="Predicts the donation sale value for a vehicle, with 90% CI and SHAP-based explanations.",
    version="1.0.0",
    docs_url=None,  # disables the Swagger UI /docs
    redoc_url=None,  # disables the ReDoc UI /redoc
    openapi_url=None,  # disables the openapi.json schema completely
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Request ID Middleware
# ==========================================
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ==========================================
# Security / Data Scrubbing
# ==========================================
def _mask_sensitive_data(payload: dict) -> dict:
    """Masks sensitive fields before logging raw request bodies.

    Checks both "zip" (current PredictRequest field name) and "vazipcode"
    (its legacy name, pre-migration): this runs on the raw, unvalidated
    JSON body in exception-handler logging paths (see below), which can
    still be a not-yet-migrated caller's payload -- one using the old field
    name would fail Pydantic validation (extra="forbid") and have its raw
    body logged right here, so both keys need masking to avoid a PII leak
    in the warning log during the transition.
    """
    if not isinstance(payload, dict):
        return payload
    safe_payload = payload.copy()
    for zip_key in ("zip", "vazipcode"):
        if zip_key in safe_payload and safe_payload[zip_key]:
            safe_payload[zip_key] = "*****"
    return safe_payload


# ==========================================
# Global Exception Handlers
# ==========================================
@app.exception_handler(HTTPException)
async def fastapi_http_exception_handler(request: Request, exc: HTTPException):
    req_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host if request.client else "unknown"
    user_id = request.headers.get("X-User-ID", "unauthenticated")

    try:
        body = await request.json()
        body = _mask_sensitive_data(body)
    except Exception:
        body = "No JSON body or unparseable"

    logger.warning(
        f"HTTP Exception: {exc.detail}",
        extra={
            "request_id": req_id,
            "endpoint": request.url.path,
            "user_id": user_id,
            "status_code": exc.status_code,
            "error": exc.detail,
            "input_data": body,
            "client_ip": client_ip,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(
    request: Request, exc: StarletteHTTPException
):
    req_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host if request.client else "unknown"
    user_id = request.headers.get("X-User-ID", "unauthenticated")

    try:
        body = await request.json()
        body = _mask_sensitive_data(body)
    except Exception:
        body = "No JSON body or unparseable"

    logger.warning(
        f"Starlette HTTP Exception: {exc.detail}",
        extra={
            "request_id": req_id,
            "endpoint": request.url.path,
            "user_id": user_id,
            "status_code": exc.status_code,
            "error": exc.detail,
            "input_data": body,
            "client_ip": client_ip,
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    req_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host if request.client else "unknown"
    user_id = request.headers.get("X-User-ID", "unauthenticated")

    missing_fields = []
    other_errors = []

    for err in exc.errors():
        msg = err.get("msg", "")
        loc = err.get("loc", ())
        if "function-after" in msg or "valid list" in msg or "valid dictionary" in msg:
            continue
        field_name = str(loc[-1]) if loc else "unknown"

        if err.get("type") == "missing":
            missing_fields.append(field_name)
        else:
            clean_msg = msg.replace("Value error, ", "")
            if field_name != "body" and not field_name.isdigit():
                other_errors.append(f"{field_name}: {clean_msg}")

    diagnostics_parts = []
    if missing_fields:
        unique_missing = sorted(list(set(missing_fields)))
        diagnostics_parts.append(
            f"These fields are missing: {', '.join(unique_missing)}"
        )
    if other_errors:
        unique_other = sorted(list(set(other_errors)))
        diagnostics_parts.extend(unique_other)

    final_diagnostics = (
        " | ".join(diagnostics_parts)
        if diagnostics_parts
        else "Invalid request payload."
    )

    try:
        raw_body = await request.json()
        raw_body = _mask_sensitive_data(raw_body)
    except Exception:
        raw_body = "Unparseable JSON body"

    logger.warning(
        "Request validation failed",
        extra={
            "request_id": req_id,
            "endpoint": request.url.path,
            "user_id": user_id,
            "diagnostics": final_diagnostics,
            "raw_request_body": raw_body,
            "client_ip": client_ip,
        },
    )
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Failed", "diagnostics": final_diagnostics},
    )


# ==========================================
# Pipelines Initialization
# ==========================================
SCRIPT17_PIPELINE: Optional[Script17Pipeline] = None
SCRIPT21_PIPELINE: Optional[Script21Pipeline] = None


@app.on_event("startup")
def load_pipelines():
    global SCRIPT17_PIPELINE, SCRIPT21_PIPELINE
    s17_dir = os.path.join(ARTIFACTS_DIR, "script17")
    s21_dir = os.path.join(ARTIFACTS_DIR, "script21")

    if os.path.exists(s17_dir):
        logger.info("Loading Script 17 artifacts...", extra={"directory": s17_dir})
        try:
            SCRIPT17_PIPELINE = Script17Pipeline(s17_dir)
            logger.info("Script 17 pipeline loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load Script 17 pipeline.", exc_info=True)

    if os.path.exists(s21_dir):
        logger.info("Loading Script 21 artifacts...", extra={"directory": s21_dir})
        try:
            SCRIPT21_PIPELINE = Script21Pipeline(s21_dir)
            logger.info("Script 21 pipeline loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load Script 21 pipeline.", exc_info=True)

    if not SCRIPT17_PIPELINE and not SCRIPT21_PIPELINE:
        logger.error("CRITICAL: NO pipelines loaded. /predict will return 503.")


# ===========================================================
# Endpoints
# ===========================================================
@app.get("/healthz")
def health(request: Request):
    req_id = getattr(request.state, "request_id", "unknown")
    client_ip = request.client.host if request.client else "unknown"
    user_id = request.headers.get("X-User-ID", "unauthenticated")

    status_ok = SCRIPT17_PIPELINE is not None or SCRIPT21_PIPELINE is not None

    if not status_ok:
        logger.error(
            "Health check failed: No pipelines loaded.",
            extra={
                "request_id": req_id,
                "endpoint": "/healthz",
                "user_id": user_id,
                "client_ip": client_ip,
            },
        )
        raise HTTPException(
            status_code=503, detail="Service Unavailable: Models not loaded."
        )

    logger.debug(
        "Health check passed.",
        extra={
            "request_id": req_id,
            "endpoint": "/healthz",
            "user_id": user_id,
            "client_ip": client_ip,
        },
    )
    return {
        "status": "running",
        # "script17_loaded": SCRIPT17_PIPELINE is not None,
        "model_loaded": SCRIPT21_PIPELINE is not None,
    }


@app.get("/models")
def list_models(request: Request):
    info = {}
    for name, pipe, sub in [
        ("script17", SCRIPT17_PIPELINE, "script17"),
        ("script21", SCRIPT21_PIPELINE, "script21"),
    ]:
        if pipe is None:
            info[name] = {"loaded": False}
            continue
        meta_path = os.path.join(ARTIFACTS_DIR, sub, "training_metadata.json")
        meta = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                pass
        info[name] = {"loaded": True, "metadata": meta}
    return info


SHAP_QUANTILE_MAP = {"p5": "q05", "p50": "q50", "p95": "q95"}


@app.post("/predict", response_model=Union[PredictResponse, List[PredictResponse]])
def predict(
    request: Request,
    payload: Union[PredictRequest, List[PredictRequest]],
    auth_info: dict = Depends(_make_auth("prediction")),
    model: str = Query(
        "script21", pattern="^(script17|script21)$", description="Which model to use"
    ),
    explain_flag: bool = Query(
        False,
        alias="explain",
        description="Generate natural-language explanation via Granite",
    ),
    shap_quantile: str = Query(
        "p50", pattern="^(p5|p50|p95)$", description="Which quantile's SHAP to compute"
    ),
    k_pos: int = Query(
        5,
        ge=-1,
        le=20,
        description="Top-K positive SHAP features; -1 returns all available",
    ),
    k_neg: int = Query(
        5,
        ge=-1,
        le=20,
        description="Top-K negative SHAP features; -1 returns all available",
    ),
    explanation_units: str = Query(
        "both",
        pattern="^(both|percentage|dollar)$",
        description="Format for explanation impacts",
    ),
    prompt_version: str = Query(
        "v1",
        pattern="^(v1|v2|v3)$",
        description=(
            "Which Granite prompt template generates the explanation. All "
            "three cover the top 5 features. v1 (default): 40-60 words, no "
            "per-feature context. v2: ~80 words, adds plain-English context "
            "per feature plus the value and why it moved the price. v3: no "
            "length limit, the most verbose form. The feature count is a "
            "ceiling capped by k_pos/k_neg. Only relevant when explain=true."
        ),
    ),
):
    req_id = getattr(request.state, "request_id", "unknown")
    user_id = auth_info["user_id"]
    client_ip = auth_info["client_ip"]

    pipeline = SCRIPT17_PIPELINE if model == "script17" else SCRIPT21_PIPELINE
    if pipeline is None:
        logger.error(
            f"Prediction failed: Model {model} is not loaded.",
            extra={
                "request_id": req_id,
                "endpoint": "/predict",
                "client_ip": client_ip,
                "user_id": user_id,
                "requested_model": model,
            },
        )
        raise HTTPException(503, f"Model '{model}' not loaded")

    qlabel_internal = SHAP_QUANTILE_MAP[shap_quantile]

    # Normalize payload to a list for unified processing
    is_single_input = not isinstance(payload, list)
    if is_single_input:
        payload = [payload]

    batch_size = len(payload)
    # Computed upfront (not built incrementally in the loop below) so the
    # "requested" log always reflects the full input, and so it's still
    # available in full even if processing fails partway through the batch.
    input_stock_ids = [(item.stock_id or "unknown") for item in payload]
    output_log_data = []
    final_responses = []

    total_t_predict = 0.0
    total_t_explain = 0.0

    # Logged unconditionally, before any processing -- this fires no matter
    # what happens next (success or failure), so /logs can filter on
    # stock_id and always find at least this entry for a given request.
    logger.info(
        f"Prediction {model} requested",
        extra={
            "request_id": req_id,
            "endpoint": "/predict",
            "model_version": model,
            "user_id": user_id,
            "client_ip": client_ip,
            "batch_size": batch_size,
            "input_stock_ids": input_stock_ids,
            "explain_requested": explain_flag,
            "shap_quantile": shap_quantile,
        },
    )

    try:
        # Loop through each item in the normalized list
        for idx, item in enumerate(payload):
            # exclude_none=False (NOT True) is required for correctness, not
            # just style: preprocessor.py's feature engineering guards
            # (`if col in X.columns`, `if set(src).issubset(X.columns)`)
            # treat an ABSENT column completely differently from a
            # PRESENT-but-NaN one. Training always sees every raw column
            # (just sometimes NaN, correctly encoded to the model's learned
            # "unknown" category -- e.g. a "make__na"-style combo index).
            # exclude_none=True silently drops any field the caller omitted,
            # so the single-row DataFrame built in Script21Pipeline.predict()
            # is MISSING that column outright -- every combo/frequency
            # feature built from it then gets skipped entirely and back-
            # filled with a bare NaN instead of the trained "unknown"
            # encoding, silently degrading predictions for any request with
            # missing fields (confirmed via validate_live_predictions.py:
            # 14 of 130 features differed for a real row with a few missing
            # attributes, fully explaining a systematic live-vs-offline MAE
            # gap). See app/inference_script21.py's None->NaN coercion for
            # the other half of this fix.
            try:
                request_dict = item.model_dump(exclude_none=False)
            except AttributeError:  # pydantic v1
                request_dict = item.dict(exclude_none=False)

            # Precomputed upfront (see input_stock_ids above), not
            # re-derived here -- keeps this in lockstep with the value
            # already logged on the "requested" log line for this request.
            stock_id = input_stock_ids[idx]

            t0 = time.time()
            pred_result = pipeline.predict(
                request_dict=request_dict,
                k_pos=k_pos,
                k_neg=k_neg,
                shap_quantile=qlabel_internal,
                explain=explain_flag,
            )
            t_predict = (time.time() - t0) * 1000
            total_t_predict += t_predict

            explanation_text = None
            t_explain = 0.0

            if explain_flag and pred_result.get("shap"):
                t1 = time.time()
                explanation_text = explain(
                    request_dict=request_dict,
                    predictions=pred_result["predictions"],
                    shap=pred_result["shap"],
                    model_used=model,
                    route=pred_result.get("route"),
                    is_cult=pred_result.get("is_cult"),
                    explanation_units=explanation_units,
                    prompt_version=prompt_version,
                )
                t_explain = (time.time() - t1) * 1000
                total_t_explain += t_explain

            feature_explanations = []
            if pred_result.get("shap"):
                for f in pred_result["shap"].get("top_positive", []):
                    feature_explanations.append(
                        {
                            "feature_name": f.get(
                                "feature_label", f.get("feature_raw_key", "Unknown")
                            ),
                            "contribution": f"{f.get('pct_of_prediction', 0):+.1f}%",
                        }
                    )
                for f in pred_result["shap"].get("top_negative", []):
                    feature_explanations.append(
                        {
                            "feature_name": f.get(
                                "feature_label", f.get("feature_raw_key", "Unknown")
                            ),
                            "contribution": f"{f.get('pct_of_prediction', 0):+.1f}%",
                        }
                    )

            output_log_data.append(
                {
                    "stock_id": stock_id,
                    "predicted_sale_value": pred_result["predictions"][
                        "predicted_price"
                    ],
                    "predicted_range": {
                        "low": pred_result["predictions"]["low"],
                        "high": pred_result["predictions"]["high"],
                    },
                    "feature_explanations": feature_explanations,
                }
            )

            final_responses.append(
                {
                    "request_id": req_id,
                    "stock_id": stock_id,
                    "model_used": model,
                    "is_cult": pred_result.get("is_cult"),
                    "predictions": pred_result["predictions"],
                    "feature_importances": pred_result.get("shap"),
                    "explanation": explanation_text,
                }
            )

        total_time = total_t_predict + total_t_explain
        inference_duration_seconds = total_time / 1000.0

        logger.info(
            f"Prediction {model} batch completed",
            extra={
                "request_id": req_id,
                "endpoint": "/predict",
                "model_version": model,
                "user_id": user_id,
                "client_ip": client_ip,
                "batch_size": batch_size,
                "inference_duration_seconds": round(inference_duration_seconds, 4),
                "input_stock_ids": input_stock_ids,
                "output_data": output_log_data,
                "total_ms": round(total_time, 2),
            },
        )

        # Unpack the response if the input was a single dictionary
        if is_single_input:
            return final_responses[0]

        return final_responses

    except Exception as e:
        logger.error(
            f"An error occurred during prediction ({model})",
            exc_info=True,
            extra={
                "request_id": req_id,
                "endpoint": "/predict",
                "model_version": model,
                "user_id": user_id,
                "client_ip": client_ip,
                "batch_size": batch_size,
                "input_stock_ids": input_stock_ids,
                "error": str(e),
            },
        )
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction."
        )


# ==========================================
# Get IBM Application Logs Endpoint
# ==========================================
@app.post(
    "/logs",
    response_model=LogsResponse,
    response_model_exclude_none=True,
)
async def get_application_logs(
    request: Request,
    payload: LogsQueryRequest,
    auth_info: dict = Depends(_make_auth("get_logs")),
):
    """
    Query, aggregate, and return downstream IBM DataPrime API logs
    in a clean, structured JSON format.
    """
    req_id = getattr(request.state, "request_id", "unknown")
    user_id = auth_info["user_id"]
    client_ip = auth_info["client_ip"]

    ibm_api_key = os.getenv("IBM_CLOUD_API_KEY", os.getenv("WATSONX_API_KEY"))
    if not ibm_api_key:
        logger.error(
            "IBM Cloud credentials missing from environment.",
            extra={
                "request_id": req_id,
                "endpoint": "/logs",
                "user_id": user_id,
                "client_ip": client_ip,
            },
        )
        raise HTTPException(
            status_code=500,
            detail="Server Misconfiguration: IBM Cloud credentials missing from environment.",
        )

    try:
        logger.info(
            "Logs query requested",
            extra={
                "request_id": req_id,
                "endpoint": "/logs",
                "user_id": user_id,
                "client_ip": client_ip,
                "query_limit": payload.limit,
                "query_stock_id": payload.stock_id,
                "query_endpoint": payload.endpoint,
                # NOT req_id above: that is THIS call's own id, whereas this
                # is the id being searched for. Two different things.
                "query_request_id": payload.request_id,
            },
        )

        start_time_exec = time.time()
        logs_payload = await fetch_and_format_logs(
            api_key=ibm_api_key,
            stock_id=payload.stock_id,
            endpoint=payload.endpoint,
            request_id=payload.request_id,
            start_time=payload.start_time,
            end_time=payload.end_time,
            days_ago=payload.days_ago,
            minutes_ago=payload.minutes_ago,
            limit=payload.limit,
        )
        fetch_duration = time.time() - start_time_exec

        logger.info(
            "Logs query completed",
            extra={
                "request_id": req_id,
                "endpoint": "/logs",
                "user_id": user_id,
                "client_ip": client_ip,
                "fetch_duration_seconds": round(fetch_duration, 4),
                "log_count_retrieved": logs_payload.get("log_count", 0),
                "time_window": logs_payload.get("time_window"),
            },
        )

        return {**logs_payload, "request_id": req_id}

    except Exception as e:
        logger.error(
            "An error occurred during log retrieval",
            exc_info=True,
            extra={
                "request_id": req_id,
                "endpoint": "/logs",
                "user_id": user_id,
                "client_ip": client_ip,
                "error": str(e),
            },
        )
        raise HTTPException(
            status_code=500, detail=f"Downstream service processing breakdown: {str(e)}"
        )


# @app.get("/")
# def root():
#     return {
#         "service": "donated-car-sale-value-predictor",
#         "endpoints": ["/predict (POST)", "/models", "/healthz", "/logs"],
#         "docs": "/docs",
#     }
