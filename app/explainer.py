"""
explainer.py
=============
Client for IBM watsonx.ai foundation model (Granite) hosted in IBM Cloud.

Auth:
    Exchanges an IBM Cloud API key for an IAM Bearer token via:
        POST https://iam.cloud.ibm.com/identity/token
    Token is cached and refreshed when expired.

Generation:
    POST {WATSONX_URL}/ml/v1/text/chat?version=2024-05-31
    Body: {"messages": [{"role": "user", "content": "..."}], "model_id": "...",
           "project_id": "...", "max_tokens": ..., "temperature": ...}
    (The older /ml/v1/text/generation "input"-style endpoint is deprecated by
    IBM in favor of this chat-completions-style endpoint; migrated 2026-08-26.)

Environment variables (all required):
    WATSONX_API_KEY    — your IBM Cloud API key
    WATSONX_PROJECT_ID — watsonx.ai project ID
    WATSONX_URL        — base URL, e.g. https://us-south.ml.cloud.ibm.com
    WATSONX_MODEL_ID   — e.g. ibm/granite-4-h-small (verify against your
                          project's available models — IBM periodically
                          retires older model IDs, e.g. ibm/granite-3-8b-instruct)

If credentials are missing or generation fails, returns a templated fallback
that includes the SHAP data but without LLM-generated prose.
"""

import os
import time
import threading
import requests
from typing import Dict, Optional, List

import logging

logger = logging.getLogger("car-sale-value-predictor")


class WatsonxGraniteClient:
    """Lazy-loaded client. Singleton via module-level get_client() below."""

    def __init__(self):
        self.api_key = os.environ.get("WATSONX_API_KEY", "").strip()
        self.project_id = os.environ.get("WATSONX_PROJECT_ID", "").strip()
        self.base_url = os.environ.get("WATSONX_URL", "").strip().rstrip("/")
        self.model_id = os.environ.get(
            "WATSONX_MODEL_ID", "ibm/granite-4-h-small"
        ).strip()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0
        self._token_lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.project_id and self.base_url)

    def _get_token(self) -> str:
        """Fetch (or reuse cached) IBM Cloud IAM bearer token."""
        now = time.time()
        with self._token_lock:
            if self._token and now < self._token_expires_at - 60:
                return self._token
            resp = requests.post(
                "https://iam.cloud.ibm.com/identity/token",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={
                    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                    "apikey": self.api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload["access_token"]
            self._token_expires_at = now + int(payload.get("expires_in", 3600))
            return self._token

    def generate(
        self, prompt: str, max_new_tokens: int = 1000, temperature: float = 0.3
    ) -> str:
        """Call watsonx.ai text chat completions. Returns the generated string.

        Uses /ml/v1/text/chat, not the older /ml/v1/text/generation endpoint
        (which IBM has deprecated and warns "will be removed soon"). The chat
        API takes a "messages" array and "max_tokens" instead of "input" and
        "max_new_tokens"/"decoding_method"/"min_new_tokens"/"repetition_penalty"
        — those older sampling knobs have no equivalent here and are dropped.
        """
        url = f"{self.base_url}/ml/v1/text/chat?version=2024-05-31"
        body = {
            "model_id": self.model_id,
            "project_id": self.project_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        if resp.status_code == 401:
            # Token expired between issue and use. Refresh once.
            self._token = None
            headers["Authorization"] = f"Bearer {self._get_token()}"
            resp = requests.post(url, headers=headers, json=body, timeout=60)
        if not resp.ok:
            # Surface IBM's error body (e.g. model_not_supported) in the logs
            # instead of a bare "404 Client Error" with no detail.
            logger.warning(
                "watsonx.ai chat call failed (%s): %s",
                resp.status_code,
                resp.text[:2000],
            )
        resp.raise_for_status()
        result = resp.json()
        generated = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return generated.strip()


_CLIENT: Optional[WatsonxGraniteClient] = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> WatsonxGraniteClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = WatsonxGraniteClient()
        return _CLIENT


# ===========================================================
# Prompt template + fallback
# ===========================================================
def _fmt_dollar(v: float) -> str:
    sign = "-" if v < 0 else "+"
    return f"{sign}${abs(v):.0f}"


def _fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"


def _build_prompt(
    request_dict: Dict,
    predictions: Dict[str, float],
    shap: Dict,
    model_used: str,
    route: Optional[str],
    is_cult: Optional[bool],
    explanation_units: str = "both",
) -> str:
    p5 = predictions["low"]
    p50 = predictions["predicted_price"]
    p95 = predictions["high"]
    baseline = shap.get("baseline_dollars", 0.0)

    car_summary_parts = []
    for field in [
        "year",
        "make",
        "model",
        "trim",
        "body_type",
        "mileage",
        "nav_condition",
        "vehicle_type",
    ]:
        val = request_dict.get(field)
        if val is not None and str(val).strip() not in ("", "nan"):
            car_summary_parts.append(f"{field}={val}")
    car_summary = (
        ", ".join(car_summary_parts)
        if car_summary_parts
        else "no descriptive details provided"
    )

    # Conditionally format per-line features based on explanation_units
    def _line(r: Dict) -> str:
        label = r.get("feature_label", r.get("feature_raw_key", "feature"))
        val = r.get("value")
        if val is not None and str(val).strip() not in ("", "nan"):
            head = f"{label} (value: {val})"
        else:
            head = label

        pct_str = _fmt_pct(r["pct_of_prediction"])
        dol_str = _fmt_dollar(r["dollar_impact"])

        if explanation_units == "dollar":
            return f"  - {head}: {dol_str}"
        elif explanation_units == "percentage":
            return f"  - {head}: {pct_str}"
        else:
            return f"  - {head}: {pct_str} ({dol_str})"

    pos_lines = [_line(r) for r in shap.get("top_positive", [])[:5]]
    neg_lines = [_line(r) for r in shap.get("top_negative", [])[:5]]

    route_note = ""
    if route == "cult":
        route_note = " (The vehicle was identified as a cult/collectible model, so it was scored with a specialized model.)"
    elif route == "standard":
        route_note = " (The vehicle was scored with the standard model.)"

    # Dynamic prompt instructions and short 40-word examples based on the selected unit
    if explanation_units == "dollar":
        instruction_5 = "5. For every feature mentioned, explicitly state its name, its actual value, and how much it shifted the base price using ONLY dollar amounts."
        instruction_8 = '8. Always express impacts in the exact format: "added approximately $Y" or "reduced the value by approximately $Y".'
        example_impacts = "added $194 and $52. Conversely, 146k miles and fair condition reduced it by $58 and $15."
    elif explanation_units == "percentage":
        instruction_5 = "5. For every feature mentioned, explicitly state its name, its actual value, and how much it shifted the base price using ONLY percentage amounts."
        instruction_8 = '8. Always express impacts in the exact format: "added approximately X%" or "reduced the value by approximately X%".'
        example_impacts = "added +19.3% and +5.2%. Conversely, 146k miles and fair condition reduced it by -5.7% and -1.5%."
    else:  # both
        instruction_5 = "5. For every feature mentioned, explicitly state its name, its actual value, and how much it shifted the base price using BOTH percentage and dollar amounts."
        instruction_8 = '8. Always express impacts in the exact format: "added approximately X% ($Y)" or "reduced the value by approximately X% ($Y)".'
        example_impacts = "added +19.3% ($194) and +5.2% ($52). Conversely, 146k miles and fair condition reduced it by -5.7% (-$58) and -1.5% (-$15)."

    prompt = f"""You are an assistant that explains a vehicle's predicted donation sale value in plain English.

Vehicle details: {car_summary}

Prediction (donation sale value, USD):
  - Baseline base price: ${baseline:.0f}
  - Median predicted value: ${p50:.0f}
  - Estimated sale value range: ${p5:.0f} to ${p95:.0f}{route_note}

Make sure to include all the features that have been mentioned here and DO NOT LEAVE ANY OUT:
Top features positive features pushing the predicted price UP from the baseline:
{chr(10).join(pos_lines) if pos_lines else "  (none)"}

Top features positive features pushing the predicted price DOWN from the baseline:
{chr(10).join(neg_lines) if neg_lines else "  (none)"}

Follow these exact instructions to write your response:
1. Start by stating the predicted donation sale value and the estimated sale value range, explicitly including the vehicle's make, model, year, and mileage.
2. Use the median estimate (${p50:.0f}) as the primary "expected to sell for around" value.
3. You MUST explicitly list and discuss EVERY SINGLE positive contributor and EVERY SINGLE negative contributor provided in the lists above. Do not skip or summarize any features; include them all.
4. Do not explain what a confidence interval is.
{instruction_5}
6. The output MUST be a single, flowing paragraph strictly Around 40 WORDS. Do not use markdown, bullet points, lists, or headings. Also do not use redundant phrases and information or connecting words like "furthermore, however, moreover". Just give the feature name its value and impact.
7. Do not mention SHAP values, feature importance, model contributions, or machine learning terminology.
{instruction_8}


Output Format Example (Mirror this exact structure, brevity, and tone, substituting the live data provided above):
"The 2007 Dodge Caliber is valued around $931 (range: $207-$1,977). Hatchback body and driveability {example_impacts}"

Explanation:"""

    return prompt


def _fallback_explanation(
    request_dict: Dict,
    predictions: Dict[str, float],
    shap: Dict,
    explanation_units: str = "both",
) -> str:
    """Templated explanation when LLM is not available or fails."""
    p5, p50, p95 = (
        predictions["low"],
        predictions["predicted_price"],
        predictions["high"],
    )
    pos = shap["top_positive"]
    neg = shap["top_negative"]
    baseline = shap.get("baseline_dollars")

    descr_parts = []
    for field in ["year", "make", "model"]:
        val = request_dict.get(field)
        if val is not None and str(val).strip() not in ("", "nan"):
            descr_parts.append(str(val))
    descr = " ".join(descr_parts) if descr_parts else "This vehicle"

    msg = (
        f"{descr} has an estimated donation sale value of about ${p50:,.0f}, "
        f"with a 90% confidence range of ${p5:,.0f}–${p95:,.0f}. "
    )

    if baseline is not None:
        msg += (
            f"For comparison, a typical donated car (average age, mileage, "
            f"and condition) would be valued around ${baseline:,.0f}. "
        )

    def _phrase(r: Dict, push_dir: str) -> str:
        label = r.get("feature_label", r.get("feature_raw_key", "a feature")).lower()
        val = r.get("value")
        if val is not None and str(val).strip() not in ("", "nan"):
            name_with_value = f"the {label} of {val}"
        else:
            name_with_value = label

        pct_str = _fmt_pct(r["pct_of_prediction"])
        dol_str = _fmt_dollar(r["dollar_impact"])

        if explanation_units == "dollar":
            impact_str = dol_str
        elif explanation_units == "percentage":
            impact_str = pct_str
        else:
            impact_str = f"{pct_str}, which is {dol_str}"

        return (
            f"The biggest factor {push_dir} the value is {name_with_value} "
            f"({impact_str}). "
        )

    if pos:
        msg += _phrase(pos[0], "pushing up")
    if neg:
        msg += _phrase(neg[0], "reducing")

    if (p95 - p5) > p50:  # very wide CI
        msg += "The wide confidence range reflects real uncertainty about this vehicle's market value."
    return msg.strip()


def explain(
    request_dict: Dict,
    predictions: Dict[str, float],
    shap: Dict,
    model_used: str,
    explanation_units: str = "both",
    route: Optional[str] = None,
    is_cult: Optional[bool] = None,
) -> str:
    """Main entry: try Granite, fall back to template on any failure."""

    client = get_client()
    if not client.configured:
        return _fallback_explanation(request_dict, predictions, shap, explanation_units)

    prompt = _build_prompt(
        request_dict,
        predictions,
        shap,
        model_used,
        route,
        is_cult,
        explanation_units,
    )

    logger.info("Generated LLM Prompt:\n%s", prompt)

    try:
        text = client.generate(prompt, max_new_tokens=1000, temperature=0.3)
        if not text or len(text) < 20:
            return _fallback_explanation(
                request_dict, predictions, shap, explanation_units
            )
        return text
    except Exception as e:
        logger.warning(
            f"Granite text generation failed ({type(e).__name__}: {e}); using fallback template.",
            exc_info=True,
            extra={"component": "explainer", "model_used": model_used, "error": str(e)},
        )
        return _fallback_explanation(request_dict, predictions, shap, explanation_units)


if __name__ == "__main__":
    import json

    # Configure basic logging for the test
    logging.basicConfig(level=logging.INFO)

    # 1. Mock request data mimicking the 2011 Lexus LFA
    mock_request = {
        "year": 2011,
        "make": "Lexus",
        "model": "LFA",
        "mileage": 12500,
        "vehicle_type": "Coupe",
        "nav_condition": "Runs & Drives",
    }

    # 2. Mock predictions (High-value collectible)
    mock_predictions = {"low": 750000.0, "predicted_price": 850000.0, "high": 980000.0}

    # 3. Mock SHAP data mimicking the LFA's value drivers
    mock_shap = {
        "baseline_dollars": 15000.0,
        "top_positive": [
            {
                "feature_label": "model",
                "value": "LFA",
                "dollar_impact": 650000.0,
                "pct_of_prediction": 76.4,
            },
            {
                "feature_label": "Collectible/cult vehicle status",
                "value": "Yes",
                "dollar_impact": 195000.0,
                "pct_of_prediction": 22.9,
            },
        ],
        "top_negative": [
            {
                "feature_label": "mileage",
                "value": "12,500",
                "dollar_impact": -8000.0,
                "pct_of_prediction": -0.9,
            },
            {
                "feature_label": "age",
                "value": "15 years",
                "dollar_impact": -2000.0,
                "pct_of_prediction": -0.2,
            },
        ],
    }

    print("==============================================")
    print("Testing Fallback Explanation (Template Based)")
    print("==============================================")
    # Temporarily hide credentials to force the fallback method
    original_key = os.environ.get("WATSONX_API_KEY")
    if original_key:
        del os.environ["WATSONX_API_KEY"]

    fallback_result = explain(
        request_dict=mock_request,
        predictions=mock_predictions,
        shap=mock_shap,
        model_used="script21",
        route="cult",
        is_cult=True,
    )
    print(f"\n{fallback_result}\n")

    print("==============================================")
    print("Testing Live Granite LLM Explanation")
    print("==============================================")
    # Restore credentials to test live WatsonX call
    if original_key:
        os.environ["WATSONX_API_KEY"] = original_key

        # Ensure other required vars are present for the test
        if not os.environ.get("WATSONX_PROJECT_ID"):
            print(
                "WARNING: WATSONX_PROJECT_ID is missing. LLM generation will likely fail."
            )
        if not os.environ.get("WATSONX_URL"):
            print("WARNING: WATSONX_URL is missing. LLM generation will likely fail.")

        llm_result = explain(
            request_dict=mock_request,
            predictions=mock_predictions,
            shap=mock_shap,
            model_used="script21",
            route="cult",
            is_cult=True,
        )
        print(f"\n{llm_result}\n")
    else:
        print(
            "WATSONX_API_KEY environment variable not set. Skipping live LLM generation test."
        )
