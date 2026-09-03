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

from app.raw_feature_mapping import (
    INTERNAL_TO_REQUEST_FIELD,
    VALUE_INTERNAL,
    VALUE_NOT_PROVIDED,
)

# Placeholder strings raw_feature_mapping puts in a group's `value` when it
# has nothing real to report. They must never reach the model: handed in as
# "(actual value: Not provided)" it can readily turn a missing input into a
# priced factor -- "the mechanical condition was not provided, reducing the
# value by 13.9%". Imported rather than restated so the two files cannot
# drift, and compared against the LOWERED string so the prompt behaviour
# survives the sentinels ever being recapitalised.
#
# Applied to the three call sites fed from a SHAP group record (v1 _line,
# v2 _line, and _fallback_explanation's _phrase). The two that read
# request_dict directly keep the plain check -- they never see a sentinel.
_VALUE_SENTINELS = ("", "nan",
                    VALUE_NOT_PROVIDED.lower(), VALUE_INTERNAL.lower())
from preprocessor import describe_picklist_value

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


def _car_summary(request_dict: Dict) -> str:
    """Human-readable 'field=value' summary of the vehicle, shared by every
    prompt version (v1, v2, ...)."""
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
        # request_dict uses the CURRENT PredictRequest field names, which
        # differ from body_type/nav_condition (kept here as the internal/
        # legacy names for this loop) since the new-schema migration --
        # fall back to the renamed request field via the same alias
        # app/raw_feature_mapping.py uses.
        val = request_dict.get(field)
        if val is None:
            request_field = INTERNAL_TO_REQUEST_FIELD.get(field)
            if request_field:
                val = request_dict.get(request_field)
        # Decode a numeric picklist ID (e.g. 22968) to its display name
        # ("Runs & Drives") so it reads naturally to the LLM instead of as
        # an opaque code -- safe no-op for every other field.
        val = describe_picklist_value(field, val)
        if val is not None and str(val).strip() not in ("", "nan"):
            car_summary_parts.append(f"{field}={val}")
    return (
        ", ".join(car_summary_parts)
        if car_summary_parts
        else "no descriptive details provided"
    )


def _route_note(route: Optional[str]) -> str:
    """Shared by every prompt version."""
    if route == "cult":
        return " (The vehicle was identified as a cult/collectible model, so it was scored with a specialized model.)"
    elif route == "standard":
        return " (The vehicle was scored with the standard model.)"
    return ""


# ---------------------------------------------------------------
# Prompt v1 (current production default) -- see docs/prompt_versions.csv
# ---------------------------------------------------------------
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

    car_summary = _car_summary(request_dict)

    # Conditionally format per-line features based on explanation_units
    def _line(r: Dict) -> str:
        label = r.get("feature_label", r.get("feature_raw_key", "feature"))
        val = r.get("value")
        if val is not None and str(val).strip().lower() not in _VALUE_SENTINELS:
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

    route_note = _route_note(route)

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
6. The output MUST be a single, flowing paragraph strictly Around 40-60 WORDS. Do not use markdown, bullet points, lists, or headings. Also do not use redundant phrases and information or connecting words like "furthermore, however, moreover". Just give the feature name its value and impact.
7. Do not mention SHAP values, feature importance, model contributions, or machine learning terminology.
{instruction_8}


Output Format Example (Mirror this exact structure, brevity, and tone, substituting the live data provided above):
"The 2007 Dodge Caliber is valued around $931 (range: $207-$1,977). Hatchback body and driveability {example_impacts}"

Explanation:"""

    return prompt


# ---------------------------------------------------------------
# Prompt v2 -- adds a plain-English blurb about what each feature actually
# represents (e.g. a market-trend/macro feature reads as "current U.S.
# economic conditions"), drops the 40-60 word cap, and only ever states a
# feature's value when one is present (many bucketed/macro features carry
# value=None -- see app/raw_feature_mapping.py's BUCKET_* groups). How many
# features are discussed (5 vs 3) is NOT a knob here -- it follows however
# many entries are already in shap["top_positive"]/["top_negative"], which
# is controlled upstream by the existing k_pos/k_neg pipeline params.
# See docs/prompt_versions.csv for the full change log and example outputs.
# ---------------------------------------------------------------

# Short, plain-English gloss of what a feature (raw key or bucket sentinel
# from app/raw_feature_mapping.py) actually represents, for the LLM to draw
# on instead of inventing an explanation. Falls back to a generic phrase
# built from the feature's label when a key isn't listed here.
FEATURE_CONTEXT: Dict[str, str] = {
    # Bucketed/macro features (BUCKET_* in raw_feature_mapping.py) -- these
    # almost always carry value=None since they don't map to one raw request
    # field, so the context blurb is the only substance the LLM has to work
    # with for them.
    "__market_trend": "current U.S. economic and used-vehicle market conditions (inflation, wholesale used-car pricing, and auto loan rates)",
    "__collectible": "the vehicle's status as a sought-after collectible or cult model among enthusiasts",
    "__location": "regional demand and pricing patterns based on where the vehicle is located",
    "__unknowns": "how many of the vehicle's six condition fields (driveability, body/paint, engine, transmission, tires, and interior) the submitter explicitly marked 'Unknown' -- fields simply left blank are NOT counted",
    "__mechanical": "the vehicle's overall mechanical condition, specifically its engine, transmission, and tires (interior and body/paint are scored separately)",
    "__time_of_sale": "the time of year the sale occurs, which affects seasonal demand",
    "__engine_specs": "the vehicle's engine specifications (size, cylinders, horsepower)",
    "__vehicle_profile": "how this vehicle's overall profile compares to similar vehicles",
    "__other_damages": "additional reported damage to the vehicle",
    # Common raw features
    "make": "the vehicle's manufacturer and its typical resale demand",
    "model": "the specific model and how well it tends to hold its value",
    "year": "the vehicle's model year",
    "age": "how many years old the vehicle is",
    "mileage": "how many miles the vehicle has been driven",
    "trim": "the specific trim or feature package of the vehicle",
    "vehicle_type": "the vehicle's body style/category",
    "body_type": "the vehicle's body style",
    "nav_condition": "the vehicle's overall driveability condition",
    "bodypaintcondition": "the condition of the vehicle's body and paint",
    "enginecondition": "the condition of the vehicle's engine",
    "transmissioncondition": "the condition of the vehicle's transmission",
    "tirecondition": "the condition of the vehicle's tires and wheels",
    "interiorcondition": "the condition of the vehicle's interior",
    "true_mileage_unknown": "whether the odometer reading can be trusted",
    "clean_title": "whether the vehicle has a clean, unbranded title",
    "nav_color": "the vehicle's exterior color",
    "vazipcode": "the vehicle's ZIP code and local market",
}


def _feature_context(raw_key: str, label: str) -> str:
    if raw_key in FEATURE_CONTEXT:
        return FEATURE_CONTEXT[raw_key]
    return f"the vehicle's {label.lower()}"


def _build_prompt_v2(
    request_dict: Dict,
    predictions: Dict[str, float],
    shap: Dict,
    model_used: str,
    route: Optional[str],
    is_cult: Optional[bool],
    explanation_units: str = "both",
    concise: bool = False,
    top_n: int = 5,
) -> str:
    p5 = predictions["low"]
    p50 = predictions["predicted_price"]
    p95 = predictions["high"]
    baseline = shap.get("baseline_dollars", 0.0)

    car_summary = _car_summary(request_dict)
    route_note = _route_note(route)

    def _line(r: Dict) -> str:
        raw_key = r.get("feature_raw_key", "")
        label = r.get("feature_label", raw_key or "feature")
        context = _feature_context(raw_key, label)
        val = r.get("value")

        # Only ever mention a value when one is actually present -- many
        # bucketed/macro features (market trend, collectible status, ...)
        # have value=None because they don't map to a single raw request
        # field, and the LLM must not invent one for those.
        if val is not None and str(val).strip().lower() not in _VALUE_SENTINELS:
            head = f"{label} (actual value: {val}) -- represents {context}"
        else:
            head = f"{label} -- represents {context}"

        pct_str = _fmt_pct(r["pct_of_prediction"])
        dol_str = _fmt_dollar(r["dollar_impact"])

        if explanation_units == "dollar":
            impact = dol_str
        elif explanation_units == "percentage":
            impact = pct_str
        else:
            impact = f"{pct_str} ({dol_str})"

        return f"  - {head}: {impact}"

    # No hardcoded [:5] slice here -- unlike v1, this shows exactly however
    # many features shap already carries, so the same builder serves both
    # a "top 5" and a "top 3" comparison purely by varying k_pos/k_neg
    # upstream when the SHAP dict was built.
    # top_n is what separates v2 (5 features) from v3/v4 (3): it caps how
    # many attributions the prompt is asked to cover. It can only ever
    # narrow what the pipeline already produced -- k_pos/k_neg decide how
    # many SHAP entries exist in the first place, so asking for 5 here when
    # k_pos=2 still yields 2.
    pos_lines = [_line(r) for r in shap.get("top_positive", [])[:top_n]]
    neg_lines = [_line(r) for r in shap.get("top_negative", [])[:top_n]]

    if explanation_units == "dollar":
        instruction_impact = 'Always express impacts in the exact format: "added approximately $Y" or "reduced the value by approximately $Y".'
    elif explanation_units == "percentage":
        instruction_impact = 'Always express impacts in the exact format: "added approximately X%" or "reduced the value by approximately X%".'
    else:
        instruction_impact = 'Always express impacts in the exact format: "added approximately X% ($Y)" or "reduced the value by approximately X% ($Y)".'

    # The only thing `concise` changes vs the unlimited variant is instruction
    # #7 (length + how much reasoning to give) and the worked example --
    # everything else (feature-context blurbs, null-value handling, how many
    # features are shown) is identical.
    if concise:
        instruction_7 = (
            'Write a single flowing summary of around 80 words -- a guide, not a '
            'hard limit: cover every feature listed above properly rather than '
            'truncating to hit a count. For each one, state the actual VALUE and '
            'explain WHY it moved the price in that direction, not just that it '
            'did. For example, do not write "mileage reduced the value by -5.7%"; '
            'write "its 150,000 miles is heavy wear for its age, cutting -5.7% '
            '(-$58)". Keep each reason to a short clause, and group several '
            'features into one sentence rather than giving each its own. Do not '
            'restate a feature you have already covered. Do not use markdown, '
            'bullet points, lists, or headings, and avoid redundant connecting '
            'words like "furthermore, however, moreover".'
        )
        example_block = (
            '"The 2007 Dodge Caliber is valued around $931 (range: $207-$1,977). Its hatchback body '
            'appeals to steady everyday demand and current U.S. economic conditions favour used cars, '
            'adding +19.3% ($194) and +5.2% ($52). Against that, its 146,000 miles is heavy wear for '
            'a car this age, cutting -5.7% (-$58), and its fair overall condition means more work '
            'before resale, taking a further -1.5% (-$15)."'
        )
    else:
        instruction_7 = (
            'Write in full, flowing prose using as many sentences as needed to clearly and completely '
            'cover every feature listed above -- there is NO word limit and NO target length. Prioritize '
            'completeness and clarity over brevity. Do not use markdown, bullet points, lists, or headings, '
            'and avoid redundant connecting words like "furthermore, however, moreover".'
        )
        example_block = (
            '"The 2007 Dodge Caliber is valued around $931, with an estimated range of $207 to $1,977. '
            'Its hatchback body style, which tends to attract steady everyday demand, added approximately '
            '+19.3% ($194). Current U.S. economic conditions, including inflation and used-vehicle market '
            'pricing trends, added a further +5.2% ($52) on top of that. On the other hand, its 146,000 miles '
            '-- reflecting how much the vehicle has been driven -- reduced the value by approximately -5.7% '
            '(-$58), and its fair overall condition reduced it by a further -1.5% (-$15)."'
        )
    length_note = (
        "NOT the length, which should stay around 80 words whatever the feature count"
        if concise
        else "NOT the length, which should scale to however many features are listed above"
    )

    prompt = f"""You are an assistant that explains a vehicle's predicted donation sale value in plain English to someone with no background in cars or data science.

Vehicle details: {car_summary}

Prediction (donation sale value, USD):
  - Baseline base price: ${baseline:.0f}
  - Median predicted value: ${p50:.0f}
  - Estimated sale value range: ${p5:.0f} to ${p95:.0f}{route_note}

Make sure to include all the features that have been mentioned here and DO NOT LEAVE ANY OUT. Each line below shows: the feature's name, its actual value (ONLY when one is given), what the feature represents, and its impact on the price.

Top features pushing the predicted price UP from the baseline:
{chr(10).join(pos_lines) if pos_lines else "  (none)"}

Top features pushing the predicted price DOWN from the baseline:
{chr(10).join(neg_lines) if neg_lines else "  (none)"}

Follow these exact instructions to write your response:
1. Start by stating the predicted donation sale value and the estimated sale value range, explicitly including the vehicle's make, model, year, and mileage.
2. Use the median estimate (${p50:.0f}) as the primary "expected to sell for around" value.
3. You MUST explicitly discuss EVERY SINGLE positive contributor and EVERY SINGLE negative contributor listed above. Do not skip or summarize any of them.
4. Do not explain what a confidence interval is.
5. For every feature, briefly explain in plain English what it represents before giving its impact -- for example, for a market-trend/economic feature say something like "current U.S. economic conditions"; for mileage, explain that it reflects how much the vehicle has been driven. Base this on what is stated for that feature above -- do not invent an unrelated explanation.
6. When a feature has an actual value given above, state that value naturally in your sentence. When NO value is given for a feature, describe it only by what it represents -- do NOT invent, guess, or state a specific number/value for it.
7. {instruction_7}
8. Do not mention SHAP values, feature importance, model contributions, or machine learning terminology.
9. {instruction_impact}

Output Format Example (mirror this exact style and tone -- {length_note} -- substituting the live data provided above):
{example_block}

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
        if val is not None and str(val).strip().lower() not in _VALUE_SENTINELS:
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
    prompt_version: str = "v1",
) -> str:
    """Main entry: try Granite, fall back to template on any failure.

    prompt_version selects the prompt template sent to Granite. Three
    versions, all covering the top 5 features and forming a ladder of
    increasing detail:

      version | length       | per-feature context | causal reasoning
      --------|--------------|---------------------|------------------
      "v1"    | 40-60 words  | no                  | no   (default)
      "v2"    | ~80 words    | yes                 | yes
      "v3"    | no limit     | yes                 | yes  (most verbose)

    "per-feature context" is a plain-English blurb about what each feature
    represents (e.g. a market-trend feature reads as "current U.S. economic
    conditions"), introduced in v2 along with explicit null-value handling.

    "causal reasoning" is v2's addition: it must state each feature's actual
    VALUE and say WHY that moved the price -- "its 150,000 miles means well
    above average wear for its age, cutting -5.7%" rather than the bare
    "mileage reduced the value by -5.7%".

    The feature count is a CEILING, not a target: k_pos/k_neg decide how
    many SHAP entries exist at all, so top-5 yields only 2 features when the
    request was made with k_pos=2.

    Wired to /predict's `prompt_version` query param. That param is
    deliberately never logged -- see the logging call below. See
    docs/prompt_versions.csv for the change log and example outputs.
    """

    client = get_client()
    if not client.configured:
        return _fallback_explanation(request_dict, predictions, shap, explanation_units)

    # v2 and v3 share one template and differ only in length and how much
    # causal reasoning is asked for -- both cover the top 5 features:
    #   v2 -- ~80 words, values + why each moved the price
    #   v3 -- no word limit, the most verbose form
    _V2_FAMILY = {
        # ~80 words, so 600 tokens is generous headroom.
        "v2": {"top_n": 5, "concise": True, "max_new_tokens": 600},
        "v3": {"top_n": 5, "concise": False, "max_new_tokens": 1500},
    }
    if prompt_version in _V2_FAMILY:
        cfg = _V2_FAMILY[prompt_version]
        prompt = _build_prompt_v2(
            request_dict,
            predictions,
            shap,
            model_used,
            route,
            is_cult,
            explanation_units,
            concise=cfg["concise"],
            top_n=cfg["top_n"],
        )
        max_new_tokens = cfg["max_new_tokens"]
    else:
        prompt = _build_prompt(
            request_dict,
            predictions,
            shap,
            model_used,
            route,
            is_cult,
            explanation_units,
        )
        max_new_tokens = 1000

    # prompt_version is deliberately NOT logged: which prompting
    # technique served a request is not part of the audit trail.
    # NOTE: the prompt body below still reveals the template in use --
    # see the caveat raised with this change.
    logger.info("Generated LLM Prompt:\n%s", prompt)

    try:
        text = client.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.3)
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
