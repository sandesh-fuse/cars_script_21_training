"""
schemas.py
==========
Pydantic request/response models for the /predict endpoint.
"""

import re
from typing import Optional, List, Any, Dict, Union
from pydantic import BaseModel, Field, model_validator, field_validator
from datetime import datetime, timezone

from app.ibm_logs_client import LOG_QUERY_ENDPOINTS

# Canonical form emitted by str(uuid.uuid4()) in main.py's request-id
# middleware -- the only shape a real request_id can have.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


# Fields whose real contract is "a number" -- 9 of these (all but zip) are
# picklist IDs the deployed script21 model has no learned encoding for as
# text at all (see each field's own docstring paragraph below for why).
# Typed Optional[Union[int, float]] below, NOT str, so the schema's own
# type signature says what it actually expects. _require_numeric_id()
# still accepts a numeric-looking STRING for caller convenience (e.g.
# "22968", a common JSON-serialization quirk) and converts it, but rejects
# genuine text (e.g. "Runs & Drives", "Iowa") with a clear 422 up front --
# previously that text was silently accepted here and reached XGBoost
# 3 layers deep inside Script21Pipeline.predict(), which raised an opaque
# "DataFrame.dtypes ... must be int, float, bool or category" ValueError
# instead. See project memory script21-vehicle-type-color-picklist-bug for
# the production incident this was written in response to.
#
# zip is grouped in here for a DIFFERENT reason: preprocessor.py's zip
# handling (.astype(str).str.extract(...), then the raw column is dropped
# entirely) is already robust to any input shape, including a string --
# it was never at the same crash risk as the other 9. It's here purely to
# keep the *contract* unambiguous: a caller must send a real number or an
# actual JSON null, never a string standing in for one ("", "null", "N/A",
# ...), which a permissive str|int|float type would otherwise allow through
# indistinguishably from a genuine numeric zip.
NUMERIC_PICKLIST_ID_FIELDS = (
    'vehicle_type', 'color',
    'vehicle_cond_picklist_id', 'engine_cond_picklist_id',
    'transmission_cond_picklist_id', 'body_paint_cond_picklist_id',
    'interior_cond_picklist_id', 'tire_cond_picklist_id',
    'state_picklist_id', 'state_title_picklist',
    'zip',
)


def _coerce_numeric_id(v, field_name):
    """Shared scalar coercion for a single picklist-ID-shaped value: accept a
    real number, or a numeric-looking string (a common JSON client quirk),
    converting either to a float; reject anything else (genuine resolved
    text, a bool, ...) with a clear ValueError. Used directly by
    _require_numeric_id below for NUMERIC_PICKLIST_ID_FIELDS, and per-element
    by _require_numeric_damage_ids for other_damage_pklist_id (which can also
    be a list of these)."""
    if isinstance(v, bool):
        raise ValueError(f"{field_name} must be a number, not a boolean.")
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)
        except ValueError:
            raise ValueError(
                f"{field_name} must be a number (e.g. 22968), not text ({v!r})."
            )
    raise ValueError(f"{field_name} must be a number, got {type(v).__name__}.")


class PredictRequest(BaseModel):
    """Request body for /predict — all car attribute fields are optional.

    The preprocessor handles missing fields gracefully (treats as NaN).
    Field names now match the NEW upstream DB schema's raw column names
    (the taegram export), NOT the legacy names preprocessor.py's feature
    engineering is internally written against — preprocessor.py renames
    them automatically (see NEW_TO_OLD_SCHEMA_MAP / map_raw_features_to_legacy
    in preprocessor.py), so nothing in app/ needs to translate between the
    two. Each field below that was renamed from a prior legacy name says so
    in its own comment, for anyone cross-referencing older logs/artifacts/
    SHAP payloads that still show the legacy name.

    Trimmed to the fields actually consumed by the currently-deployed
    script21 model (per artifacts/script21/training_metadata.json:
    use_dataone=False, enabled_new_features=[]), PLUS the fields the live
    upstream payload always sends regardless (vin, other_damage_pklist_id,
    oem_body_style, drive_type, msrp — see their own notes below). Config.
    extra is "forbid" (not "allow"), so every field a real request can
    carry must be declared here or the whole request 422s.

    Two toggle-gated groups are still intentionally omitted below because
    the deployed config has them off AND the live payload never sends them:
      - Remaining DataOne spec fields (engine_name, engineconfiguration,
        enginecylinders, enginehp, displacementl, valvetraindesign,
        transmission_name, us_style_name) — only read when a model is
        retrained with --use-dataone.
      - NEW_FEATURE_COLS fields (true_mileage_unknown, clean_title,
        gvm_range, tonnage, engine_type) — only read when a model is
        retrained with --enable-new-features.
    Re-add any of the above explicitly here if/when a deployed model
    starts using it and the upstream payload starts sending it.
    Also dropped for being unused in every config, not just this one —
    never populated by any known training data source: vin_id, oem_doors,
    rear_axle, model_number.

    vin_hin_no (legacy name: vin) is dropped outright by preprocessor.py's
    DROP_COLS_NO_ZIP — it never reaches the model as a feature, regardless
    of config. Declared here purely so the live payload (which always
    includes it) validates under extra="forbid"; it has zero effect on
    predictions.

    other_damage_pklist_id (legacy name: other_damages) IS genuinely
    consumed: preprocessor.py's _parse_other_damages() derives
    has_other_damage / n_other_damages / other_damages_normalized /
    per-damage-type indicator features from it, then drops the raw column.
    Now typed Optional[Union[int, float, List[Union[int, float]]]] and
    validated by _require_numeric_damage_ids below, matching the numeric-
    picklist-ID contract of NUMERIC_PICKLIST_ID_FIELDS above: a vehicle can
    carry more than one damage type, so this field additionally accepts a
    JSON array of numeric IDs, not just a single scalar. A resolved-text
    value (e.g. "mold, other*") is rejected here with a 422 instead of
    silently reaching preprocessor.py's parser as an unrecognized token --
    same rationale as the other picklist-ID fields (project memory
    script21-vehicle-type-color-picklist-bug). preprocessor.py's
    _parse_other_damages() itself still accepts plain text and the legacy
    JSON-encoded {'id','name'}-dict string format too, since it's also used
    to ingest historical training data written before this migration --
    only the live /predict request contract is restricted to numeric ID(s)
    here.

    oem_body_style / drive_type / msrp are DataOne spec fields (see
    DATAONE_FEATURES in train_save_script21.py), gated by --use-dataone at
    training time. The currently-deployed model was trained with
    use_dataone=False, so these three sit outside its feature_cols_ and
    have zero effect on today's predictions — but the live payload
    consistently includes them, so they're declared here (rather than left
    to extra="allow") purely to keep real requests from 422ing. msrp is
    typed Optional[Union[str, int, float]], the same str-or-number
    convention as zip/doors/vehicle_type/color below, because the
    live payload sends it as a JSON string ('"17500"') despite being
    numeric, and preprocessor.py has no bespoke coercion for it (unlike
    zip's regex-extract) — it's a raw DataOne passthrough.

    vehicle_type / color (legacy name: nav_color) are numeric picklist IDs:
    in the taegram training export, these two arrive as unresolved numeric
    picklist IDs (e.g. 23101.0), not text. vehicle_type is the SAME raw
    column name in both schemas (no NEW_TO_OLD_SCHEMA_MAP rename needed)
    and is now in train_save_script21.py's SCRIPT21_RAW_COLUMNS whitelist
    -- it was excluded from the initial migration's whitelist, which
    measurably regressed accuracy (it's ~98.5% populated and drives 4
    derived interaction features on top of itself), so it was added back;
    color already arrived as a raw numeric ID even before this migration
    (no color_name sibling ever existed). The deployed model was trained on
    those raw numbers, never int-encoded, so at inference time a number for
    either field reaches the model as-is and predicts fine. See
    NUMERIC_PICKLIST_ID_FIELDS / _require_numeric_id above the class for
    why genuine text is now rejected with a 422 instead of silently
    reaching XGBoost and crashing there (project memory:
    script21-vehicle-type-color-picklist-bug).

    doors is ALSO typed str-or-number, for a related but distinct reason:
    the raw taegram 'doors' column isn't cleanly numeric across the full
    training set (stray junk values like '2500', '2.4l', 'lt' show up
    alongside real door counts), so the WHOLE column got fit as a text
    category, not a numeric passthrough (confirmed:
    preprocessor_standard.joblib's int_maps_['doors'] has string keys like
    '4.0'/'4' as DIFFERENT trained categories, not a numeric feature at
    all). A caller sending doors=4 as a plain int previously got silently
    coerced to Python int 4 by the old `Optional[int]` typing, which
    doesn't match either learned string key -- degrading to the model's
    "unknown door count" category every time. Send a genuine number here
    (int or float both work) and it's rendered to match the dominant
    '<N>.0'-style trained category, same convention as vehicle_type/
    color; a string still passes through unchanged too.

    zip (legacy name: vazipcode) requires a numeric value now too (via
    NUMERIC_PICKLIST_ID_FIELDS / _require_numeric_id), but for a DIFFERENT
    reason than the picklist-ID fields above: preprocessor.py's zip
    handling does `.astype(str).str.extract(r'(\\d{1,5})')` on it
    regardless of input type, so "52732.0"/52732.0/52732/"52732" all
    extract to the identical '52732' before being zero-padded and fanned
    out into zip_region/zip_first2/zip_first3/zip_lat/zip_lon/
    zip_full_freq, then the raw column is dropped entirely -- it was never
    at crash risk from a string the way the other 9 fields were (verified:
    even outright garbage like "not-a-zip" degrades gracefully to the "no
    zip" bucket instead of erroring). It's restricted to numeric here
    purely to keep the request *contract* unambiguous -- a missing zip
    must be sent as an actual JSON null, not a string standing in for one
    ("", "null", "N/A", ...), which the old str|int|float typing would
    otherwise accept indistinguishably from a genuine numeric zip.

    accessible_for_tow_truck (legacy: accessiblefortwotruck) /
    located_at_donation_c_a (legacy: locatedatdonationca) stay
    Optional[str], NOT bool, even though they're plain true/false flags:
    both are int-encoded categoricals trained on the STRING keys
    'true'/'false' (preprocessor_standard.joblib's int_maps_ confirms
    exactly {'true': 0, 'false': 1} for each). A genuine Python bool lands
    as `bool` dtype in the single-row DataFrame Script21Pipeline.predict()
    builds, which preprocessor.py's _normalize_text() silently skips (it
    only touches object-dtype columns) -- so it would never get
    lowercased/stringified and would fail to match either trained category,
    same failure mode vehicle_type/color/doors had before their fixes
    above. _require_true_false below now coerces a genuine bool to the
    literal string "true"/"false" (fixing that silently-wrong case), and
    rejects anything else that isn't (case-insensitively) "true" or "false"
    with a clear 422 instead of letting an unrecognized string reach the
    model as an unseen category.

    vehicle_cond_picklist_id / engine_cond_picklist_id /
    transmission_cond_picklist_id / body_paint_cond_picklist_id /
    interior_cond_picklist_id / tire_cond_picklist_id (legacy names:
    nav_condition, enginecondition, transmissioncondition,
    bodypaintcondition, interiorcondition, tirecondition) require the
    numeric picklist ID now, enforced by _require_numeric_id above the
    class. preprocessor.py's severity dicts (NAV_CONDITION_SEV / BODY_SEV /
    ENGINE_SEV / TRANS_SEV / TIRE_SEV / INTERIOR_SEV) still carry both the
    numeric-ID keys AND the original text-label keys internally (so a
    legacy text value would still resolve to the correct severity number
    if it ever got that far) -- but a raw picklist-ID column is ALSO fed
    straight through into the final feature matrix as its own feature
    (alongside whatever's derived from it), and since these columns are
    numeric-only at fit time under the new schema, there's no learned
    encoding for text there at all. A resolved-text value (e.g.
    "Runs & Drives") used to be silently accepted here and crash 3 layers
    deep in XGBoost instead -- see project memory
    script21-vehicle-type-color-picklist-bug for the production incident.
    Rejected here now with a clear 422 instead.

    state_title_picklist / state_picklist_id (legacy names:
    state_province_of_title, vstate_name) require the numeric picklist ID
    for the same reason, but more strictly still: these are opaque,
    non-ordinal categoricals with no severity-dict-style text fallback at
    all (see preprocessor.py's FREQ_COLS_BASE) -- a resolved text value
    like "Iowa" was never correctly encoded here even before it crashed,
    unlike the condition fields above which at least degrade correctly on
    text everywhere except the raw passthrough.
    """

    # Identifiers
    stock_id: Optional[str] = None
    # vin_hin_no: Optional[str] = None  # legacy name: vin

    # Core vehicle attrs
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    trim: Optional[str] = None
    vehicle_type: Optional[Union[int, float]] = None
    vehicle_category: Optional[str] = None  # legacy name: body_type
    body_subtype: Optional[str] = None
    mileage: Optional[float] = None

    # Condition & Visual attrs
    vehicle_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: nav_condition
    color: Optional[Union[int, float]] = None  # legacy name: nav_color
    body_paint_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: bodypaintcondition
    engine_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: enginecondition
    transmission_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: transmissioncondition
    tire_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: tirecondition
    interior_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: interiorcondition
    other_damage_pklist_id: Optional[Union[int, float, List[Union[int, float]]]] = None  # legacy name: other_damages
    comment: Optional[str] = None  # legacy name: all_clean_notes

    # Geo, Admin, and Flags
    zip: Optional[Union[int, float]] = None  # legacy name: vazipcode
    state_picklist_id: Optional[Union[int, float]] = None  # legacy name: vstate_name
    state_title_picklist: Optional[Union[int, float]] = None  # legacy name: state_province_of_title
    accessible_for_tow_truck: Optional[str] = None  # legacy name: accessiblefortwotruck
    located_at_donation_c_a: Optional[str] = None  # legacy name: locatedatdonationca

    # Date — defaults to today if not provided
    creation_datetime: Optional[str] = None  # legacy name: record_creation_date

    @field_validator(*NUMERIC_PICKLIST_ID_FIELDS, mode="before")
    @classmethod
    def _require_numeric_id(cls, v, info):
        """Accept a real number, or a numeric-looking string (a common JSON
        client quirk -- see NUMERIC_PICKLIST_ID_FIELDS above the class),
        converting either to a float. Reject anything else -- genuine
        resolved text (e.g. "Runs & Drives"/"Iowa") for the 9 picklist-ID
        fields, or a string standing in for null (e.g. "", "null", "N/A")
        for zip -- with a clear error instead of letting it through to
        crash inside XGBoost later (or, for zip, silently degrade to the
        same "no zip" bucket as a real null while claiming to be a value).
        Message is deliberately generic ("a number", not "a picklist ID")
        since zip is grouped in here too and isn't one."""
        if v is None:
            return v
        return _coerce_numeric_id(v, info.field_name)

    @field_validator("other_damage_pklist_id", mode="before")
    @classmethod
    def _require_numeric_damage_ids(cls, v):
        """other_damage_pklist_id is a picklist-ID field like the ones in
        NUMERIC_PICKLIST_ID_FIELDS above, except a vehicle can carry more
        than one damage type -- so this field additionally accepts a JSON
        array of numeric IDs, not just a single scalar. Each element goes
        through the same number-or-numeric-string coercion as
        _require_numeric_id; genuine text (a damage-type name instead of
        its ID, e.g. "mold, other*") is rejected with a 422 rather than
        silently reaching preprocessor.py's other-damages parser as an
        unrecognized token."""
        if v is None:
            return v
        if isinstance(v, list):
            return [_coerce_numeric_id(item, "other_damage_pklist_id") for item in v]
        return _coerce_numeric_id(v, "other_damage_pklist_id")

    @field_validator("accessible_for_tow_truck", "located_at_donation_c_a", mode="before")
    @classmethod
    def _require_true_false(cls, v, info):
        """The model was trained on exactly the string categories
        'true'/'false' for these two flags. Coerce a genuine bool to the
        matching string (fixes a silent-miss case -- see docstring above);
        reject anything else that isn't (case-insensitively) "true" or
        "false" instead of letting an unrecognized string reach the model
        as an unseen category."""
        if v is None:
            return v
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
            return v.strip().lower()
        raise ValueError(
            f'{info.field_name} must be "true" or "false" (the model was trained '
            f"on exactly those two string categories), got {v!r}."
        )

    # Unrecognized fields are rejected (422) rather than silently ignored —
    # every field a real request can carry must be declared explicitly above.
    class Config:
        extra = "forbid"


class ShapFeatureRecord(BaseModel):
    """A user-facing SHAP contribution, collapsed from engineered features to raw input."""

    feature_raw_key: str  # raw feature key (e.g. 'make', '__collectible')
    feature_label: str  # user-facing label (e.g. 'Make', 'Collectible/cult status')
    value: Optional[str] = None  # raw value from request, or None
    dollar_impact: (
        float  # summed marginal $ impact across underlying engineered features
    )
    pct_of_prediction: float  # summed % of prediction
    # n_underlying: int  # how many engineered features collapsed into this
    # top_underlying: (
    #     str  # name of the highest-magnitude engineered contributor (for audit)
    # )


class ShapPayload(BaseModel):
    # quantile_explained: str
    # baseline_dollars: float
    # final_pred_dollars: float
    top_positive: List[ShapFeatureRecord]
    top_negative: List[ShapFeatureRecord]


class PredictResponse(BaseModel):
    # model_used: str
    # The same UUID as the X-Request-ID response header, and the same value
    # on every item of a batch response: it identifies the HTTP request, not
    # the row (pair it with stock_id to point at a single row). Echoed in
    # the body as well as the header because clients that keep only the
    # parsed JSON -- and anyone pasting a response into a bug report --
    # otherwise lose the join key needed to look the call up via /logs,
    # where it is now also accepted as a filter.
    #
    # Optional rather than required so a future code path that forgets to
    # set it degrades to a null field instead of turning a perfectly good
    # prediction into a response-validation 500. (Pydantic v2's
    # protected_namespaces warning only fires for "model_"-prefixed field
    # names, so "request_id" needs no model_config change.)
    request_id: Optional[str] = None
    stock_id: Optional[str] = None  # Add this line
    is_cult: Optional[bool] = None
    # route: Optional[str] = None
    predictions: Dict[str, float]
    feature_importances: Optional[ShapPayload] = None
    explanation: Optional[str] = None
    # elapsed_ms: Dict[str, float]


class LogsQueryRequest(BaseModel):
    """Request body for fetching logs.

    Only two time-window modes are accepted:
      - relative: days_ago and/or minutes_ago
      - absolute: start_time AND end_time (both required together)
    Mixing the two modes in one request is rejected.

    Note the field names are start_time/end_time, NOT start_date/end_date --
    the latter is IBM's own naming for the downstream metadata.start_date/
    end_date sent in ibm_logs_client.py, easy to confuse with this field.
    extra="forbid" below exists so that mix-up (or any other typo) gets a
    clear 422 instead of being silently dropped and quietly falling back
    to the default relative window.
    """

    class Config:
        extra = "forbid"

    # A single ID (str) or several (list) -- fetch_and_format_logs ORs
    # together one arrayContains(...) filter per ID in the latter case.
    stock_id: Optional[Union[str, List[str]]] = None
    # Optional endpoint filter: one path, or several. Left unset, the
    # downstream query keeps its previous behaviour of returning all three
    # endpoints, so existing callers see no change.
    endpoint: Optional[Union[str, List[str]]] = None
    # Optional request-id filter: one id, or several. Pairs with the
    # request_id now returned in the /predict response body and the
    # X-Request-ID header -- a caller quotes theirs, this pulls back every
    # log line for that exact call.
    request_id: Optional[Union[str, List[str]]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    days_ago: Optional[int] = None
    minutes_ago: Optional[int] = None
    limit: Optional[int] = Field(default=200, ge=1, le=200)

    @field_validator("stock_id")
    @classmethod
    def _reject_query_breaking_chars(cls, v):
        # stock_id is spliced verbatim into a single-quoted DataPrime string
        # literal in ibm_logs_client.py (`arrayContains('{sid}')`). A stray
        # `'` closes that literal early and turns the rest of the value into
        # live query syntax -- a DataPrime-injection analogue of SQL
        # injection. Rather than guess at DataPrime's escape convention
        # (unconfirmed, and untestable from here without live IBM creds),
        # reject the character outright: no legitimate stock ID needs it.
        if v is None:
            return v
        candidates = [v] if isinstance(v, str) else v
        for sid in candidates:
            if "'" in sid or "\\" in sid:
                raise ValueError(
                    f"stock_id {sid!r} contains a character (' or \\) that "
                    "cannot be safely used in the downstream log query."
                )
        return v

    @field_validator("endpoint")
    @classmethod
    def _restrict_to_known_endpoints(cls, v):
        # Like stock_id, this value is spliced into a single-quoted
        # DataPrime string literal ($d.endpoint == '{ep}'), but unlike
        # stock_id it is drawn from a closed set the service itself defines
        # -- so it gets an allowlist rather than stock_id's denylist of
        # query-breaking characters. That is strictly stronger: a denylist
        # only blocks the metacharacters we thought of (and we have no live
        # DataPrime to confirm the full set), whereas an allowlist means the
        # only strings that ever reach the query are constants from our own
        # source, so the escaping question never arises. It also turns a
        # typo like "/predicts" into an immediate 422 instead of a silently
        # empty result set that reads like "no traffic".
        if v is None:
            return v
        candidates = [v] if isinstance(v, str) else v
        for ep in candidates:
            if ep not in LOG_QUERY_ENDPOINTS:
                raise ValueError(
                    f"endpoint {ep!r} is not one of "
                    f"{', '.join(LOG_QUERY_ENDPOINTS)}."
                )
        return v

    @field_validator("request_id")
    @classmethod
    def _require_uuid_format(cls, v):
        # Same reasoning as the endpoint allowlist above, by shape rather
        # than by enumeration: every real request_id is a str(uuid.uuid4())
        # from main.py's middleware, and a UUID is hex digits and dashes
        # only -- no quote, no backslash. Validating the shape makes the
        # interpolation into '$d.request_id == '{rid}'' provably safe
        # rather than denylist-safe.
        #
        # This does reject the literal "unknown" that handlers fall back to
        # via getattr(request.state, "request_id", "unknown"). That fallback
        # should never fire (the middleware sets state before call_next),
        # and such an entry could not correlate to a real request anyway.
        if v is None:
            return v
        candidates = [v] if isinstance(v, str) else v
        for rid in candidates:
            if not _UUID_RE.match(rid):
                raise ValueError(
                    f"request_id {rid!r} is not a UUID. Use the value from "
                    "the X-Request-ID response header or the response body."
                )
        return v

    @field_validator("start_time", "end_time")
    @classmethod
    def _require_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError(
                "start_time/end_time must be UTC with an explicit offset, "
                "e.g. '2026-01-01T00:00:00Z'."
            )
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _validate_time_window(self):
        has_start = self.start_time is not None
        has_end = self.end_time is not None
        has_relative = self.days_ago is not None or self.minutes_ago is not None

        if has_start != has_end:
            raise ValueError(
                "start_time and end_time must both be provided together."
            )

        has_range = has_start and has_end

        if has_range and has_relative:
            raise ValueError(
                "Provide either days_ago/minutes_ago or start_time/end_time, not both."
            )

        if not has_range and not has_relative:
            self.days_ago = 7  # default relative window when nothing is specified

        return self


class LogsResponse(BaseModel):
    """Response structure for the logs endpoint."""

    # Mirrors PredictResponse.request_id -- see the comment there. Note the
    # dict returned by fetch_and_format_logs also carries "query_executed",
    # which this model deliberately omits and FastAPI therefore strips: the
    # generated DataPrime query is an internal detail, not part of the
    # response contract.
    request_id: Optional[str] = None
    time_window: Dict[str, str]
    log_count: int
    logs: List[Dict[str, Any]]
