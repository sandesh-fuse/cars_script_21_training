"""
schemas.py
==========
Pydantic request/response models for the /predict endpoint.
"""

from typing import Optional, List, Any, Dict, Union
from pydantic import BaseModel, Field, model_validator, field_validator
from datetime import datetime, timezone


# Fields whose real contract is "a numeric picklist ID" -- the deployed
# script21 model has no learned encoding for genuine text on any of these
# columns (see each field's own docstring paragraph below for why). Typed
# Optional[Union[int, float]] below, NOT str, so the schema's own type
# signature says what it actually expects. _require_numeric_id() still
# accepts a numeric-looking STRING for caller convenience (e.g. "22968",
# a common JSON-serialization quirk) and converts it, but rejects genuine
# resolved text (e.g. "Runs & Drives", "Iowa") with a clear 422 up front --
# previously that text was silently accepted here and reached XGBoost
# 3 layers deep inside Script21Pipeline.predict(), which raised an opaque
# "DataFrame.dtypes ... must be int, float, bool or category" ValueError
# instead. See project memory script21-vehicle-type-color-picklist-bug for
# the production incident this was written in response to.
NUMERIC_PICKLIST_ID_FIELDS = (
    'vehicle_type', 'color',
    'vehicle_cond_picklist_id', 'engine_cond_picklist_id',
    'transmission_cond_picklist_id', 'body_paint_cond_picklist_id',
    'interior_cond_picklist_id', 'tire_cond_picklist_id',
    'state_picklist_id', 'state_title_picklist',
)


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
    Typed Optional[Union[str, int, float, List[Any]]] because that method
    accepts several shapes: a bare numeric picklist ID (the new-schema
    format — decoded via preprocessor.py's OTHER_DAMAGE_ID_TO_LABEL),
    plain text ('mold, other*'), a JSON-encoded string of {'id','name'}
    dicts (the training-data legacy format), or — per its own inline
    comment covering live API callers specifically — an actual JSON list of
    numeric picklist IDs, plain strings, or {'name': ...} dicts.

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

    zip (legacy name: vazipcode) accepts str-or-number too, but for a
    benign reason (unlike the three above): preprocessor.py's zip handling
    does `.astype(str).str.extract(r'(\\d{1,5})')` on it regardless of
    input type, so "52732.0"/52732.0/52732/"52732" all extract to the
    identical '52732' before being zero-padded and fanned out into
    zip_region/zip_first2/zip_first3/zip_lat/zip_lon/zip_full_freq
    (verified). The old Optional[str]-only typing didn't match anything
    wrong -- it just rejected a caller sending the ZIP as a JSON number
    with an avoidable 422 before ever reaching that already-robust code.

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
    doors: Optional[Union[str, int, float]] = None
    mileage: Optional[float] = None

    # Condition & Visual attrs
    vehicle_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: nav_condition
    color: Optional[Union[int, float]] = None  # legacy name: nav_color
    body_paint_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: bodypaintcondition
    engine_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: enginecondition
    transmission_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: transmissioncondition
    tire_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: tirecondition
    interior_cond_picklist_id: Optional[Union[int, float]] = None  # legacy name: interiorcondition
    other_damage_pklist_id: Optional[Union[str, int, float, List[Any]]] = None  # legacy name: other_damages
    comment: Optional[str] = None  # legacy name: all_clean_notes

    # Geo, Admin, and Flags
    zip: Optional[Union[str, int, float]] = None  # legacy name: vazipcode
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
        converting either to a float. Reject anything else (genuine
        resolved text, e.g. "Runs & Drives"/"Iowa") with a clear error
        instead of letting it through to crash inside XGBoost later."""
        if v is None:
            return v
        if isinstance(v, bool):
            raise ValueError(f"{info.field_name} must be a numeric picklist ID, not a boolean.")
        if isinstance(v, (int, float)):
            return v
        if isinstance(v, str):
            s = v.strip()
            try:
                return float(s)
            except ValueError:
                raise ValueError(
                    f"{info.field_name} must be a numeric picklist ID (e.g. 22968), "
                    f"not resolved text ({v!r}). The deployed model was trained "
                    "exclusively on numeric picklist IDs for this field and has "
                    "no encoding for text."
                )
        raise ValueError(f"{info.field_name} must be a numeric picklist ID, got {type(v).__name__}.")

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
    """

    stock_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    days_ago: Optional[int] = None
    minutes_ago: Optional[int] = None
    limit: Optional[int] = Field(default=200, ge=1, le=200)

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

    time_window: Dict[str, str]
    log_count: int
    logs: List[Dict[str, Any]]
