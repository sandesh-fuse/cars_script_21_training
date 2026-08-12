"""
schemas.py
==========
Pydantic request/response models for the /predict endpoint.
"""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class PredictRequest(BaseModel):
    """Request body for /predict — all car attribute fields are optional.

    The preprocessor handles missing fields gracefully (treats as NaN).
    Field names must match the column names in your training CSV.
    """

    # Identifiers
    stock_id: Optional[str] = None
    vin_id: Optional[str] = None

    # Core vehicle attrs
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    trim: Optional[str] = None
    vehicle_type: Optional[str] = None
    body_type: Optional[str] = None
    body_subtype: Optional[str] = None
    drive_type: Optional[str] = None
    doors: Optional[int] = None
    oem_doors: Optional[int] = None
    oem_body_style: Optional[str] = None
    us_style_name: Optional[str] = None
    model_number: Optional[str] = None
    msrp: Optional[float] = None
    mileage: Optional[float] = None

    # Engine & Transmission attrs
    engine_name: Optional[str] = None
    engineconfiguration: Optional[str] = None
    enginecylinders: Optional[float] = None
    enginehp: Optional[float] = None
    displacementl: Optional[float] = None
    valvetraindesign: Optional[str] = None
    transmission_name: Optional[str] = None
    rear_axle: Optional[str] = None

    # Condition & Visual attrs
    nav_condition: Optional[str] = None
    nav_color: Optional[str] = None
    bodypaintcondition: Optional[str] = None
    enginecondition: Optional[str] = None
    transmissioncondition: Optional[str] = None
    tirecondition: Optional[str] = None
    interiorcondition: Optional[str] = None

    # Geo, Admin, and Flags
    # is_valid_vin: Optional[bool] = None
    vazipcode: Optional[str] = None
    vstate_name: Optional[str] = None
    state_province_of_title: Optional[str] = None
    accessiblefortwotruck: Optional[str] = None
    locatedatdonationca: Optional[str] = None

    # Date — defaults to today if not provided
    record_creation_date: Optional[str] = None

    # Any extra fields the client sends are silently ignored at preprocessing
    class Config:
        extra = "allow"


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
    """Request body for fetching logs."""

    stock_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    days_ago: Optional[int] = 7
    minutes_ago: Optional[int] = None
    limit: Optional[int] = None


class LogsResponse(BaseModel):
    """Response structure for the logs endpoint."""

    query_executed: str
    time_window: Dict[str, str]
    log_count: int
    logs: List[Dict[str, Any]]
