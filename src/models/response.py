"""Pydantic response models for the SOOCHAK prediction API."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SHAPContribution(BaseModel):
    """Single feature contribution from SHAP explainability."""

    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Encoded feature value")
    contribution: float = Field(..., description="SHAP value (log-odds impact)")
    contribution_prob: float = Field(..., description="Approximate probability-space contribution")


class PredictResponse(BaseModel):
    """Response model for accident severity prediction.

    Returns calibrated probability, SHAP explanations, and top contributing
    risk factors. The confidence field indicates prediction reliability.
    """

    prediction_id: int = Field(..., description="SQLite row ID")
    incident_id: int = Field(..., description="SQLite row ID from incidents table")
    predicted_class: int = Field(
        ..., ge=0, le=1, description="0 = Slight Injury, 1 = Serious/Fatal Injury"
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated probability of Serious/Fatal injury"
    )
    raw_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Raw ONNX model probability (before calibration)"
    )
    confidence: str = Field(..., description="Prediction confidence: low | medium | high")
    inference_ms: float = Field(..., description="End-to-end inference latency (ms)")
    threshold_used: float = Field(..., description="Threshold applied for class assignment")

    # Explainability
    shap_base_value: float = Field(..., description="SHAP expected value (probability space)")
    top_features: List[SHAPContribution] = Field(..., description="Top 3 contributing features")

    # Metadata
    model_version: str = Field(default="soochak_v1", description="ONNX model version")
    calibrated: bool = Field(default=True, description="Whether isotonic calibration was applied")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Feature vector (for debugging / drift detection)
    feature_vector: Optional[List[float]] = Field(
        default=None, description="14-element feature vector sent to model"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": 42,
                "incident_id": 42,
                "predicted_class": 1,
                "probability": 0.72,
                "raw_probability": 0.68,
                "confidence": "medium",
                "threshold_used": 0.18,
                "inference_ms": 12.5,
                "shap_base_value": 0.5308,
                "top_features": [
                    {
                        "feature": "Speed_Limit",
                        "value": 80.0,
                        "contribution": 0.45,
                        "contribution_prob": 0.11,
                    },
                    {
                        "feature": "State_Risk_Score",
                        "value": 0.18,
                        "contribution": 0.32,
                        "contribution_prob": 0.08,
                    },
                    {
                        "feature": "Weather_Risk",
                        "value": 0.15,
                        "contribution": 0.18,
                        "contribution_prob": 0.04,
                    },
                ],
                "model_version": "soochak_v1",
                "calibrated": True,
                "timestamp": "2026-06-11T13:55:00Z",
                "feature_vector": None,
            }
        }


class CounterfactualResponse(BaseModel):
    original_prob: float
    new_prob: float
    delta_prob: float
    delta_pct: float
    interpretation: str


class ExplainResponse(BaseModel):
    groq_explanation: str
    top_features: List[SHAPContribution]
    recommendation: Optional[str] = None


class BatchStatusResponse(BaseModel):
    task_id: str
    status: str
    progress_pct: float
    results_url: Optional[str] = None
    estimated_seconds: Optional[int] = None
    warning: str = "Jobs are stored in-memory and lost on container restart."


class IncidentResponse(BaseModel):
    id: int
    datetime: Optional[str] = None
    road_type: Optional[str] = None
    speed_limit: Optional[int] = None
    weather: Optional[str] = None
    lighting: Optional[str] = None
    junction: Optional[str] = None
    junction_ctrl: Optional[str] = None
    vehicle_type: Optional[str] = None
    driver_age: Optional[str] = None
    urban_rural: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    severity_true: Optional[int] = None
    source: str = "api"
    created_at: Optional[str] = None


class IncidentListResponse(BaseModel):
    incidents: List[IncidentResponse]
    total: int
    page: int
    pages: int


class ModelMetricsResponse(BaseModel):
    model_version: str
    f1_score: float
    roc_auc: float
    precision_fatal: float
    recall_fatal: float
    predictions_today: int
    cache_hit_rate: float


class DriftFeature(BaseModel):
    feature: str
    p_value: float
    drifted: bool


class DriftResponse(BaseModel):
    overall_drift: bool
    features: List[DriftFeature]
    checked_at: str


class FeedbackResponse(BaseModel):
    feedback_id: int
    correction_rate_7d: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    db_connected: bool
    model_loaded: bool
    uptime_s: float
    optimal_threshold: float = 0.0
    global_mean: float = 0.0


class LatencyResponse(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    last_1000_requests: int
