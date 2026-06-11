from typing import Optional
from pydantic import BaseModel


class Incident(BaseModel):
    id: Optional[int] = None
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
    source: Optional[str] = "api"
    created_at: Optional[str] = None


class Prediction(BaseModel):
    id: Optional[int] = None
    incident_id: int
    model_version: Optional[str] = None
    predicted_class: Optional[int] = None
    probability: Optional[float] = None
    shap_json: Optional[str] = None
    shap_base_value: Optional[float] = None
    groq_explanation: Optional[str] = None
    inference_ms: Optional[float] = None
    cache_hit: Optional[int] = 0
    created_at: Optional[str] = None


class Counterfactual(BaseModel):
    id: Optional[int] = None
    incident_id: int
    changed_feature: Optional[str] = None
    original_value: Optional[str] = None
    new_value: Optional[str] = None
    original_prob: Optional[float] = None
    new_prob: Optional[float] = None
    delta_prob: Optional[float] = None
    created_at: Optional[str] = None


class Feedback(BaseModel):
    id: Optional[int] = None
    prediction_id: int
    was_correct: Optional[int] = None
    human_label: Optional[int] = None
    feedback_source: Optional[str] = "ui"
    created_at: Optional[str] = None
