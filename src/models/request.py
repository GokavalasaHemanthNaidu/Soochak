import html
from typing import Literal, Optional, List
from pydantic import BaseModel, Field, model_validator, field_validator

class PredictRequest(BaseModel):
    road_type: Literal["National Highway", "State Highway", "City Road", "Rural Road", "Expressway"]
    speed_limit: Optional[int] = Field(ge=10, le=120)
    weather: Literal["Clear", "Rainy", "Foggy", "Snow", "Dust Storm", "Cloudy"]
    lighting: Literal["Daylight", "Darkness - well lit", "Darkness - no lighting", "Dusk", "Dawn"]
    junction: Literal["None", "Crossroads", "T-Junction", "Y-Junction", "Roundabout"]
    junction_ctrl: Optional[Literal["None", "Traffic Light", "Stop Sign", "Yield Sign"]]
    vehicle_type: str = Field(max_length=50)
    driver_age: Literal["Under 18", "18-25", "26-40", "41-60", "Over 60"]
    urban_rural: Literal["Urban", "Rural"]
    state: str = Field(max_length=50)
    city: str = Field(max_length=50)

    @field_validator("speed_limit")
    @classmethod
    def validate_speed(cls, v):
        if v not in [10, 20, 30, 40, 50, 60, 70, 80, 100, 120]:
            raise ValueError("Speed limit must be a standard value (10, 20... 120)")
        return v

    @field_validator("vehicle_type", "state", "city")
    @classmethod
    def sanitize_text(cls, v):
        if v:
            clean = html.escape(v.strip())
            clean = clean.replace("script", "").replace("javascript:", "")
            return clean
        return v

class CounterfactualRequest(BaseModel):
    incident_id: int = Field(gt=0)
    feature_to_change: Literal["speed_limit", "road_type", "lighting", "weather", "junction_ctrl"]
    new_value: str = Field(max_length=50)

    @model_validator(mode='after')
    def validate_domain(self):
        feat = self.feature_to_change
        val = self.new_value
        
        if feat == "speed_limit":
            if val not in ["10", "20", "30", "40", "50", "60", "70", "80", "100", "120"]:
                raise ValueError("Invalid speed limit")
        elif feat == "road_type":
            valid = ["National Highway", "State Highway", "City Road", "Rural Road", "Expressway"]
            if val not in valid:
                raise ValueError(f"Invalid road_type. Must be one of {valid}")
        elif feat == "lighting":
            valid = ["Daylight", "Darkness - well lit", "Darkness - no lighting", "Dusk", "Dawn"]
            if val not in valid:
                raise ValueError(f"Invalid lighting. Must be one of {valid}")
        elif feat == "weather":
            valid = ["Clear", "Rainy", "Foggy", "Snow", "Dust Storm", "Cloudy"]
            if val not in valid:
                raise ValueError(f"Invalid weather. Must be one of {valid}")
        elif feat == "junction_ctrl":
            valid = ["None", "Traffic Light", "Stop Sign", "Yield Sign"]
            if val not in valid:
                raise ValueError(f"Invalid junction_ctrl. Must be one of {valid}")
        
        return self

class ExplainRequest(BaseModel):
    prediction_id: int = Field(gt=0)

class BatchPredictRequest(BaseModel):
    incidents: List[PredictRequest] = Field(max_length=1000)

class FeedbackRequest(BaseModel):
    prediction_id: int = Field(gt=0)
    was_correct: Optional[int] = Field(ge=0, le=1)
    human_label: Optional[int] = Field(ge=0, le=1)

class IncidentQueryParams(BaseModel):
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=50, ge=1, le=200)
    severity: Optional[Literal["Fatal", "Non-Fatal"]] = None
    state: Optional[str] = None
    city: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
