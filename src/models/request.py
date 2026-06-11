"""Pydantic request models for the SOOCHAK prediction API.

Relaxed from strict Literal constraints to generic str to support
any geographic dataset (Ethiopian, Indian, etc.) via OrdinalEncoder fallback.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List



class PredictRequest(BaseModel):
    """Request model for accident severity prediction.

    All categorical fields accept any string. The production pipeline uses
    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    to safely map unseen categories to -1 without crashing.

    Target-encoded features (State_Risk_Score, Weather_Risk) are computed
    internally from the raw categorical inputs.
    """

    # ── Categorical Features (10) ────────────────────────────────────────────
    Road_Type: str = Field(
        ..., max_length=100,
        description="Road geometry type, e.g. 'Undivided Two way', 'National Highway'",
        examples=["Undivided Two way", "Double carriageway (median)"]
    )
    Weather: str = Field(
        ..., max_length=50,
        description="Weather conditions at time of accident",
        examples=["Normal", "Rain", "Fog or mist"]
    )
    Lighting: str = Field(
        ..., max_length=50,
        description="Lighting conditions",
        examples=["Daylight", "Darkness - lights lit", "Dawn"]
    )
    Junction: str = Field(
        ..., max_length=100,
        description="Junction type where accident occurred",
        examples=["No junction", "T Junction", "Crossing"]
    )
    Junction_Control: str = Field(
        ..., max_length=100,
        description="Behavioral proxy from cause of accident (dataset limitation)",
        examples=["No distancing", "Changing lane to the right", "Other"]
    )
    Vehicle_Type: str = Field(
        ..., max_length=50,
        description="Type of vehicle involved",
        examples=["Automobile", "Lorry (41?100Q)", "Public (12 seats)"]
    )
    Driver_Age: str = Field(
        ..., max_length=30,
        description="Age band of driver",
        examples=["18-30", "31-50", "Over 51", "Under 18"]
    )
    Urban_Rural: str = Field(
        ..., max_length=100,
        description="Area classification",
        examples=["Residential areas", "Office areas", "Rural village areas"]
    )
    State: str = Field(
        ..., max_length=100,
        description="Road alignment / state proxy (dataset uses road alignment)",
        examples=["Tangent road with flat terrain", "Steep grade"]
    )
    City: str = Field(
        ..., max_length=50,
        description="Temporal proxy from day of week (dataset limitation)",
        examples=["Monday", "Friday", "Sunday"]
    )

    # ── Numerical Feature (1) ────────────────────────────────────────────────
    Speed_Limit: int = Field(
        ..., ge=0, le=200,
        description="Speed limit of the road (km/h). Proxy derived from road geometry.",
        examples=[60, 80, 40]
    )

    # ── Optional: Raw target-encoding inputs ─────────────────────────────────
    # These are used internally; clients do not send them directly.
    # State_Risk_Score and Weather_Risk are computed from State and Weather.

    @field_validator('Road_Type', 'Weather', 'Lighting', 'Junction', 
                     'Junction_Control', 'Vehicle_Type', 'Driver_Age',
                     'Urban_Rural', 'State', 'City', mode='before')
    @classmethod
    def strip_strings(cls, v):
        """Strip whitespace from all string inputs to prevent encoding mismatches."""
        if isinstance(v, str):
            return v.strip()
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "Road_Type": "Undivided Two way",
                "Speed_Limit": 60,
                "Weather": "Normal",
                "Lighting": "Daylight",
                "Junction": "No junction",
                "Junction_Control": "No distancing",
                "Vehicle_Type": "Automobile",
                "Driver_Age": "18-30",
                "Urban_Rural": "Residential areas",
                "State": "Tangent road with flat terrain",
                "City": "Monday"
            }
        }


class CounterfactualRequest(BaseModel):
    """Request model for what-if counterfactual simulation."""
    incident_id: int = Field(..., description="Original incident ID in database")
    feature_to_change: str = Field(..., description="Feature name to change (e.g. 'Speed_Limit', 'Weather')")
    new_value: str = Field(..., description="New value for the feature")

    class Config:
        json_schema_extra = {
            "example": {
                "incident_id": 1,
                "feature_to_change": "Speed_Limit",
                "new_value": "40"
            }
        }


class BatchPredictRequest(BaseModel):
    """Request model for batch incident predictions."""
    incidents: List[PredictRequest] = Field(..., description="List of incidents to predict")

    class Config:
        json_schema_extra = {
            "example": {
                "incidents": [
                    {
                        "Road_Type": "Undivided Two way",
                        "Speed_Limit": 60,
                        "Weather": "Normal",
                        "Lighting": "Daylight",
                        "Junction": "No junction",
                        "Junction_Control": "No distancing",
                        "Vehicle_Type": "Automobile",
                        "Driver_Age": "18-30",
                        "Urban_Rural": "Residential areas",
                        "State": "Tangent road with flat terrain",
                        "City": "Monday"
                    }
                ]
            }
        }


class ExplainRequest(BaseModel):
    """Request model for prediction explanations."""
    prediction_id: int = Field(..., description="SQLite row ID from predictions table")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": 1
            }
        }


class FeedbackRequest(BaseModel):
    prediction_id: int = Field(..., description="SQLite row ID from predictions table")
    was_correct: Optional[int] = Field(None, ge=0, le=1, description="1 if prediction was correct, 0 if wrong")
    human_label: Optional[int] = Field(None, ge=0, le=1, description="Human-assigned correct label")



