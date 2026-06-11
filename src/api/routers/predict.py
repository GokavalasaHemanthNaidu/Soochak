"""Prediction router — Phase 5: Real ML inference.

Replaces the Phase 3 mock implementation with production-grade ONNX inference,
isotonic calibration, and SHAP explainability.
"""

import time
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
import aiosqlite

from src.models.request import PredictRequest, CounterfactualRequest
from src.models.response import PredictResponse, CounterfactualResponse
from src.services.ml_service import MLService, get_ml_service
from src.api.dependencies import get_db_dependency

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "/",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict accident severity",
    description="""
    Production inference endpoint using ONNX runtime with isotonic calibration
    and SHAP TreeExplainer explainability.

    The model accepts any string for categorical fields via OrdinalEncoder
    fallback (unknown categories map to -1). Target-encoded features
    (State_Risk_Score, Weather_Risk) use global mean fallback for unseen
    categories.
    """
)
async def predict(
    request: PredictRequest,
    ml_service: MLService = Depends(get_ml_service),
    db: aiosqlite.Connection = Depends(get_db_dependency),
) -> PredictResponse:
    """Run prediction and persist to database."""
    t_start = time.time()

    # Convert Pydantic model to dict for MLService
    data = request.model_dump()

    try:
        result = ml_service.predict(data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature engineering error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

    # Persist to SQLite using async connection
    cursor = await db.execute("""
        INSERT INTO incidents (
            datetime, road_type, speed_limit, weather, lighting,
            junction, junction_ctrl, vehicle_type, driver_age,
            urban_rural, state, city
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        request.Road_Type,
        request.Speed_Limit,
        request.Weather,
        request.Lighting,
        request.Junction,
        request.Junction_Control,
        request.Vehicle_Type,
        request.Driver_Age,
        request.Urban_Rural,
        request.State,
        request.City,
    ))
    await db.commit()
    incident_id = cursor.lastrowid
    
    # 2. Insert prediction outputs referencing incident_id
    model_version = "soochak_v1"
    cursor = await db.execute("""
        INSERT INTO predictions (
            incident_id, model_version, predicted_class, probability,
            shap_json, shap_base_value, inference_ms, cache_hit
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        incident_id,
        model_version,
        result["predicted_class"],
        result["probability"],
        json.dumps(result["top_features"]),
        result["shap_base_value"],
        result["inference_ms"],
        0, # Not cached in this basic predict call
    ))
    await db.commit()
    prediction_id = cursor.lastrowid

    # Build response
    return PredictResponse(
        prediction_id=prediction_id,
        incident_id=incident_id,
        predicted_class=result["predicted_class"],
        probability=result["probability"],
        raw_probability=result["raw_probability"],
        confidence=result["confidence"],
        threshold_used=result["threshold_used"],
        inference_ms=result["inference_ms"],
        shap_base_value=result["shap_base_value"],
        top_features=result["top_features"],
        feature_vector=result["feature_vector"],
        timestamp=datetime.utcnow(),
    )


@router.get("/health", summary="ML service health check")
async def health(ml_service: MLService = Depends(get_ml_service)) -> dict:
    """Return ML service health status."""
    return ml_service.health_check()


# ── Counterfactual Simulator Router & Endpoint ─────────────────────────────────
cf_router = APIRouter(tags=["Counterfactual"])

# Map request feature names (often lowercase) to model feature names (PascalCase)
FEATURE_MAPPING = {
    "road_type": "Road_Type",
    "speed_limit": "Speed_Limit",
    "weather": "Weather",
    "lighting": "Lighting",
    "junction": "Junction",
    "junction_control": "Junction_Control",
    "junction_ctrl": "Junction_Control",
    "vehicle_type": "Vehicle_Type",
    "driver_age": "Driver_Age",
    "urban_rural": "Urban_Rural",
    "state": "State",
    "city": "City"
}

# Map DB columns (lowercase) to model feature names (PascalCase)
DB_TO_MODEL_MAPPING = {
    "road_type": "Road_Type",
    "speed_limit": "Speed_Limit",
    "weather": "Weather",
    "lighting": "Lighting",
    "junction": "Junction",
    "junction_ctrl": "Junction_Control",
    "vehicle_type": "Vehicle_Type",
    "driver_age": "Driver_Age",
    "urban_rural": "Urban_Rural",
    "state": "State",
    "city": "City"
}

@cf_router.post(
    "/counterfactual",
    response_model=CounterfactualResponse,
    status_code=status.HTTP_200_OK,
    summary="Run what-if counterfactual simulation",
    description="Modify a single feature of an existing incident and predict the change in severity."
)
async def counterfactual(
    request: CounterfactualRequest,
    db: aiosqlite.Connection = Depends(get_db_dependency),
    ml_service: MLService = Depends(get_ml_service)
) -> CounterfactualResponse:
    """Modify a single feature of an existing incident and predict the change in severity."""
    # 1. Fetch original incident
    async with db.execute("SELECT * FROM incidents WHERE id = ?", (request.incident_id,)) as cursor:
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident with ID {request.incident_id} not found."
        )
    
    incident_dict = dict(row)
    
    # 2. Get or calculate original probability
    async with db.execute(
        "SELECT probability FROM predictions WHERE incident_id = ? ORDER BY id DESC LIMIT 1",
        (request.incident_id,)
    ) as cursor:
        pred_row = await cursor.fetchone()
        
    if pred_row:
        original_prob = float(pred_row["probability"])
    else:
        # Re-predict using stored features
        orig_input = {}
        for db_col, model_col in DB_TO_MODEL_MAPPING.items():
            val = incident_dict.get(db_col)
            if model_col == "Speed_Limit" and val is not None:
                val = int(val)
            orig_input[model_col] = val
        orig_res = ml_service.predict(orig_input)
        original_prob = orig_res["probability"]
    
    # 3. Resolve feature to change
    feat_lower = request.feature_to_change.lower()
    mapped_feat = FEATURE_MAPPING.get(feat_lower)
    if not mapped_feat:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid feature to change: '{request.feature_to_change}'."
        )
    
    # 4. Construct modified input
    modified_input = {}
    for db_col, model_col in DB_TO_MODEL_MAPPING.items():
        if model_col == mapped_feat:
            val = request.new_value
            if model_col == "Speed_Limit":
                try:
                    val = int(val)
                except ValueError:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="Speed_Limit must be an integer."
                    )
            modified_input[model_col] = val
        else:
            val = incident_dict.get(db_col)
            if model_col == "Speed_Limit" and val is not None:
                val = int(val)
            modified_input[model_col] = val
            
    # Domain validation for Speed_Limit
    if "Speed_Limit" in modified_input and (modified_input["Speed_Limit"] < 0 or modified_input["Speed_Limit"] > 200):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Speed_Limit must be between 0 and 200."
        )
        
    # 5. Predict on modified input
    try:
        new_res = ml_service.predict(modified_input)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during counterfactual simulation: {str(e)}"
        )
        
    new_prob = new_res["probability"]
    delta_prob = new_prob - original_prob
    delta_pct = (delta_prob / original_prob * 100) if original_prob > 0 else 0.0
    
    direction = "increases" if delta_prob > 0 else "decreases"
    abs_pct = abs(delta_pct)
    interpretation = f"Changing {request.feature_to_change} to {request.new_value} {direction} fatal probability by {abs_pct:.1f}% (from {original_prob:.1%} to {new_prob:.1%})"
    
    # 6. Save counterfactual to database
    await db.execute("""
        INSERT INTO counterfactuals (
            incident_id, changed_feature, original_value, new_value,
            original_prob, new_prob, delta_prob
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        request.incident_id,
        request.feature_to_change,
        str(incident_dict.get(feat_lower if feat_lower != 'junction_control' else 'junction_ctrl', "")),
        request.new_value,
        original_prob,
        new_prob,
        delta_prob
    ))
    await db.commit()
    
    return CounterfactualResponse(
        original_prob=original_prob,
        new_prob=new_prob,
        delta_prob=delta_prob,
        delta_pct=delta_pct,
        interpretation=interpretation
    )
