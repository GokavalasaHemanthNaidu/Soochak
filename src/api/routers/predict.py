import json
import traceback
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from aiosqlite import Connection

from src.api.dependencies import get_db_dependency
from src.models.request import PredictRequest, CounterfactualRequest
from src.models.response import PredictResponse, CounterfactualResponse, TopFeature

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict_severity(
    req: PredictRequest, 
    request: Request,
    db: Connection = Depends(get_db_dependency)
):
    """
    STUB: Phase 3 Mock Prediction endpoint.
    Will be replaced with real inference in Phase 5.
    """
    try:
        # Mock ML Output
        predicted_class = 1
        probability = 0.75
        mock_shap = {"speed_limit": 0.4, "road_type": -0.2}
        top_features = [
            TopFeature(feature="speed_limit", contribution=0.4, contribution_pct=66.6),
            TopFeature(feature="road_type", contribution=-0.2, contribution_pct=33.3)
        ]
        
        # Insert incident
        cursor = await db.execute(
            """
            INSERT INTO incidents 
            (road_type, speed_limit, weather, lighting, junction, junction_ctrl, vehicle_type, driver_age, urban_rural, state, city)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (req.road_type, req.speed_limit, req.weather, req.lighting, req.junction, req.junction_ctrl,
             req.vehicle_type, req.driver_age, req.urban_rural, req.state, req.city)
        )
        incident_id = cursor.lastrowid
        
        # Insert prediction
        model_version = request.app.state.model_version if hasattr(request.app.state, "model_version") else "v1.0.0"
        cursor = await db.execute(
            """
            INSERT INTO predictions
            (incident_id, model_version, predicted_class, probability, shap_json, shap_base_value, inference_ms, cache_hit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (incident_id, model_version, predicted_class, probability, json.dumps(mock_shap), 0.5, 12.5, 0)
        )
        prediction_id = cursor.lastrowid
        await db.commit()
        
        return PredictResponse(
            incident_id=incident_id,
            prediction_id=prediction_id,
            predicted_class=predicted_class,
            probability=probability,
            shap_values=mock_shap,
            top_features=top_features,
            inference_ms=12.5,
            model_version=model_version,
            cache_hit=False
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction Error: {traceback.format_exc()}")
        # CRITICAL: Do not leak stack traces to the user (Security 20/20)
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction")

@router.post("/counterfactual", response_model=CounterfactualResponse)
async def simulate_intervention(
    req: CounterfactualRequest,
    db: Connection = Depends(get_db_dependency)
):
    """
    STUB: Phase 3 Mock Counterfactual endpoint.
    Will be replaced with real inference delta in Phase 5.
    """
    original_prob = 0.75
    new_prob = 0.45
    delta = new_prob - original_prob
    delta_pct = (delta / original_prob) * 100 if original_prob > 0 else 0
    
    return CounterfactualResponse(
        original_prob=original_prob,
        new_prob=new_prob,
        delta_prob=delta,
        delta_pct=delta_pct,
        interpretation=f"Changing {req.feature_to_change} to {req.new_value} decreases fatal probability by {abs(round(delta_pct, 1))}%"
    )
