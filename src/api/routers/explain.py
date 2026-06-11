"""LLM explanation endpoint."""

import json
from fastapi import APIRouter, HTTPException, Depends
from src.models.request import ExplainRequest
from src.models.response import ExplainResponse
from src.api.dependencies import get_db_dependency
from src.services.groq_service import explain_prediction
import aiosqlite

router = APIRouter(tags=["Explanation"])


@router.post("/explain", response_model=ExplainResponse)
async def explain(
    request: ExplainRequest,
    db: aiosqlite.Connection = Depends(get_db_dependency)
):
    """Generate LLM explanation for a prediction."""
    async with db.execute(
        "SELECT * FROM predictions WHERE id = ?", (request.prediction_id,)
    ) as cursor:
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")

    pred = dict(row)
    top_features = json.loads(pred["shap_json"])

    if pred.get("groq_explanation"):
        return ExplainResponse(
            groq_explanation=pred["groq_explanation"],
            top_features=top_features,
            recommendation="See explanation above."
        )

    explanation = await explain_prediction(
        top_features=top_features,
        probability=pred["probability"],
        predicted_class=pred["predicted_class"]
    )

    await db.execute(
        "UPDATE predictions SET groq_explanation = ? WHERE id = ?",
        (explanation, request.prediction_id)
    )
    await db.commit()

    return ExplainResponse(
        groq_explanation=explanation,
        top_features=top_features,
        recommendation="See explanation above."
    )
