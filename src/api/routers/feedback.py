"""Human feedback endpoint for drift monitoring."""

from fastapi import APIRouter, HTTPException, Depends
from src.models.request import FeedbackRequest
from src.models.response import FeedbackResponse
from src.api.dependencies import get_db_dependency
import aiosqlite

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest, db: aiosqlite.Connection = Depends(get_db_dependency)
):
    """Log human feedback on a prediction."""
    async with db.execute(
        "SELECT id FROM predictions WHERE id = ?", (request.prediction_id,)
    ) as cursor:
        if not await cursor.fetchone():
            raise HTTPException(status_code=404, detail="Prediction not found")

    cursor = await db.execute(
        """INSERT INTO feedback (prediction_id, was_correct, human_label, feedback_source)
           VALUES (?, ?, ?, ?)""",
        (request.prediction_id, request.was_correct, request.human_label, "ui"),
    )
    feedback_id = cursor.lastrowid
    await db.commit()

    async with db.execute(
        """SELECT COUNT(*) as total, SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as corrections
        FROM feedback WHERE created_at >= datetime('now', '-7 days')"""
    ) as cursor:
        row = await cursor.fetchone()
        total, corrections = row[0], row[1]
        correction_rate = (corrections / total * 100) if total > 0 else None

    return FeedbackResponse(
        feedback_id=feedback_id,
        correction_rate_7d=round(correction_rate, 2) if correction_rate is not None else None,
    )
