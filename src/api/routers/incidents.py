"""Incident registry and prediction detail endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from src.models.response import IncidentListResponse, IncidentResponse
from src.api.dependencies import get_db_dependency
import aiosqlite

router = APIRouter()


@router.get("/incidents", response_model=IncidentListResponse)
async def list_incidents(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    severity: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    db: aiosqlite.Connection = Depends(get_db_dependency)
):
    """Paginated incident list with filters."""
    conditions = []
    params = []

    if severity:
        conditions.append("severity_true = ?")
        params.append(1 if severity == "fatal" else 0)
    if state:
        conditions.append("state = ?")
        params.append(state)
    if city:
        conditions.append("city = ?")
        params.append(city)
    if date_from:
        conditions.append("datetime >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("datetime <= ?")
        params.append(date_to)

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    count_sql = f"SELECT COUNT(*) FROM incidents {where_clause}"
    async with db.execute(count_sql, params) as cursor:
        total = (await cursor.fetchone())[0]

    offset = (page - 1) * limit
    sql = f"""SELECT * FROM incidents {where_clause}
              ORDER BY created_at DESC LIMIT ? OFFSET ?"""
    async with db.execute(sql, params + [limit, offset]) as cursor:
        rows = await cursor.fetchall()

    incidents = [dict(row) for row in rows]
    pages = (total + limit - 1) // limit

    return IncidentListResponse(
        incidents=[IncidentResponse(**inc) for inc in incidents],
        total=total,
        page=page,
        pages=pages
    )


@router.get("/incidents/{incident_id}")
async def get_incident(incident_id: int, db: aiosqlite.Connection = Depends(get_db_dependency)):
    """Get full incident with prediction and counterfactuals."""
    async with db.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)) as cursor:
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Incident not found")

    incident = dict(row)

    async with db.execute(
        "SELECT * FROM predictions WHERE incident_id = ?", (incident_id,)
    ) as cursor:
        predictions = [dict(r) for r in await cursor.fetchall()]

    async with db.execute(
        "SELECT * FROM counterfactuals WHERE incident_id = ?", (incident_id,)
    ) as cursor:
        counterfactuals = [dict(r) for r in await cursor.fetchall()]

    return {"incident": incident, "predictions": predictions, "counterfactuals": counterfactuals}


@router.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: int, db: aiosqlite.Connection = Depends(get_db_dependency)):
    """Get prediction with SHAP and explanation."""
    async with db.execute(
        "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
    ) as cursor:
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")

    return dict(row)
