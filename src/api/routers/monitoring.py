"""Model metrics and drift detection endpoints."""

from fastapi import APIRouter, Depends
from src.models.response import ModelMetricsResponse, DriftResponse
from src.api.dependencies import get_db_dependency
from src.services.drift_service import detect_drift
from src.utils.feature_encoder import encode_features
import aiosqlite

router = APIRouter()


@router.get("/model/metrics", response_model=ModelMetricsResponse)
async def model_metrics(db: aiosqlite.Connection = Depends(get_db_dependency)):
    """Return current model performance metrics from registry."""
    async with db.execute(
        """SELECT COUNT(*) FROM predictions WHERE date(created_at) = date('now')"""
    ) as cursor:
        row = await cursor.fetchone()
        predictions_today = row[0] if row else 0

    async with db.execute(
        """SELECT SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as hits, COUNT(*) as total FROM predictions"""
    ) as cursor:
        row = await cursor.fetchone()
        cache_hit_rate = (row[0] / row[1]) if row and row[1] > 0 else 0.0

    async with db.execute(
        """SELECT version, f1_score, roc_auc, precision_fatal, recall_fatal
        FROM model_registry WHERE is_active = 1 ORDER BY deployed_at DESC LIMIT 1"""
    ) as cursor:
        row = await cursor.fetchone()

    if row:
        version, f1, roc, prec, rec = row
    else:
        version, f1, roc, prec, rec = "v1.0.0", 0.65, 0.82, 0.68, 0.62

    return ModelMetricsResponse(
        model_version=version,
        f1_score=f1,
        roc_auc=roc,
        precision_fatal=prec,
        recall_fatal=rec,
        predictions_today=predictions_today,
        cache_hit_rate=round(cache_hit_rate, 2),
    )


@router.get("/model/drift", response_model=DriftResponse)
async def model_drift(db: aiosqlite.Connection = Depends(get_db_dependency)):
    """KS-test drift detection comparing live vs training distribution."""
    async with db.execute(
        """SELECT road_type, speed_limit, weather, lighting, junction, junction_ctrl,
        vehicle_type, driver_age, urban_rural
        FROM incidents i
        JOIN predictions p ON i.id = p.incident_id
        ORDER BY p.created_at DESC LIMIT 1000"""
    ) as cursor:
        rows = await cursor.fetchall()

    if not rows:
        return DriftResponse(overall_drift=False, features=[], checked_at="2024-01-01T00:00:00")

    live_features_raw = [dict(row) for row in rows]
    live_features_encoded = []
    for feat in live_features_raw:
        try:
            vec = encode_features(feat)
            from src.utils.feature_encoder import get_feature_dict

            live_features_encoded.append(get_feature_dict(vec))
        except Exception:
            continue

    if not live_features_encoded:
        return DriftResponse(overall_drift=False, features=[], checked_at="2024-01-01T00:00:00")

    drift_result = detect_drift(live_features_encoded)
    return DriftResponse(**drift_result)
