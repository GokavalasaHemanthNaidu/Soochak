from fastapi import APIRouter, Depends
from aiosqlite import Connection

from src.api.dependencies import get_db_dependency
from src.models.response import ModelMetricsResponse, DriftResponse

router = APIRouter(prefix="/model")

@router.get("/metrics", response_model=ModelMetricsResponse)
async def get_metrics(db: Connection = Depends(get_db_dependency)):
    """
    STUB: Phase 3 Mock Metrics.
    Will pull from DB and model_registry in Phase 8.
    """
    return ModelMetricsResponse(
        model_version="v1.0.0",
        f1_score=0.65,
        roc_auc=0.82,
        precision_fatal=0.68,
        recall_fatal=0.62,
        predictions_today=42,
        cache_hit_rate=0.15
    )

@router.get("/drift", response_model=DriftResponse)
async def check_drift(db: Connection = Depends(get_db_dependency)):
    """
    STUB: Phase 3 Mock Drift.
    Will use KS-test and lazy-loaded baselines in Phase 8.
    """
    from datetime import datetime
    return DriftResponse(
        overall_drift=False,
        features=[],
        checked_at=datetime.utcnow().isoformat() + "Z"
    )
