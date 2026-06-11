import os
import time
from fastapi import APIRouter, Depends
from aiosqlite import Connection
import numpy as np

from src.api.dependencies import get_db_dependency
from src.models.response import HealthResponse, LatencyResponse
from src.services.ml_service import get_ml_service

router = APIRouter()

start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Connection = Depends(get_db_dependency)):
    """
    Health check endpoint.
    Verifies DB connectivity and dummy ONNX inference (safe loading).
    """
    db_connected = False
    model_loaded = False

    # 1. Check DB
    try:
        await db.execute("SELECT 1")
        db_connected = True
    except Exception:
        pass

    # 2. Check ONNX Model (Lazy Import to prevent circular dependencies)
    optimal_threshold = 0.0
    global_mean = 0.0
    try:
        ml = get_ml_service()
        ml_health = ml.health_check()
        model_loaded = ml_health.get("model_loaded", True)
        optimal_threshold = ml_health.get("optimal_threshold", 0.0)
        global_mean = ml_health.get("global_mean", 0.0)
    except Exception:
        pass

    status = "ok" if (db_connected and model_loaded) else "degraded"

    return HealthResponse(
        status=status,
        model_version=os.getenv("MODEL_VERSION", "v1.0.0"),
        db_connected=db_connected,
        model_loaded=model_loaded,
        uptime_s=round(time.time() - start_time, 2),
        optimal_threshold=optimal_threshold,
        global_mean=global_mean,
    )


@router.get("/latency", response_model=LatencyResponse)
async def get_latency_stats(db: Connection = Depends(get_db_dependency)):
    """Computes p50, p95, p99 latency from recent request logs."""
    async with db.execute(
        "SELECT latency_ms FROM request_logs ORDER BY created_at DESC LIMIT 1000"
    ) as cursor:
        rows = await cursor.fetchall()

    if not rows:
        return LatencyResponse(p50_ms=0, p95_ms=0, p99_ms=0, last_1000_requests=0)

    latencies = [row["latency_ms"] for row in rows]

    return LatencyResponse(
        p50_ms=np.percentile(latencies, 50),
        p95_ms=np.percentile(latencies, 95),
        p99_ms=np.percentile(latencies, 99),
        last_1000_requests=len(latencies),
    )
