"""Batch prediction and task status endpoints."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from src.models.request import BatchPredictRequest
from src.models.response import BatchStatusResponse
from src.services.batch_service import create_task, get_task_status

router = APIRouter(prefix="/batch", tags=["Batch Processing"])


@router.post("/predict", response_model=BatchStatusResponse)
async def batch_predict(
    request: BatchPredictRequest,
):
    """Queue a batch prediction job."""
    if len(request.incidents) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 incidents per batch")

    incidents = [inc.model_dump() for inc in request.incidents]
    task_id = create_task(incidents)

    estimated_seconds = max(1, len(incidents) * 0.1)

    return BatchStatusResponse(
        task_id=task_id,
        status="queued",
        progress_pct=0.0,
        estimated_seconds=int(estimated_seconds)
    )


@router.get("/tasks/{task_id}", response_model=BatchStatusResponse)
async def get_task(task_id: str):
    """Poll batch task status."""
    task = get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return BatchStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        progress_pct=task["progress_pct"],
        results_url=f"/v1/batch/tasks/{task_id}/results" if task["status"] == "complete" else None,
        estimated_seconds=None
    )


@router.get("/tasks/{task_id}/results")
async def get_task_results(task_id: str):
    """Get results of completed batch task."""
    task = get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "complete":
        raise HTTPException(status_code=400, detail="Task results are not ready yet")
    return task["results"]
