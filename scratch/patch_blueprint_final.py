import re
from pathlib import Path

blueprint_path = Path(r"C:\Users\Hemanth\Downloads\ROADRISK_ZERO_FLAW_BLUEPRINT.md")
workspace_dir = Path(r"c:\Users\Hemanth\SOOCHAK")

# Read current content of blueprint
with open(blueprint_path, "r", encoding="utf-8") as f:
    bp_content = f.read()

def read_workspace_file(rel_path: str) -> str:
    path = workspace_dir / rel_path
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# Read actual files from workspace
sql_initial = read_workspace_file("src/db/migrations/001_initial.sql")
req_models = read_workspace_file("src/models/request.py")
resp_models = read_workspace_file("src/models/response.py")
health_router = read_workspace_file("src/api/routers/health.py")
main_api = read_workspace_file("src/api/main.py")
ml_service = read_workspace_file("src/services/ml_service.py")
drift_logger = read_workspace_file("src/services/drift_logger.py")
predict_router = read_workspace_file("src/api/routers/predict.py")
api_key_gate = read_workspace_file("src/middleware/api_key_gate.py")
test_api = read_workspace_file("tests/test_api.py")
test_security = read_workspace_file("tests/test_security.py")

# Helper to replace block in markdown
def replace_file_block(content: str, heading_prefix: str, filepath: str, new_code: str) -> str:
    escaped_fp = re.escape(filepath)
    # We match: ## Step X.Y: ... \n\n```bash\ncat > filepath << 'EOF'\n(any contents)\nEOF\n```
    pattern = r"(##\s*" + re.escape(heading_prefix) + r"[^\n]*\n+```bash\ncat\s*>\s*" + escaped_fp + r"\s*<<\s*'EOF'\n).*?(\nEOF\n```)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"WARNING: Could not find block for heading prefix '{heading_prefix}' and file '{filepath}'")
        return content
    replacement = r"\g<1>" + new_code + r"\g<2>"
    return re.sub(pattern, replacement, content, flags=re.DOTALL)

# 1. Replace all Phase 1-5 code blocks with active workspace files
bp_content = replace_file_block(bp_content, "Step 2.1:", "src/db/migrations/001_initial.sql", sql_initial)
bp_content = replace_file_block(bp_content, "Step 3.2:", "src/middleware/api_key_gate.py", api_key_gate)
bp_content = replace_file_block(bp_content, "Step 3.3:", "src/models/request.py", req_models)
bp_content = replace_file_block(bp_content, "Step 3.4:", "src/models/response.py", resp_models)
bp_content = replace_file_block(bp_content, "Step 3.6:", "src/api/routers/health.py", health_router)
bp_content = replace_file_block(bp_content, "Step 3.9:", "src/api/main.py", main_api)
bp_content = replace_file_block(bp_content, "Step 3.10:", "tests/test_api.py", test_api)
bp_content = replace_file_block(bp_content, "Step 3.11:", "tests/test_security.py", test_security)

# Step 5.1 & 5.2 consolidation
bp_content = bp_content.replace(
    "## Step 5.1: Create Feature Encoder Utility (With Shape Validation)\n\n```bash\ncat > src/utils/feature_encoder.py",
    "## Step 5.1: Create Eager Singleton ML Service (ONNX + Calibrator + SHAP)\n\n```bash\ncat > src/services/ml_service.py"
)
bp_content = replace_file_block(bp_content, "Step 5.1:", "src/services/ml_service.py", ml_service)

bp_content = bp_content.replace(
    "## Step 5.2: Create Inference Service (Lazy Load + Shape Check)\n\n```bash\ncat > src/services/inference.py",
    "## Step 5.2: Create Drift Logger Service (Drift Event Tracking)\n\n```bash\ncat > src/services/drift_logger.py"
)
bp_content = replace_file_block(bp_content, "Step 5.2:", "src/services/drift_logger.py", drift_logger)

# Step 5.6 predict.py replacement (use workspace version)
bp_content = bp_content.replace(
    "## Step 5.6: Update Predict Router with Real Inference + prediction_id\n\n```bash\ncat > src/api/routers/predict.py",
    "## Step 5.6: Update Predict Router with Eager ML Service & Two-Table Normalized Insert\n\n```bash\ncat > src/api/routers/predict.py"
)
bp_content = replace_file_block(bp_content, "Step 5.6:", "src/api/routers/predict.py", predict_router)

# 2. Fix Step 3.7 Prediction Router (Stub) to use PascalCase payload fields and trailing slash
predict_stub_new = """\"\"\"Prediction and counterfactual endpoints.\"\"\"

import time
import traceback
import logging
from fastapi import APIRouter, HTTPException, Depends, status
from src.models.request import PredictRequest, CounterfactualRequest
from src.models.response import PredictResponse, CounterfactualResponse
from src.api.dependencies import get_db_dependency
import aiosqlite

router = APIRouter()
logger = logging.getLogger("soochak")


@router.post("/predict/", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    db: aiosqlite.Connection = Depends(get_db_dependency)
):
    \"\"\"Predict accident severity with SHAP explainability.\"\"\"
    try:
        start = time.time()
        mock_shap = [
            {"feature": "Road_Type", "value": 0.0, "contribution": 0.15, "contribution_prob": 0.03},
            {"feature": "Vehicle_Type", "value": 0.0, "contribution": 0.12, "contribution_prob": 0.02},
        ]

        cursor = await db.execute(
            \"\"\"INSERT INTO incidents
               (datetime, road_type, speed_limit, weather, lighting, junction,
                junction_ctrl, vehicle_type, driver_age, urban_rural, state, city)
               VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\"\"\",
            (request.Road_Type, request.Speed_Limit, request.Weather,
             request.Lighting, request.Junction, request.Junction_Control,
             request.Vehicle_Type, request.Driver_Age, request.Urban_Rural,
             request.State, request.City)
        )
        incident_id = cursor.lastrowid
        await db.commit()

        inference_ms = round((time.time() - start) * 1000, 2)

        return PredictResponse(
            prediction_id=1,
            predicted_class=1,
            probability=0.75,
            raw_probability=0.68,
            confidence="high",
            threshold_used=0.18,
            shap_base_value=0.53,
            top_features=mock_shap,
            inference_ms=inference_ms,
            model_version="soochak_v1",
            calibrated=True
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Prediction could not be completed."
        )
"""
bp_content = replace_file_block(bp_content, "Step 3.7:", "src/api/routers/predict.py", predict_stub_new)

# 3. Fix Step 5.7: src/api/routers/explain.py
explain_new = """\"\"\"LLM explanation endpoint.\"\"\"

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
    \"\"\"Generate LLM explanation for a prediction.\"\"\"
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
"""
bp_content = replace_file_block(bp_content, "Step 5.7:", "src/api/routers/explain.py", explain_new)

# 4. Fix Step 5.9: tests/test_predict.py
test_predict_new = """import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_predict_with_real_inference():
    payload = {
        "Road_Type": "Undivided Two way",
        "Speed_Limit": 80,
        "Weather": "Windy",
        "Lighting": "Darkness - no lighting",
        "Junction": "Crossing",
        "Junction_Control": "Drunk driving",
        "Vehicle_Type": "Motorcycle",
        "Driver_Age": "Under 18",
        "Urban_Rural": "Rural village areas",
        "State": "Steep grade upward with mountainous terrain",
        "City": "Saturday"
    }
    response = client.post("/v1/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["probability"] <= 1
    assert data["predicted_class"] in [0, 1]
    assert len(data["top_features"]) > 0
    assert data["inference_ms"] < 1000
    assert "prediction_id" in data
    assert data["prediction_id"] > 0


def test_explain_endpoint():
    # First create a prediction
    payload = {
        "Road_Type": "Undivided Two way",
        "Speed_Limit": 80,
        "Weather": "Windy",
        "Lighting": "Darkness - no lighting",
        "Junction": "Crossing",
        "Junction_Control": "Drunk driving",
        "Vehicle_Type": "Motorcycle",
        "Driver_Age": "Under 18",
        "Urban_Rural": "Rural village areas",
        "State": "Steep grade upward with mountainous terrain",
        "City": "Saturday"
    }
    pred_response = client.post("/v1/predict/", json=payload)
    prediction_id = pred_response.json()["prediction_id"]

    explain_response = client.post("/v1/explain", json={"prediction_id": prediction_id})
    assert explain_response.status_code == 200
    data = explain_response.json()
    assert "groq_explanation" in data
    assert len(data["top_features"]) > 0
"""
bp_content = replace_file_block(bp_content, "Step 5.9:", "tests/test_predict.py", test_predict_new)

# 5. Fix Step 6.1: src/services/batch_service.py to use MLService
batch_service_new = """\"\"\"Async batch prediction queue using asyncio.

NOTE: This demonstrates the asyncio.Queue producer-consumer pattern.
Jobs are stored in-memory and will be lost on container restart.
For production, upgrade to Redis + Celery.
\"\"\"

import asyncio
import uuid
import time
from typing import Dict, List
from src.services.ml_service import get_ml_service
from src.services.cache_service import get as cache_get, set as cache_set

TASKS: Dict[str, dict] = {}
BATCH_QUEUE = asyncio.Queue()


async def process_batch_task(task_id: str, incidents: List[dict]):
    \"\"\"Process a batch of incidents asynchronously using MLService.\"\"\"
    TASKS[task_id]["status"] = "processing"
    total = len(incidents)
    results = []
    ml_service = get_ml_service()

    for i, incident in enumerate(incidents):
        try:
            cached = cache_get(incident)
            if cached:
                result = cached
            else:
                result = ml_service.predict(incident)
                cache_set(incident, result)

            results.append({
                "incident": incident,
                "predicted_class": result["predicted_class"],
                "probability": result["probability"],
                "raw_probability": result["raw_probability"],
                "confidence": result["confidence"],
                "threshold_used": result["threshold_used"],
                "top_features": [
                    {
                        "feature": f["feature"],
                        "value": f["value"],
                        "contribution": f["contribution"],
                        "contribution_prob": f["contribution_prob"]
                    } for f in result["top_features"]
                ],
                "inference_ms": result["inference_ms"],
            })

        except Exception as e:
            results.append({"incident": incident, "error": str(e)})

        TASKS[task_id]["progress_pct"] = round((i + 1) / total * 100, 1)

    TASKS[task_id]["status"] = "complete"
    TASKS[task_id]["results"] = results
    TASKS[task_id]["completed_at"] = time.time()


async def batch_consumer():
    \"\"\"Background consumer that processes queued batch tasks.\"\"\"
    while True:
        try:
            task_id, incidents = await BATCH_QUEUE.get()
            await process_batch_task(task_id, incidents)
            BATCH_QUEUE.task_done()
        except Exception as e:
            print(f"Batch consumer error: {e}")
            await asyncio.sleep(1)


def create_task(incidents: List[dict]) -> str:
    \"\"\"Create a new batch task and enqueue it.\"\"\"
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "progress_pct": 0.0,
        "results": None,
        "created_at": time.time(),
        "completed_at": None,
    }
    BATCH_QUEUE.put_nowait((task_id, incidents))
    return task_id


def get_task_status(task_id: str) -> dict:
    \"\"\"Get status of a batch task.\"\"\"
    if task_id not in TASKS:
        return None
    task = TASKS[task_id].copy()
    if task["results"]:
        task["results_count"] = len(task["results"])
    return task
"""
bp_content = replace_file_block(bp_content, "Step 6.1:", "src/services/batch_service.py", batch_service_new)

# 6. Fix Step 6.2: src/api/routers/batch.py
batch_router_new = """\"\"\"Batch prediction and task status endpoints.\"\"\"

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from src.models.request import BatchPredictRequest
from src.models.response import BatchStatusResponse
from src.services.batch_service import create_task, get_task_status

router = APIRouter(prefix="/batch", tags=["Batch Processing"])


@router.post("/predict", response_model=BatchStatusResponse)
async def batch_predict(
    request: BatchPredictRequest,
):
    \"\"\"Queue a batch prediction job.\"\"\"
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
    \"\"\"Poll batch task status.\"\"\"
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
    \"\"\"Get results of completed batch task.\"\"\"
    task = get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "complete":
        raise HTTPException(status_code=400, detail="Task results are not ready yet")
    return task["results"]
"""
bp_content = replace_file_block(bp_content, "Step 6.2:", "src/api/routers/batch.py", batch_router_new)

# 7. Update any remaining curl commands in the blueprint (Phases 10, README, etc.) to use PascalCase
bp_content = bp_content.replace(
    '{"road_type":"National Highway","speed_limit":80,"weather":"Clear","lighting":"Daylight","junction":"None","state":"Delhi","city":"New Delhi"}',
    '{"Road_Type":"National Highway","Speed_Limit":80,"Weather":"Clear","Lighting":"Daylight","Junction":"None","State":"Delhi","City":"New Delhi"}'
)
bp_content = bp_content.replace(
    '{"road_type": "National Highway", "speed_limit": 80, "weather": "Clear", "lighting": "Daylight", "junction": "None", "state": "Delhi", "city": "New Delhi"}',
    '{"Road_Type": "National Highway", "Speed_Limit": 80, "Weather": "Clear", "Lighting": "Daylight", "Junction": "None", "State": "Delhi", "City": "New Delhi"}'
)

# 8. Save updated blueprint
with open(blueprint_path, "w", encoding="utf-8") as f:
    f.write(bp_content)

print("Blueprint AUDIT and REPLACEMENT completed successfully!")
