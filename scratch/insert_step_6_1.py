import re
from pathlib import Path

blueprint_path = Path(r"C:\Users\Hemanth\Downloads\ROADRISK_ZERO_FLAW_BLUEPRINT.md")

with open(blueprint_path, "r", encoding="utf-8") as f:
    bp_content = f.read()

# Define the Step 6.1 block to insert
step_6_1 = """## Step 6.1: Create Batch Service

```bash
cat > src/services/batch_service.py << 'EOF'
\"\"\"Async batch prediction queue using asyncio.

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
EOF
```

"""

if "## Step 6.1: Create Batch Service" not in bp_content:
    print("Step 6.1 not found, inserting...")
    # Insert right before Step 6.2
    bp_content = bp_content.replace("## Step 6.2: Create Batch Router", step_6_1 + "## Step 6.2: Create Batch Router")
    
    with open(blueprint_path, "w", encoding="utf-8") as f:
        f.write(bp_content)
    print("Step 6.1 successfully inserted!")
else:
    print("Step 6.1 already exists in the file!")
