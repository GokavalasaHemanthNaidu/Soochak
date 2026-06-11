import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app
from src.api.dependencies import get_db_dependency

@pytest.fixture
async def client(test_db):
    async def _override():
        yield test_db
    app.dependency_overrides[get_db_dependency] = _override
    
    from httpx import AsyncClient, ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
        
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_predict_with_real_inference(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
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
    response = await client.post("/v1/predict/", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["probability"] <= 1
    assert data["predicted_class"] in [0, 1]
    assert len(data["top_features"]) > 0
    assert data["inference_ms"] < 1000
    assert "prediction_id" in data
    assert data["prediction_id"] > 0


@pytest.mark.asyncio
async def test_explain_endpoint(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
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
    pred_response = await client.post("/v1/predict/", json=payload, headers=headers)
    prediction_id = pred_response.json()["prediction_id"]

    explain_response = await client.post("/v1/explain", json={"prediction_id": prediction_id}, headers=headers)
    assert explain_response.status_code == 200
    data = explain_response.json()
    assert "groq_explanation" in data
    assert len(data["top_features"]) > 0


@pytest.mark.asyncio
async def test_batch_prediction_flow(client):
    import asyncio
    headers = {"X-API-Key": "soochak-demo-2024"}
    payload = {
        "incidents": [
            {
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
        ]
    }
    response = await client.post("/v1/batch/predict", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "queued"
    task_id = data["task_id"]

    # Poll task status
    poll_response = await client.get(f"/v1/batch/tasks/{task_id}", headers=headers)
    assert poll_response.status_code == 200
    poll_data = poll_response.json()
    assert poll_data["task_id"] == task_id

    # Give the consumer a brief moment to process the task
    await asyncio.sleep(0.5)

    poll_response = await client.get(f"/v1/batch/tasks/{task_id}", headers=headers)
    assert poll_response.status_code == 200
    poll_data = poll_response.json()
    
    # Check results URL or completed results
    if poll_data["status"] == "complete":
        results_response = await client.get(f"/v1/batch/tasks/{task_id}/results", headers=headers)
        assert results_response.status_code == 200
        results_data = results_response.json()
        assert len(results_data) == 1
        assert "probability" in results_data[0]

