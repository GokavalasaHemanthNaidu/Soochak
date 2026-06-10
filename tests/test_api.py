import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app

from src.api.dependencies import get_db_dependency

@pytest.fixture
async def client(test_db):
    async def _override():
        yield test_db
    app.dependency_overrides[get_db_dependency] = _override
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
        
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_health_endpoint(client, test_db):
    response = await client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "db_connected" in data

@pytest.mark.asyncio
async def test_predict_endpoint(client, test_db):
    # Missing API key shouldn't trigger 403 here because it's test env?
    # Actually, api_key_gate checks DEMO_API_KEY. We can inject the header.
    headers = {"X-API-Key": "soochak-demo-2024"}
    
    payload = {
        "road_type": "National Highway",
        "speed_limit": 80,
        "weather": "Clear",
        "lighting": "Daylight",
        "junction": "None",
        "junction_ctrl": "None",
        "vehicle_type": "Car",
        "driver_age": "26-40",
        "urban_rural": "Urban",
        "state": "Delhi",
        "city": "New Delhi"
    }
    response = await client.post("/v1/predict", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "probability" in data
    assert "shap_values" in data

@pytest.mark.asyncio
async def test_counterfactual_endpoint(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
    payload = {
        "incident_id": 1,
        "feature_to_change": "speed_limit",
        "new_value": "40"
    }
    response = await client.post("/v1/counterfactual", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "delta_prob" in data

@pytest.mark.asyncio
async def test_invalid_speed_limit(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
    payload = {
        "road_type": "National Highway",
        "speed_limit": 85,  # 85 is not in the allowed list [10, 20... 120]
        "weather": "Clear",
        "lighting": "Daylight",
        "junction": "None",
        "junction_ctrl": "None",
        "vehicle_type": "Car",
        "driver_age": "26-40",
        "urban_rural": "Urban",
        "state": "Delhi",
        "city": "New Delhi"
    }
    response = await client.post("/v1/predict", json=payload, headers=headers)
    assert response.status_code == 422  # Pydantic validation error

@pytest.mark.asyncio
async def test_xss_sanitization(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
    payload = {
        "road_type": "National Highway",
        "speed_limit": 80,
        "weather": "Clear",
        "lighting": "Daylight",
        "junction": "None",
        "junction_ctrl": "None",
        "vehicle_type": "<script>alert(1)</script>Car",
        "driver_age": "26-40",
        "urban_rural": "Urban",
        "state": "Delhi",
        "city": "New Delhi"
    }
    response = await client.post("/v1/predict", json=payload, headers=headers)
    # The XSS should be stripped and it shouldn't crash
    assert response.status_code == 200
    
@pytest.mark.asyncio
async def test_model_metrics(client, test_db):
    response = await client.get("/v1/model/metrics")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_model_drift(client, test_db):
    response = await client.get("/v1/model/drift")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_dashboard_page(client, test_db):
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
