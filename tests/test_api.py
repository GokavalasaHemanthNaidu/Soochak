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
    assert data["status"] in ["ok", "degraded", "healthy"]
    assert "db_connected" in data

@pytest.mark.asyncio
async def test_predict_endpoint(client, test_db):
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
    assert "prediction_id" in data
    assert "probability" in data
    assert "raw_probability" in data
    assert "confidence" in data
    assert "threshold_used" in data

@pytest.mark.asyncio
async def test_counterfactual_endpoint(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
    
    # First insert a mock incident to base the counterfactual on
    await test_db.execute(
        """INSERT INTO incidents (id, road_type, speed_limit, weather, lighting, junction, junction_ctrl, vehicle_type, driver_age, urban_rural, state, city)
           VALUES (1, 'Undivided Two way', 80, 'Windy', 'Darkness - no lighting', 'Crossing', 'Drunk driving', 'Motorcycle', 'Under 18', 'Rural village areas', 'Steep grade upward with mountainous terrain', 'Saturday')"""
    )
    await test_db.execute(
        """INSERT INTO predictions (incident_id, model_version, predicted_class, probability, shap_json, shap_base_value, inference_ms)
           VALUES (1, 'soochak_v1', 1, 0.1985, '[]', 0.53, 5.0)"""
    )
    await test_db.commit()

    payload = {
        "incident_id": 1,
        "feature_to_change": "speed_limit",
        "new_value": "40"
    }
    response = await client.post("/v1/counterfactual", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "delta_prob" in data
    assert "original_prob" in data
    assert "new_prob" in data
    assert "interpretation" in data

@pytest.mark.asyncio
async def test_invalid_speed_limit(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
    payload = {
        "Road_Type": "Undivided Two way",
        "Speed_Limit": 250,  # 250 is out of bounds [0, 200]
        "Weather": "Normal",
        "Lighting": "Daylight",
        "Junction": "No junction",
        "Junction_Control": "No distancing",
        "Vehicle_Type": "Automobile",
        "Driver_Age": "18-30",
        "Urban_Rural": "Residential areas",
        "State": "Tangent road with flat terrain",
        "City": "Monday"
    }
    response = await client.post("/v1/predict/", json=payload, headers=headers)
    assert response.status_code == 422  # Pydantic validation error

@pytest.mark.asyncio
async def test_xss_sanitization(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
    payload = {
        "Road_Type": "Undivided Two way",
        "Speed_Limit": 80,
        "Weather": "Windy",
        "Lighting": "Darkness - no lighting",
        "Junction": "Crossing",
        "Junction_Control": "Drunk driving",
        "Vehicle_Type": "<script>alert(1)</script>Motorcycle",
        "Driver_Age": "Under 18",
        "Urban_Rural": "Rural village areas",
        "State": "Steep grade upward with mountainous terrain",
        "City": "Saturday"
    }
    response = await client.post("/v1/predict/", json=payload, headers=headers)
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


@pytest.mark.asyncio
async def test_list_incidents(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
    # Insert a dummy incident
    await test_db.execute(
        """INSERT INTO incidents (id, road_type, speed_limit, weather, lighting, junction, junction_ctrl, vehicle_type, driver_age, urban_rural, state, city, severity_true)
           VALUES (1, 'Undivided Two way', 80, 'Windy', 'Darkness - no lighting', 'Crossing', 'Drunk driving', 'Motorcycle', 'Under 18', 'Rural village areas', 'Steep grade upward with mountainous terrain', 'Saturday', 1)"""
    )
    await test_db.commit()

    # Query list
    response = await client.get("/v1/incidents", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "incidents" in data
    assert len(data["incidents"]) >= 1
    assert data["incidents"][0]["road_type"] == "Undivided Two way"
    assert data["total"] >= 1

    # Test filters
    response_filtered = await client.get("/v1/incidents?severity=fatal", headers=headers)
    assert response_filtered.status_code == 200
    assert len(response_filtered.json()["incidents"]) >= 1

    response_empty = await client.get("/v1/incidents?severity=slight", headers=headers)
    assert response_empty.status_code == 200
    assert len(response_empty.json()["incidents"]) == 0


@pytest.mark.asyncio
async def test_get_incident_detail(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
    await test_db.execute(
        """INSERT INTO incidents (id, road_type, speed_limit, weather, lighting, junction, junction_ctrl, vehicle_type, driver_age, urban_rural, state, city)
           VALUES (2, 'Undivided Two way', 80, 'Windy', 'Darkness - no lighting', 'Crossing', 'Drunk driving', 'Motorcycle', 'Under 18', 'Rural village areas', 'Steep grade upward with mountainous terrain', 'Saturday')"""
    )
    await test_db.execute(
        """INSERT INTO predictions (incident_id, model_version, predicted_class, probability, shap_json, shap_base_value, inference_ms)
           VALUES (2, 'soochak_v1', 1, 0.1985, '[]', 0.53, 5.0)"""
    )
    await test_db.commit()

    response = await client.get("/v1/incidents/2", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "incident" in data
    assert data["incident"]["id"] == 2
    assert "predictions" in data
    assert len(data["predictions"]) == 1


@pytest.mark.asyncio
async def test_get_prediction_detail(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
    # Insert prediction
    await test_db.execute(
        """INSERT INTO incidents (id, road_type, speed_limit, weather, lighting, junction, junction_ctrl, vehicle_type, driver_age, urban_rural, state, city)
           VALUES (3, 'Undivided Two way', 80, 'Windy', 'Darkness - no lighting', 'Crossing', 'Drunk driving', 'Motorcycle', 'Under 18', 'Rural village areas', 'Steep grade upward with mountainous terrain', 'Saturday')"""
    )
    await test_db.execute(
        """INSERT INTO predictions (id, incident_id, model_version, predicted_class, probability, shap_json, shap_base_value, inference_ms)
           VALUES (42, 3, 'soochak_v1', 1, 0.1985, '[]', 0.53, 5.0)"""
    )
    await test_db.commit()

    response = await client.get("/v1/predictions/42", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 42
    assert data["model_version"] == "soochak_v1"


@pytest.mark.asyncio
async def test_feedback_loop(client, test_db):
    headers = {"X-API-Key": "soochak-demo-2024"}
    # Insert prerequisite prediction
    await test_db.execute(
        """INSERT INTO incidents (id, road_type, speed_limit, weather, lighting, junction, junction_ctrl, vehicle_type, driver_age, urban_rural, state, city)
           VALUES (4, 'Undivided Two way', 80, 'Windy', 'Darkness - no lighting', 'Crossing', 'Drunk driving', 'Motorcycle', 'Under 18', 'Rural village areas', 'Steep grade upward with mountainous terrain', 'Saturday')"""
    )
    await test_db.execute(
        """INSERT INTO predictions (id, incident_id, model_version, predicted_class, probability, shap_json, shap_base_value, inference_ms)
           VALUES (10, 4, 'soochak_v1', 1, 0.1985, '[]', 0.53, 5.0)"""
    )
    await test_db.commit()

    payload = {
        "prediction_id": 10,
        "was_correct": 1,
        "human_label": 1
    }
    response = await client.post("/v1/feedback", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "feedback_id" in data
    assert data["correction_rate_7d"] == 0.0

