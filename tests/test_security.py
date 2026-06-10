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
async def test_rate_limiter(client, test_db):
    """
    Test rate limiter asserts 429 on the 35th request.
    The limit is 30 per minute.
    """
    responses = []
    for _ in range(35):
        responses.append(await client.get("/v1/health"))
    
    # First 30 should be 200
    assert responses[0].status_code == 200
    
    # Last one should be 429
    assert responses[-1].status_code == 429

@pytest.mark.asyncio
async def test_api_key_required_for_post(client, test_db):
    """POST without X-API-Key should be blocked with 403."""
    payload = {"dummy": "data"}
    # No headers
    response = await client.post("/v1/predict", json=payload)
    assert response.status_code == 403
    
@pytest.mark.asyncio
async def test_dashboard_referer_allowed_for_post(client, test_db):
    """POST from dashboard (Referer ends with /) is allowed."""
    headers = {"Referer": "http://test/"}
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
