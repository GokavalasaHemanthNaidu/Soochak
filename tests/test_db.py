import pytest
import aiosqlite

@pytest.mark.asyncio
async def test_all_tables_created(test_db):
    async with test_db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
        tables = [row["name"] for row in await cursor.fetchall()]
        
    expected_tables = [
        "incidents", "predictions", "counterfactuals", 
        "feedback", "model_registry", "request_logs"
    ]
    for t in expected_tables:
        assert t in tables

@pytest.mark.asyncio
async def test_indexes_exist(test_db):
    async with test_db.execute("SELECT name FROM sqlite_master WHERE type='index'") as cursor:
        indexes = [row["name"] for row in await cursor.fetchall()]
        
    expected_indexes = [
        "idx_predictions_incident", "idx_incidents_datetime", 
        "idx_incidents_state", "idx_incidents_city", 
        "idx_predictions_created", "idx_feedback_prediction", 
        "idx_request_logs_endpoint", "idx_request_logs_ip"
    ]
    for idx in expected_indexes:
        assert idx in indexes

@pytest.mark.asyncio
async def test_foreign_keys(test_db):
    # Test 1: Insert incident and prediction, then delete incident and verify cascade
    await test_db.execute("INSERT INTO incidents (id, state) VALUES (1, 'MH')")
    await test_db.execute("INSERT INTO predictions (incident_id, predicted_class) VALUES (1, 1)")
    await test_db.commit()
    
    # Delete incident should cascade and delete prediction
    await test_db.execute("DELETE FROM incidents WHERE id=1")
    await test_db.commit()
    
    async with test_db.execute("SELECT COUNT(*) as count FROM predictions WHERE incident_id=1") as cursor:
        row = await cursor.fetchone()
        assert row["count"] == 0
        
    # Test 2: Foreign key violation raises Exception
    with pytest.raises(aiosqlite.IntegrityError):
        await test_db.execute("INSERT INTO predictions (incident_id, predicted_class) VALUES (999, 1)")
        await test_db.commit()

@pytest.mark.asyncio
async def test_no_plaintext_ip_field(test_db):
    async with test_db.execute("PRAGMA table_info(request_logs)") as cursor:
        columns = [row["name"] for row in await cursor.fetchall()]
        
    assert "client_ip_hash" in columns
    assert "client_ip" not in columns
