import os
import aiosqlite
import logging

logger = logging.getLogger(__name__)
DB_PATH = os.getenv("DB_PATH", "./data/soochak.db")

async def log_drift_event(feature_name: str, value_raw: str, event_type: str = "unknown_category"):
    """Log a drift event (such as an unknown category) to the database asynchronously."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO drift_events (feature_name, value_raw, event_type)
                VALUES (?, ?, ?)
                """,
                (feature_name, str(value_raw), event_type)
            )
            await db.commit()
    except Exception as e:
        logger.error(f"Failed to log drift event: {e}")
