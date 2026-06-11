import os
import glob
import logging
from typing import AsyncGenerator
from dotenv import load_dotenv

import aiosqlite

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "./data/soochak.db")

logger = logging.getLogger(__name__)


async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    db = await aiosqlite.connect(DB_PATH)
    try:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("PRAGMA foreign_keys=ON")
        db.row_factory = aiosqlite.Row
        yield db
    finally:
        await db.close()


async def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")

        migration_files = sorted(glob.glob("src/db/migrations/*.sql"))

        for file in migration_files:
            logger.info(f"Applying migration: {file}")
            with open(file, "r") as f:
                sql_script = f.read()
            await db.executescript(sql_script)

        await db.commit()
