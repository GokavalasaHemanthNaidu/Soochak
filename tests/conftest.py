import os
import glob
import tempfile
import pytest
import pytest_asyncio
import aiosqlite

@pytest_asyncio.fixture
async def test_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    db = await aiosqlite.connect(path)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    db.row_factory = aiosqlite.Row
    
    # Load migrations
    migration_files = sorted(glob.glob("src/db/migrations/*.sql"))
    for file in migration_files:
        with open(file, "r") as f:
            sql_script = f.read()
        await db.executescript(sql_script)
    await db.commit()
    
    yield db
    
    await db.close()
    
    # Cleanup files
    try:
        os.unlink(path)
        if os.path.exists(path + "-wal"):
            os.unlink(path + "-wal")
        if os.path.exists(path + "-shm"):
            os.unlink(path + "-shm")
    except OSError:
        pass
