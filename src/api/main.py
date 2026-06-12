import os
import time
import asyncio
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.routers import explain, batch, incidents, feedback
from src.api.routers import health, predict, monitoring
from src.api.routers.predict import cf_router
from src.db.database import init_db, DB_PATH
from src.middleware.rate_limiter import rate_limiter
from src.middleware.api_key_gate import api_key_gate
from src.services.ml_service import get_ml_service

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Initializing database...")
    await init_db()

    logger.info("Eagerly loading ML Service artifacts...")
    try:
        get_ml_service()
    except Exception as e:
        logger.error(f"Fatal error loading ML Service: {e}")
        raise e

    app.state.model = None
    app.state.calibrator = None
    app.state.model_version = os.getenv("MODEL_VERSION", "v1.0.0")

    # Phase 6: Start background batch consumer task — store ref so we can cancel cleanly
    from src.services.batch_service import batch_consumer

    consumer_task = asyncio.create_task(batch_consumer())

    logger.info("Application startup complete.")
    yield
    # Graceful shutdown: cancel the batch consumer before the event loop closes
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    logger.info("Application shutting down.")


app = FastAPI(title="SOOCHAK", version="2.1.0", lifespan=lifespan)

# Static & Templates — ensure directories exist (CI checkout may lack empty dirs)
os.makedirs("src/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

# CORS Middleware (Restricted to known origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://soochak.onrender.com",
        "http://localhost:8000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom Middlewares
@app.middleware("http")
async def apply_custom_middlewares(request: Request, call_next):
    try:
        # 1. Rate Limiting (Only for API endpoints)
        if request.url.path.startswith("/v1/"):
            await rate_limiter.check(request)

        # 2. API Key Gate for bot protection
        await api_key_gate(request)
    except Exception as e:
        if hasattr(e, "status_code"):
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        raise e

    # Process request and measure latency
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000

    # 3. Request Logging (Fire and forget async task)
    if request.url.path.startswith("/v1/"):
        asyncio.create_task(_log_request(request, response.status_code, latency_ms))

    return response


async def _log_request(request: Request, status_code: int, latency_ms: float):
    """Non-blocking request logging to SQLite."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    ip_hash = rate_limiter._hash_ip(client_ip)
    db_path = os.getenv("DB_PATH", DB_PATH)

    # Use direct connect (not the generator) to avoid async generator teardown leak
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                INSERT INTO request_logs
                (client_ip_hash, endpoint, method, status_code, latency_ms, rate_limited)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ip_hash,
                    request.url.path,
                    request.method,
                    status_code,
                    latency_ms,
                    1 if status_code == 429 else 0,
                ),
            )
            await db.commit()
    except Exception as e:
        logger.error(f"Failed to log request: {e}")


app.include_router(health.router, prefix="/v1")
app.include_router(predict.router, prefix="/v1")
app.include_router(cf_router, prefix="/v1")
app.include_router(explain.router, prefix="/v1")
app.include_router(batch.router, prefix="/v1")
app.include_router(monitoring.router, prefix="/v1")
app.include_router(incidents.router, prefix="/v1")
app.include_router(feedback.router, prefix="/v1")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serves the frontend HTML dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
