from fastapi import Request
from src.db.database import get_db


async def get_db_dependency():
    """Dependency to get a database connection per request."""
    async for db in get_db():
        yield db


def get_model(request: Request):
    """Dependency to get the globally loaded ONNX model."""
    return request.app.state.model


def get_calibrator(request: Request):
    """Dependency to get the globally loaded probability calibrator."""
    return request.app.state.calibrator
