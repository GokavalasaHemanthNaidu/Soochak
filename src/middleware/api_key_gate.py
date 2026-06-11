import os
from fastapi import Request, HTTPException
import logging

logger = logging.getLogger(__name__)

DEMO_KEY = os.getenv("DEMO_API_KEY", "soochak-demo-2024")

async def api_key_gate(request: Request):
    """
    Protects POST endpoints from bot abuse.
    Allows local dashboard form submissions if Referer ends with '/'.
    Otherwise requires X-API-Key header.
    (ADR-004: No user auth, just bot protection).
    """
    if request.method == "GET":
        return

    # Allow dashboard form submissions
    referer = request.headers.get("referer", "")
    if request.url.path in ("/v1/predict", "/v1/predict/") and referer.endswith("/"):
        return


    api_key = request.headers.get("X-API-Key")
    if api_key != DEMO_KEY:
        logger.warning("Invalid or missing API Key")
        raise HTTPException(status_code=403, detail="Invalid API Key")
