"""In-memory SHA256-keyed cache with TTL."""

import time
import hashlib
import json
from typing import Optional, Any

CACHE = {}
TTL_SECONDS = 300


def _make_key(data: dict) -> str:
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def get(data: dict) -> Optional[Any]:
    key = _make_key(data)
    if key in CACHE:
        result, timestamp = CACHE[key]
        if time.time() - timestamp < TTL_SECONDS:
            return result
        else:
            del CACHE[key]
    return None


def set(data: dict, result: Any) -> None:
    key = _make_key(data)
    CACHE[key] = (result, time.time())


def get_stats() -> dict:
    now = time.time()
    valid = sum(1 for _, ts in CACHE.values() if now - ts < TTL_SECONDS)
    return {
        "total_entries": len(CACHE),
        "valid_entries": valid,
        "ttl_seconds": TTL_SECONDS,
    }
