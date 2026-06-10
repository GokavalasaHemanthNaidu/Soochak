import time
import hashlib
from collections import defaultdict
from fastapi import Request, HTTPException
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    In-memory rate limiter using a sliding window.
    
    LIMITATION (ADR-007): This resets on container restart and is not distributed.
    It is sufficient for the SOOCHAK demo portfolio. For production, upgrade to Redis.
    """
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.ip_records = defaultdict(list)
        
    def _hash_ip(self, ip: str) -> str:
        """Hash IP with SHA-256 for privacy."""
        return hashlib.sha256(ip.encode()).hexdigest()[:16]

    async def check(self, request: Request):
        client_ip = request.client.host if request.client else "127.0.0.1"
        ip_hash = self._hash_ip(client_ip)
        current_time = time.time()
        
        # Clean up entries older than 60 seconds
        self.ip_records[ip_hash] = [
            t for t in self.ip_records[ip_hash] 
            if current_time - t < 60.0
        ]
        
        if len(self.ip_records[ip_hash]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for hashed IP {ip_hash}")
            raise HTTPException(status_code=429, detail="Too Many Requests")
            
        self.ip_records[ip_hash].append(current_time)

rate_limiter = RateLimiter(requests_per_minute=30)
