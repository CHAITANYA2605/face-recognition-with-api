from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import time
from collections import defaultdict

class RequestTracker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RequestTracker, cls).__new__(cls)
            cls._instance.request_counts = defaultdict(int)
            cls._instance.start_time = time.time()
        return cls._instance

    def record_request(self, path: str):
         self.request_counts[path] += 1

    def get_stats(self):
        uptime_seconds = time.time() - self.start_time
        uptime_minutes = uptime_seconds / 60 if uptime_seconds > 0 else 1
        
        stats = {}
        for path, count in self.request_counts.items():
            stats[path] = {
                "total_requests": count,
                "rpm": round(count / uptime_minutes, 2)
            }
        return stats

request_tracker = RequestTracker()

class StatsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/v1"):
            # We can also strip query params here if needed, but path usually doesn't have them
            request_tracker.record_request(path)
            
        response = await call_next(request)
        return response