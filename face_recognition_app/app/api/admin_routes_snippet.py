import resource
from app.services.vector_db import vector_db

# ... existing imports ...

@router.get("/admin/stats")
async def get_system_stats(request: Request):
    # Memory Usage
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Mac, ru_maxrss is in bytes, on Linux it's in KB. 
    # Python docs say: "ru_maxrss: maximum resident set size"
    # Let's assume bytes for Mac based on experience or check platform.
    # Actually on Mac it is bytes, on Linux KB.
    # Let's return it as is or convert to MB.
    memory_mb = usage / (1024 * 1024) # Assuming bytes
    if memory_mb < 1: # If it was KB
         memory_mb = usage / 1024
    
    # DB Stats
    try:
        collection_info = vector_db.get_collection_info()
        db_count = collection_info.vectors_count
        db_segments = collection_info.segments_count
    except Exception:
        db_count = "Unavailable"
        db_segments = "Unavailable"

    # API RPM
    # We need access to the middleware instance. 
    # Proper way: Attach middleware stats calculate function to app state or singleton.
    # For now, I'll access it via `request.app.state.stats_middleware` if I attach it there.

    api_stats = {}
    if hasattr(request.app.state, "stats_middleware"):
        api_stats = request.app.state.stats_middleware.get_stats()

    return {
        "memory_usage_mb": round(memory_mb, 2),
        "total_face_vectors": db_count,
        "db_segments": db_segments,
        "api_performance": api_stats
    }
