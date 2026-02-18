from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Request
import resource
from app.services.face_recognition import face_service
from app.services.vector_db import vector_db
from app.schemas.face import FaceRegisterResponse, FaceSearchResponse, FaceMatch, MessageResponse
from app.middleware.stats import request_tracker
import numpy as np

router = APIRouter()

@router.post("/register", response_model=FaceRegisterResponse)
async def register_face(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    phone_number: str = Form(...)
):
    # Input Validation
    if not name.strip() or len(name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Name must be at least 2 characters long")
    
    if not phone_number.isdigit() or not (10 == len(phone_number)):
        raise HTTPException(status_code=400, detail="Phone number must be between 10 digits")

    # Check for duplicate registration
    if vector_db.is_user_registered(name, phone_number):
        raise HTTPException(
            status_code=400, 
            detail=f"User with name '{name}' and phone number '{phone_number}' is already registered."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    content = await file.read()
    embedding, face_b64 = face_service.analyze_face(content)
    
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    
    metadata = {
        "name": name,
        "age": age,
        "phone_number": phone_number,
        "filename": file.filename,
        "face_image": face_b64
    }
    
    face_id = vector_db.insert_face(embedding, metadata=metadata)
    
    return FaceRegisterResponse(id=face_id, message="Face registered successfully", face_image=face_b64)

@router.post("/recognize", response_model=FaceSearchResponse)
async def recognize_face(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    content = await file.read()
    embedding, _ = face_service.analyze_face(content)
    
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    
    results = vector_db.search_face(embedding)
    
    matches = []
    for result in results:
        matches.append(FaceMatch(
            id=result.id,
            score=result.score,
            metadata=result.payload,
            face_image=result.payload.get("face_image") if result.payload else None
        ))
        
    return FaceSearchResponse(matches=matches)

@router.delete("/face", response_model=MessageResponse)
async def delete_face(name: str, phone_number: str):
    if not vector_db.is_user_registered(name, phone_number):
         raise HTTPException(status_code=404, detail=f"User with name '{name}' and phone number '{phone_number}' not found")

    vector_db.delete_face_by_metadata(name, phone_number)
    return MessageResponse(message=f"Face(s) for user '{name}' deleted successfully")

@router.get("/admin/stats")
async def get_system_stats(request: Request):
    # Memory Usage
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Mac, ru_maxrss is in bytes, on Linux it's often KB.
    # Assuming bytes for Mac.
    memory_mb = usage / (1024 * 1024) 
    
    # DB Stats
    db_count = 0
    db_segments = 0
    try:
        collection_info = vector_db.get_collection_info()
        db_count = collection_info.vectors_count if collection_info.vectors_count is not None else 0
        db_segments = collection_info.segments_count if collection_info.segments_count is not None else 0
    except Exception:
        db_count = "Unavailable"
        db_segments = "Unavailable"

    # API RPM
    api_stats = request_tracker.get_stats()

    return {
        "memory_usage_mb": round(memory_mb, 2),
        "total_face_vectors": db_count,
        "db_segments": db_segments,
        "api_performance": api_stats
    }
