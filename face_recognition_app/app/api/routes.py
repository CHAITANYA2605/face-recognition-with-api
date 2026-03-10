from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Request, Response
import resource
import base64
import numpy as np
from app.core.config import settings
from app.services.face_recognition import face_service
from app.services.vector_db import vector_db
from app.schemas.face import FaceRegisterResponse, FaceSearchResponse, FaceMatch, MessageResponse, Base64Request, FaceDetection
from app.middleware.stats import request_tracker

router = APIRouter()

@router.post("/decode-image")
async def decode_base64_image(request: Base64Request):
    try:
        # Remove header if present (e.g., data:image/jpeg;base64,)
        b64_str = request.base64_string
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
            
        image_data = base64.b64decode(b64_str)
        return Response(content=image_data, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 string: {str(e)}")

@router.get("/face/{face_id}/view")
async def view_face_image(face_id: str):
    try:
        # Qdrant client's retrieve method
        points = vector_db.client.retrieve(
            collection_name=settings.COLLECTION_NAME,
            ids=[face_id],
            with_payload=True
        )
        
        if not points:
            raise HTTPException(status_code=404, detail="Face not found")
            
        payload = points[0].payload
        face_b64 = payload.get("face_image")
        
        if not face_b64:
            raise HTTPException(status_code=404, detail="No image stored for this face")
            
        image_data = base64.b64decode(face_b64)
        return Response(content=image_data, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register", response_model=FaceRegisterResponse)
async def register_face(
    front_image: UploadFile = File(...),
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
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

    views = [
        ("front", front_image),
        ("left_profile", left_image),
        ("right_profile", right_image),
    ]

    embeddings = {}
    face_images_b64 = {}

    for view, upload in views:
        if not upload.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{view} image must be an image file")

        content = await upload.read()
        embedding, face_b64 = face_service.analyze_face(content)

        if embedding is None:
            raise HTTPException(status_code=400, detail=f"No face detected in the {view} image")

        embeddings[view] = embedding
        face_images_b64[view] = face_b64

    face_id = vector_db.register_user(
        front_vec=embeddings["front"],
        left_vec=embeddings["left_profile"],
        right_vec=embeddings["right_profile"],
        metadata={"name": name, "phone_number": phone_number}
    )

    return FaceRegisterResponse(
        id=face_id,
        message="Face registered successfully with front and 2 side profile images",
        face_images=face_images_b64
    )

@router.post("/recognize", response_model=FaceSearchResponse)
async def recognize_face(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    content = await file.read()
    detections = face_service.analyze_all_faces(content)
    
    if not detections:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    
    all_results = []
    for embedding, _ in detections:
        results = vector_db.search_face(embedding, limit=5)

        matches = [
            FaceMatch(
                id=r["payload"].get("face_id", ""),
                score=round(r["score"], 4),
                metadata=r["payload"],
                face_image=None
            )
            for r in results
        ]

        all_results.append(FaceDetection(results=matches))

    return FaceSearchResponse(detections=all_results)

@router.delete("/face", response_model=MessageResponse)
async def delete_face(name: str, phone_number: str):
    if not vector_db.is_user_registered(name, phone_number):
         raise HTTPException(status_code=404, detail=f"User with name '{name}' and phone number '{phone_number}' not found")

    vector_db.delete_face_by_metadata(name, phone_number)
    return MessageResponse(message=f"Face(s) for user '{name}' deleted successfully")

@router.get("/admin/stats")
async def get_system_stats(request: Request):
    # Memory Usage
    import platform
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Mac, ru_maxrss is in bytes; on Linux it's in KB
    memory_mb = usage / (1024 * 1024) if platform.system() == "Darwin" else usage / 1024
    
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
