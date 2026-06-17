from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Request, Response
import newrelic.agent
import logging
import resource

logger = logging.getLogger(__name__)
import base64
import numpy as np
import requests
from app.core.config import settings
from app.services.face_recognition import face_service
from app.services.vector_db import vector_db
from app.schemas.face import FaceRegisterResponse, FaceSearchResponse, FaceMatch, MessageResponse, Base64Request, FaceDetection
from app.middleware.stats import request_tracker

router = APIRouter()


def _match_quality(score: float, quality: dict) -> str:
    occlusion_score = float((quality or {}).get("occlusion_score", 0.0))
    if score >= 0.72 and occlusion_score < 0.45:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _register_face_with_user_api(name: str, face_id: str, token: str) -> requests.Response:
    return requests.post(
        settings.USER_LIST_API_URL,
        headers={
            "Content-Type": "application/json",
            "token": token,
        },
        json={
            "userLabel": name,
            "key": face_id,
            "type": "FACE_RECOGNITION",
        },
        timeout=settings.USER_LIST_API_TIMEOUT_SECONDS,
    )

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
    request: Request,
    front_image: UploadFile = File(...),
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...)
):
    logger.info("Register face request for name=%s", name)
    # Input Validation
    if not name.strip() or len(name.strip()) < 2:
        logger.warning("Registration rejected: name too short name=%s", name)
        raise HTTPException(status_code=400, detail="Name must be at least 2 characters long")

    # Check for duplicate registration
    if vector_db.is_user_registered(name):
        logger.warning("Registration rejected: duplicate name=%s", name)
        raise HTTPException(
            status_code=400,
            detail=f"User with name '{name}' is already registered."
        )

    views = [
        ("front", front_image),
        ("left_profile", left_image),
        ("right_profile", right_image),
    ]

    embeddings = {}
    region_vectors = {}
    occlusion_vectors = {}
    face_images_b64 = {}
    profile_quality = {}
    profile_visible_regions = {}
    profile_occlusion_model_used = {}

    for view, upload in views:
        if not upload.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{view} image must be an image file")

        content = await upload.read()
        analysis = face_service.analyze_primary_face_details(content)

        if analysis is None:
            logger.warning("No face detected in %s image for name=%s", view, name)
            raise HTTPException(status_code=400, detail=f"No face detected in the {view} image")

        embeddings[view] = analysis.embedding
        region_vectors[view] = analysis.regions
        occlusion_vectors[view] = analysis.occlusion_embedding if analysis.occlusion_model_used else None
        face_images_b64[view] = analysis.face_image
        profile_quality[view] = analysis.quality
        profile_visible_regions[view] = analysis.visible_regions
        profile_occlusion_model_used[view] = analysis.occlusion_model_used

    face_id = vector_db.register_user(
        front_vec=embeddings["front"],
        left_vec=embeddings["left_profile"],
        right_vec=embeddings["right_profile"],
        region_vectors=region_vectors,
        occlusion_vectors=occlusion_vectors,
        metadata={
            "name": name,
            "age": age,
            "profile_quality": profile_quality,
            "profile_visible_regions": profile_visible_regions,
            "profile_occlusion_model_used": profile_occlusion_model_used,
            "face_images_stored": settings.STORE_FACE_IMAGES_IN_DB,
            "occlusion_model_used": any(profile_occlusion_model_used.values())
        }
        | ({
            "face_image": face_images_b64["front"],
        } if settings.STORE_FACE_IMAGES_IN_DB else {})
    )

    try:
        user_api_response = _register_face_with_user_api(
            name=name,
            face_id=face_id,
            token=request.headers.get("token", "")
        )
    except requests.RequestException as exc:
        vector_db.delete_face_by_id(face_id)
        logger.error("User API request failed for name=%s face_id=%s error=%s", name, face_id, exc)
        newrelic.agent.record_custom_event("FaceRegistration", {
            "status": "failure",
            "name": name,
            "reason": "user_api_request_error",
        })
        raise HTTPException(status_code=502, detail=f"User list API request failed: {str(exc)}")

    if not user_api_response.ok:
        vector_db.delete_face_by_id(face_id)
        logger.error("User API returned error status=%s for name=%s face_id=%s", user_api_response.status_code, name, face_id)
        newrelic.agent.record_custom_event("FaceRegistration", {
            "status": "failure",
            "name": name,
            "reason": f"user_api_http_{user_api_response.status_code}",
        })
        return Response(
            content=user_api_response.content,
            status_code=user_api_response.status_code,
            media_type=user_api_response.headers.get("content-type", "application/json")
        )

    logger.info("Face registered successfully name=%s face_id=%s age=%s", name, face_id, age)
    newrelic.agent.record_custom_event("FaceRegistration", {
        "status": "success",
        "name": name,
        "age": age,
        "face_id": face_id,
        "occlusion_model_used": any(profile_occlusion_model_used.values()),
        "views_processed": len(views),
    })
    return FaceRegisterResponse(
        id=face_id,
        message="Face registered successfully with front and 2 side profile images",
        face_images=face_images_b64
    )

@router.post("/recognize", response_model=FaceSearchResponse)
async def recognize_face(file: UploadFile = File(...)):
    logger.info("Recognize face request filename=%s content_type=%s", file.filename, file.content_type)
    if not file.content_type.startswith("image/"):
        logger.warning("Recognize rejected: not an image content_type=%s", file.content_type)
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    detections = face_service.analyze_all_faces(content)

    if not detections:
        logger.info("No faces detected in uploaded image")
        return FaceSearchResponse(
            detections=[],
            message="No usable face detected. The image may be too occluded, blurred, dark, or angled for recognition."
        )
    
    all_results = []
    for detection in detections:
        results = vector_db.staged_search_face(
            detection.embedding,
            region_vectors=detection.regions,
            visible_regions=detection.visible_regions,
            occlusion_vector=detection.occlusion_embedding,
            occlusion_model_used=detection.occlusion_model_used,
            quality=detection.quality,
            limit=5
        )

        matches = [
            FaceMatch(
                id=r["payload"].get("face_id", ""),
                score=round(r["score"], 4),
                metadata=r["payload"],
                face_image=r["payload"].get("face_image"),
                match_quality=_match_quality(r["score"], detection.quality),
                score_breakdown=r.get("score_breakdown", [])[:8],
                recognition_stages=r.get("recognition_stages", []),
                confidence_reason=r.get("confidence_reason"),
                fused_score=round(r["fused_score"], 4) if "fused_score" in r else None,
                stage_scores=r.get("stage_scores")
            )
            for r in results
        ]

        message = None
        if detection.quality.get("occlusion_score", 0) >= 0.45:
            message = "Partial or low-visibility face detected; results are best-effort."
        elif results:
            message = results[0].get("confidence_reason")

        all_results.append(FaceDetection(
            results=matches,
            query_face_image=detection.face_image,
            visible_regions=detection.visible_regions,
            quality=detection.quality,
            fallback_used=detection.fallback_used,
            occlusion_model_used=detection.occlusion_model_used,
            message=message
        ))

    for i, (detection, result_set) in enumerate(zip(detections, all_results)):
        top_score = result_set.results[0].score if result_set.results else 0.0
        top_id = result_set.results[0].id if result_set.results else None
        logger.info(
            "Recognition result face_index=%s matched=%s top_score=%.4f top_id=%s",
            i, bool(result_set.results), top_score, top_id,
        )
        newrelic.agent.record_custom_event("FaceRecognition", {
            "face_index": i,
            "matched": len(result_set.results) > 0,
            "top_score": round(top_score, 4),
            "match_quality": result_set.results[0].match_quality if result_set.results else "none",
            "occlusion_model_used": detection.occlusion_model_used,
            "fallback_used": detection.fallback_used,
            "occlusion_score": round(float((detection.quality or {}).get("occlusion_score", 0.0)), 3),
            "visible_region_count": len(detection.visible_regions or []),
        })

    return FaceSearchResponse(detections=all_results)

@router.delete("/face", response_model=MessageResponse)
async def delete_face(name: str):
    logger.info("Delete face request name=%s", name)
    if not vector_db.is_user_registered(name):
        logger.warning("Delete rejected: user not found name=%s", name)
        raise HTTPException(status_code=404, detail=f"User with name '{name}' not found")

    vector_db.delete_face_by_metadata(name)
    logger.info("Face deleted successfully name=%s", name)
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
