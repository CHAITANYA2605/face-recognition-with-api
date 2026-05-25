from pydantic import BaseModel
from typing import List, Optional

class FaceRegisterResponse(BaseModel):
    id: str
    message: str
    face_images: Optional[dict] = None  # {"front": b64, "left_profile": b64, "right_profile": b64}

class FaceMetadata(BaseModel):
    name: str
    age: int
    filename: Optional[str] = None

class FaceMatch(BaseModel):
    id: str
    score: float
    metadata: Optional[dict] = None # Will contain FaceMetadata fields
    # face_image: Optional[str] = None # Base64 encoded crop from DB (if stored) or query crop
    match_quality: Optional[str] = None
    score_breakdown: Optional[List[dict]] = None
    recognition_stages: Optional[List[str]] = None
    confidence_reason: Optional[str] = None
    fused_score: Optional[float] = None
    stage_scores: Optional[dict] = None



class FaceDetection(BaseModel):
    results: List[FaceMatch] # Matching results for this specific face
    # query_face_image: Optional[str] = None
    # visible_regions: Optional[List[str]] = None
    # quality: Optional[dict] = None
    # fallback_used: Optional[str] = None
    # occlusion_model_used: Optional[bool] = None
    # message: Optional[str] = None

class FaceSearchResponse(BaseModel):
    detections: List[FaceDetection]
    message: Optional[str] = None

class MessageResponse(BaseModel):
    message: str

class Base64Request(BaseModel):
    base64_string: str
