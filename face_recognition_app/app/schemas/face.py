from pydantic import BaseModel
from typing import List, Optional

class FaceRegisterResponse(BaseModel):
    id: str
    message: str
    face_image: Optional[str] = None # Base64 encoded crop

class FaceMetadata(BaseModel):
    name: str
    age: int
    phone_number: str
    filename: Optional[str] = None

class FaceMatch(BaseModel):
    id: str
    score: float
    metadata: Optional[dict] = None # Will contain FaceMetadata fields
    face_image: Optional[str] = None # Base64 encoded crop from DB (if stored) or query crop



class FaceSearchResponse(BaseModel):
    matches: List[FaceMatch]

class MessageResponse(BaseModel):
    message: str
