import os

class Settings:
    PROJECT_NAME: str = "Face Recognition App"
    API_V1_STR: str = "/api/v1"
    
    # Face Recognition Settings
    DETECTION_MODEL: str = "buffalo_l" # InsightFace default model pack
    # buffalo_l includes SCRFD-10G for detection and ArcFace-R100 for recognition
    
    # Vector DB Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "faces"
    VECTOR_SIZE: int = 512

settings = Settings()
