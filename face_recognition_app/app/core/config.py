import os

class Settings:
    PROJECT_NAME: str = "Face Recognition App"
    API_V1_STR: str = "/api/v1"
    
    # Face Recognition Settings
    DETECTION_MODEL: str = "buffalo_l" # InsightFace default model pack
    # buffalo_l includes SCRFD-10G for detection and ArcFace-R100 for recognition
    INSIGHTFACE_PROVIDERS: list = ["CPUExecutionProvider"]
    SUPPRESS_INSIGHTFACE_MODEL_LOGS: bool = os.environ.get("SUPPRESS_INSIGHTFACE_MODEL_LOGS", "true").lower() == "true"
    OCCLUSION_MODEL_PATH: str = os.environ.get("OCCLUSION_MODEL_PATH", "")
    OCCLUSION_VECTOR_SIZE: int = int(os.environ.get("OCCLUSION_VECTOR_SIZE", "256"))
    STORE_FACE_IMAGES_IN_DB: bool = os.environ.get("STORE_FACE_IMAGES_IN_DB", "false").lower() == "true"
    
    # Vector DB Settings
    QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))
    COLLECTION_NAME: str = "faces-4.0"
    VECTOR_SIZE: int = 512
    REGION_VECTOR_SIZE: int = 128

    # New Relic APM
    NEW_RELIC_LICENSE_KEY: str = os.environ.get("NEW_RELIC_LICENSE_KEY", "")
    NEW_RELIC_APP_NAME: str = os.environ.get("NEW_RELIC_APP_NAME", "face-recognition-app")
    NEW_RELIC_USER_KEY: str = os.environ.get("NEW_RELIC_USER_KEY", "")

    # Kapisa user mapping API
    USER_LIST_API_URL: str = os.environ.get("USER_LIST_API_URL", "https://backend.kapisa.co.in/v1/user/list")
    USER_LIST_API_TIMEOUT_SECONDS: float = float(os.environ.get("USER_LIST_API_TIMEOUT_SECONDS", "100"))

settings = Settings()
