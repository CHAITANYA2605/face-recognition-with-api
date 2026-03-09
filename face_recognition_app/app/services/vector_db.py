from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings
import uuid
import numpy as np

class VectorDBService:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_checked = False

    def _ensure_collection_exists(self):
        if self.collection_checked:
            return
            
        try:
            self.client.get_collection(settings.COLLECTION_NAME)
        except Exception:
            # If collection doesn't exist, create it
            # This might fail if Qdrant is not running, but that's expected when we actually try to use it
            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=settings.VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
        self.collection_checked = True

    def insert_face(self, vector: np.ndarray, metadata: dict = None) -> str:
        self._ensure_collection_exists()
        point_id = str(uuid.uuid4())
        
        # Save face_id to metadata for easier access
        if metadata is None:
            metadata = {}
        metadata["face_id"] = point_id
        
        self.client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=metadata
                )
            ]
        )
        return point_id

    def search_face(self, vector: np.ndarray, limit: int = 1) -> list:
        self._ensure_collection_exists()
        results = self.client.query_points(
            collection_name=settings.COLLECTION_NAME,
            query=vector.tolist(),
            limit=limit,
            with_payload=True
        ).points
        return results

    def is_user_registered(self, name: str, phone_number: str) -> bool:
        self._ensure_collection_exists()
        count_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="name",
                    match=models.MatchValue(value=name)
                ),
                models.FieldCondition(
                    key="phone_number",
                    match=models.MatchValue(value=phone_number)
                )
            ]
        )
        count_result = self.client.count(
            collection_name=settings.COLLECTION_NAME,
            count_filter=count_filter
        )
        return count_result.count > 0

    def delete_face(self, point_id: str):
        self._ensure_collection_exists()
        self.client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=models.PointIdsList(
                points=[point_id]
            )
        )

    def delete_face_by_metadata(self, name: str, phone_number: str):
        self._ensure_collection_exists()
        self.client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="name",
                            match=models.MatchValue(value=name)
                        ),
                        models.FieldCondition(
                            key="phone_number",
                            match=models.MatchValue(value=phone_number)
                        )
                    ]
                )
            )
        )

    def get_collection_info(self):
        self._ensure_collection_exists()
        return self.client.get_collection(settings.COLLECTION_NAME)

vector_db = VectorDBService()