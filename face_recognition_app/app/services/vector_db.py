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
            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config={
                    "front": models.VectorParams(size=settings.VECTOR_SIZE, distance=models.Distance.COSINE),
                    "left_profile": models.VectorParams(size=settings.VECTOR_SIZE, distance=models.Distance.COSINE),
                    "right_profile": models.VectorParams(size=settings.VECTOR_SIZE, distance=models.Distance.COSINE),
                }
            )
        self.collection_checked = True

    def register_user(self, front_vec: np.ndarray, left_vec: np.ndarray, right_vec: np.ndarray, metadata: dict = None) -> str:
        self._ensure_collection_exists()
        point_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}
        metadata["face_id"] = point_id

        self.client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        "front": front_vec.tolist(),
                        "left_profile": left_vec.tolist(),
                        "right_profile": right_vec.tolist(),
                    },
                    payload=metadata
                )
            ]
        )
        return point_id

    def search_face(self, vector: np.ndarray, limit: int = 5) -> list:
        """Search across all 3 named vectors and return aggregated results."""
        self._ensure_collection_exists()
        groups = {}
        for view_name in ["front", "left_profile", "right_profile"]:
            results = self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=vector.tolist(),
                using=view_name,
                limit=limit,
                with_payload=True
            ).points
            for r in results:
                pid = r.id
                if pid not in groups:
                    groups[pid] = {"scores": [], "payload": r.payload}
                groups[pid]["scores"].append(r.score)

        ranked = sorted(
            groups.values(),
            key=lambda x: sum(x["scores"]) / len(x["scores"]),
            reverse=True
        )
        return [
            {"score": sum(g["scores"]) / len(g["scores"]), "payload": g["payload"]}
            for g in ranked[:limit]
        ]

    def is_user_registered(self, name: str, phone_number: str) -> bool:
        self._ensure_collection_exists()
        count_result = self.client.count(
            collection_name=settings.COLLECTION_NAME,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(key="name", match=models.MatchValue(value=name)),
                    models.FieldCondition(key="phone_number", match=models.MatchValue(value=phone_number))
                ]
            )
        )
        return count_result.count > 0

    def delete_face_by_metadata(self, name: str, phone_number: str):
        self._ensure_collection_exists()
        self.client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="name", match=models.MatchValue(value=name)),
                        models.FieldCondition(key="phone_number", match=models.MatchValue(value=phone_number))
                    ]
                )
            )
        )

    def get_collection_info(self):
        self._ensure_collection_exists()
        return self.client.get_collection(settings.COLLECTION_NAME)

vector_db = VectorDBService()
