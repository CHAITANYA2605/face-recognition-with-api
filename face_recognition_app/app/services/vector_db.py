from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings
import uuid
import numpy as np

PROFILE_VIEWS = ("front", "left_profile", "right_profile")
REGION_NAMES = ("upper_face", "lower_face", "left_half", "right_half", "center_face")
FULL_VECTOR_WEIGHT=0.52
REGION_VECTOR_WEIGHT=0.33
OCCLUSION_VECTOR_WEIGHT=0.15
FULL_SCORE_THRESHOLD=0.68
REGION_SCORE_THRESHOLD=0.55
AMBIGUITY_MARGIN=0.06
MIN_ACCEPTED_SCORE=0.7
MIN_ACCEPTED_MARGIN=0.04
MIN_WEAK_MARGIN_SCORE=0.82
MIN_FULL_SUPPORTING_CHANNEL_SCORE=0.5
MIN_REGION_SUPPORTING_CHANNEL_SCORE=0.55
MIN_OCCLUSION_SUPPORTING_CHANNEL_SCORE=0.62
MIN_SUPPORTING_CHANNELS=2
MIN_FULL_STAGE_SUPPORT_SCORE=0.4


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
            vectors_config = {
                view: models.VectorParams(size=settings.VECTOR_SIZE, distance=models.Distance.COSINE)
                for view in PROFILE_VIEWS
            }
            vectors_config.update({
                f"{view}_{region}": models.VectorParams(size=settings.REGION_VECTOR_SIZE, distance=models.Distance.COSINE)
                for view in PROFILE_VIEWS
                for region in REGION_NAMES
            })
            vectors_config.update({
                f"{view}_occlusion": models.VectorParams(size=settings.OCCLUSION_VECTOR_SIZE, distance=models.Distance.COSINE)
                for view in PROFILE_VIEWS
            })

            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=vectors_config
            )
        self.collection_checked = True

    def register_user(
        self,
        front_vec: np.ndarray,
        left_vec: np.ndarray,
        right_vec: np.ndarray,
        region_vectors: dict = None,
        occlusion_vectors: dict = None,
        metadata: dict = None
    ) -> str:
        self._ensure_collection_exists()
        point_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}
        metadata["face_id"] = point_id
        region_vectors = region_vectors or {}
        occlusion_vectors = occlusion_vectors or {}

        vectors = {
            "front": front_vec.tolist(),
            "left_profile": left_vec.tolist(),
            "right_profile": right_vec.tolist(),
        }
        for view in PROFILE_VIEWS:
            for region in REGION_NAMES:
                region_key = f"{view}_{region}"
                descriptor = region_vectors.get(view, {}).get(region)
                vectors[region_key] = descriptor if descriptor is not None else [0.0] * settings.REGION_VECTOR_SIZE
            occlusion_key = f"{view}_occlusion"
            occlusion_descriptor = occlusion_vectors.get(view)
            vectors[occlusion_key] = (
                occlusion_descriptor
                if occlusion_descriptor is not None
                else [0.0] * settings.OCCLUSION_VECTOR_SIZE
            )

        self.client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=metadata
                )
            ]
        )
        return point_id

    def _add_scores(self, groups: dict, results: list, channel: str, weight: float):
        for result in results:
            pid = result.id
            if pid not in groups:
                groups[pid] = {
                    "weighted_score": 0.0,
                    "weight": 0.0,
                    "payload": result.payload,
                    "score_breakdown": [],
                    "channel_scores": {},
                    "channels_used": []
                }
            score = float(result.score)
            groups[pid]["weighted_score"] += score * weight
            groups[pid]["weight"] += weight
            groups[pid]["channel_scores"][channel] = max(
                score,
                groups[pid]["channel_scores"].get(channel, 0.0)
            )
            groups[pid]["channels_used"].append(channel)
            groups[pid]["score_breakdown"].append({
                "channel": channel,
                "score": round(score, 4),
                "weight": round(weight, 4),
            })

    def _rank_groups(self, groups: dict, limit: int) -> list:
        ranked = sorted(
            groups.values(),
            key=lambda x: self._channel_display_score(x),
            reverse=True
        )
        return [
            {
                "score": self._channel_display_score(g),
                "channel_fused_score": g["weighted_score"] / g["weight"] if g["weight"] else 0,
                "channel_scores": {
                    channel: round(score, 4)
                    for channel, score in g["channel_scores"].items()
                },
                "payload": g["payload"],
                "score_breakdown": sorted(g["score_breakdown"], key=lambda item: item["score"], reverse=True)
            }
            for g in ranked[:limit]
        ]

    def _channel_display_score(self, group: dict) -> float:
        fused_score = group["weighted_score"] / group["weight"] if group["weight"] else 0.0
        best_channel_score = max(group.get("channel_scores", {}).values(), default=0.0)
        channel_count = len(set(group.get("channels_used", [])))
        agreement_bonus = 0.01 if channel_count >= 2 else 0.0
        return round(min(1.0, max(fused_score, best_channel_score) + agreement_bonus), 4)

    def _search_full_face(self, vector: np.ndarray, limit: int) -> list:
        groups = {}
        per_full_view_weight = 1.0 / len(PROFILE_VIEWS)
        for view_name in PROFILE_VIEWS:
            results = self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=vector.tolist(),
                using=view_name,
                limit=limit,
                with_payload=True
            ).points
            self._add_scores(groups, results, view_name, per_full_view_weight)
        return self._rank_groups(groups, limit)

    def _search_region_face(self, region_vectors: dict, visible_regions: list, limit: int) -> list:
        groups = {}
        usable_regions = [region for region in REGION_NAMES if region in visible_regions and region in region_vectors]
        if not usable_regions:
            return []

        per_region_weight = 1.0 / (len(usable_regions) * len(PROFILE_VIEWS))
        for region in usable_regions:
            for view_name in PROFILE_VIEWS:
                channel = f"{view_name}_{region}"
                results = self.client.query_points(
                    collection_name=settings.COLLECTION_NAME,
                    query=region_vectors[region],
                    using=channel,
                    limit=limit,
                    with_payload=True
                ).points
                self._add_scores(groups, results, channel, per_region_weight)
        return self._rank_groups(groups, limit)

    def _search_occlusion_face(self, occlusion_vector: list, limit: int) -> list:
        groups = {}
        if not occlusion_vector:
            return []

        per_view_weight = 1.0 / len(PROFILE_VIEWS)
        for view_name in PROFILE_VIEWS:
            channel = f"{view_name}_occlusion"
            results = self.client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=occlusion_vector,
                using=channel,
                limit=limit,
                with_payload=True
            ).points
            self._add_scores(groups, results, channel, per_view_weight)
        return self._rank_groups(groups, limit)

    def _merge_stage_results(self, stages: list, limit: int) -> list:
        groups = {}
        for stage in stages:
            for result in stage["results"]:
                pid = result["payload"].get("face_id")
                if not pid:
                    continue
                if pid not in groups:
                    groups[pid] = {
                        "weighted_score": 0.0,
                        "weight": 0.0,
                        "payload": result["payload"],
                        "score_breakdown": [],
                        "stages_used": [],
                        "stage_scores": {}
                    }
                score = float(result["score"])
                groups[pid]["weighted_score"] += score * stage["weight"]
                groups[pid]["weight"] += stage["weight"]
                groups[pid]["stages_used"].append(stage["name"])
                groups[pid]["stage_scores"][stage["name"]] = score
                for item in result.get("score_breakdown", []):
                    enriched = dict(item)
                    enriched["stage"] = stage["name"]
                    enriched["stage_weight"] = round(stage["weight"], 4)
                    groups[pid]["score_breakdown"].append(enriched)

        ranked = sorted(
            groups.values(),
            key=lambda x: self._display_score(x),
            reverse=True
        )
        return [
            {
                "score": self._display_score(g),
                "fused_score": g["weighted_score"] / g["weight"] if g["weight"] else 0,
                "stage_scores": {
                    stage_name: round(score, 4)
                    for stage_name, score in g["stage_scores"].items()
                },
                "payload": g["payload"],
                "score_breakdown": sorted(g["score_breakdown"], key=lambda item: item["score"], reverse=True),
                "stages_used": sorted(set(g["stages_used"])),
            }
            for g in ranked[:limit]
        ]

    def _display_score(self, group: dict) -> float:
        fused_score = group["weighted_score"] / group["weight"] if group["weight"] else 0.0
        best_stage_score = max(group.get("stage_scores", {}).values(), default=0.0)
        stage_count = len(set(group.get("stages_used", [])))
        agreement_bonus = 0.02 if stage_count >= 2 else 0.0
        detail_lift = min(0.10, max(0.0, best_stage_score - fused_score) * 0.6)
        return round(min(1.0, fused_score + detail_lift + agreement_bonus), 4)

    def _confidence_reason(self, final_results: list, stages: list, quality: dict) -> str:
        if not final_results:
            return "No confident face match found."

        top_score = final_results[0]["score"]
        second_score = final_results[1]["score"] if len(final_results) > 1 else 0.0
        margin = top_score - second_score
        active_stage_names = [stage["name"] for stage in stages if stage["results"]]

        if quality and quality.get("occlusion_score", 0) >= 0.45:
            return "Occlusion is high, so confidence relies on visible regions and any occlusion-model signal."
        if margin < AMBIGUITY_MARGIN:
            return "Top candidates are close together; treat this as ambiguous."
        if "full_face" in active_stage_names and top_score >= FULL_SCORE_THRESHOLD:
            return "Full-face match is strong and separated from alternatives."
        if "region" in active_stage_names:
            return "Full-face evidence was weak or partial, so visible facial regions were used."
        return "Best available candidate from staged matching."

    def _accepted_results(self, final_results: list) -> list:
        if not final_results:
            return []

        top_result = final_results[0]
        top_score = float(top_result["score"])
        second_score = float(final_results[1]["score"]) if len(final_results) > 1 else 0.0
        margin = top_score - second_score

        if top_score < MIN_ACCEPTED_SCORE:
            return []

        if margin < MIN_ACCEPTED_MARGIN and top_score < MIN_WEAK_MARGIN_SCORE:
            return []

        if self._supporting_channel_count(top_result) < MIN_SUPPORTING_CHANNELS:
            return []

        if not self._has_full_stage_support(top_result):
            return []

        return [top_result]

    def _has_full_stage_support(self, result: dict) -> bool:
        full_score = float((result.get("stage_scores") or {}).get("full_face", 0.0))
        return full_score >= MIN_FULL_STAGE_SUPPORT_SCORE

    def _supporting_channel_count(self, result: dict) -> int:
        channels = {
            item.get("channel")
            for item in result.get("score_breakdown", [])
            if item.get("channel") and self._is_supporting_channel(item)
        }
        return len(channels)

    def _is_supporting_channel(self, score_item: dict) -> bool:
        score = float(score_item.get("score", 0.0))
        stage = score_item.get("stage")
        channel = score_item.get("channel", "")

        if stage == "region" or any(channel.endswith(f"_{region}") for region in REGION_NAMES):
            return score >= MIN_REGION_SUPPORTING_CHANNEL_SCORE
        if stage == "occlusion_model" or channel.endswith("_occlusion"):
            return score >= MIN_OCCLUSION_SUPPORTING_CHANNEL_SCORE
        return score >= MIN_FULL_SUPPORTING_CHANNEL_SCORE

    def staged_search_face(
        self,
        vector: np.ndarray,
        region_vectors: dict = None,
        visible_regions: list = None,
        occlusion_vector: list = None,
        occlusion_model_used: bool = False,
        quality: dict = None,
        limit: int = 5
    ) -> list:
        """Run production-style staged search: full face, conditional regions, optional occlusion model."""
        self._ensure_collection_exists()

        full_results = self._search_full_face(vector, limit)
        region_vectors = region_vectors or {}
        visible_regions = visible_regions or list(region_vectors.keys())
        top_full_score = full_results[0]["score"] if full_results else 0.0
        second_full_score = full_results[1]["score"] if len(full_results) > 1 else 0.0
        top_full_supporting_channels = self._supporting_channel_count(full_results[0]) if full_results else 0
        occlusion_score = float((quality or {}).get("occlusion_score", 0.0))

        needs_region_stage = (
            top_full_score < FULL_SCORE_THRESHOLD
            or occlusion_score >= 0.35
            or (top_full_score - second_full_score) < AMBIGUITY_MARGIN
            or top_full_supporting_channels < MIN_SUPPORTING_CHANNELS
        )

        region_results = self._search_region_face(region_vectors, visible_regions, limit) if needs_region_stage else []
        top_region_score = region_results[0]["score"] if region_results else 0.0
        needs_occlusion_stage = bool(
            occlusion_model_used
            and
            occlusion_vector
            and (
                occlusion_score >= 0.35
                or top_region_score < REGION_SCORE_THRESHOLD
                or top_full_score < FULL_SCORE_THRESHOLD
            )
        )
        occlusion_results = self._search_occlusion_face(occlusion_vector, limit) if needs_occlusion_stage else []

        stages = [{"name": "full_face", "weight": FULL_VECTOR_WEIGHT, "results": full_results}]
        if region_results:
            stages.append({"name": "region", "weight": REGION_VECTOR_WEIGHT, "results": region_results})
        if occlusion_results:
            stages.append({"name": "occlusion_model", "weight": OCCLUSION_VECTOR_WEIGHT, "results": occlusion_results})

        final_results = self._accepted_results(self._merge_stage_results(stages, limit))
        reason = self._confidence_reason(final_results, stages, quality or {})
        for result in final_results:
            result["recognition_stages"] = [stage["name"] for stage in stages if stage["results"]]
            result["confidence_reason"] = reason
            result["full_face_score"] = round(top_full_score, 4)
            result["region_stage_used"] = bool(region_results)
            result["occlusion_stage_used"] = bool(occlusion_results)
        return final_results

    def search_face(
        self,
        vector: np.ndarray,
        region_vectors: dict = None,
        visible_regions: list = None,
        limit: int = 5
    ) -> list:
        """Backward-compatible wrapper around staged search."""
        return self.staged_search_face(
            vector=vector,
            region_vectors=region_vectors,
            visible_regions=visible_regions,
            occlusion_model_used=False,
            limit=limit
        )

    def is_user_registered(self, name: str) -> bool:
        self._ensure_collection_exists()
        count_result = self.client.count(
            collection_name=settings.COLLECTION_NAME,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(key="name", match=models.MatchValue(value=name))
                ]
            )
        )
        return count_result.count > 0

    def delete_face_by_metadata(self, name: str):
        self._ensure_collection_exists()
        self.client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="name", match=models.MatchValue(value=name))
                    ]
                )
            )
        )

    def delete_face_by_id(self, face_id: str):
        self._ensure_collection_exists()
        self.client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=models.PointIdsList(points=[face_id])
        )

    def get_collection_info(self):
        self._ensure_collection_exists()
        return self.client.get_collection(settings.COLLECTION_NAME)

vector_db = VectorDBService()
