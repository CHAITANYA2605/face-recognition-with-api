from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from app.services.vector_db import VectorDBService


def _point(face_id, score):
    return SimpleNamespace(
        id=face_id,
        score=score,
        payload={"face_id": face_id, "name": f"User {face_id}"},
    )


def test_staged_search_uses_only_full_face_for_strong_unambiguous_match():
    service = VectorDBService()
    service.collection_checked = True
    service.client = MagicMock()
    service.client.query_points.side_effect = [
        SimpleNamespace(points=[_point("a", 0.9), _point("b", 0.5)]),
        SimpleNamespace(points=[_point("a", 0.88), _point("b", 0.48)]),
        SimpleNamespace(points=[_point("a", 0.89), _point("b", 0.47)]),
    ]

    results = service.staged_search_face(
        np.zeros(512),
        region_vectors={"upper_face": [0.0] * 128},
        visible_regions=["upper_face"],
        quality={"occlusion_score": 0.1},
    )

    assert results[0]["payload"]["face_id"] == "a"
    assert len(results) == 1
    assert results[0]["recognition_stages"] == ["full_face"]
    assert results[0]["region_stage_used"] is False
    assert results[0]["score"] >= 0.9
    assert results[0]["stage_scores"]["full_face"] >= 0.9
    assert service.client.query_points.call_count == 3


def test_staged_search_returns_only_the_accepted_top_match():
    service = VectorDBService()
    service.collection_checked = True
    service.client = MagicMock()
    service.client.query_points.side_effect = [
        SimpleNamespace(points=[_point("a", 0.9), _point("b", 0.74), _point("c", 0.71)]),
        SimpleNamespace(points=[_point("a", 0.88), _point("b", 0.52), _point("c", 0.51)]),
        SimpleNamespace(points=[_point("a", 0.89), _point("b", 0.5), _point("c", 0.49)]),
    ]

    results = service.staged_search_face(
        np.zeros(512),
        region_vectors={"upper_face": [0.0] * 128},
        visible_regions=["upper_face"],
        quality={"occlusion_score": 0.1},
        limit=5,
    )

    assert [result["payload"]["face_id"] for result in results] == ["a"]


def test_staged_search_returns_no_result_for_single_lucky_channel():
    service = VectorDBService()
    service.collection_checked = True
    service.client = MagicMock()
    service.client.query_points.side_effect = [
        SimpleNamespace(points=[_point("a", 0.95), _point("b", 0.5)]),
        SimpleNamespace(points=[_point("a", 0.31), _point("b", 0.3)]),
        SimpleNamespace(points=[_point("a", 0.3), _point("b", 0.29)]),
        SimpleNamespace(points=[_point("a", 0.42), _point("b", 0.41)]),
        SimpleNamespace(points=[_point("a", 0.4), _point("b", 0.39)]),
        SimpleNamespace(points=[_point("a", 0.41), _point("b", 0.4)]),
    ]

    results = service.staged_search_face(
        np.zeros(512),
        region_vectors={"upper_face": [0.0] * 128},
        visible_regions=["upper_face"],
        quality={"occlusion_score": 0.1},
    )

    assert results == []


def test_staged_search_adds_region_and_occlusion_when_full_face_is_weak():
    service = VectorDBService()
    service.collection_checked = True
    service.client = MagicMock()
    service.client.query_points.side_effect = [
        SimpleNamespace(points=[_point("a", 0.52), _point("b", 0.51)]),
        SimpleNamespace(points=[_point("a", 0.5), _point("b", 0.49)]),
        SimpleNamespace(points=[_point("a", 0.53), _point("b", 0.52)]),
        SimpleNamespace(points=[_point("a", 0.72), _point("b", 0.45)]),
        SimpleNamespace(points=[_point("a", 0.7), _point("b", 0.44)]),
        SimpleNamespace(points=[_point("a", 0.71), _point("b", 0.43)]),
        SimpleNamespace(points=[_point("a", 0.74), _point("b", 0.41)]),
        SimpleNamespace(points=[_point("a", 0.75), _point("b", 0.42)]),
        SimpleNamespace(points=[_point("a", 0.76), _point("b", 0.43)]),
    ]

    results = service.staged_search_face(
        np.zeros(512),
        region_vectors={"upper_face": [0.0] * 128},
        visible_regions=["upper_face"],
        occlusion_vector=[0.0] * 256,
        occlusion_model_used=True,
        quality={"occlusion_score": 0.6},
    )

    assert results[0]["payload"]["face_id"] == "a"
    assert results[0]["region_stage_used"] is True
    assert results[0]["occlusion_stage_used"] is True
    assert set(results[0]["recognition_stages"]) == {"full_face", "region", "occlusion_model"}
    assert results[0]["score"] > results[0]["fused_score"]
    assert results[0]["score"] < max(results[0]["stage_scores"].values())


def test_staged_search_skips_occlusion_stage_when_model_was_not_used():
    service = VectorDBService()
    service.collection_checked = True
    service.client = MagicMock()
    service.client.query_points.side_effect = [
        SimpleNamespace(points=[_point("a", 0.52), _point("b", 0.51)]),
        SimpleNamespace(points=[_point("a", 0.5), _point("b", 0.49)]),
        SimpleNamespace(points=[_point("a", 0.53), _point("b", 0.52)]),
        SimpleNamespace(points=[_point("a", 0.72), _point("b", 0.45)]),
        SimpleNamespace(points=[_point("a", 0.7), _point("b", 0.44)]),
        SimpleNamespace(points=[_point("a", 0.71), _point("b", 0.43)]),
    ]

    results = service.staged_search_face(
        np.zeros(512),
        region_vectors={"upper_face": [0.0] * 128},
        visible_regions=["upper_face"],
        occlusion_vector=[0.0] * 256,
        occlusion_model_used=False,
        quality={"occlusion_score": 0.6},
    )

    assert results[0]["region_stage_used"] is True
    assert results[0]["occlusion_stage_used"] is False
    assert set(results[0]["recognition_stages"]) == {"full_face", "region"}
    assert service.client.query_points.call_count == 6
