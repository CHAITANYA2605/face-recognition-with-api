from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


@pytest.fixture
def mock_face_service():
    with patch("app.api.routes.face_service") as mock:
        yield mock


@pytest.fixture
def mock_vector_db():
    with patch("app.api.routes.vector_db") as mock:
        yield mock


@pytest.fixture
def mock_user_api():
    with patch("app.api.routes._register_face_with_user_api") as mock:
        mock.return_value = SimpleNamespace(
            ok=True,
            status_code=200,
            content=b'{"success":true}',
            headers={"content-type": "application/json"},
        )
        yield mock


def _analysis_result():
    return SimpleNamespace(
        embedding=np.zeros(512),
        face_image="base64encodedstring",
        regions={
            "upper_face": [0.0] * 128,
            "lower_face": [0.0] * 128,
            "left_half": [0.0] * 128,
            "right_half": [0.0] * 128,
            "center_face": [0.0] * 128,
        },
        occlusion_embedding=[0.0] * 256,
        occlusion_model_used=False,
        visible_regions=["upper_face", "left_half", "right_half", "center_face"],
        quality={
            "occlusion_score": 0.2,
            "blur_score": 100.0,
            "landmarks_detected": True,
            "alignment_used": True,
            "occlusion_model_used": False,
        },
        fallback_used="original",
    )


def _registration_files(content_type="image/jpeg"):
    image_content = b"fake-image-content"
    return {
        "front_image": ("front.jpg", image_content, content_type),
        "left_image": ("left.jpg", image_content, content_type),
        "right_image": ("right.jpg", image_content, content_type),
    }


def test_register_face_success(mock_face_service, mock_vector_db, mock_user_api):
    mock_vector_db.is_user_registered.return_value = False
    mock_face_service.analyze_primary_face_details.return_value = _analysis_result()
    mock_vector_db.register_user.return_value = "test-uuid"

    response = client.post(
        "/api/v1/register",
        files=_registration_files(),
        data={"name": "John Doe", "age": 30},
        headers={"token": "incoming-token"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "test-uuid"
    assert response.json()["message"] == "Face registered successfully with front and 2 side profile images"
    assert response.json()["face_images"]["front"] == "base64encodedstring"
    mock_vector_db.register_user.assert_called_once()
    _, kwargs = mock_vector_db.register_user.call_args
    assert kwargs["metadata"]["age"] == 30
    assert "face_image" not in kwargs["metadata"]
    assert kwargs["metadata"]["face_images_stored"] is False
    assert "upper_face" in kwargs["region_vectors"]["front"]
    assert kwargs["occlusion_vectors"]["front"] is None
    assert kwargs["metadata"]["profile_occlusion_model_used"]["front"] is False
    assert kwargs["metadata"]["occlusion_model_used"] is False
    mock_user_api.assert_called_once_with(
        name="John Doe",
        face_id="test-uuid",
        token="incoming-token",
    )


def test_register_face_user_api_error_rolls_back(mock_face_service, mock_vector_db, mock_user_api):
    mock_vector_db.is_user_registered.return_value = False
    mock_face_service.analyze_primary_face_details.return_value = _analysis_result()
    mock_vector_db.register_user.return_value = "test-uuid"
    mock_user_api.return_value = SimpleNamespace(
        ok=False,
        status_code=422,
        content=b'{"message":"Invalid user mapping"}',
        headers={"content-type": "application/json"},
    )

    response = client.post(
        "/api/v1/register",
        files=_registration_files(),
        data={"name": "John Doe", "age": 30},
        headers={"token": "incoming-token"},
    )

    assert response.status_code == 422
    assert response.json() == {"message": "Invalid user mapping"}
    mock_vector_db.delete_face_by_id.assert_called_once_with("test-uuid")


def test_register_face_duplicate(mock_face_service, mock_vector_db):
    mock_vector_db.is_user_registered.return_value = True

    response = client.post(
        "/api/v1/register",
        files=_registration_files(),
        data={"name": "Jane Doe", "age": 25},
    )

    assert response.status_code == 400
    assert "already registered" in response.json()["detail"]
    mock_face_service.analyze_primary_face_details.assert_not_called()


def test_register_face_invalid_input(mock_vector_db):
    response = client.post(
        "/api/v1/register",
        files=_registration_files(),
        data={"name": "A", "age": 25},
    )
    assert response.status_code == 400
    assert "Name must be at least 2 characters" in response.json()["detail"]


def test_register_face_empty_body_returns_clean_error():
    response = client.post("/api/v1/register")

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            "Invalid registration request."
        )
    }


def test_register_face_missing_file_returns_clean_error():
    files = _registration_files()
    files.pop("right_image")

    response = client.post(
        "/api/v1/register",
        files=files,
        data={"name": "John Doe", "age": 30},
    )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Invalid registration request")
    assert isinstance(response.json()["detail"], str)


def test_recognize_face_success(mock_face_service, mock_vector_db):
    mock_face_service.analyze_all_faces.return_value = [_analysis_result()]
    mock_vector_db.staged_search_face.return_value = [
        {
            "score": 0.76,
            "payload": {
                "face_id": "test-uuid",
                "name": "John Doe",
                "age": 30,
                "face_image": "dbbase64string",
            },
            "score_breakdown": [{"channel": "front", "score": 0.8, "weight": 0.1933}],
            "recognition_stages": ["full_face"],
            "confidence_reason": "Full-face match is strong and separated from alternatives.",
        }
    ]

    response = client.post(
        "/api/v1/recognize",
        files={"file": ("test.jpg", b"fake-image-content", "image/jpeg")},
    )

    assert response.status_code == 200
    json_response = response.json()
    assert len(json_response["detections"]) == 1
    match = json_response["detections"][0]["results"][0]
    assert match["id"] == "test-uuid"
    assert match["score"] == 0.76
    assert match["metadata"]["name"] == "John Doe"
    assert match.get("face_image") in (None, "dbbase64string")
    assert match["match_quality"] == "high"
    assert match["recognition_stages"] == ["full_face"]
    assert "Full-face match" in match["confidence_reason"]
    assert json_response["detections"][0].get("occlusion_model_used") in (False, None)
    mock_vector_db.staged_search_face.assert_called_once()
    _, kwargs = mock_vector_db.staged_search_face.call_args
    assert kwargs["occlusion_model_used"] is False


def test_recognize_face_no_usable_face(mock_face_service, mock_vector_db):
    mock_face_service.analyze_all_faces.return_value = []

    response = client.post(
        "/api/v1/recognize",
        files={"file": ("test.jpg", b"fake-image-content", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["detections"] == []
    assert "No usable face detected" in response.json()["message"]
    mock_vector_db.staged_search_face.assert_not_called()


def test_register_face_no_face_detected(mock_face_service, mock_vector_db):
    mock_vector_db.is_user_registered.return_value = False
    mock_face_service.analyze_primary_face_details.return_value = None

    response = client.post(
        "/api/v1/register",
        files=_registration_files(),
        data={"name": "John Doe", "age": 30},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "No face detected in the front image"


def test_delete_face_success(mock_vector_db):
    mock_vector_db.is_user_registered.return_value = True

    response = client.delete("/api/v1/face?name=John%20Doe")

    assert response.status_code == 200
    assert response.json()["message"] == "Face(s) for user 'John Doe' deleted successfully"
    mock_vector_db.delete_face_by_metadata.assert_called_with("John Doe")


def test_delete_face_not_found(mock_vector_db):
    mock_vector_db.is_user_registered.return_value = False

    response = client.delete("/api/v1/face?name=John%20Doe")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_system_stats(mock_vector_db):
    mock_info = MagicMock()
    mock_info.vectors_count = 100
    mock_info.segments_count = 2
    mock_vector_db.get_collection_info.return_value = mock_info

    with patch("app.api.routes.request_tracker") as mock_tracker:
        mock_tracker.get_stats.return_value = {"/api/v1/test": {"total_requests": 10, "rpm": 5.0}}

        response = client.get("/api/v1/admin/stats")

        assert response.status_code == 200
        json_response = response.json()
        assert "memory_usage_mb" in json_response
        assert json_response["total_face_vectors"] == 100
        assert json_response["db_segments"] == 2
        assert json_response["api_performance"]["/api/v1/test"]["total_requests"] == 10
