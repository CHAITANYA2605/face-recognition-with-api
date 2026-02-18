from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from main import app # Import app instead of router
from app.services.face_recognition import face_service
from app.services.vector_db import vector_db

# Create a TestClient instance using the full app
client = TestClient(app)

@pytest.fixture
def mock_face_service():
    with patch("app.api.routes.face_service") as mock:
        yield mock

@pytest.fixture
def mock_vector_db():
    with patch("app.api.routes.vector_db") as mock:
        yield mock

def test_register_face_success(mock_face_service, mock_vector_db):
    # Mocking check for duplicate
    mock_vector_db.is_user_registered.return_value = False
    
    # Mocking the embedding generation and face crop
    mock_face_service.analyze_face.return_value = (np.zeros(512), "base64encodedstring")
    
    # Mocking the database insertion
    mock_vector_db.insert_face.return_value = "test-uuid"
    
    # Create a dummy image
    image_content = b"fake-image-content"
    files = {"file": ("test.jpg", image_content, "image/jpeg")}
    data = {
        "name": "John Doe",
        "age": 30,
        "phone_number": "1234567890"
    }
    
    response = client.post("/api/v1/register", files=files, data=data) # Note the prefix
    
    assert response.status_code == 200
    assert response.json() == {
        "id": "test-uuid",
        "message": "Face registered successfully",
        "face_image": "base64encodedstring"
    }

def test_register_face_duplicate(mock_face_service, mock_vector_db):
    # Mocking duplicate found
    mock_vector_db.is_user_registered.return_value = True
    
    image_content = b"fake-image-content"
    files = {"file": ("test.jpg", image_content, "image/jpeg")}
    data = {
        "name": "Jane Doe",
        "age": 25,
        "phone_number": "0987654321"
    }
    
    response = client.post("/api/v1/register", files=files, data=data)
    
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"]

def test_register_face_invalid_input(mock_vector_db):
    # Input validation happens before db check
    image_content = b"fake-image-content"
    files = {"file": ("test.jpg", image_content, "image/jpeg")}
    
    # Test short name
    response = client.post("/api/v1/register", files=files, data={
        "name": "A", "age": 25, "phone_number": "1234567890"
    })
    assert response.status_code == 400
    assert "Name must be at least 2 characters" in response.json()["detail"]

    # Test invalid phone (letters)
    response = client.post("/api/v1/register", files=files, data={
        "name": "Alice", "age": 25, "phone_number": "123abc4567"
    })
    assert response.status_code == 400
    assert "Phone number must be between 10 digits" in response.json()["detail"] # Message might need adjustment if I only check digits first

    # Test invalid phone length
    response = client.post("/api/v1/register", files=files, data={
        "name": "Alice", "age": 25, "phone_number": "123"
    })
    assert response.status_code == 400
    assert "Phone number must be between 10 digits" in response.json()["detail"]


def test_recognize_face_success(mock_face_service, mock_vector_db):
    # Mocking the embedding generation
    mock_face_service.analyze_face.return_value = (np.zeros(512), "querybase64")
    
    # Mocking the database search
    mock_match = MagicMock()
    mock_match.id = "test-uuid"
    mock_match.score = 0.95
    mock_match.payload = {
        "name": "John Doe", 
        "age": 30, 
        "phone_number": "1234567890",
        "filename": "test.jpg",
        "face_image": "dbbase64string"
    }
    mock_vector_db.search_face.return_value = [mock_match]
    
    # Create a dummy image
    image_content = b"fake-image-content"
    files = {"file": ("test.jpg", image_content, "image/jpeg")}
    
    response = client.post("/api/v1/recognize", files=files) # Note the prefix
    
    assert response.status_code == 200
    json_response = response.json()
    assert len(json_response["matches"]) == 1
    assert json_response["matches"][0]["id"] == "test-uuid"
    assert json_response["matches"][0]["score"] == 0.95
    assert json_response["matches"][0]["metadata"]["name"] == "John Doe"
    assert json_response["matches"][0]["face_image"] == "dbbase64string"


def test_register_face_no_face_detected(mock_face_service, mock_vector_db):
    # Mocking no duplicate
    mock_vector_db.is_user_registered.return_value = False
    
    # Mocking no face detected
    mock_face_service.analyze_face.return_value = (None, None)
    
    image_content = b"fake-image-content"
    files = {"file": ("test.jpg", image_content, "image/jpeg")}
    data = {
        "name": "John Doe",
        "age": 30,
        "phone_number": "1234567890"
    }
    
    response = client.post("/api/v1/register", files=files, data=data)
    
    assert response.status_code == 400
    assert response.json()["detail"] == "No face detected in the image"

def test_delete_face_success(mock_vector_db):
    # Mock user exists
    mock_vector_db.is_user_registered.return_value = True
    
    response = client.delete("/api/v1/face?name=John%20Doe&phone_number=1234567890")
    
    assert response.status_code == 200
    assert response.json()["message"] == "Face(s) for user 'John Doe' deleted successfully"
    mock_vector_db.delete_face_by_metadata.assert_called_with("John Doe", "1234567890")

def test_delete_face_not_found(mock_vector_db):
    # Mock user does not exist
    mock_vector_db.is_user_registered.return_value = False
    
    response = client.delete("/api/v1/face?name=John%20Doe&phone_number=1234567890")
    
def test_get_system_stats(mock_vector_db):
    # Mock DB info
    mock_info = MagicMock()
    mock_info.vectors_count = 100
    mock_info.segments_count = 2
    mock_vector_db.get_collection_info.return_value = mock_info
    
    # Mock request tracker
    with patch("app.api.routes.request_tracker") as mock_tracker:
        mock_tracker.get_stats.return_value = {"/api/v1/test": {"total_requests": 10, "rpm": 5.0}}
        
        response = client.get("/api/v1/admin/stats")
        
        assert response.status_code == 200
        json_response = response.json()
        assert "memory_usage_mb" in json_response
        assert json_response["total_face_vectors"] == 100
        assert json_response["db_segments"] == 2
        assert json_response["api_performance"]["/api/v1/test"]["total_requests"] == 10
