import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from app.services.face_recognition import FaceRecognitionService

def test_decode_image_opencv_success():
    service = FaceRecognitionService()
    
    # Mock cv2.imdecode to return a valid image
    # We need to mock numpy frombuffer as well in _decode_image because it's used before imdecode
    with patch("app.services.face_recognition.cv2.imdecode") as mock_imdecode:
        mock_imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = service._decode_image(b"fake_jpg_bytes")
        assert result is not None
        assert result.shape == (100, 100, 3)

def test_decode_image_heic_fallback():
    service = FaceRecognitionService()
    
    # Mock cv2.imdecode to return None (simulating failure/unsupported format)
    with patch("app.services.face_recognition.cv2.imdecode", return_value=None):
        # Mock PIL Image.open
        with patch("app.services.face_recognition.Image.open") as mock_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            # Simulate numpy array conversion of PIL image
            # We patch np.array but it's used inside the function
            # Alternatively mock the PIL Image object to support __array__
            
            # Better approach: mock the np.array call inside the module if possible, 
            # or allow the code to call np.array on the mock.
            # But simpler: make PIL image mock behave like array.
            
            # Let's just mock the whole process
            mock_open.return_value = mock_image
            
            # Mocking np.array in the module
            with patch("app.services.face_recognition.np.array", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
                 result = service._decode_image(b"fake_heic_bytes")
                 assert result is not None
                 # Verify usage of PIL
                 mock_open.assert_called_once()
