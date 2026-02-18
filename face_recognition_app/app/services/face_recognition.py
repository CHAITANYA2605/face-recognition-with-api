import numpy as np
import cv2
import base64
import io
from PIL import Image
import pillow_heif
from insightface.app import FaceAnalysis
from app.core.config import settings

# Register HEIF opener
pillow_heif.register_heif_opener()

class FaceRecognitionService:
    def __init__(self):
        # Initialize FaceAnalysis with the specified model pack
        # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] can be used if GPU is available
        # For now, we default to CPU to ensure it runs everywhere
        self.model_name = settings.DETECTION_MODEL
        self.app = None

    def _load_model(self):
        if self.app is None:
            self.app = FaceAnalysis(name=self.model_name)
            self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        # Try OpenCV first (faster)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        
        # Try Pillow (supports HEIC via pillow-heif)
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')
            img = np.array(image)
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception:
            return None

    def analyze_face(self, image_bytes: bytes):
        self._load_model()
        
        img = self._decode_image(image_bytes)
        
        if img is None:
            raise ValueError("Could not decode image")

        # Perform inference
        faces = self.app.get(img)
        
        if not faces:
            return None, None
            
        # Sort by size to get the largest face
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]
        
        # Crop the face (with some padding if needed, but bbox is usually tight)
        bbox = face.bbox.astype(int)
        # Ensure bounds
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
        face_crop = img[y1:y2, x1:x2]
        
        # Encode face crop to base64
        _, buffer = cv2.imencode('.jpg', face_crop)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return face.normed_embedding, face_base64

face_service = FaceRecognitionService()
