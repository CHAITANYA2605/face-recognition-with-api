import numpy as np
import cv2
import base64
import io
import contextlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import pillow_heif
from insightface.app import FaceAnalysis
from app.core.config import settings

try:
    import onnxruntime as ort
except Exception:
    ort = None

# Register HEIF opener
pillow_heif.register_heif_opener()

REGION_NAMES = ("upper_face", "lower_face", "left_half", "right_half", "center_face")
REGION_VECTOR_SIZE = 128


@dataclass
class FaceAnalysisResult:
    embedding: np.ndarray
    face_image: str
    bbox: list
    detection_confidence: float
    regions: dict
    visible_regions: list
    quality: dict
    occlusion_embedding: list = None
    occlusion_model_used: bool = False
    fallback_used: str = "original"


class FaceRecognitionService:
    def __init__(self):
        # Initialize FaceAnalysis with the specified model pack
        # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] can be used if GPU is available
        # For now, we default to CPU to ensure it runs everywhere
        self.model_name = settings.DETECTION_MODEL
        self.app = None
        self.occlusion_session = None
        self.occlusion_input_name = None
        print(f"FaceRecognitionService initialized with model: {self.model_name}")

    def _load_model(self):
        if self.app is None:
            with self._insightface_log_context():
                self.app = FaceAnalysis(name=self.model_name, providers=settings.INSIGHTFACE_PROVIDERS)
                self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _insightface_log_context(self):
        warning_context = warnings.catch_warnings()
        warning_context.__enter__()
        warnings.filterwarnings(
            "ignore",
            message="`estimate` is deprecated.*",
            category=FutureWarning,
            module="insightface.utils.face_align",
        )

        if settings.SUPPRESS_INSIGHTFACE_MODEL_LOGS:
            stdout_context = contextlib.redirect_stdout(io.StringIO())
            stderr_context = contextlib.redirect_stderr(io.StringIO())
            stdout_context.__enter__()
            stderr_context.__enter__()
        else:
            stdout_context = None
            stderr_context = None

        @contextlib.contextmanager
        def manager():
            try:
                yield
            finally:
                if stderr_context is not None:
                    stderr_context.__exit__(None, None, None)
                if stdout_context is not None:
                    stdout_context.__exit__(None, None, None)
                warning_context.__exit__(None, None, None)

        return manager()

    def _load_occlusion_model(self):
        if not settings.OCCLUSION_MODEL_PATH or ort is None:
            return None

        model_path = Path(settings.OCCLUSION_MODEL_PATH)
        if not model_path.exists():
            return None

        if self.occlusion_session is None:
            self.occlusion_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.occlusion_input_name = self.occlusion_session.get_inputs()[0].name
        return self.occlusion_session

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

    def _image_variants(self, img: np.ndarray):
        yield "original", img

        largest_side = max(img.shape[:2])
        if largest_side < 1000:
            scale = 1000 / largest_side
            upscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            yield "upscaled", upscaled

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
        yield "contrast_enhanced", cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        yield "sharpened", cv2.filter2D(img, -1, sharpen_kernel)

        for angle in (-15, 15):
            yield f"rotated_{angle}", self._rotate_image(img, angle)

    def _rotate_image(self, img: np.ndarray, angle: int) -> np.ndarray:
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _detect_faces_with_fallbacks(self, img: np.ndarray):
        for variant_name, variant in self._image_variants(img):
            with self._insightface_log_context():
                faces = self.app.get(variant)
            if faces:
                return variant_name, variant, faces
        return "none", img, []

    def _encode_crop(self, crop: np.ndarray) -> str:
        if crop.size == 0:
            return ""
        _, buffer = cv2.imencode(".jpg", crop)
        return base64.b64encode(buffer).decode("utf-8")

    def _crop_face(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox.astype(int)
        height, width = img.shape[:2]
        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.08)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        return img[y1:y2, x1:x2]

    def _region_crops(self, face_crop: np.ndarray) -> dict:
        height, width = face_crop.shape[:2]
        if height == 0 or width == 0:
            return {}

        return {
            "upper_face": face_crop[0:int(height * 0.58), :],
            "lower_face": face_crop[int(height * 0.42):height, :],
            "left_half": face_crop[:, 0:int(width * 0.62)],
            "right_half": face_crop[:, int(width * 0.38):width],
            "center_face": face_crop[int(height * 0.18):int(height * 0.82), int(width * 0.18):int(width * 0.82)],
        }

    def _aligned_face_crop(self, img: np.ndarray, face) -> np.ndarray:
        if not hasattr(face, "kps") or face.kps is None or len(face.kps) < 2:
            return self._crop_face(img, face.bbox)

        keypoints = np.asarray(face.kps, dtype=np.float32)
        left_eye, right_eye = keypoints[0], keypoints[1]
        eye_center = ((left_eye + right_eye) / 2.0).astype(np.float32)
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        angle = np.degrees(np.arctan2(dy, dx))

        desired_width = 160
        desired_height = 192
        desired_left_eye = (0.35, 0.38)
        desired_right_eye_x = 1.0 - desired_left_eye[0]
        dist = np.sqrt((dx ** 2) + (dy ** 2))
        desired_dist = (desired_right_eye_x - desired_left_eye[0]) * desired_width
        scale = desired_dist / dist if dist > 0 else 1.0

        matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)
        matrix[0, 2] += desired_width * 0.5 - eye_center[0]
        matrix[1, 2] += desired_height * desired_left_eye[1] - eye_center[1]

        return cv2.warpAffine(
            img,
            matrix,
            (desired_width, desired_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _region_descriptor(self, crop: np.ndarray) -> list:
        if crop.size == 0:
            return [0.0] * REGION_VECTOR_SIZE

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray)
        descriptor = gray.astype(np.float32).reshape(-1)
        descriptor = descriptor - float(descriptor.mean())
        norm = float(np.linalg.norm(descriptor))
        if norm > 0:
            descriptor = descriptor / norm

        if descriptor.size >= REGION_VECTOR_SIZE:
            descriptor = descriptor[:REGION_VECTOR_SIZE]
        else:
            descriptor = np.pad(descriptor, (0, REGION_VECTOR_SIZE - descriptor.size))
        return descriptor.astype(float).tolist()

    def _quality_metrics(self, crop: np.ndarray, face=None) -> dict:
        if crop.size == 0:
            return {"blur_score": 0.0, "brightness": 0.0, "contrast": 0.0, "occlusion_score": 1.0}

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        dark_ratio = float((gray < 35).mean())
        bright_ratio = float((gray > 245).mean())
        occlusion_score = min(1.0, dark_ratio + bright_ratio + (0.25 if contrast < 18 else 0.0))

        quality = {
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "occlusion_score": round(occlusion_score, 3),
        }
        if face is not None and hasattr(face, "kps"):
            quality["landmarks_detected"] = bool(face.kps is not None)
        return quality

    def _occlusion_embedding(self, aligned_crop: np.ndarray) -> tuple:
        session = self._load_occlusion_model()
        if session is None or aligned_crop.size == 0:
            return [0.0] * settings.OCCLUSION_VECTOR_SIZE, False

        model_input = cv2.resize(aligned_crop, (112, 112), interpolation=cv2.INTER_AREA)
        model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB).astype(np.float32)
        model_input = (model_input - 127.5) / 128.0
        model_input = np.transpose(model_input, (2, 0, 1))[None, ...]

        outputs = session.run(None, {self.occlusion_input_name: model_input})
        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm

        if embedding.size >= settings.OCCLUSION_VECTOR_SIZE:
            embedding = embedding[:settings.OCCLUSION_VECTOR_SIZE]
        else:
            embedding = np.pad(embedding, (0, settings.OCCLUSION_VECTOR_SIZE - embedding.size))
        return embedding.astype(float).tolist(), True

    def _visible_regions(self, region_crops: dict) -> list:
        visible = []
        for name, crop in region_crops.items():
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            contrast = float(gray.std())
            dark_ratio = float((gray < 35).mean())
            bright_ratio = float((gray > 245).mean())
            if contrast >= 12 and dark_ratio < 0.65 and bright_ratio < 0.65:
                visible.append(name)
        return visible

    def _build_result(self, face, img: np.ndarray, fallback_used: str) -> FaceAnalysisResult:
        bbox = face.bbox.astype(int)
        raw_face_crop = self._crop_face(img, bbox)
        aligned_face_crop = self._aligned_face_crop(img, face)
        region_crops = self._region_crops(aligned_face_crop)
        regions = {
            name: self._region_descriptor(crop)
            for name, crop in region_crops.items()
        }
        visible_regions = self._visible_regions(region_crops)
        quality = self._quality_metrics(aligned_face_crop, face)
        detection_confidence = float(getattr(face, "det_score", 0.0) or 0.0)
        occlusion_embedding, occlusion_model_used = self._occlusion_embedding(aligned_face_crop)
        quality["alignment_used"] = bool(hasattr(face, "kps") and face.kps is not None)
        quality["raw_crop_available"] = bool(raw_face_crop.size)
        quality["occlusion_model_used"] = occlusion_model_used

        return FaceAnalysisResult(
            embedding=face.normed_embedding,
            face_image=self._encode_crop(aligned_face_crop),
            bbox=[int(v) for v in bbox.tolist()],
            detection_confidence=round(detection_confidence, 4),
            regions=regions,
            visible_regions=visible_regions,
            quality=quality,
            occlusion_embedding=occlusion_embedding,
            occlusion_model_used=occlusion_model_used,
            fallback_used=fallback_used,
        )

    def analyze_face(self, image_bytes: bytes):
        """Analyze the largest face in the image (Legacy/Simple)."""
        faces = self.analyze_all_faces(image_bytes)
        if not faces:
            return None, None
        
        # Sort by size and return the largest (it's already sorted in analyze_all_faces)
        return faces[0].embedding, faces[0].face_image

    def analyze_primary_face_details(self, image_bytes: bytes):
        faces = self.analyze_all_faces(image_bytes)
        return faces[0] if faces else None

    def analyze_all_faces(self, image_bytes: bytes):
        """Analyze all faces detected in the image."""
        self._load_model()
        
        img = self._decode_image(image_bytes)
        
        if img is None:
            raise ValueError("Could not decode image")

        fallback_used, detection_img, faces = self._detect_faces_with_fallbacks(img)
        
        if not faces:
            return []
            
        # Sort by size to get larger faces first
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        
        results = []
        for face in faces:
            results.append(self._build_result(face, detection_img, fallback_used))
        
        return results

face_service = FaceRecognitionService()
