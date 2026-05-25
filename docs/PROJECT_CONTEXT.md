# Project Context

This project is a FastAPI face recognition service backed by InsightFace for full-face embeddings, landmark-aligned regional descriptors for partial-face matching, optional occlusion-aware ONNX embeddings, and Qdrant for named-vector search.

## Layout

| Path | Purpose |
| --- | --- |
| `face_recognition_app/main.py` | Creates the FastAPI app, configures CORS and stats middleware, includes API routes at `/api/v1`, and exposes `/health`. |
| `face_recognition_app/app/api/routes.py` | Main HTTP API: base64 decode, image viewing, registration, recognition, deletion, and admin stats. |
| `face_recognition_app/app/services/face_recognition.py` | Image decoding, robust fallback detection, landmark alignment, full-face embeddings, regional descriptors, optional occlusion-model embeddings, visibility, and quality metrics. |
| `face_recognition_app/app/services/vector_db.py` | Qdrant collection setup, full/region/occlusion vector storage, staged search, duplicate checks, deletion, and stats access. |
| `face_recognition_app/app/schemas/face.py` | Pydantic request/response models, including enriched recognition results and stage reasoning. |
| `face_recognition_app/app/middleware/stats.py` | Request counting and per-path RPM metrics. |
| `face_recognition_app/app/core/config.py` | Configuration constants and environment-driven Qdrant/model settings. |
| `face_recognition_app/tests/` | Pytest tests for current API behavior, image decoding/alignment, and staged vector search. |

## Current API Contract

| Method | Path | Input | Output |
| --- | --- | --- | --- |
| `GET` | `/health` | None | `{"status": "ok"}` |
| `POST` | `/api/v1/decode-image` | JSON body with `base64_string` | JPEG response bytes |
| `GET` | `/api/v1/face/{face_id}/view` | Path face ID | Stored front-face JPEG bytes when `STORE_FACE_IMAGES_IN_DB=true`; otherwise 404 |
| `POST` | `/api/v1/register` | Multipart fields: `front_image`, `left_image`, `right_image`, `name`, `age` | `FaceRegisterResponse` |
| `POST` | `/api/v1/recognize` | Multipart field: `file` | `FaceSearchResponse` with enriched `detections` |
| `DELETE` | `/api/v1/face` | Query params: `name` | `MessageResponse` |
| `GET` | `/api/v1/admin/stats` | None | Memory, Qdrant, and API request stats |

## Hybrid Recognition Pipeline

### Registration

1. `register_face` validates `name` and all three upload content types.
2. It checks duplicates through `vector_db.is_user_registered(name)`.
3. Each registration image goes through `face_service.analyze_primary_face_details`.
4. Face analysis tries the original image first, then fallback variants: upscaled, contrast-enhanced, sharpened, and rotated.
5. The detected face is landmark-aligned using the eye keypoints before region descriptors, quality metrics, and stored crop output are generated.
6. The largest detected face becomes a `FaceAnalysisResult` containing:
   - InsightFace full-face embedding.
   - Base64 aligned face crop.
   - Regional descriptors for `upper_face`, `lower_face`, `left_half`, `right_half`, and `center_face`.
   - Optional occlusion-aware ONNX embedding.
   - Visible region list.
   - Quality metrics such as blur, brightness, contrast, occlusion score, landmark presence, alignment use, and occlusion-model use.
7. `vector_db.register_user` stores one Qdrant point containing three full-face vectors, 15 region vectors, and three occlusion vectors. Occlusion vectors are real only when the optional model ran; otherwise they are stored as zero vectors to satisfy the named-vector schema.
8. Payload metadata stores `name`, `age`, profile quality, profile visible regions, `profile_occlusion_model_used`, top-level `occlusion_model_used`, and `face_images_stored`.
9. Base64 face images are not stored in Qdrant by default. Set `STORE_FACE_IMAGES_IN_DB=true` to store the front face image for `/face/{face_id}/view`.

### Recognition

1. `recognize_face` validates the uploaded query file is an image.
2. `face_service.analyze_all_faces` runs the robust fallback detector and returns all detected faces.
3. If no usable face is found, the route returns a 200 response with empty `detections` and a message explaining that the image may be occluded, blurred, dark, or angled.
4. For each detected face, `vector_db.staged_search_face` runs production-style staged recognition:
   - Stage 1: full-face embedding search against `front`, `left_profile`, and `right_profile`.
   - Stage 2: visible-region search only when full-face evidence is weak, occlusion is high, or top candidates are too close.
   - Stage 3: optional occlusion-model search only when the query actually produced an occlusion-model embedding and the case is weak, occluded, or ambiguous.
5. Stage scores are merged with `FULL_VECTOR_WEIGHT = 0.52`, `REGION_VECTOR_WEIGHT = 0.33`, and `OCCLUSION_VECTOR_WEIGHT = 0.15`.
6. Within each stage, the public stage score preserves the best matching channel so a strong front match is not dragged down by weaker side-profile channels. Conservative channel averages remain available in score breakdown data.
7. The final public `score` preserves the strongest supporting stage score, with a small agreement bonus when multiple stages support the same person. The conservative weighted average is returned separately as `fused_score`.
8. Each `FaceMatch` includes score, fused score, stage scores, payload metadata, optional stored image, match quality, score breakdown, recognition stages, and confidence reason.
9. Each `FaceDetection` includes the query aligned crop, visible regions, quality metrics, fallback used, occlusion-model flag, and a best-effort or confidence message.

### Deletion

1. `delete_face` checks whether the name exists.
2. `vector_db.delete_face_by_metadata` deletes all matching Qdrant points.

### Stats

1. `StatsMiddleware.dispatch` records `/api/v1` request paths.
2. `get_system_stats` reads process memory through `resource.getrusage`.
3. It reads Qdrant collection info through `vector_db.get_collection_info`.
4. It reads request counts and RPM through `request_tracker.get_stats`.

## Vector Schema

Qdrant defaults to `localhost:6333`. Override with `QDRANT_HOST` and `QDRANT_PORT`.

The configured collection is `faces-4.0`.

| Vector channel | Size | Meaning |
| --- | --- | --- |
| `front`, `left_profile`, `right_profile` | 512 | InsightFace full-face embeddings. |
| `{view}_upper_face` | 128 | Descriptor for upper face area from landmark-aligned crop. |
| `{view}_lower_face` | 128 | Descriptor for lower face area from landmark-aligned crop. |
| `{view}_left_half` | 128 | Descriptor for left half of aligned face crop. |
| `{view}_right_half` | 128 | Descriptor for right half of aligned face crop. |
| `{view}_center_face` | 128 | Descriptor for central face area from aligned crop. |
| `{view}_occlusion` | `OCCLUSION_VECTOR_SIZE`, default 256 | Optional ONNX occlusion-aware embedding, or zero vector when no model is configured. |

The regional descriptors are deterministic image descriptors, not separate neural embeddings. The occlusion channel is designed for a real trained ONNX model configured through `OCCLUSION_MODEL_PATH`.

Image payload storage is off by default with `STORE_FACE_IMAGES_IN_DB=false`. This keeps Qdrant records much smaller. Existing oversized records must be deleted/re-registered or migrated to benefit from the lean payload.

## Optional Occlusion-Aware Model

Set `OCCLUSION_MODEL_PATH` to a local ONNX model path. The service will:

1. Load it with `onnxruntime`.
2. Feed a landmark-aligned 112x112 RGB crop normalized to `[-1, 1]`.
3. Normalize the first model output.
4. Pad or trim it to `OCCLUSION_VECTOR_SIZE`.
5. Store/search it through `{view}_occlusion` vectors and mark `profile_occlusion_model_used` in the Qdrant payload.

If no model path is configured or the file does not exist, the system stores zero occlusion vectors, sets `occlusion_model_used` to false, and skips occlusion-stage search.

## Dependencies and Runtime Assumptions

| Dependency | Used for |
| --- | --- |
| `fastapi`, `uvicorn`, `starlette` | HTTP server, routing, middleware, test client integration. |
| `insightface` | Face detection, landmarks, and full-face recognition embeddings. |
| `onnxruntime` | Optional occlusion-aware ONNX embedding model. |
| `opencv-python` / `cv2` | Image byte decoding, preprocessing fallbacks, landmark alignment, cropping, JPEG encoding, regional descriptors, and quality metrics. |
| `Pillow`, `pillow-heif` | Fallback image decoding, especially HEIC/HEIF. |
| `numpy` | Image buffers, embeddings, alignment, and descriptor math. |
| `qdrant-client` | Named-vector storage and similarity search. |

InsightFace is configured with `INSIGHTFACE_PROVIDERS = ["CPUExecutionProvider"]`. This is the safest default on macOS and avoids ONNX Runtime trying CUDA, which is not available on typical Mac hardware.

InsightFace's third-party model inventory prints are suppressed by default with `SUPPRESS_INSIGHTFACE_MODEL_LOGS=true`. Set it to `false` when debugging model loading.

## Known Sharp Edges

| Area | Note |
| --- | --- |
| Existing data | The new schema uses collection `faces-4.0`; users registered into earlier collections must be re-registered or migrated. |
| Existing oversized records | Turning off image payload storage only affects new registrations. Old records that already contain base64 images must be deleted/re-registered or migrated. |
| Occlusion model | The code supports a real ONNX occlusion model, but the repo does not include model weights. Configure `OCCLUSION_MODEL_PATH` when a trained model is available. |
| Regional descriptors | These are handcrafted descriptors for partial-face support. They are useful fallback signals, but should be calibrated against real validation data. |
| Detector limits | The fallback pipeline helps with difficult images, but fully hidden or very low-quality faces may still produce no detection. |
| Duplicate checking | Duplicate identity is exact-match `name`; no normalization is applied beyond route validation. |
| `admin_routes_snippet.py` | Appears to be unused reference code and is not registered with the app. |
| Model loading | InsightFace loads lazily on first face analysis using CPU-only ONNX Runtime. First recognition/register call may be slower and may require model files. Model-load stdout/stderr chatter is suppressed by default. |
| Qdrant availability | Most vector service calls ensure collection existence and will fail if Qdrant is unavailable. Tests mock Qdrant-facing services where appropriate. |

## Good Starting Points For Future Tasks

| Task type | Start here |
| --- | --- |
| Add or change API behavior | `face_recognition_app/app/api/routes.py`, then update schemas in `app/schemas/face.py`. |
| Change detection, fallback preprocessing, alignment, quality, descriptors, or occlusion ONNX preprocessing | `face_recognition_app/app/services/face_recognition.py`. |
| Change storage, vector schema, stage thresholds, fusion weights, or staged search behavior | `face_recognition_app/app/services/vector_db.py`. |
| Add a specific occlusion-aware model | Set `OCCLUSION_MODEL_PATH`, confirm input/output shape, then adjust `_occlusion_embedding` preprocessing if needed. |
| Add observability | `face_recognition_app/app/middleware/stats.py` and `get_system_stats`. |
| Change runtime config | `face_recognition_app/app/core/config.py` and environment variables. |

## Suggested Cleanup Backlog

1. Normalize phone/name values before duplicate checks if user input may vary.
2. Either wire, remove, or archive `app/api/admin_routes_snippet.py`.
3. Add route-level tests for `decode_base64_image` and `view_face_image`.
4. Calibrate score thresholds and fusion weights with a real validation set.
5. Add a migration or admin utility if old collection data must be moved into `faces-4.0`.
