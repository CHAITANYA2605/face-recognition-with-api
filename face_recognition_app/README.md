# Face Recognition System

A production-grade face recognition API built with **InsightFace** (ArcFace + SCRFD) and **Qdrant** vector database. Supports multi-view registration, regional facial descriptors, occlusion-aware matching, and a staged multi-channel similarity search pipeline.

---

## Architecture Overview

```
                        ┌─────────────────────────────────┐
                        │          FastAPI Server           │
                        │                                   │
  Image Upload ────────►│  /register   /recognize  /delete  │
                        │                                   │
                        └────────────┬────────────┬─────────┘
                                     │            │
                        ┌────────────▼──┐  ┌──────▼──────────────┐
                        │ InsightFace   │  │   Qdrant Vector DB   │
                        │ (SCRFD-10G +  │  │                      │
                        │  ArcFace-R100)│  │  18 vectors/face:    │
                        │               │  │  • 3 full-face views │
                        │  512-dim      │  │  • 15 region vectors │
                        │  embeddings   │  │  • 3 occlusion vecs  │
                        └───────────────┘  └──────────────────────┘
```

**Multi-Channel Fusion Weights**

| Channel | Dimensions | Weight |
|---|---|---|
| Full-face (front/left/right) | 512 | 52% |
| Regional descriptors (5 zones × 3 views) | 128 each | 33% |
| Occlusion embedding | configurable | 15% |

---

## Features

- **Multi-view registration** — front + left profile + right profile for robust angle coverage
- **Regional face descriptors** — 5 facial zones (upper, lower, left half, right half, center) stored and searched independently to handle partial occlusion
- **Occlusion-aware search** — pluggable ONNX model detects visible regions at query time; only available channels are searched
- **Staged similarity pipeline** — configurable score thresholds, ambiguity margin checks, and minimum supporting-channel requirements before accepting a match
- **Match confidence levels** — each result is labeled `high`, `medium`, or `low` with a human-readable `confidence_reason`
- **HEIF/HEIC support** — accepts iPhone photos natively
- **Admin stats endpoint** — real-time memory usage, vector count, and per-route RPM tracking

---

## Prerequisites

- Python 3.9+
- Docker & Docker Compose

---

## Quick Start

### 1. Clone & install

```bash
cd face_recognition_app
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker-compose up -d
```

Qdrant REST API will be available at `localhost:6333`.

### 3. Run the server

```bash
uvicorn main:app --reload
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API Reference

### `POST /api/v1/register`

Register a new face with three views.

| Field | Type | Description |
|---|---|---|
| `front_image` | file | Front-facing photo |
| `left_image` | file | Left-profile photo |
| `right_image` | file | Right-profile photo |
| `name` | string | Person's name (min 2 chars, must be unique) |
| `age` | int | Person's age |

**Response**
```json
{
  "id": "uuid",
  "message": "Face registered successfully with front and 2 side profile images"
}
```

---

### `POST /api/v1/recognize`

Recognize one or more faces in an image.

| Field | Type | Description |
|---|---|---|
| `file` | file | Image file (JPEG, PNG, HEIF, etc.) |

**Response**
```json
{
  "detections": [
    {
      "results": [
        {
          "id": "uuid",
          "score": 0.8412,
          "fused_score": 0.8731,
          "match_quality": "high",
          "confidence_reason": "Strong match across 3 channels",
          "metadata": { "name": "John Doe", "age": 30 },
          "score_breakdown": [],
          "recognition_stages": []
        }
      ]
    }
  ]
}
```

---

### `DELETE /api/v1/face?name={name}`

Delete all stored vectors for a registered person.

---

### `GET /api/v1/admin/stats`

Returns memory usage, total face vectors in DB, and per-endpoint request stats.

```json
{
  "memory_usage_mb": 312.5,
  "total_face_vectors": 1024,
  "db_segments": 3,
  "api_performance": {
    "/api/v1/recognize": { "total_requests": 540, "rpm": 9.3 }
  }
}
```

---

### `GET /api/v1/face/{face_id}/view`

Returns the stored face crop image for a registered face ID (only available when `STORE_FACE_IMAGES_IN_DB=true`).

---

## Configuration

All settings are in [app/core/config.py](app/core/config.py) and can be overridden via environment variables.

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `STORE_FACE_IMAGES_IN_DB` | `false` | Store cropped face images in Qdrant payload |
| `OCCLUSION_MODEL_PATH` | `` | Path to custom ONNX occlusion model |
| `OCCLUSION_VECTOR_SIZE` | `256` | Output dimension of the occlusion model |
| `SUPPRESS_INSIGHTFACE_MODEL_LOGS` | `true` | Suppress verbose model download logs |
| `USER_LIST_API_TIMEOUT_SECONDS` | `100` | Timeout for downstream user registry API calls |

---

## Recognition Thresholds

Tuned via parameter sweep (see [scripts/recognize_parameter_sweep.py](scripts/recognize_parameter_sweep.py) and `recognize_score_sweep.xlsx`):

| Threshold | Value | Purpose |
|---|---|---|
| `FULL_SCORE_THRESHOLD` | 0.68 | Minimum full-face cosine similarity to proceed |
| `REGION_SCORE_THRESHOLD` | 0.55 | Minimum regional channel score |
| `MIN_ACCEPTED_SCORE` | 0.70 | Minimum fused score to accept a match |
| `MIN_ACCEPTED_MARGIN` | 0.04 | Minimum gap between top-1 and top-2 scores |
| `MIN_WEAK_MARGIN_SCORE` | 0.82 | Score above which margin check is bypassed |
| `MIN_SUPPORTING_CHANNELS` | 2 | Channels that must agree for a valid match |

---

## Project Structure

```
face_recognition_app/
├── main.py                      # FastAPI app entry point
├── docker-compose.yml           # Qdrant setup
├── requirements.txt
├── app/
│   ├── api/
│   │   └── routes.py            # All API endpoints
│   ├── core/
│   │   └── config.py            # Settings
│   ├── middleware/
│   │   └── stats.py             # RPM tracking middleware
│   ├── schemas/
│   │   └── face.py              # Pydantic request/response models
│   └── services/
│       ├── face_recognition.py  # InsightFace embedding + occlusion logic
│       └── vector_db.py         # Qdrant operations + staged search pipeline
├── scripts/
│   └── recognize_parameter_sweep.py   # Threshold tuning experiments
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_recognition.py
    └── test_vector_db.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker Deployment

Build and run everything with Docker:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t face-recognition .
docker run -p 8000:8000 \
  -e QDRANT_HOST=host.docker.internal \
  face-recognition
```

---

## Tech Stack

| Component | Library |
|---|---|
| API framework | FastAPI + Uvicorn |
| Face detection | InsightFace SCRFD-10G |
| Face recognition | InsightFace ArcFace-R100 |
| Vector database | Qdrant |
| Occlusion model | ONNX Runtime |
| Image processing | OpenCV, Pillow, pillow-heif |
| Testing | pytest, httpx |
