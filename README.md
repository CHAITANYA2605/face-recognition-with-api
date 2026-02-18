# Face Recognition Software Using InsightFace & Qdrant

This project is a high-performance face recognition system that uses **InsightFace** (ArcFace/CosFace) for embedding generation and **Qdrant** as a vector database for efficient similarity search.

## Features

- **State-of-the-Art Accuracy**: Uses ArcFace methodology for robust face recognition.
- **Scalable Vector Search**: Leverages Qdrant for fast retrieval of millions of faces.
- **REST API**: Built with FastAPI for easy integration.
- **Rich Metadata**: Stores Name, Age, and Phone Number with every face.

## Prerequisites

- **Python 3.9+**
- **Docker & Docker Compose** (for the vector database)

## Quick Start (Local Machine)

### 1. Set up the Environment

1.  Clone the repository and navigate to the project root:
    ```bash
    cd face_recognition_app
    ```

2.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Start the Vector Database

Start Qdrant using Docker Compose:

```bash
docker-compose up -d
```
This will start Qdrant on `localhost:6333`.

### 3. Run the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### 4. Usage (API Documentation)

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

#### Register a New Face
- **Endpoint**: `POST /api/v1/register`
- **Parameters**:
    - `file`: Image file containing a face.
    - `name`: Name of the person.
    - `age`: Age of the person.
    - `phone_number`: Phone number.

#### Recognize a Face
- **Endpoint**: `POST /api/v1/recognize`
- **Parameters**:
    - `file`: Image file to recognize.
- **Response**: Returns matching faces with similarity scores and metadata (Name, Age, Phone).

## Server Deployment Guide

### Option 1: Docker (Recommended)

1.  **Create a `Dockerfile`** in the project root:
    ```dockerfile
    FROM python:3.9-slim

    WORKDIR /app

    # Install system dependencies for OpenCV
    RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    COPY . .

    EXPOSE 8000

    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    ```

2.  **Update `docker-compose.yml`** to include the app:
    ```yaml
    version: '3.8'
    services:
      qdrant:
        image: qdrant/qdrant
        ports:
          - "6333:6333"
        volumes:
          - ./qdrant_storage:/qdrant/storage

      api:
        build: .
        ports:
          - "8000:8000"
        depends_on:
          - qdrant
        environment:
          - QDRANT_HOST=qdrant
          - QDRANT_PORT=6333
    ```

3.  **Deploy**:
    ```bash
    docker-compose up -d --build
    ```

### Option 2: Manual Server Setup (Ubuntu/Debian)

1.  **Install Python & Pip**:
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip python3-venv libgl1-mesa-glx libglib2.0-0
    ```

2.  **Install Docker**: Follow official Docker installation guide.

3.  **Clone & Setup**:
    ```bash
    git clone <your-repo-url>
    cd face_recognition_app
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4.  **Run Qdrant**:
    ```bash
    docker run -d -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
    ```

5.  **Run with Gunicorn (Production)**:
    ```bash
    pip install gunicorn
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --daemon
    ```

## Testing

Run unit tests:
```bash
pytest tests/
```
