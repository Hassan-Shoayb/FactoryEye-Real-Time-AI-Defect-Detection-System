# FactoryEye — Real-Time AI Defect Detection System

> End-to-end computer vision + MLOps system that detects manufacturing defects using YOLO26, served via FastAPI, tracked with MLflow, and containerised with Docker.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![YOLO26](https://img.shields.io/badge/YOLO26-Ultralytics-00BFFF?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.13-0194E2?style=flat&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## What is FactoryEye?

FactoryEye is a production-grade defect detection system built for the manufacturing industry. It takes images or video frames of steel surfaces, runs them through a fine-tuned YOLO26 model, and returns bounding boxes around detected defects — with confidence scores, alert notifications, and full experiment tracking.

**The project is not just about detecting defects.** It is about building the infrastructure that a real factory would need:

- Model versioning and promotion via MLflow Model Registry
- A REST API that any downstream system can call
- Containerised deployment so it runs identically in dev and production
- Automated alerts when defect confidence exceeds a threshold
- A simple web UI for human review

---

## Demo

Upload a steel surface image to the web UI → FactoryEye returns an annotated image with bounding boxes and a JSON defect report in under 100ms on CPU.

```
POST /predict
→ {"detections": [{"label": "scratches", "confidence": 0.91, "bbox": [120, 45, 310, 180]}],
   "defect_count": 1, "defect_detected": true, "inference_ms": 38.4}
```

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Stack](#stack)
- [Dataset](#dataset)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Docker Deployment](#docker-deployment)
- [Running Tests](#running-tests)
- [Deploying to Production](#deploying-to-production)
- [Interview Talking Points](#interview-talking-points)
- [Roadmap](#roadmap)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Compose                        │
│                                                              │
│   ┌──────────────────┐          ┌────────────────────────┐  │
│   │   FastAPI (8000) │          │   MLflow UI (5000)     │  │
│   │                  │          │                        │  │
│   │  POST /predict   │          │  Experiment tracking   │  │
│   │  POST /predict-  │          │  Model registry        │  │
│   │        video     │          │  Artifact store        │  │
│   │  GET  /health    │          │                        │  │
│   └────────┬─────────┘          └────────────────────────┘  │
│            │                                                  │
│   ┌────────▼─────────┐                                       │
│   │  inference.py    │  ← loads best.pt once at startup      │
│   │  YOLO26 model    │  ← runs on CPU or GPU                 │
│   └────────┬─────────┘                                       │
│            │                                                  │
│   ┌────────▼─────────┐                                       │
│   │  alerts.py       │  ← Slack webhook, cooldown timer      │
│   └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘

Training pipeline (runs once, outside Docker):
  data/processed/ → training/train.py → MLflow run → Model Registry → best.pt
```

---

## Project Structure

```
defect-detection/
│
├── data/
│   ├── raw/                    # Original Roboflow download — never modified
│   ├── processed/              # Train/val/test split used by YOLO26
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── samples/                # Demo images and short test video
│
├── training/
│   ├── config.yaml             # Dataset paths + class names for YOLO26
│   ├── train.py                # Fine-tune YOLO26, log to MLflow, register model
│   ├── evaluate.py             # Validation metrics + per-class AP50
│   └── runs/                   # Auto-created by YOLO26 (gitignored)
│       └── train/weights/
│           ├── best.pt         # Weights used by the API
│           └── last.pt
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app — routes, CORS, startup
│   ├── inference.py            # Model loading + predict() function
│   ├── schemas.py              # Pydantic v2 request/response models
│   └── alerts.py               # Slack webhook with cooldown
│
├── mlflow_tracking/
│   ├── mlruns/                 # MLflow metadata (gitignored)
│   └── artifacts/              # Logged weights, confusion matrices (gitignored)
│
├── docker/
│   ├── Dockerfile              # Multi-stage build — builder + slim runtime
│   └── docker-compose.yml      # api (8000) + mlflow (5000) services
│
├── frontend/
│   └── index.html              # Upload UI — no framework, no build step
│
├── tests/
│   └── test_api.py             # pytest — health, predict, error handling
│
├── .env.example                # All environment variables documented
├── .gitignore
├── requirements.txt            # Pinned production dependencies
├── requirements-dev.txt        # Adds ruff, mypy, pytest
└── README.md
```

---

## Stack

| Layer | Tool | Version | Why |
|---|---|---|---|
| Detection model | Ultralytics YOLO26 | 8.4.x | Latest YOLO release — NMS-free inference, 43% faster than prior versions on CPU |
| API framework | FastAPI | 0.115 | Async, automatic OpenAPI docs, native Pydantic v2 support |
| Experiment tracking | MLflow | 3.13 | Model registry, run comparison, artifact store — industry standard for MLOps |
| Image processing | OpenCV | 4.10 | Frame decoding, annotation drawing, video I/O |
| Schema validation | Pydantic v2 | 2.13 | Typed request/response contracts, automatic validation |
| Containerisation | Docker + Compose | — | Reproducible deployment, single command to start everything |
| Testing | pytest + httpx | — | Async-compatible, FastAPI TestClient integration |
| Linting | ruff | 0.9 | Replaces flake8 + black in a single tool |

---

## Dataset

**NEU Surface Defect Database** — steel surface images with 6 defect classes:

| Class | Description |
|---|---|
| `crazing` | Network of fine cracks across the surface |
| `inclusion` | Foreign material embedded in the surface |
| `patches` | Irregular surface patches with different texture |
| `pitted_surface` | Small pits or holes |
| `rolled-in_scale` | Scale pressed into the surface during rolling |
| `scratches` | Linear surface scratches |

**Download**: [Roboflow Universe — NEU Surface Defect](https://universe.roboflow.com/steel-surface-defects/neu-surface-defect-database)

Select **YOLO26 format** when downloading. Unzip into `data/processed/`.

---

## Setup

### Prerequisites

- Python 3.12+
- Docker Desktop (for containerised deployment)
- 4GB RAM minimum (8GB recommended for training)
- GPU optional — CPU inference works fine for demo purposes

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/FactoryEye-Real-Time-AI-Defect-Detection-System.git
cd FactoryEye-Real-Time-AI-Defect-Detection-System
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For development (adds linter, type checker, test tools):

```bash
pip install -r requirements-dev.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set your values:

```env
MODEL_PATH=training/runs/train/weights/best.pt
CONFIDENCE_THRESHOLD=0.50
DEVICE=cpu

MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=defect-detection

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
ALERT_COOLDOWN_SECONDS=30

API_HOST=0.0.0.0
API_PORT=8000
```

> `SLACK_WEBHOOK_URL` is optional. Leave blank to disable alerts.

---

## Running the Project

Follow these steps in order the first time. After the model is trained, you only need steps 3 and 4.

### Step 1 — Download and prepare the dataset

1. Go to [Roboflow Universe](https://universe.roboflow.com) and search for **NEU Surface Defect Database**
2. Export in **YOLO26 format** → Download zip to computer
3. Unzip into `data/processed/` so the structure matches:

```
data/processed/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
  data.yaml
```

4. Open `data.yaml` from the download and copy the `path`, `names`, and split fields into `training/config.yaml`

### Step 2 — Start MLflow tracking server

Open a dedicated terminal and run:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri ./mlflow_tracking/mlruns \
  --default-artifact-root ./mlflow_tracking/artifacts
```

Keep this running. Open [http://localhost:5000](http://localhost:5000) — you should see an empty MLflow UI.

### Step 3 — Train the model

```bash
python training/train.py
```

This will:
- Fine-tune YOLO26n for 50 epochs on your dataset
- Log all hyperparameters and metrics to MLflow automatically
- Save best weights to `training/runs/train/weights/best.pt`
- Register the model in MLflow Model Registry if mAP50 ≥ 0.70

Expected output:
```
MLflow run ID: a3f2c9e1...
Epoch 1/50: loss=2.341, mAP50=0.142
...
Epoch 50/50: loss=0.612, mAP50=0.847
Model registered as version 1 (mAP50=0.847)
Training complete. Final mAP50: 0.8472
```

Training time: ~20 minutes on CPU, ~5 minutes with GPU.

### Step 4 — Evaluate the model

```bash
python training/evaluate.py --weights training/runs/train/weights/best.pt
```

Output example:
```
── Validation Results ──────────────────────────────
  mAP50:      0.8472
  mAP50-95:   0.6103
  Precision:  0.8891
  Recall:     0.8224

── Per-class AP50 ──────────────────────────────────
  Class 0 (crazing):         0.8821
  Class 1 (inclusion):       0.7934
  Class 2 (patches):         0.9012
  Class 3 (pitted_surface):  0.8445
  Class 4 (rolled-in_scale): 0.8109
  Class 5 (scratches):       0.9021
```

If mAP50 is below 0.70, try increasing epochs to 100 or switching to `yolov26s.pt` (small) instead of nano.

### Step 5 — Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify it is running:

```bash
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":true,"version":"1.0.0"}
```

Swagger docs available at [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 6 — Open the demo frontend

Open `frontend/index.html` directly in your browser. Upload any steel surface image and click **Run detection**.

---

## API Reference

### `GET /health`

Liveness check. Returns model load status.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true, "version": "1.0.0"}
```

---

### `POST /predict`

Upload a single image. Returns detections and a base64-encoded annotated image.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/samples/test_surface.jpg"
```

**Request:** `multipart/form-data` — field name `file`, accepts `image/jpeg` or `image/png`

**Response:**

```json
{
  "detections": [
    {
      "label": "scratches",
      "confidence": 0.9134,
      "bbox": [120, 45, 310, 180]
    }
  ],
  "defect_count": 1,
  "defect_detected": true,
  "inference_ms": 38.4,
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**bbox** format: `[x1, y1, x2, y2]` in pixel coordinates.

---

### `POST /predict-video`

Upload an MP4 or AVI video. Processes every 5th frame and returns a per-frame summary.

```bash
curl -X POST http://localhost:8000/predict-video \
  -F "file=@data/samples/test_video.mp4"
```

**Response:**

```json
{
  "total_frames_processed": 120,
  "defect_frames": 14,
  "defect_rate": 0.117,
  "frame_results": [
    {"frame": 0, "defect_count": 0, "detections": []},
    {"frame": 5, "defect_count": 1, "detections": [{"label": "crazing", ...}]}
  ]
}
```

---

## MLflow Experiment Tracking

Every training run is automatically logged. To compare runs:

1. Open [http://localhost:5000](http://localhost:5000)
2. Select the `defect-detection` experiment
3. Click **Compare** to view metrics side by side across runs
4. Under **Models**, view registered versions and their promotion status

**What gets logged per run:**

| Category | Values |
|---|---|
| Parameters | base_model, epochs, image_size, batch_size, learning_rate |
| Metrics | mAP50, mAP50-95, precision, recall |
| Artifacts | best.pt weights, training charts |
| Tags | Auto-tagged if model is promoted to registry |

**Model promotion logic** (in `training/train.py`):

The model is only registered in the MLflow Model Registry if `mAP50 ≥ 0.70`. This prevents accidentally deploying a degraded model. In production you would extend this to compare against the currently deployed version before promoting.

---

## Docker Deployment

Builds and starts both services with one command:

```bash
cd docker
docker compose up --build
```

Services:
- **API** → [http://localhost:8000](http://localhost:8000)
- **MLflow UI** → [http://localhost:5000](http://localhost:5000)

The trained weights in `training/runs/` are mounted as a read-only volume — you do not need to rebuild the Docker image after retraining.

To stop:

```bash
docker compose down
```

To rebuild after code changes:

```bash
docker compose up --build --force-recreate
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_api.py::test_health           PASSED
tests/test_api.py::test_predict_image    PASSED
tests/test_api.py::test_predict_rejects_non_image  PASSED

3 passed in 2.41s
```

Tests use FastAPI's `TestClient` — no server needs to be running.

---

## Deploying to Production

### Option A — Railway (recommended for demos)

1. Push your code to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your repo → Railway auto-detects the Dockerfile
4. Set environment variables in the Railway dashboard (copy from `.env`)
5. Deploy — you get a live URL in ~2 minutes

### Option B — Docker Hub + any cloud

```bash
# Build and push image
docker build -f docker/Dockerfile -t yourdockerhubname/factoryeye:latest .
docker push yourdockerhubname/factoryeye:latest

# Pull and run on any server
docker pull yourdockerhubname/factoryeye:latest
docker run -p 8000:8000 --env-file .env yourdockerhubname/factoryeye:latest
```

---

## Roadmap

- [ ] Automated retraining pipeline triggered by confidence score drift
- [ ] Prometheus metrics endpoint for production monitoring
- [ ] S3 artifact backend for MLflow (replacing local filesystem)
- [ ] CI/CD pipeline with GitHub Actions — run tests on every push
- [ ] A/B testing between model versions in production traffic
- [ ] Support for RTSP camera streams (IP cameras)

---

## License

MIT License — see `LICENSE` for details.

---
