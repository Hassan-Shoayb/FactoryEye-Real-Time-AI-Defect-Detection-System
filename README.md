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
│   ├── main.py                 # FastAPI app — routes, WebSocket, CORS & UI mount
│   ├── inference.py            # Multi-backend engine (PyTorch .pt + ONNX Runtime)
│   ├── explainability.py       # Saliency attention heatmaps & jet colormap overlays (/explain)
│   ├── severity.py             # Defect severity grading & QA action recommendation engine
│   ├── certificate.py          # ISO Metallurgical quality inspection certificate generator (/audit/certificate)
│   ├── canary.py               # Production A/B Canary traffic router, live SLA comparative telemetry & promotion engine
│   ├── retrain.py              # Automated continuous retraining pipeline, SLA gate verification & canary deployment
│   ├── rca.py                  # Automated Root Cause Analysis (RCA), defect spatial clustering & maintenance dispatch
│   ├── ledger.py               # Cryptographic tamper-evident quality audit ledger, SHA-256 Merkle root engine & HMAC digital seal
│   ├── digital_twin.py         # Coil digital twin 3D/2.5D surface topology & automated flying shear-cut plan optimizer
│   ├── anomaly_detector.py     # Zero-shot edge anomaly detector, dual-domain FFT spectral residual & novel flaw discovery
│   ├── schemas.py              # Pydantic v2 request/response contracts
│   ├── alerts.py               # Slack webhook alerting with debouncing
│   ├── metrics.py              # Prometheus latency histogram & defect metrics (/metrics)
│   ├── drift.py                # Rolling confidence drift monitor & Active Learning queue
│   ├── mqtt_publisher.py       # Industrial MQTT telemetry for PLC pneumatic reject arms
│   ├── database.py             # SQLite QA defect audit log, ROI crops, Pareto trends & multi-station telemetry (/audit/defects, /audit/defects/{id}/crop, /audit/stations)
│   ├── rtsp_stream.py          # Multi-threaded zero-lag RTSP camera ingestion worker
│   └── config.py               # Environment configuration & dynamic device resolver
│
├── scripts/
│   ├── model_gate.py           # Champion vs Challenger model governance & regression gating
│   ├── benchmark_export.py     # Edge latency benchmarker & ONNX optimization exporter
│   ├── batch_predict.py        # High-throughput offline batch folder inference CLI
│   ├── generate_synthetic.py   # Procedural synthetic metal defect generator
│   └── stream_simulator.py     # Industrial camera video stream simulator
│
├── .github/
│   └── workflows/
│       └── ci.yml              # Automated GitHub Actions CI & model validation
│
├── docker/
│   ├── Dockerfile              # Multi-stage build — builder + slim runtime
│   └── docker-compose.yml      # api (8000) + mlflow (5000) services
│
├── frontend/
│   └── index.html              # Modern dark-mode UI with Image, Stream & QA Audit tabs
│
├── tests/
│   └── test_api.py             # pytest suite — health, predict, metrics, drift & audit tests
│
├── Makefile                    # Developer task runner (run, test, benchmark, gate, batch)
├── .env.example                # All environment variables documented
├── .gitignore
├── requirements.txt            # Pinned production dependencies
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

### `GET /active-learning/queue`

Fetches candidate ambiguous defect samples ($0.30 \le \text{confidence} \le 0.55$) automatically sequestered during live factory inspection.

```bash
curl http://localhost:8000/active-learning/queue?limit=50
```

**Response:**
```json
[
  {
    "filename": "sample_1788814279327_conf_48.jpg",
    "filepath": "/path/to/data/active_learning_queue/sample_1788814279327_conf_48.jpg",
    "confidence_estimate": 0.48,
    "timestamp_utc": 1788814279.327
  }
]
```

---

### `POST /active-learning/review`

Submits human verification decisions (`approve`, `relabel`, `discard`). Approved and relabeled samples are automatically promoted to `data/curated_training_set/images/` and tagged in `data/curated_training_set/labels/` for next-generation model fine-tuning.

```bash
curl -X POST http://localhost:8000/active-learning/review \
  -H "Content-Type: application/json" \
  -d '{"filename": "sample_1788814279327_conf_48.jpg", "action": "approve", "verified_class": "scratches"}'
```

**Response:**
```json
{
  "status": "success",
  "action": "approve",
  "filename": "sample_1788814279327_conf_48.jpg",
  "promoted_to": "data/curated_training_set/images/sample_1788814279327_conf_48.jpg",
  "verified_class": "scratches",
  "remaining_queue_size": 4
}
```

---

### `GET /canary/config` & `POST /canary/config`

Inspects or dynamically updates live A/B canary routing percentages and candidate challenger models without container restart:

```bash
# View active canary status
curl http://localhost:8000/canary/config

# Update split: route 25% traffic to candidate weights
curl -X POST http://localhost:8000/canary/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "canary_percentage": 25.0, "canary_path": "training/runs/train/weights/best.pt"}'
```

---

### `GET /canary/metrics`

Returns live side-by-side SLA comparative performance analytics between Champion and Canary models (request throughput, mean/P95 latency, defect discovery rate):

```bash
curl http://localhost:8000/canary/metrics
```

---

### `POST /canary/rollback` & `POST /canary/promote`

- **`POST /canary/rollback`**: Emergency kill-switch. Instantly redirects 100% of factory traffic to primary Champion.
- **`POST /canary/promote`**: Zero-downtime model promotion. Sets candidate model as the new primary Champion and resets canary traffic to 0%.

---

### `GET /retrain/curated-summary`

Returns metrics and defect class breakdown of verified active learning samples ready for fine-tuning in `data/curated_training_set/`:

```bash
curl http://localhost:8000/retrain/curated-summary
```

**Response:**
```json
{
  "total_curated_images": 24,
  "total_curated_labels": 24,
  "class_distribution": {
    "crazing": 8,
    "scratches": 12,
    "patches": 4
  },
  "ready_for_retraining": true
}
```

---

### `POST /retrain/trigger`

Asynchronously launches background continuous retraining, preparing dataset YAML, running YOLO fine-tuning, benchmarking against SLA latency/accuracy gates, and auto-mounting verified candidate weights into the Canary router:

```bash
curl -X POST http://localhost:8000/retrain/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 20,
    "batch_size": 16,
    "auto_mount_canary": true,
    "canary_split_percent": 20.0,
    "sla_max_latency_ms": 30.0,
    "dry_run": false
  }'
```

---

### `GET /retrain/status/{job_id}` & `GET /retrain/jobs`

- **`GET /retrain/status/{job_id}`**: Real-time polling endpoint reporting job lifecycle phase (`PENDING`, `DATASET_PREP`, `TRAINING`, `GATE_EVALUATION`, `CANARY_MOUNT`, `COMPLETED`, `FAILED`), progress percentage, streaming logs, SLA verification status, and canary deployment state.
- **`GET /retrain/jobs`**: Returns chronological audit history of all retraining runs executed on the cluster.

---

### `GET /rca/diagnostics`

Returns real-time automated Root Cause Analysis (RCA) including cross-strip transverse lane distribution (Left Edge, Center Strip, Right Edge), roll eccentricity periodic pitch recurrence harmonics ($P = \pi \times D$), and diagnosed machine subsystem fault attribution:

```bash
curl http://localhost:8000/rca/diagnostics?limit=100
```

**Response:**
```json
{
  "analysis_timestamp_utc": 1788814299.12,
  "total_analyzed_defects": 84,
  "spatial_lanes": {
    "left_edge_count": 12,
    "center_count": 62,
    "right_edge_count": 10,
    "left_edge_percent": 14.3,
    "center_percent": 73.8,
    "right_edge_percent": 11.9,
    "dominant_lane": "CENTER"
  },
  "periodicity": {
    "pitch_detected": true,
    "dominant_pitch_mm": 314.2,
    "recurrence_confidence": 0.88,
    "suspect_roll_diameter_mm": 100.0,
    "explanation": "Harmonic periodicity detected: Repeating defect mark every 314.2 mm, matching a Ø100mm Work Roll circumference."
  },
  "primary_fault": {
    "fault_code": "WORK_ROLL_SURFACE_PITTING",
    "suspect_subsystem": "WORK_ROLL_STAND_02",
    "subsystem_label": "Work Roll Stand #2 (Top Roll)",
    "fault_probability": 92.0,
    "severity": "HIGH",
    "root_cause_explanation": "Periodic mechanical imprint of pitted_surface detected along strip length. Indicates roll thermal fatigue cracking or work roll surface pitting.",
    "corrective_action": "Schedule immediate roll grind or swap for Stand #2 Top Work Roll. Inspect cooling water nozzles for blockage."
  },
  "equipment_status": "ACTION_REQUIRED"
}
```

---

### `GET /rca/spatial-map`

Returns normalized 2D defect coordinates across transverse strip width (0.0 to 1.0) and longitudinal progression for visual spatial defect scatter heatmaps:

```bash
curl http://localhost:8000/rca/spatial-map?limit=100
```

---

### `POST /rca/work-orders` & `GET /rca/work-orders`

- **`POST /rca/work-orders`**: Generates and persists a structured corrective maintenance work order dispatched directly to the plant's CMMS / maintenance team.
- **`GET /rca/work-orders`**: Retrieves history of all active (`OPEN`, `ACKNOWLEDGED`, `RESOLVED`) maintenance work orders.

---

### `GET /ledger/status` & `GET /ledger/verify`

Provides cryptographic audit and tamper verification of the immutable ledger chain:
- **`GET /ledger/status`**: Returns current blockchain height, genesis hash, latest block hash, and total sealed defect records.
- **`GET /ledger/verify`**: Traverses the entire cryptographic hash-chain, recomputing block SHA-256 digests, validating previous block linkage, and verifying HMAC digital signatures to guarantee zero alterations.

```bash
curl http://localhost:8000/ledger/verify
```

```json
{
  "verified": true,
  "total_blocks_verified": 3,
  "chain_status": "CHAIN_IMMUTABLE_AND_VALID",
  "audit_timestamp_utc": 1789551300.0,
  "message": "✓ All 3 blocks in cryptographic chain verified successfully. Zero alterations detected.",
  "findings": []
}
```

---

### `POST /ledger/seal` & `GET /ledger/blocks`

- **`POST /ledger/seal`**: Cryptographically seals an inspection batch and its defect records into an immutable block with a SHA-256 Merkle tree root and HMAC digital signature.
- **`GET /ledger/blocks`**: Chronological block explorer returning all sealed blocks in the immutable chain.

```bash
curl -X POST http://localhost:8000/ledger/seal \
  -H "Content-Type: application/json" \
  -d '{"batch_id": "BATCH-2026-COIL-A", "notes": "Production cold-rolled coil certified."}'
```

---

### `GET /digital-twin/coil-geometry` & `GET /digital-twin/defect-profile`

Provides 2.5D/3D physical coiler topology and longitudinal flaw density profiling:
- **`GET /digital-twin/coil-geometry`**: Computes Archimedean spiral winding dimensions (Outer Diameter $OD$, coil volume, total weight in tonnes, total wrap laps, and build-up ratio).
- **`GET /digital-twin/defect-profile`**: Discretizes the continuous strip into 10-meter segments and classifies each segment into `GRADE_A_PRIME`, `GRADE_B_COMMERCIAL`, or `SCRAP_REJECT`.

```bash
curl "http://localhost:8000/digital-twin/coil-geometry?strip_length_m=1200&strip_thickness_mm=1.2&inner_diameter_mm=508"
```

```json
{
  "strip_length_m": 1200.0,
  "strip_width_mm": 1250.0,
  "strip_thickness_mm": 1.2,
  "inner_diameter_mm": 508.0,
  "outer_diameter_mm": 1564.2,
  "coil_volume_m3": 1.8,
  "coil_weight_kg": 14130.0,
  "coil_weight_tonnes": 14.13,
  "total_wraps": 440,
  "coil_build_up_ratio": 3.08
}
```

---

### `POST /digital-twin/shear-cut-plan` & `GET /digital-twin/export-cqm`

- **`POST /digital-twin/shear-cut-plan`**: Optimizes flying shear cut coordinates along the continuous strip to excise defect clusters while maximizing continuous Prime Grade-A sections ($\ge L_{\text{min}}$).
- **`GET /digital-twin/export-cqm`**: Generates and downloads standardized Coil Quality Map (CQM) JSON for MES, ERP, and CNC flying shear PLCs.

```bash
curl -X POST http://localhost:8000/digital-twin/shear-cut-plan \
  -H "Content-Type: application/json" \
  -d '{"batch_id": "BATCH-2026-COIL-A", "min_prime_length_m": 200.0}'
```

---

### Zero-Shot Edge Anomaly Detection & Novel Flaw Discovery

Industrial metal rolling and casting processes frequently encounter unseen or rare defect morphologies that supervised object detectors like YOLO have never been trained on. FactoryEye integrates a dual-domain Zero-Shot Edge Anomaly Engine:
- **Dual-Domain Analysis**: Computes 2D Fast Fourier Transform (FFT) log-spectral residual saliency combined with spatial Sobel gradient variance in $< 20\text{ms}$ on CPU.
- **Novel Flaw Quarantine**: Irregularities exceeding sensitivity thresholds ($\tau \ge 0.45$) with Shannon spectral entropy spikes are automatically quarantined into `data/novel_flaw_candidates/`.
- **Human-in-the-Loop Triage**: Quality metallurgists can inspect isolated crops in the UI, assign new defect classes, and promote samples directly into the active learning retraining pipeline.

**Anomaly Endpoints:**
- **`POST /anomaly/detect`**: Multipart image inspection returning composite `anomaly_score`, `is_anomalous`, `spectral_entropy`, localized `anomaly_bboxes`, and Base64 Jet heatmap.
- **`GET /anomaly/stats`**: Real-time telemetry including rolling mean anomaly score, OOD event rate, and texture baseline stability.
- **`GET /anomaly/novel-flaws`**: Quarantined novel flaw candidate gallery.
- **`GET /anomaly/novel-flaws/{filename}/crop`**: High-resolution localized thumbnail crop of candidate anomaly.
- **`POST /anomaly/classify-novel`**: Triage action (`promote` to active retraining pool or `discard`).

```bash
# Scan surface for novel zero-shot anomalies
curl -X POST "http://localhost:8000/anomaly/detect?threshold=0.40&quarantine=true" \
  -F "file=@data/samples/sample_surface.jpg"

# View real-time surface texture stability
curl http://localhost:8000/anomaly/stats
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

FactoryEye includes an end-to-end integration and API verification suite testing all 34 endpoints and subsystems:

```bash
make test
# Or: python3 tests/test_api.py
```

Expected output:

```
Running FactoryEye API Tests...
  ✓ test_health_endpoint passed
  ✓ test_prometheus_metrics_endpoint passed
  ✓ test_drift_stats_endpoint passed
  ✓ test_audit_defects_and_summary_endpoints passed
  ✓ test_audit_export_endpoint passed
  ✓ test_audit_defect_trends_endpoint passed
  ✓ test_audit_stations_endpoint passed
  ✓ test_system_hardware_info_endpoint passed
  ✓ test_defect_roi_crop_endpoint passed
  ✓ test_quality_certificate_endpoint passed
  ✓ test_explainability_heatmap_endpoint passed
  ✓ test_predict_image_success passed
  ✓ test_predict_rejects_non_image passed
  ✓ test_predict_rejects_empty_file passed
  ✓ test_active_learning_queue_endpoint passed
  ✓ test_active_learning_review_endpoint passed
  ✓ test_canary_config_endpoints passed
  ✓ test_canary_routing_and_metrics passed
  ✓ test_canary_rollback_and_promote passed
  ✓ test_retrain_curated_summary passed
  ✓ test_retrain_trigger_and_status passed
  ✓ test_retrain_jobs_list passed
  ✓ test_rca_diagnostics_endpoint passed
  ✓ test_rca_spatial_map_endpoint passed
  ✓ test_rca_work_orders_endpoints passed
  ✓ test_ledger_status_and_blocks passed
  ✓ test_ledger_seal_batch passed
  ✓ test_ledger_tamper_verification passed
  ✓ test_digital_twin_coil_geometry passed
  ✓ test_digital_twin_defect_profile passed
  ✓ test_digital_twin_shear_cut_optimization passed
  ✓ test_anomaly_detect_endpoint passed
  ✓ test_anomaly_stats_and_novel_pool passed
  ✓ test_anomaly_classify_and_promote passed

🎉 ALL 34 API TESTS PASSED SUCCESSFULLY!
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

- [x] Automated retraining dataset promotion & Active Learning human-in-the-loop curation queue
- [x] Prometheus metrics endpoint (`/metrics`) for production monitoring
- [x] CI/CD pipeline with GitHub Actions (`.github/workflows/ci.yml`)
- [x] Multi-station yield analytics, Pareto charts & flaw thumbnail crop service
- [x] Saliency attention explainability heatmaps (`/explain`)
- [x] ISO Metallurgical Quality Compliance Certificate generator (`/audit/certificate`)
- [x] Champion vs Challenger SLA model regression gating (`scripts/model_gate.py`)
- [x] Support for RTSP camera streams (`api/rtsp_stream.py`)
- [x] A/B canary testing between model versions in production traffic (`api/canary.py`)
- [x] Zero-shot edge anomaly detection & out-of-distribution novel flaw discovery engine (`api/anomaly_detector.py`)
- [ ] S3 artifact backend for MLflow (replacing local filesystem)

---

## License

MIT License — see `LICENSE` for details.

---
