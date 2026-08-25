import os
import io
import asyncio
import tempfile
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, UploadFile, Query, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from api.config import API_VERSION, CONFIDENCE_THRESHOLD, MODEL_PATH, DEVICE
from api.schemas import PredictResponse, VideoPredictResponse, VideoFrameResult, HealthResponse, Detection
from api.inference import engine
from api.alerts import alert_manager
from api.metrics import metrics_collector
from api.drift import drift_monitor
from api.mqtt_publisher import mqtt_publisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("factoryeye.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FactoryEye Defect Detection API initializing...")
    if not engine.model_loaded:
        engine.load_model()
    logger.info(f"✓ Active model: {engine.model_path} ({engine.backend_type})")
    yield
    logger.info("🛑 FactoryEye API shutting down...")

app = FastAPI(
    title="FactoryEye — Real-Time AI Defect Detection API",
    description="Production-grade REST & WebSocket Computer Vision API with Prometheus observability and drift tracking.",
    version=API_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 1. Liveness & Health Probe ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=engine.model_loaded,
        model_path=str(engine.model_path),
        version=API_VERSION,
        device=f"{engine.device} ({engine.backend_type})"
    )

# ── 2. Prometheus Observability Metrics ─────────────────────────────────────
@app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
async def prometheus_metrics():
    """Exposes Prometheus-formatted metrics (P95 latency, defect counts, active streams)."""
    return Response(
        content=metrics_collector.generate_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

# ── 3. Data Drift & MLOps Telemetry ─────────────────────────────────────────
@app.get("/drift/stats", tags=["MLOps"])
async def drift_statistics():
    """Returns rolling confidence statistics and active learning queue status."""
    return drift_monitor.get_stats()

# ── 4. REST Single Image Inference ──────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_image(
    file: UploadFile = File(..., description="Steel surface image file (JPEG/PNG)"),
    conf: float = Query(CONFIDENCE_THRESHOLD, ge=0.05, le=1.0, description="Confidence threshold"),
    return_annotated: bool = Query(True, description="Include base64 annotated image")
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid image (JPEG/PNG).")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    frame = engine.decode_image_bytes(contents)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Corrupted file.")

    annotated, detections, inference_ms = await asyncio.to_thread(
        engine.run_inference, frame, conf, return_annotated
    )

    defect_count = len(detections)

    # 1. Record Prometheus metrics
    metrics_collector.record_inference(inference_ms, detections)

    # 2. Check Data Drift & Active Learning
    drift_monitor.analyze_predictions(frame, detections)

    # 3. Trigger Alerts & Industrial MQTT if defects found
    if defect_count > 0:
        asyncio.create_task(alert_manager.send_defect_alert(
            defect_count=defect_count,
            detections=[d.model_dump() for d in detections],
            source=f"Image: {file.filename}"
        ))
        mqtt_publisher.publish_defect_event(
            defect_count=defect_count,
            detections=[d.model_dump() for d in detections],
            source=f"REST: {file.filename}"
        )

    annotated_b64 = engine.encode_frame_to_base64(annotated) if return_annotated else None

    return PredictResponse(
        detections=detections,
        defect_count=defect_count,
        defect_detected=defect_count > 0,
        inference_ms=inference_ms,
        annotated_image=annotated_b64
    )

# ── 5. REST Video Clip Inspection ───────────────────────────────────────────
@app.post("/predict-video", response_model=VideoPredictResponse, tags=["Inference"])
async def predict_video(
    file: UploadFile = File(..., description="Video clip (MP4, AVI, MOV)"),
    frame_stride: int = Query(5, ge=1, le=30, description="Sample every N-th frame"),
    conf: float = Query(CONFIDENCE_THRESHOLD, ge=0.05, le=1.0, description="Confidence threshold")
):
    suffix = Path(file.filename or "video.mp4").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        processed_count = 0
        defect_frames_count = 0
        frame_results = []
        t0 = asyncio.get_event_loop().time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                _, detections, inf_ms = await asyncio.to_thread(engine.run_inference, frame, conf, False)
                processed_count += 1
                if len(detections) > 0:
                    defect_frames_count += 1

                metrics_collector.record_inference(inf_ms, detections)
                drift_monitor.analyze_predictions(frame, detections)

                frame_results.append(VideoFrameResult(
                    frame=frame_idx,
                    timestamp_sec=round(frame_idx / fps, 2),
                    defect_count=len(detections),
                    detections=detections
                ))

            frame_idx += 1

        cap.release()
        total_time_ms = round((asyncio.get_event_loop().time() - t0) * 1000, 2)
        defect_rate = round(defect_frames_count / processed_count, 4) if processed_count > 0 else 0.0

        return VideoPredictResponse(
            total_frames_processed=processed_count,
            defect_frames=defect_frames_count,
            defect_rate=defect_rate,
            inference_total_ms=total_time_ms,
            frame_results=frame_results
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── 6. Live WebSocket Camera Streaming ──────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_video_stream(websocket: WebSocket):
    await websocket.accept()
    metrics_collector.stream_connected()
    logger.info("WebSocket camera client connected.")

    try:
        while True:
            data = await websocket.receive_text()
            frame = engine.decode_base64_frame(data)

            if frame is None:
                continue

            annotated, detections, inf_ms = await asyncio.to_thread(
                engine.run_inference, frame, CONFIDENCE_THRESHOLD, True
            )

            metrics_collector.record_frame_streamed()
            metrics_collector.record_inference(inf_ms, detections)
            drift_monitor.analyze_predictions(frame, detections)

            await websocket.send_json({
                "frame": engine.encode_frame_to_base64(annotated, quality=75),
                "detections": [d.model_dump() for d in detections],
                "defect_count": len(detections),
                "defect_detected": len(detections) > 0,
                "inference_ms": inf_ms
            })

            if len(detections) > 0:
                mqtt_publisher.publish_defect_event(
                    defect_count=len(detections),
                    detections=[d.model_dump() for d in detections],
                    source="WebSocket Live Stream"
                )

    except WebSocketDisconnect:
        metrics_collector.stream_disconnected()
        logger.info("WebSocket camera client disconnected.")
    except Exception as e:
        metrics_collector.stream_disconnected()
        logger.error(f"WebSocket error: {e}")

# ── 7. Operator Web Frontend Static Mount ───────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
