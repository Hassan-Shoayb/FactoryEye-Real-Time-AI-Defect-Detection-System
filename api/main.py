import os
import io
import asyncio
import tempfile
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, UploadFile, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from api.config import API_VERSION, CONFIDENCE_THRESHOLD, MODEL_PATH, DEVICE
from api.schemas import PredictResponse, VideoPredictResponse, VideoFrameResult, HealthResponse, Detection
from api.inference import engine
from api.alerts import alert_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("factoryeye.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: Runs startup and shutdown hooks."""
    logger.info("🚀 FactoryEye Defect Detection API initializing...")
    if not engine.model_loaded:
        engine.load_model()
    logger.info(f"✓ Active model path: {engine.model_path}")
    yield
    logger.info("🛑 FactoryEye API shutting down...")

app = FastAPI(
    title="FactoryEye — Real-Time AI Defect Detection API",
    description="Production-grade REST & WebSocket Computer Vision API for steel surface defect detection.",
    version=API_VERSION,
    lifespan=lifespan
)

# Enable CORS for web clients and cross-origin tools
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
    """Liveness probe returning active model status, device, and API version."""
    return HealthResponse(
        status="ok",
        model_loaded=engine.model_loaded,
        model_path=str(engine.model_path),
        version=API_VERSION,
        device=str(engine.device)
    )

# ── 2. REST Single Image Inference ──────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_image(
    file: UploadFile = File(..., description="Steel surface image file (JPEG/PNG)"),
    conf: float = Query(CONFIDENCE_THRESHOLD, ge=0.05, le=1.0, description="Detection confidence threshold"),
    return_annotated: bool = Query(True, description="Whether to include base64 annotated image in response")
):
    """Upload a single steel surface image and receive detected defects and bounding boxes."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid image (JPEG/PNG).")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    frame = engine.decode_image_bytes(contents)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Corrupted file.")

    # Run inference in worker thread
    annotated, detections, inference_ms = await asyncio.to_thread(
        engine.run_inference, frame, conf, return_annotated
    )

    annotated_b64 = engine.encode_frame_to_base64(annotated) if return_annotated else None
    defect_count = len(detections)

    # Trigger alert if defect detected
    if defect_count > 0:
        asyncio.create_task(alert_manager.send_defect_alert(
            defect_count=defect_count,
            detections=[d.model_dump() for d in detections],
            source=f"Image: {file.filename}"
        ))

    return PredictResponse(
        detections=detections,
        defect_count=defect_count,
        defect_detected=defect_count > 0,
        inference_ms=inference_ms,
        annotated_image=annotated_b64
    )

# ── 3. REST Video Clip Inspection ───────────────────────────────────────────
@app.post("/predict-video", response_model=VideoPredictResponse, tags=["Inference"])
async def predict_video(
    file: UploadFile = File(..., description="Video clip (MP4, AVI, MOV)"),
    frame_stride: int = Query(5, ge=1, le=30, description="Sample every N-th frame"),
    conf: float = Query(CONFIDENCE_THRESHOLD, ge=0.05, le=1.0, description="Confidence threshold")
):
    """Upload a manufacturing line video clip and inspect sampled frames for defects."""
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
                _, detections, _ = await asyncio.to_thread(engine.run_inference, frame, conf, False)
                processed_count += 1
                if len(detections) > 0:
                    defect_frames_count += 1

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

# ── 4. Live Real-Time WebSocket Video Streaming ─────────────────────────────
@app.websocket("/ws/stream")
async def websocket_video_stream(websocket: WebSocket):
    """
    High-FPS bidirectional WebSocket endpoint for live camera streams.
    Client sends: Base64 JPEG frame string
    Server responds: JSON with annotated base64 frame, detections, latency
    """
    await websocket.accept()
    logger.info("WebSocket camera client connected.")

    try:
        while True:
            data = await websocket.receive_text()
            frame = engine.decode_base64_frame(data)

            if frame is None:
                continue

            # Run inference without blocking event loop
            annotated, detections, inf_ms = await asyncio.to_thread(
                engine.run_inference, frame, CONFIDENCE_THRESHOLD, True
            )

            # Send back annotated frame + detections
            await websocket.send_json({
                "frame": engine.encode_frame_to_base64(annotated, quality=75),
                "detections": [d.model_dump() for d in detections],
                "defect_count": len(detections),
                "defect_detected": len(detections) > 0,
                "inference_ms": inf_ms
            })

            # Check alert asynchronously
            if len(detections) > 0 and alert_manager.can_alert():
                asyncio.create_task(alert_manager.send_defect_alert(
                    defect_count=len(detections),
                    detections=[d.model_dump() for d in detections],
                    source="Live WebSocket Stream"
                ))

    except WebSocketDisconnect:
        logger.info("WebSocket camera client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}")

# ── 5. Operator Web Frontend Static Mount ───────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
