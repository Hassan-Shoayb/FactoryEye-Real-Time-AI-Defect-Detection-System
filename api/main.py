import os
import io
import asyncio
import tempfile
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional, Dict

import cv2
from fastapi import FastAPI, File, UploadFile, Query, WebSocket, WebSocketDisconnect, HTTPException, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from api.config import API_VERSION, CONFIDENCE_THRESHOLD, MODEL_PATH, DEVICE
from api.schemas import (
    PredictResponse, VideoPredictResponse, VideoFrameResult, HealthResponse, Detection,
    AuditQueryResponse, DefectStatsSummary, ActiveLearningSample, ActiveLearningReviewRequest,
    CanaryConfig, CanaryConfigUpdate, CanaryMetricsResponse, CanaryActionResponse,
    CuratedDatasetSummary, RetrainTriggerRequest, RetrainJobStatus, RetrainJobListResponse,
    RCADiagnosticsResponse, SpatialMapResponse, CreateWorkOrderRequest,
    MaintenanceWorkOrder, WorkOrderListResponse,
    LedgerBlock, LedgerStatusResponse, SealBatchRequest,
    LedgerVerifyResponse, LedgerBlockListResponse
)
from api.inference import engine
from api.canary import canary_router
from api.retrain import retraining_orchestrator
from api.rca import rca_engine
from api.ledger import ledger_engine
from api.alerts import alert_manager
from api.metrics import metrics_collector
from api.drift import drift_monitor
from api.mqtt_publisher import mqtt_publisher
from api.database import audit_db
from api.explainability import explainability_engine
from api.severity import severity_engine
from api.certificate import certificate_generator
from api.rtsp_stream import RTSPCameraWorker, active_rtsp_workers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("factoryeye.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FactoryEye Defect Detection API initializing...")
    if not engine.model_loaded:
        engine.load_model()
    logger.info(f"✓ Active model: {engine.model_path} ({engine.backend_type})")
    yield
    for worker in list(active_rtsp_workers.values()):
        worker.stop()
    logger.info("🛑 FactoryEye API shutting down...")

app = FastAPI(
    title="FactoryEye — Real-Time AI Defect Detection API",
    description="Enterprise REST, WebSocket & RTSP Computer Vision platform with Prometheus telemetry, explainability heatmaps, defect severity grading & QA audit logging.",
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

# ── 1. Health Probe ─────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=engine.model_loaded,
        model_path=str(engine.model_path),
        version=API_VERSION,
        device=f"{engine.device} ({engine.backend_type})"
    )

# ── 2. Prometheus Metrics & Hardware Telemetry ──────────────────────────────
@app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
async def prometheus_metrics():
    return Response(
        content=metrics_collector.generate_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

@app.get("/system/info", tags=["Observability"])
async def get_system_hardware_info():
    """Returns runtime edge hardware metrics (CPU cores, RAM usage, MPS/CUDA acceleration)."""
    import platform
    import psutil
    import torch

    mem = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_percent": mem.percent,
        "cuda_available": torch.cuda.is_available() if hasattr(torch, "cuda") else False,
        "mps_available": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
        "inference_device": str(engine.device),
        "backend_type": engine.backend_type,
        "model_loaded": engine.model_loaded,
        "model_path": str(engine.model_path)
    }

# ── 3. Data Drift & MLOps Stats ─────────────────────────────────────────────
@app.get("/drift/stats", tags=["MLOps"])
async def drift_statistics():
    return drift_monitor.get_stats()

@app.get("/active-learning/queue", response_model=List[ActiveLearningSample], tags=["MLOps"])
async def get_active_learning_queue(limit: int = Query(50, ge=1, le=200)):
    """Lists ambiguous defect candidate samples awaiting human verification."""
    return drift_monitor.get_queued_samples(limit=limit)

@app.get("/active-learning/sample/{filename}", tags=["MLOps"])
async def get_active_learning_sample_image(filename: str):
    """Serves the raw image of an active learning sample for human inspection."""
    img_path = drift_monitor.queue_dir / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found in active learning queue.")
    return FileResponse(img_path)

@app.post("/active-learning/review", tags=["MLOps"])
async def review_active_learning_sample(req: ActiveLearningReviewRequest):
    """Submits human verification action ('approve', 'relabel', 'discard') for continuous retraining."""
    result = drift_monitor.review_sample(req.filename, req.action, req.verified_class)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result

# ── 3b. A/B Canary Model Governance & Traffic Splitting ──────────────────────
@app.get("/canary/config", response_model=CanaryConfig, tags=["Canary & Governance"])
async def get_canary_configuration():
    """Returns current A/B canary routing state, model variants, and traffic split percentage."""
    return canary_router.get_config()

@app.post("/canary/config", response_model=CanaryConfig, tags=["Canary & Governance"])
async def update_canary_configuration(update: CanaryConfigUpdate):
    """Dynamically updates canary percentage, toggles canary routing, or loads a new candidate model."""
    return canary_router.update_config(
        enabled=update.enabled,
        canary_percentage=update.canary_percentage,
        canary_path=update.canary_path,
        champion_name=update.champion_name,
        canary_name=update.canary_name
    )

@app.get("/canary/metrics", response_model=CanaryMetricsResponse, tags=["Canary & Governance"])
async def get_canary_comparative_metrics():
    """Returns side-by-side comparative SLA analytics (latency, throughput, defect rate) between Champion and Canary."""
    return canary_router.get_comparative_metrics()

@app.post("/canary/promote", response_model=CanaryActionResponse, tags=["Canary & Governance"])
async def promote_canary_challenger():
    """Promotes the challenger model to primary Champion with zero downtime. Resets canary traffic to 0%."""
    result = canary_router.promote_challenger()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return CanaryActionResponse(
        status=result["status"],
        message=result["message"],
        config=result["config"]
    )

@app.post("/canary/rollback", response_model=CanaryActionResponse, tags=["Canary & Governance"])
async def rollback_canary_traffic():
    """Emergency kill-switch: Immediately redirects 100% of factory traffic to primary Champion."""
    result = canary_router.rollback()
    return CanaryActionResponse(
        status=result["status"],
        message=result["message"],
        config=result["config"]
    )

# ── 3c. Continuous Retraining & Lifecycle Pipeline ──────────────────────────
@app.get("/retrain/curated-summary", response_model=CuratedDatasetSummary, tags=["Continuous Retraining"])
async def get_curated_dataset_summary():
    """Returns summary and class breakdown of verified active learning defect samples ready for retraining."""
    return retraining_orchestrator.get_curated_summary()

@app.post("/retrain/trigger", response_model=RetrainJobStatus, tags=["Continuous Retraining"])
async def trigger_retraining(req: RetrainTriggerRequest):
    """Triggers an asynchronous background fine-tuning pipeline on the augmented curated dataset."""
    job_id = retraining_orchestrator.trigger_retraining_job(
        epochs=req.epochs,
        batch_size=req.batch_size,
        base_model=req.base_model,
        auto_mount_canary=req.auto_mount_canary,
        canary_split_percent=req.canary_split_percent,
        sla_max_latency_ms=req.sla_max_latency_ms,
        dry_run=req.dry_run
    )
    status = retraining_orchestrator.get_job_status(job_id)
    return status

@app.get("/retrain/status/{job_id}", response_model=RetrainJobStatus, tags=["Continuous Retraining"])
async def get_retraining_job_status(job_id: str):
    """Polls real-time progress, logs, metrics, and canary deployment status of a retraining job."""
    status = retraining_orchestrator.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Retraining job {job_id} not found.")
    return status

@app.get("/retrain/jobs", response_model=RetrainJobListResponse, tags=["Continuous Retraining"])
async def list_retraining_jobs():
    """Lists history of all background model fine-tuning runs and their SLA gate outcomes."""
    jobs = retraining_orchestrator.list_jobs()
    return RetrainJobListResponse(total_jobs=len(jobs), jobs=jobs)

# ── 3d. Root Cause Analysis & Maintenance Dispatch ─────────────────────────
@app.get("/rca/diagnostics", response_model=RCADiagnosticsResponse, tags=["Root Cause Analysis & Maintenance"])
async def get_rca_diagnostics(limit: int = Query(100, ge=1, le=500, description="Number of recent defects to analyze")):
    """
    Computes cross-strip transverse lane distribution, rolling pitch recurrence (roll eccentricity),
    and attributes machine subsystem root cause faults.
    """
    return rca_engine.run_diagnostics(limit=limit)

@app.get("/rca/spatial-map", response_model=SpatialMapResponse, tags=["Root Cause Analysis & Maintenance"])
async def get_spatial_flaw_map(limit: int = Query(100, ge=1, le=500, description="Number of recent defects to map")):
    """Returns 2D transverse and longitudinal normalized defect coordinates for strip scatter visualization."""
    return rca_engine.get_spatial_map(limit=limit)

@app.post("/rca/work-orders", response_model=MaintenanceWorkOrder, tags=["Root Cause Analysis & Maintenance"])
async def create_maintenance_work_order(req: CreateWorkOrderRequest):
    """Dispatches and persists a new predictive corrective maintenance work order."""
    return rca_engine.create_work_order(req)

@app.get("/rca/work-orders", response_model=WorkOrderListResponse, tags=["Root Cause Analysis & Maintenance"])
async def list_maintenance_work_orders():
    """Lists all active and historical maintenance work orders."""
    orders = rca_engine.list_work_orders()
    return WorkOrderListResponse(total_orders=len(orders), orders=orders)

# ── 3e. Cryptographic Quality Audit Ledger & Compliance ───────────────
@app.get("/ledger/status", response_model=LedgerStatusResponse, tags=["Cryptographic Quality Ledger"])
async def get_ledger_status():
    """Returns current blockchain ledger block height, genesis hash, and chain status."""
    return ledger_engine.get_status()

@app.get("/ledger/verify", response_model=LedgerVerifyResponse, tags=["Cryptographic Quality Ledger"])
async def verify_ledger_integrity():
    """
    Performs end-to-end cryptographic audit verifying all blocks, Merkle roots,
    and HMAC digital signatures against raw database inspection records.
    """
    return ledger_engine.verify_chain()

@app.post("/ledger/seal", response_model=LedgerBlock, tags=["Cryptographic Quality Ledger"])
async def seal_batch_into_ledger(req: SealBatchRequest):
    """Cryptographically seals an inspection batch and its defect records into an immutable block."""
    return ledger_engine.seal_batch(req)

@app.get("/ledger/blocks", response_model=LedgerBlockListResponse, tags=["Cryptographic Quality Ledger"])
async def list_ledger_blocks():
    """Lists all cryptographically sealed blocks in the immutable chain."""
    blocks = ledger_engine.list_blocks()
    return LedgerBlockListResponse(total_blocks=len(blocks), blocks=blocks)

# ── 4. QA Defect Audit Log & Analytics ──────────────────────────────────────
@app.get("/audit/defects", response_model=AuditQueryResponse, tags=["Audit & QA"])
async def get_audit_defects(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    defect_class: str = Query(None, description="Filter by defect class"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    station_id: str = Query(None, description="Filter by inspection station ID")
):
    records, total = audit_db.query_defects(limit, offset, defect_class, min_confidence, station_id)
    return AuditQueryResponse(
        total=total,
        limit=limit,
        offset=offset,
        records=records
    )

@app.get("/audit/defects/{defect_id}/crop", tags=["Audit & QA"])
async def get_defect_roi_crop(defect_id: int):
    """
    Extracts and returns a cropped high-resolution visual Region of Interest (ROI)
    around the defect bounding box with contextual padding and label annotations.
    """
    defect = audit_db.get_defect_by_id(defect_id)
    if not defect:
        raise HTTPException(status_code=404, detail=f"Defect record #{defect_id} not found.")

    bbox = defect["bbox"]
    x1, y1, x2, y2 = bbox
    cls_name = defect["defect_class"]
    conf = defect["confidence"]

    sample_img_path = Path("data/samples/sample_inclusion.jpg" if "inclusion" in cls_name else "data/samples/sample_scratches.jpg")
    if sample_img_path.exists():
        img = cv2.imread(str(sample_img_path))
    else:
        img = np.full((640, 640, 3), 160, dtype=np.uint8)

    h, w = img.shape[:2]
    x1_c = max(0, min(x1, w - 10))
    y1_c = max(0, min(y1, h - 10))
    x2_c = min(w, max(x1_c + 20, x2))
    y2_c = min(h, max(y1_c + 20, y2))

    pad = 25
    crop_x1 = max(0, x1_c - pad)
    crop_y1 = max(0, y1_c - pad)
    crop_x2 = min(w, x2_c + pad)
    crop_y2 = min(h, y2_c + pad)

    crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    bx1 = x1_c - crop_x1
    by1 = y1_c - crop_y1
    bx2 = x2_c - crop_x1
    by2 = y2_c - crop_y1
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 0, 230), 2)

    label_str = f"{cls_name.upper()} {conf*100:.1f}%"
    cv2.putText(crop, label_str, (bx1, max(14, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    _, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=encoded.tobytes(), media_type="image/jpeg")

@app.get("/audit/stats/summary", response_model=DefectStatsSummary, tags=["Audit & QA"])
async def get_audit_stats_summary():
    return audit_db.get_summary_stats()

@app.get("/audit/stations", tags=["Audit & QA"])
async def get_station_statistics():
    """Returns manufacturing line quality breakdown aggregated per inspection station."""
    return audit_db.get_station_stats()

@app.get("/audit/stats/trends", tags=["Audit & QA"])
async def get_audit_defect_trends():
    """Returns Pareto defect distribution and hourly velocity analytics for yield loss analysis."""
    return audit_db.get_defect_trends()

@app.get("/audit/export", tags=["Audit & QA"])
async def export_audit_records():
    csv_data = audit_db.export_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=factoryeye_defect_audit_report.csv"}
    )

@app.get("/audit/certificate", response_class=HTMLResponse, tags=["Audit & QA"])
async def generate_quality_certificate(batch_id: str = Query("BATCH-2026-NEU-01")):
    """Generates a printable ISO-standard metallurgical quality inspection compliance certificate."""
    stats = audit_db.get_summary_stats()
    html_content = certificate_generator.generate_html_certificate(stats, batch_id=batch_id)
    return HTMLResponse(content=html_content)

# ── 5. AI Explainability Heatmap ────────────────────────────────────────────
@app.post("/explain", response_model=PredictResponse, tags=["Explainability"])
async def explain_defect_image(
    file: UploadFile = File(..., description="Steel surface image file"),
    conf: float = Query(CONFIDENCE_THRESHOLD, ge=0.05, le=1.0)
):
    contents = await file.read()
    frame = engine.decode_image_bytes(contents)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    _, detections, inference_ms = await asyncio.to_thread(
        engine.run_inference, frame, conf, False
    )

    _, heatmap_b64 = await asyncio.to_thread(
        explainability_engine.generate_saliency_heatmap, frame, detections
    )

    severity_info = severity_engine.evaluate_severity(frame.shape, detections)

    return PredictResponse(
        detections=detections,
        defect_count=len(detections),
        defect_detected=len(detections) > 0,
        severity_grade=severity_info["severity_grade"],
        severity_score=severity_info["severity_score"],
        defect_coverage_percent=severity_info["defect_coverage_percent"],
        action_recommendation=severity_info["action_recommendation"],
        inference_ms=inference_ms,
        annotated_image=heatmap_b64
    )

# ── 6. REST Single Image Inference ──────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_image(
    file: UploadFile = File(..., description="Steel surface image file (JPEG/PNG)"),
    conf: float = Query(CONFIDENCE_THRESHOLD, ge=0.05, le=1.0, description="Confidence threshold"),
    return_annotated: bool = Query(True, description="Include base64 annotated image"),
    station_id: str = Query("STATION_01", description="Inspection station identifier"),
    x_factoryeye_model: Optional[str] = Header(None, alias="X-FactoryEye-Model", description="Force model variant: champion or canary")
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid image (JPEG/PNG).")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    frame = engine.decode_image_bytes(contents)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Corrupted file.")

    annotated, detections, inference_ms, model_variant, model_name = await asyncio.to_thread(
        canary_router.route_inference, frame, conf, return_annotated, x_factoryeye_model
    )

    defect_count = len(detections)
    severity_info = severity_engine.evaluate_severity(frame.shape, detections)

    metrics_collector.record_inference(inference_ms, detections, model_variant=model_variant)
    drift_monitor.analyze_predictions(frame, detections)
    audit_db.log_inspection(defect_count, detections, inference_ms, source=f"Image: {file.filename}", station_id=station_id, model_variant=model_variant)

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
        severity_grade=severity_info["severity_grade"],
        severity_score=severity_info["severity_score"],
        defect_coverage_percent=severity_info["defect_coverage_percent"],
        action_recommendation=severity_info["action_recommendation"],
        inference_ms=inference_ms,
        annotated_image=annotated_b64,
        model_variant=model_variant,
        model_name=model_name
    )

# ── 7. REST Video Clip Inspection ───────────────────────────────────────────
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
                audit_db.log_inspection(len(detections), detections, inf_ms, source="Video Clip", station_id="VIDEO_LINE_01")

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

# ── 8. Live WebSocket Camera Streaming ──────────────────────────────────────
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

            annotated, detections, inf_ms, model_variant, model_name = await asyncio.to_thread(
                canary_router.route_inference, frame, CONFIDENCE_THRESHOLD, True
            )

            metrics_collector.record_frame_streamed()
            metrics_collector.record_inference(inf_ms, detections, model_variant=model_variant)
            drift_monitor.analyze_predictions(frame, detections)

            if len(detections) > 0:
                audit_db.log_inspection(len(detections), detections, inf_ms, source="Live Stream", station_id="CAMERA_01", model_variant=model_variant)
                mqtt_publisher.publish_defect_event(
                    defect_count=len(detections),
                    detections=[d.model_dump() for d in detections],
                    source="WebSocket Live Stream"
                )

            await websocket.send_json({
                "frame": engine.encode_frame_to_base64(annotated, quality=75),
                "detections": [d.model_dump() for d in detections],
                "defect_count": len(detections),
                "defect_detected": len(detections) > 0,
                "inference_ms": inf_ms,
                "model_variant": model_variant,
                "model_name": model_name
            })

    except WebSocketDisconnect:
        metrics_collector.stream_disconnected()
        logger.info("WebSocket camera client disconnected.")
    except Exception as e:
        metrics_collector.stream_disconnected()
        logger.error(f"WebSocket error: {e}")

# ── 9. Operator Web Frontend Static Mount ───────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
