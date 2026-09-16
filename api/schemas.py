from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Detection(BaseModel):
    label: str = Field(..., description="Defect class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model prediction confidence score")
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="Bounding box pixel coordinates [x1, y1, x2, y2]")

class PredictResponse(BaseModel):
    detections: List[Detection] = Field(default_factory=list, description="List of detected defects")
    defect_count: int = Field(..., ge=0, description="Total number of defects detected")
    defect_detected: bool = Field(..., description="True if one or more defects were found")
    severity_grade: str = Field(default="NONE", description="Severity category: NONE, MINOR, MAJOR, CRITICAL")
    severity_score: float = Field(default=0.0, description="Composite defect severity index (0-100)")
    defect_coverage_percent: float = Field(default=0.0, description="Surface area coverage percentage")
    action_recommendation: str = Field(default="PASS", description="Recommended shop-floor QA action (PASS, REWORK, SCRAP)")
    inference_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")
    annotated_image: Optional[str] = Field(None, description="Base64 data URL of the annotated image with bounding boxes")
    model_variant: str = Field(default="champion", description="Variant that performed inference: champion or canary")
    model_name: Optional[str] = Field(default=None, description="Descriptive identifier of the model instance")

class VideoFrameResult(BaseModel):
    frame: int = Field(..., ge=0, description="Frame index processed")
    timestamp_sec: float = Field(..., ge=0.0, description="Video timestamp in seconds")
    defect_count: int = Field(..., ge=0, description="Number of defects in this frame")
    detections: List[Detection] = Field(default_factory=list, description="Defects in this frame")

class VideoPredictResponse(BaseModel):
    total_frames_processed: int = Field(..., ge=0, description="Total sample frames processed")
    defect_frames: int = Field(..., ge=0, description="Number of frames with defects")
    defect_rate: float = Field(..., ge=0.0, le=1.0, description="Proportion of defective frames")
    inference_total_ms: float = Field(..., ge=0.0, description="Total processing time in ms")
    frame_results: List[VideoFrameResult] = Field(default_factory=list, description="Per-frame detection details")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status ('ok')")
    model_loaded: bool = Field(..., description="True if YOLO weights are loaded in memory")
    model_path: str = Field(..., description="Path to active model weights")
    version: str = Field(..., description="API Version")
    device: str = Field(..., description="Inference device (cpu, cuda, mps)")

class AuditDefectItem(BaseModel):
    id: int
    inspection_id: int
    timestamp_utc: float
    datetime_iso: str
    station_id: str
    defect_class: str
    confidence: float
    bbox: List[int]
    model_variant: Optional[str] = Field(default="champion", description="Model variant: champion or canary")

class AuditQueryResponse(BaseModel):
    total: int
    limit: int
    offset: int
    records: List[AuditDefectItem]

class ClassBreakdownItem(BaseModel):
    class_: str = Field(..., alias="class")
    count: int
    avg_confidence: float

class DefectStatsSummary(BaseModel):
    total_inspections: int
    clean_inspections: int
    defective_inspections: int
    defect_rate_percent: float
    quality_yield_percent: float
    mean_inference_ms: float
    defect_class_breakdown: List[Dict[str, Any]]

class ActiveLearningSample(BaseModel):
    filename: str
    filepath: str
    confidence_estimate: float
    timestamp_utc: float

class ActiveLearningReviewRequest(BaseModel):
    filename: str
    action: str = Field(..., description="'approve', 'relabel', or 'discard'")
    verified_class: Optional[str] = Field(None, description="Human-verified defect class")

class CanaryConfig(BaseModel):
    enabled: bool = Field(..., description="Whether canary traffic splitting is active")
    canary_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of traffic routed to candidate model (0-100)")
    champion_name: str = Field(..., description="Name / tag of the champion production model")
    champion_path: str = Field(..., description="Filesystem path of the champion weights")
    canary_name: str = Field(..., description="Name / tag of the challenger candidate model")
    canary_path: Optional[str] = Field(None, description="Filesystem path of the canary weights")
    canary_loaded: bool = Field(..., description="Whether the canary candidate model is currently loaded in memory")
    routing_strategy: str = Field(default="weighted_random", description="Routing mode: weighted_random or header_override")

class CanaryConfigUpdate(BaseModel):
    enabled: Optional[bool] = Field(None, description="Enable or disable canary routing")
    canary_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Target canary traffic percentage")
    canary_path: Optional[str] = Field(None, description="File path to candidate weights to load")
    champion_name: Optional[str] = Field(None, description="Updated name for champion")
    canary_name: Optional[str] = Field(None, description="Updated name for canary")

class CanaryModelTelemetry(BaseModel):
    model_name: str
    model_path: str
    model_variant: str
    inferences_count: int
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    defective_inferences: int
    defect_detection_rate_percent: float
    total_defects_found: int

class CanaryMetricsResponse(BaseModel):
    canary_enabled: bool
    canary_percentage: float
    routing_strategy: str
    total_inferences: int
    champion: CanaryModelTelemetry
    canary: CanaryModelTelemetry

class CanaryActionResponse(BaseModel):
    status: str
    message: str
    config: CanaryConfig

class CuratedDatasetSummary(BaseModel):
    total_curated_images: int
    total_curated_labels: int
    class_distribution: Dict[str, int]
    ready_for_retraining: bool

class RetrainTriggerRequest(BaseModel):
    epochs: int = Field(default=30, ge=1, le=200, description="Number of fine-tuning epochs")
    batch_size: int = Field(default=16, ge=1, le=128, description="Training batch size")
    base_model: str = Field(default="yolov8n.pt", description="Base checkpoint backbone")
    auto_mount_canary: bool = Field(default=True, description="Automatically load candidate weights into Canary router if SLA gate passes")
    canary_split_percent: float = Field(default=20.0, ge=0.0, le=100.0, description="Initial traffic split to canary (0-100%)")
    sla_max_latency_ms: float = Field(default=30.0, description="P95 latency SLA budget threshold")
    dry_run: bool = Field(default=False, description="Run in fast mock mode for automated testing")

class RetrainJobStatus(BaseModel):
    job_id: str
    status: str = Field(..., description="Job lifecycle state: PENDING, DATASET_PREP, TRAINING, GATE_EVALUATION, CANARY_MOUNT, COMPLETED, FAILED")
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    started_at: float
    completed_at: Optional[float] = None
    duration_sec: float = 0.0
    epochs_total: int
    current_epoch: int = 0
    curated_samples_used: int = 0
    map50: Optional[float] = None
    gate_passed: Optional[bool] = None
    candidate_weights_path: Optional[str] = None
    canary_mounted: bool = False
    message: str = ""
    logs: List[str] = Field(default_factory=list)

class RetrainJobListResponse(BaseModel):
    total_jobs: int
    jobs: List[RetrainJobStatus]

# ── 6. Root Cause Analysis (RCA) & Maintenance Dispatch Schemas ────────────
class SpatialLaneBreakdown(BaseModel):
    left_edge_count: int = Field(default=0, description="Defects in 0-20% strip width")
    center_count: int = Field(default=0, description="Defects in 20-80% strip width")
    right_edge_count: int = Field(default=0, description="Defects in 80-100% strip width")
    left_edge_percent: float = Field(default=0.0, description="Left edge defect share %")
    center_percent: float = Field(default=0.0, description="Center defect share %")
    right_edge_percent: float = Field(default=0.0, description="Right edge defect share %")
    dominant_lane: str = Field(default="BALANCED", description="Dominant spatial lane: LEFT_EDGE, CENTER, RIGHT_EDGE, or BALANCED")

class PeriodicPitchFinding(BaseModel):
    pitch_detected: bool = Field(default=False, description="True if a repeating spatial pitch is detected")
    dominant_pitch_mm: Optional[float] = Field(default=None, description="Dominant defect repeat pitch in mm")
    recurrence_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in periodic recurrence")
    suspect_roll_diameter_mm: Optional[float] = Field(default=None, description="Calculated work roll diameter D = Pitch / pi")
    explanation: str = Field(default="No periodic recurrence detected.", description="Textual description of periodicity findings")

class MachineFaultAttribution(BaseModel):
    fault_code: str = Field(default="NORMAL", description="Diagnosed mechanical fault code")
    suspect_subsystem: str = Field(default="NONE", description="Suspect machine subsystem or stand")
    subsystem_label: str = Field(default="Line Operating Normally", description="Human-readable subsystem label")
    fault_probability: float = Field(default=0.0, ge=0.0, le=100.0, description="Probability percentage")
    severity: str = Field(default="LOW", description="Fault severity: LOW, MEDIUM, HIGH, CRITICAL")
    root_cause_explanation: str = Field(default="", description="Detailed root cause explanation")
    corrective_action: str = Field(default="", description="Recommended maintenance task")

class RCADiagnosticsResponse(BaseModel):
    analysis_timestamp_utc: float
    total_analyzed_defects: int
    spatial_lanes: SpatialLaneBreakdown
    periodicity: PeriodicPitchFinding
    primary_fault: MachineFaultAttribution
    secondary_faults: List[MachineFaultAttribution] = Field(default_factory=list)
    equipment_status: str = Field(default="OPTIMAL", description="Overall line status: OPTIMAL, WARNING, ACTION_REQUIRED, CRITICAL")

class SpatialFlawPoint(BaseModel):
    id: int
    defect_class: str
    confidence: float
    x_norm: float = Field(..., ge=0.0, le=1.0, description="Normalized transverse position across strip width (0.0 = Left Edge, 1.0 = Right Edge)")
    y_norm: float = Field(..., ge=0.0, description="Normalized longitudinal position or relative frame/timestamp offset")
    station_id: str
    lane: str

class SpatialMapResponse(BaseModel):
    total_points: int
    strip_width_px: int = 200
    flaws: List[SpatialFlawPoint]

class CreateWorkOrderRequest(BaseModel):
    suspect_subsystem: str = Field(..., description="Target machine subsystem (e.g. WORK_ROLL_STAND_02)")
    fault_code: str = Field(default="UNSPECIFIED", description="Identified root cause fault code")
    priority: str = Field(default="HIGH", description="Priority level: LOW, MEDIUM, HIGH, CRITICAL")
    recommended_action: str = Field(..., description="Action instructions for maintenance team")
    station_id: Optional[str] = Field(default="ALL_STATIONS", description="Originating line or inspection station")
    notes: Optional[str] = Field(default="", description="Optional operator notes")

class MaintenanceWorkOrder(BaseModel):
    order_id: str
    created_at: float
    datetime_iso: str
    suspect_subsystem: str
    fault_code: str
    priority: str
    recommended_action: str
    station_id: str
    notes: str
    status: str = Field(default="OPEN", description="Work order lifecycle status: OPEN, ACKNOWLEDGED, RESOLVED")
    resolved_at: Optional[float] = None

class WorkOrderListResponse(BaseModel):
    total_orders: int
    orders: List[MaintenanceWorkOrder]

# ── 7. Cryptographic Quality Audit Ledger Schemas ───────────────────────────
class LedgerBlock(BaseModel):
    block_height: int = Field(..., ge=0, description="Sequential monotonic block index")
    block_hash: str = Field(..., description="SHA-256 hash of this block header and payload")
    previous_hash: str = Field(..., description="SHA-256 hash of the preceding block (genesis uses zeros)")
    merkle_root: str = Field(..., description="SHA-256 Merkle tree root of defect records sealed in this block")
    timestamp_utc: float = Field(..., description="Block creation timestamp")
    datetime_iso: str = Field(..., description="ISO 8601 UTC timestamp")
    batch_id: str = Field(..., description="Production coil or inspection batch identifier")
    record_count: int = Field(default=0, ge=0, description="Number of defect records sealed in this block")
    signature: str = Field(..., description="HMAC-SHA256 digital provenance signature")
    sealed_by: str = Field(default="FactoryEye-Immutability-Engine", description="Authority or operator that sealed block")
    notes: Optional[str] = Field(default="", description="Operator batch sealing notes")

class LedgerStatusResponse(BaseModel):
    block_height: int = Field(..., ge=0, description="Latest sealed block height")
    genesis_hash: str = Field(..., description="Hash of the genesis block")
    latest_block_hash: str = Field(..., description="Hash of the latest sealed block")
    total_sealed_records: int = Field(default=0, ge=0, description="Total defect events cryptographically sealed")
    chain_integrity: str = Field(default="VALID", description="Chain status: VALID, VERIFIED, or TAMPER_DETECTED")
    last_verified_at: Optional[float] = None

class SealBatchRequest(BaseModel):
    batch_id: str = Field(default="BATCH-2026-NEU-01", description="Production coil or inspection batch identifier")
    station_id: Optional[str] = Field(default="ALL_STATIONS", description="Originating inspection station")
    notes: Optional[str] = Field(default="Routine production batch cryptographic sealing.", description="Sealing notes")

class TamperAuditFinding(BaseModel):
    block_height: int
    batch_id: str
    expected_hash: str
    recomputed_hash: str
    status: str = Field(..., description="PASS or TAMPERED")
    details: str

class LedgerVerifyResponse(BaseModel):
    verified: bool = Field(..., description="True if 100% of blocks and Merkle roots match database records")
    total_blocks_verified: int
    chain_status: str = Field(..., description="CHAIN_IMMUTABLE_AND_VALID or TAMPER_DETECTED")
    audit_timestamp_utc: float
    message: str
    findings: List[TamperAuditFinding] = Field(default_factory=list)

class LedgerBlockListResponse(BaseModel):
    total_blocks: int
    blocks: List[LedgerBlock]




