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

