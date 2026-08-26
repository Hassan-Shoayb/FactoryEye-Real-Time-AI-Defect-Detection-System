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
    inference_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")
    annotated_image: Optional[str] = Field(None, description="Base64 data URL of the annotated image with bounding boxes")

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
