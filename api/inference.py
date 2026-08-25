import cv2
import time
import base64
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
from ultralytics import YOLO

from api.config import MODEL_PATH, CONFIDENCE_THRESHOLD, DEVICE
from api.schemas import Detection

logger = logging.getLogger("factoryeye.inference")

CLASS_COLORS = {
    "crazing": (255, 100, 50),
    "inclusion": (50, 180, 255),
    "patches": (50, 220, 50),
    "pitted_surface": (220, 50, 220),
    "rolled-in_scale": (50, 220, 220),
    "scratches": (0, 0, 255)
}
DEFAULT_COLOR = (0, 165, 255)

class YOLOInferenceEngine:
    """
    High-throughput YOLO Inference Engine supporting both native PyTorch (.pt)
    and hardware-optimized ONNX Runtime (.onnx) backends.
    """
    def __init__(self, model_path: str = MODEL_PATH, device: str = DEVICE):
        self.model_path = model_path
        self.device = device
        self.model: Optional[YOLO] = None
        self.model_loaded: bool = False
        self.backend_type: str = "PyTorch"
        self.load_model()

    def load_model(self):
        target_path = Path(self.model_path)
        
        # Prioritize ONNX for fast CPU inference if available
        onnx_candidate = target_path.with_suffix(".onnx")
        if onnx_candidate.exists():
            target_path = onnx_candidate
            self.backend_type = "ONNX Runtime"

        if not target_path.exists():
            candidates = [
                Path("training/runs/train/weights/best.onnx"),
                Path("training/runs/train/weights/best.pt"),
                Path("training/runs/train/weights/last.pt"),
                Path("best.pt"),
                Path("yolov8n.pt")
            ]
            for c in candidates:
                if c.exists():
                    target_path = c
                    if target_path.suffix == ".onnx":
                        self.backend_type = "ONNX Runtime"
                    break

        if not target_path.exists():
            target_path = Path("yolov8n.pt")

        try:
            logger.info(f"Loading YOLO model from {target_path} (Backend: {self.backend_type})...")
            self.model = YOLO(str(target_path))
            self.model_path = str(target_path.resolve())
            self.model_loaded = True
            logger.info(f"✓ Model loaded successfully ({len(self.model.names)} classes).")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            self.model_loaded = False

    def decode_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def decode_base64_frame(self, b64_str: str) -> Optional[np.ndarray]:
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            return self.decode_image_bytes(img_bytes)
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return None

    def encode_frame_to_base64(self, frame: np.ndarray, quality: int = 80) -> str:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        b64_encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_encoded}"

    def run_inference(
        self,
        frame: np.ndarray,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        annotate: bool = True
    ) -> Tuple[np.ndarray, List[Detection], float]:
        if not self.model_loaded or self.model is None:
            return frame, [], 0.0

        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )[0]
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        detections: List[Detection] = []
        annotated_frame = frame.copy() if annotate else frame

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.model.names.get(cls_id, f"defect_{cls_id}")

            detections.append(Detection(
                label=label,
                confidence=round(conf, 4),
                bbox=[x1, y1, x2, y2]
            ))

            if annotate:
                color = CLASS_COLORS.get(label, DEFAULT_COLOR)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated_frame, (x1, max(y1 - th - 6, 0)), (x1 + tw + 6, max(y1, th + 6)), color, -1)
                cv2.putText(annotated_frame, text, (x1 + 3, max(y1 - 4, th + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_frame, detections, inference_ms

engine = YOLOInferenceEngine()
