import cv2
import time
import base64
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
from ultralytics import YOLO

from api.config import MODEL_PATH, CONFIDENCE_THRESHOLD, DEVICE, DEFECT_CLASSES
from api.schemas import Detection

logger = logging.getLogger("factoryeye.inference")

# Color palette for defect classes (BGR format for OpenCV)
CLASS_COLORS = {
    "crazing": (255, 100, 50),       # Blue-ish
    "inclusion": (50, 180, 255),     # Orange
    "patches": (50, 220, 50),        # Green
    "pitted_surface": (220, 50, 220),# Purple
    "rolled-in_scale": (50, 220, 220),# Yellow
    "scratches": (0, 0, 255)         # Red
}
DEFAULT_COLOR = (0, 165, 255)

class YOLOInferenceEngine:
    """
    Singleton YOLO Inference Engine for FactoryEye.
    Loads fine-tuned weights on startup and provides fast image, frame, and video batch inference.
    """
    def __init__(self, model_path: str = MODEL_PATH, device: str = DEVICE):
        self.model_path = model_path
        self.device = device
        self.model: Optional[YOLO] = None
        self.model_loaded: bool = False
        self.load_model()

    def load_model(self):
        """Loads YOLO weights with fallback to yolov8n.pt if custom weights are not found yet."""
        target_path = Path(self.model_path)
        
        if not target_path.exists():
            # Search alternative paths
            candidates = [
                Path("training/runs/train/weights/best.pt"),
                Path("training/runs/train/weights/last.pt"),
                Path("best.pt"),
                Path("yolov8n.pt")
            ]
            for c in candidates:
                if c.exists():
                    target_path = c
                    break

        if not target_path.exists():
            logger.warning(f"Target model '{self.model_path}' not found on disk. Initializing with default yolov8n.pt...")
            target_path = Path("yolov8n.pt")

        try:
            logger.info(f"Loading YOLO model from {target_path} on device='{self.device}'...")
            self.model = YOLO(str(target_path))
            self.model_path = str(target_path.resolve())
            self.model_loaded = True
            logger.info(f"✓ Model loaded successfully ({len(self.model.names)} classes).")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            self.model_loaded = False

    def decode_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Converts raw image bytes to an OpenCV BGR numpy array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def decode_base64_frame(self, b64_str: str) -> Optional[np.ndarray]:
        """Converts a base64-encoded image string to an OpenCV BGR array."""
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            return self.decode_image_bytes(img_bytes)
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return None

    def encode_frame_to_base64(self, frame: np.ndarray, quality: int = 80) -> str:
        """Converts an OpenCV BGR frame to a JPEG base64 data URL."""
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        b64_encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_encoded}"

    def run_inference(
        self,
        frame: np.ndarray,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        annotate: bool = True
    ) -> Tuple[np.ndarray, List[Detection], float]:
        """
        Executes YOLO inference on a single OpenCV frame.
        Returns:
            - annotated_frame (np.ndarray)
            - detections (List[Detection])
            - inference_ms (float)
        """
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
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                # Draw text badge with background
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated_frame, (x1, max(y1 - th - 6, 0)), (x1 + tw + 6, max(y1, th + 6)), color, -1)
                cv2.putText(annotated_frame, text, (x1 + 3, max(y1 - 4, th + 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_frame, detections, inference_ms

engine = YOLOInferenceEngine()
