import os
import cv2
import time
import logging
from pathlib import Path
from collections import deque
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger("factoryeye.drift")

class DataDriftMonitor:
    """
    Monitors inference confidence distributions to detect environmental or camera drift.
    Automatically captures ambiguous samples (low-to-moderate confidence) into an Active Learning queue.
    """
    def __init__(
        self,
        window_size: int = 100,
        drift_confidence_threshold: float = 0.55,
        queue_dir: str = "data/active_learning_queue"
    ):
        self.window_size = window_size
        self.drift_threshold = drift_confidence_threshold
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_history = deque(maxlen=window_size)
        self.total_samples_monitored = 0
        self.drift_events_detected = 0

    def analyze_predictions(self, frame: np.ndarray, detections: List[Dict]):
        if not detections:
            return

        confidences = [d.get("confidence", 0.0) if isinstance(d, dict) else getattr(d, "confidence", 0.0) for d in detections]
        for c in confidences:
            self.confidence_history.append(c)
            self.total_samples_monitored += 1

            # Active Learning trigger: Ambiguous sample (0.30 <= conf <= 0.55)
            if 0.30 <= c <= 0.55:
                self._save_to_active_learning_queue(frame, c)

        # Check rolling average confidence drift
        if len(self.confidence_history) >= 20:
            avg_conf = sum(self.confidence_history) / len(self.confidence_history)
            if avg_conf < self.drift_threshold:
                self.drift_events_detected += 1
                logger.warning(
                    f"⚠️ DATA DRIFT DETECTED! Rolling avg confidence ({avg_conf:.3f}) is below threshold ({self.drift_threshold}). "
                    "Check factory camera lens cleanliness or lighting conditions."
                )

    def _save_to_active_learning_queue(self, frame: np.ndarray, confidence: float):
        try:
            timestamp = int(time.time() * 1000)
            filename = f"sample_{timestamp}_conf_{int(confidence * 100)}.jpg"
            filepath = self.queue_dir / filename
            # Limit queue size to 500 images
            if len(list(self.queue_dir.glob("*.jpg"))) < 500:
                cv2.imwrite(str(filepath), frame)
        except Exception as e:
            logger.error(f"Failed to save active learning frame: {e}")

    def get_stats(self) -> Dict:
        avg_conf = (sum(self.confidence_history) / len(self.confidence_history)) if self.confidence_history else 1.0
        return {
            "window_size": self.window_size,
            "samples_in_window": len(self.confidence_history),
            "rolling_avg_confidence": round(avg_conf, 4),
            "drift_detected": avg_conf < self.drift_threshold if len(self.confidence_history) >= 20 else False,
            "drift_threshold": self.drift_threshold,
            "total_drift_events": self.drift_events_detected,
            "active_learning_queue_size": len(list(self.queue_dir.glob("*.jpg")))
        }

drift_monitor = DataDriftMonitor()
