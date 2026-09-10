import os
import cv2
import time
import logging
from pathlib import Path
from collections import deque
from typing import List, Dict, Optional, Union
import numpy as np

logger = logging.getLogger("factoryeye.drift")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_QUEUE_DIR = BASE_DIR / "data" / "active_learning_queue"

class DataDriftMonitor:
    """
    Monitors inference confidence distributions to detect environmental or camera drift.
    Automatically captures ambiguous samples (low-to-moderate confidence) into an Active Learning queue.
    """
    def __init__(
        self,
        window_size: int = 100,
        drift_confidence_threshold: float = 0.55,
        queue_dir: Optional[Union[str, Path]] = None
    ):
        self.window_size = window_size
        self.drift_threshold = drift_confidence_threshold
        self.queue_dir = Path(queue_dir) if queue_dir is not None else DEFAULT_QUEUE_DIR
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

    def get_queued_samples(self, limit: int = 50) -> List[Dict]:
        """Lists pending ambiguous defect samples waiting for human verification."""
        samples = []
        for p in sorted(self.queue_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)[:limit]:
            # Extract estimated conf from filename e.g. sample_1787622537295_conf_44.jpg
            parts = p.stem.split("_")
            conf_est = 0.45
            try:
                if "conf" in parts:
                    conf_idx = parts.index("conf")
                    conf_est = float(parts[conf_idx + 1]) / 100.0
            except Exception:
                pass

            samples.append({
                "filename": p.name,
                "filepath": str(p.resolve()),
                "confidence_estimate": conf_est,
                "timestamp_utc": p.stat().st_mtime
            })
        return samples

    def review_sample(self, filename: str, action: str, verified_class: Optional[str] = None) -> Dict:
        """
        Reviews a queued ambiguous defect sample:
        - 'approve' / 'relabel': promotes to data/curated_training_set for continuous model retraining.
        - 'discard': purges sample from review queue.
        """
        src = self.queue_dir / filename
        if not src.exists():
            return {"status": "error", "message": f"Sample {filename} not found in queue."}

        curated_dir = self.queue_dir.parent / "curated_training_set"
        curated_images = curated_dir / "images"
        curated_labels = curated_dir / "labels"
        curated_images.mkdir(parents=True, exist_ok=True)
        curated_labels.mkdir(parents=True, exist_ok=True)

        if action in ("approve", "relabel"):
            dst_img = curated_images / filename
            os.replace(src, dst_img)
            
            # Save verified class label annotation
            lbl_name = src.stem + ".txt"
            dst_lbl = curated_labels / lbl_name
            with open(dst_lbl, "w") as f:
                f.write(f"# Human-verified class: {verified_class or 'verified_defect'}\n")
            
            logger.info(f"✓ Active learning sample {filename} promoted to curated retraining dataset.")
            return {
                "status": "success",
                "action": action,
                "filename": filename,
                "promoted_to": str(dst_img),
                "verified_class": verified_class or "verified_defect",
                "remaining_queue_size": len(list(self.queue_dir.glob("*.jpg")))
            }
        elif action == "discard":
            os.remove(src)
            logger.info(f"🗑️ Active learning sample {filename} discarded.")
            return {
                "status": "success",
                "action": "discard",
                "filename": filename,
                "remaining_queue_size": len(list(self.queue_dir.glob("*.jpg")))
            }
        else:
            return {"status": "error", "message": f"Unsupported action '{action}'. Use 'approve', 'relabel', or 'discard'."}

drift_monitor = DataDriftMonitor()
