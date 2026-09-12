import random
import logging
import threading
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from api.inference import YOLOInferenceEngine, engine as default_champion_engine
from api.config import MODEL_PATH

logger = logging.getLogger("factoryeye.canary")

class CanaryRouter:
    """
    Production A/B Canary Model Routing and Governance Engine.
    Manages live traffic splitting between Champion (Primary) and Challenger (Canary) models,
    accumulates comparative SLA telemetry, and supports zero-downtime promotion and rollback.
    """
    def __init__(self, champion_engine: Optional[YOLOInferenceEngine] = None):
        self._lock = threading.RLock()
        self.champion_engine: YOLOInferenceEngine = champion_engine or default_champion_engine
        self.champion_name: str = "Champion-Production"
        
        self.canary_engine: Optional[YOLOInferenceEngine] = None
        self.canary_name: str = "Challenger-Candidate"
        self.canary_path: Optional[str] = None

        self.enabled: bool = False
        self.canary_percentage: float = 20.0
        self.routing_strategy: str = "weighted_random"

        # In-memory telemetry accumulators (thread-safe)
        self.telemetry: Dict[str, Dict[str, Any]] = {
            "champion": {
                "requests": 0,
                "latencies": deque(maxlen=500),
                "total_latency_ms": 0.0,
                "defective_requests": 0,
                "total_defects": 0
            },
            "canary": {
                "requests": 0,
                "latencies": deque(maxlen=500),
                "total_latency_ms": 0.0,
                "defective_requests": 0,
                "total_defects": 0
            }
        }

    def load_canary_model(self, model_path: str) -> bool:
        """Loads a candidate challenger YOLO model into memory."""
        p = Path(model_path)
        if not p.is_absolute():
            base_p = Path(__file__).resolve().parent.parent / model_path
            if base_p.exists():
                p = base_p

        if not p.exists():
            logger.warning(f"Challenger model file not found: {model_path}")
            return False

        try:
            logger.info(f"Initializing Canary challenger model from: {model_path}")
            canary_eng = YOLOInferenceEngine(model_path=str(p.resolve()))
            if canary_eng.model_loaded:
                with self._lock:
                    self.canary_engine = canary_eng
                    self.canary_path = str(p.resolve())
                logger.info("✓ Canary challenger model loaded successfully into memory.")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed loading canary model from {model_path}: {e}")
            return False

    def route_inference(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.50,
        annotate: bool = True,
        forced_variant: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Any], float, str, str]:
        """
        Routes inference request to Champion or Canary based on active traffic split or header override.
        Returns: (annotated_frame, detections, inference_ms, variant_tag, model_name)
        """
        variant = "champion"

        with self._lock:
            canary_ready = self.canary_engine is not None and self.canary_engine.model_loaded
            
            if forced_variant == "canary" and canary_ready:
                variant = "canary"
            elif forced_variant == "champion":
                variant = "champion"
            elif self.enabled and canary_ready and self.canary_percentage > 0.0:
                roll = random.uniform(0.0, 100.0)
                if roll < self.canary_percentage:
                    variant = "canary"
                else:
                    variant = "champion"
            else:
                variant = "champion"

            selected_engine = self.canary_engine if (variant == "canary" and canary_ready) else self.champion_engine
            selected_name = self.canary_name if variant == "canary" else self.champion_name

        # Execute inference outside lock to avoid blocking other concurrent requests
        annotated, detections, inf_ms = selected_engine.run_inference(frame, conf_threshold, annotate)

        # Record comparative metrics
        with self._lock:
            stats = self.telemetry[variant]
            stats["requests"] += 1
            stats["total_latency_ms"] += inf_ms
            stats["latencies"].append(inf_ms)
            if len(detections) > 0:
                stats["defective_requests"] += 1
                stats["total_defects"] += len(detections)

        return annotated, detections, inf_ms, variant, selected_name

    def get_config(self) -> Dict[str, Any]:
        """Returns current canary router configuration."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "canary_percentage": self.canary_percentage,
                "champion_name": self.champion_name,
                "champion_path": str(self.champion_engine.model_path) if self.champion_engine else "N/A",
                "canary_name": self.canary_name,
                "canary_path": self.canary_path,
                "canary_loaded": self.canary_engine is not None and self.canary_engine.model_loaded,
                "routing_strategy": self.routing_strategy
            }

    def update_config(
        self,
        enabled: Optional[bool] = None,
        canary_percentage: Optional[float] = None,
        canary_path: Optional[str] = None,
        champion_name: Optional[str] = None,
        canary_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Dynamically adjusts routing parameters or loads a new candidate model without restarting."""
        if canary_path:
            self.load_canary_model(canary_path)

        with self._lock:
            if enabled is not None:
                self.enabled = enabled
            if canary_percentage is not None:
                self.canary_percentage = max(0.0, min(100.0, canary_percentage))
            if champion_name:
                self.champion_name = champion_name
            if canary_name:
                self.canary_name = canary_name

        logger.info(f"Canary routing updated: enabled={self.enabled}, split={self.canary_percentage}%")
        return self.get_config()

    def get_comparative_metrics(self) -> Dict[str, Any]:
        """Calculates side-by-side performance analytics between Champion and Canary models."""
        with self._lock:
            def _compile_stats(variant: str, engine_obj: Optional[YOLOInferenceEngine], name: str):
                data = self.telemetry[variant]
                reqs = data["requests"]
                lats = list(data["latencies"])
                mean_lat = round(sum(lats) / len(lats), 2) if lats else 0.0
                min_lat = round(min(lats), 2) if lats else 0.0
                max_lat = round(max(lats), 2) if lats else 0.0
                p95_lat = round(float(np.percentile(lats, 95)), 2) if lats else 0.0
                det_rate = round((data["defective_requests"] / reqs) * 100.0, 2) if reqs > 0 else 0.0

                return {
                    "model_name": name,
                    "model_path": str(engine_obj.model_path) if engine_obj else "N/A",
                    "model_variant": variant,
                    "inferences_count": reqs,
                    "mean_latency_ms": mean_lat,
                    "min_latency_ms": min_lat,
                    "max_latency_ms": max_lat,
                    "p95_latency_ms": p95_lat,
                    "defective_inferences": data["defective_requests"],
                    "defect_detection_rate_percent": det_rate,
                    "total_defects_found": data["total_defects"]
                }

            champ_stats = _compile_stats("champion", self.champion_engine, self.champion_name)
            canary_stats = _compile_stats("canary", self.canary_engine, self.canary_name)

            return {
                "canary_enabled": self.enabled,
                "canary_percentage": self.canary_percentage,
                "routing_strategy": self.routing_strategy,
                "total_inferences": champ_stats["inferences_count"] + canary_stats["inferences_count"],
                "champion": champ_stats,
                "canary": canary_stats
            }

    def promote_challenger(self) -> Dict[str, Any]:
        """
        Promotes the challenger model into primary Champion status.
        Resets canary traffic to 0% and disables canary mode.
        """
        with self._lock:
            if self.canary_engine is None or not self.canary_engine.model_loaded:
                return {
                    "status": "error",
                    "message": "Cannot promote: No valid challenger model is currently loaded.",
                    "config": self.get_config()
                }

            promoted_name = self.canary_name
            self.champion_engine = self.canary_engine
            self.champion_name = f"{promoted_name} (Promoted)"
            
            # Reset canary
            self.canary_engine = None
            self.canary_path = None
            self.canary_name = "Challenger-Candidate"
            self.enabled = False
            self.canary_percentage = 0.0

            logger.info(f"🏆 Challenger model '{promoted_name}' successfully PROMOTED to Champion!")

            return {
                "status": "success",
                "message": f"Challenger model '{promoted_name}' successfully promoted to primary Champion. 100% traffic restored to Champion.",
                "config": self.get_config()
            }

    def rollback(self) -> Dict[str, Any]:
        """
        Instant emergency kill-switch.
        Immediately diverts 100% of factory traffic to Champion with zero downtime.
        """
        with self._lock:
            self.enabled = False
            self.canary_percentage = 0.0
            logger.warning("🚨 CANARY ROLLBACK TRIGGERED: 100% traffic redirected to Champion.")
            return {
                "status": "success",
                "message": "Canary traffic successfully killed. 100% traffic routed to primary Champion.",
                "config": self.get_config()
            }

# Singleton router instance initialized with the primary production engine
canary_router = CanaryRouter(champion_engine=default_champion_engine)
