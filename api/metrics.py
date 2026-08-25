import time
import threading
from collections import defaultdict
from typing import Dict, List

class MetricsCollector:
    """
    Thread-safe Prometheus-compatible metrics collector for FactoryEye.
    Tracks inference latency distributions, defect counts by class, and active streams.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.images_processed_total = 0
        self.frames_streamed_total = 0
        self.defects_total = defaultdict(int)
        self.active_streams = 0
        self.latency_buckets = [5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        self.latency_counts = defaultdict(int)
        self.latency_sum_ms = 0.0
        self.latency_count = 0

    def record_inference(self, latency_ms: float, detections: List[Dict]):
        with self._lock:
            self.images_processed_total += 1
            self.latency_count += 1
            self.latency_sum_ms += latency_ms
            for bucket in self.latency_buckets:
                if latency_ms <= bucket:
                    self.latency_counts[bucket] += 1
            for d in detections:
                label = d.get("label", "unknown") if isinstance(d, dict) else getattr(d, "label", "unknown")
                self.defects_total[label] += 1

    def record_frame_streamed(self):
        with self._lock:
            self.frames_streamed_total += 1

    def stream_connected(self):
        with self._lock:
            self.active_streams += 1

    def stream_disconnected(self):
        with self._lock:
            self.active_streams = max(0, self.active_streams - 1)

    def generate_prometheus_metrics(self) -> str:
        """Generates Prometheus text-formatted exposition output."""
        lines = [
            "# HELP factoryeye_images_processed_total Total number of images processed",
            "# TYPE factoryeye_images_processed_total counter",
            f"factoryeye_images_processed_total {self.images_processed_total}",
            "",
            "# HELP factoryeye_frames_streamed_total Total number of WebSocket video frames streamed",
            "# TYPE factoryeye_frames_streamed_total counter",
            f"factoryeye_frames_streamed_total {self.frames_streamed_total}",
            "",
            "# HELP factoryeye_active_streams Current number of active live camera WebSocket streams",
            "# TYPE factoryeye_active_streams gauge",
            f"factoryeye_active_streams {self.active_streams}",
            "",
            "# HELP factoryeye_defects_total Total detected defects categorized by class",
            "# TYPE factoryeye_defects_total counter"
        ]
        for defect_cls, count in self.defects_total.items():
            lines.append(f'factoryeye_defects_total{{defect_class="{defect_cls}"}} {count}')

        lines.extend([
            "",
            "# HELP factoryeye_inference_latency_ms Inference latency in milliseconds",
            "# TYPE factoryeye_inference_latency_ms histogram"
        ])
        for b in self.latency_buckets:
            lines.append(f'factoryeye_inference_latency_ms_bucket{{le="{b}"}} {self.latency_counts[b]}')
        lines.append(f'factoryeye_inference_latency_ms_bucket{{le="+Inf"}} {self.latency_count}')
        lines.append(f"factoryeye_inference_latency_ms_sum {self.latency_sum_ms:.2f}")
        lines.append(f"factoryeye_inference_latency_ms_count {self.latency_count}")

        return "\n".join(lines) + "\n"

metrics_collector = MetricsCollector()
