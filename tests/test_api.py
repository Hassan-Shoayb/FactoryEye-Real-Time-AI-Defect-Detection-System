import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.config import API_VERSION

client = TestClient(app)

def create_synthetic_image_bytes(width: int = 300, height: int = 300) -> bytes:
    img = np.random.randint(100, 180, (height, width, 3), dtype=np.uint8)
    cv2.line(img, (20, 20), (280, 280), (50, 50, 50), 3)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "model_path" in data
    assert data["version"] == API_VERSION

def test_prometheus_metrics_endpoint():
    """Verify /metrics endpoint returns valid Prometheus exposition text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "factoryeye_images_processed_total" in text
    assert "factoryeye_inference_latency_ms" in text

def test_drift_stats_endpoint():
    """Verify /drift/stats returns rolling window statistics."""
    response = client.get("/drift/stats")
    assert response.status_code == 200
    data = response.json()
    assert "rolling_avg_confidence" in data
    assert "drift_detected" in data
    assert "active_learning_queue_size" in data

def test_predict_image_success():
    img_bytes = create_synthetic_image_bytes()
    files = {"file": ("test_surface.jpg", img_bytes, "image/jpeg")}
    
    response = client.post("/predict?conf=0.20", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "defect_count" in data
    assert "defect_detected" in data
    assert "inference_ms" in data
    assert isinstance(data["detections"], list)
    assert isinstance(data["inference_ms"], (int, float))

def test_predict_rejects_non_image():
    files = {"file": ("test.txt", b"This is not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400

def test_predict_rejects_empty_file():
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400

if __name__ == "__main__":
    print("Running FactoryEye API Tests...")
    test_health_endpoint()
    print("  ✓ test_health_endpoint passed")
    test_prometheus_metrics_endpoint()
    print("  ✓ test_prometheus_metrics_endpoint passed")
    test_drift_stats_endpoint()
    print("  ✓ test_drift_stats_endpoint passed")
    test_predict_image_success()
    print("  ✓ test_predict_image_success passed")
    test_predict_rejects_non_image()
    print("  ✓ test_predict_rejects_non_image passed")
    test_predict_rejects_empty_file()
    print("  ✓ test_predict_rejects_empty_file passed")
    print("\n🎉 ALL API TESTS PASSED SUCCESSFULLY!")
