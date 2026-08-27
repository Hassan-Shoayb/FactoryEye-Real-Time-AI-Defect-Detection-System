import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.config import API_VERSION
from api.database import audit_db

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
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "factoryeye_images_processed_total" in text
    assert "factoryeye_inference_latency_ms" in text

def test_drift_stats_endpoint():
    response = client.get("/drift/stats")
    assert response.status_code == 200
    data = response.json()
    assert "rolling_avg_confidence" in data
    assert "drift_detected" in data

def test_audit_defects_and_summary_endpoints():
    stats_res = client.get("/audit/stats/summary")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert "total_inspections" in stats
    assert "quality_yield_percent" in stats

    defects_res = client.get("/audit/defects?limit=10")
    assert defects_res.status_code == 200
    data = defects_res.json()
    assert "total" in data
    assert "records" in data

def test_audit_export_endpoint():
    """Verify /audit/export returns CSV content."""
    response = client.get("/audit/export")
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    assert "ID,Inspection_ID,Datetime_UTC" in response.text

def test_explainability_heatmap_endpoint():
    """Verify /explain generates jet-colormap saliency maps."""
    img_bytes = create_synthetic_image_bytes()
    files = {"file": ("test_surface.jpg", img_bytes, "image/jpeg")}
    response = client.post("/explain?conf=0.20", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "annotated_image" in data
    assert data["annotated_image"].startswith("data:image/jpeg;base64,")

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
    test_audit_defects_and_summary_endpoints()
    print("  ✓ test_audit_defects_and_summary_endpoints passed")
    test_audit_export_endpoint()
    print("  ✓ test_audit_export_endpoint passed")
    test_explainability_heatmap_endpoint()
    print("  ✓ test_explainability_heatmap_endpoint passed")
    test_predict_image_success()
    print("  ✓ test_predict_image_success passed")
    test_predict_rejects_non_image()
    print("  ✓ test_predict_rejects_non_image passed")
    test_predict_rejects_empty_file()
    print("  ✓ test_predict_rejects_empty_file passed")
    print("\n🎉 ALL 9 API TESTS PASSED SUCCESSFULLY!")
