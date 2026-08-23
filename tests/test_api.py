import io
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.config import API_VERSION

client = TestClient(app)

def create_synthetic_image_bytes(width: int = 300, height: int = 300) -> bytes:
    """Creates a synthetic test JPEG image in-memory."""
    img = np.random.randint(100, 180, (height, width, 3), dtype=np.uint8)
    cv2.line(img, (20, 20), (280, 280), (50, 50, 50), 3)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()

def test_health_endpoint():
    """Verify /health returns 200 and expected schema keys."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "model_path" in data
    assert data["version"] == API_VERSION

def test_predict_image_success():
    """Verify /predict processes a valid image and returns schema."""
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
    """Verify /predict rejects text or invalid MIME types with HTTP 400."""
    files = {"file": ("test.txt", b"This is not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400

def test_predict_rejects_empty_file():
    """Verify /predict rejects 0-byte file with HTTP 400."""
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400

if __name__ == "__main__":
    print("Running FactoryEye API Tests...")
    test_health_endpoint()
    print("  ✓ test_health_endpoint passed")
    test_predict_image_success()
    print("  ✓ test_predict_image_success passed")
    test_predict_rejects_non_image()
    print("  ✓ test_predict_rejects_non_image passed")
    test_predict_rejects_empty_file()
    print("  ✓ test_predict_rejects_empty_file passed")
    print("\n🎉 ALL API TESTS PASSED SUCCESSFULLY!")

