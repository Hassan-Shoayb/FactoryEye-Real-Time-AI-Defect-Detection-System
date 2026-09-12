import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from api.config import API_VERSION
from api.database import audit_db
from api.drift import drift_monitor
from api.canary import canary_router

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
    response = client.get("/audit/export")
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    assert "ID,Inspection_ID,Datetime_UTC" in response.text

def test_audit_defect_trends_endpoint():
    """Verify /audit/stats/trends returns Pareto and hourly velocity analytics."""
    response = client.get("/audit/stats/trends")
    assert response.status_code == 200
    data = response.json()
    assert "total_defects_recorded" in data
    assert "pareto_class_distribution" in data
    assert "hourly_defect_velocity" in data
    assert isinstance(data["pareto_class_distribution"], list)

def test_audit_stations_endpoint():
    """Verify /audit/stations returns station breakdown."""
    response = client.get("/audit/stations")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "station_id" in data[0]
        assert "quality_yield_percent" in data[0]

def test_system_hardware_info_endpoint():
    """Verify /system/info returns runtime hardware and acceleration metrics."""
    response = client.get("/system/info")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_count" in data
    assert "ram_total_gb" in data
    assert "inference_device" in data
    assert "backend_type" in data

def test_defect_roi_crop_endpoint():
    """Verify /audit/defects/{id}/crop returns a cropped flaw thumbnail."""
    # Ensure at least one defect record exists
    insp_id = audit_db.log_inspection(
        defect_count=1,
        detections=[{"label": "scratches", "confidence": 0.88, "bbox": [40, 40, 180, 180]}],
        inference_ms=12.5,
        source="ROI Unit Test",
        station_id="TEST_STATION"
    )
    records, total = audit_db.query_defects(limit=1, offset=0)
    assert total > 0
    defect_id = records[0]["id"]

    response = client.get(f"/audit/defects/{defect_id}/crop")
    assert response.status_code == 200
    assert "image/jpeg" in response.headers.get("content-type", "")
    assert len(response.content) > 100

def test_quality_certificate_endpoint():
    """Verify /audit/certificate returns printable compliance sheet."""
    response = client.get("/audit/certificate?batch_id=TEST-BATCH-01")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "FactoryEye Quality Compliance Certificate" in response.text

def test_explainability_heatmap_endpoint():
    img_bytes = create_synthetic_image_bytes()
    files = {"file": ("test_surface.jpg", img_bytes, "image/jpeg")}
    response = client.post("/explain?conf=0.20", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "annotated_image" in data
    assert "severity_grade" in data
    assert "action_recommendation" in data
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
    assert "severity_grade" in data
    assert "severity_score" in data
    assert "defect_coverage_percent" in data
    assert "action_recommendation" in data
    assert "inference_ms" in data

def test_predict_rejects_non_image():
    files = {"file": ("test.txt", b"This is not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400

def test_predict_rejects_empty_file():
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400

def test_active_learning_queue_endpoint():
    """Verify /active-learning/queue lists ambiguous defect candidates."""
    response = client.get("/active-learning/queue?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        sample = data[0]
        assert "filename" in sample
        assert "confidence_estimate" in sample
        assert "timestamp_utc" in sample

def test_active_learning_review_endpoint():
    """Verify /active-learning/review approves/promotes and discards samples."""
    # 1. Create temporary sample in queue
    dummy_name = "test_eval_conf_45.jpg"
    dummy_path = drift_monitor.queue_dir / dummy_name
    dummy_path.write_bytes(create_synthetic_image_bytes(100, 100))

    # 2. Verify serving raw sample image
    img_res = client.get(f"/active-learning/sample/{dummy_name}")
    assert img_res.status_code == 200
    assert len(img_res.content) > 0

    # 3. Approve sample and assert dataset promotion
    rev_res = client.post(
        "/active-learning/review",
        json={"filename": dummy_name, "action": "approve", "verified_class": "crazing"}
    )
    assert rev_res.status_code == 200
    res_data = rev_res.json()
    assert res_data["status"] == "success"
    assert res_data["action"] == "approve"
    assert res_data["verified_class"] == "crazing"

    promoted_img = drift_monitor.queue_dir.parent / "curated_training_set" / "images" / dummy_name
    promoted_lbl = drift_monitor.queue_dir.parent / "curated_training_set" / "labels" / "test_eval_conf_45.txt"
    assert promoted_img.exists()
    assert promoted_lbl.exists()

    # Clean up promoted test files
    if promoted_img.exists():
        promoted_img.unlink()
    if promoted_lbl.exists():
        promoted_lbl.unlink()

    # 4. Discard sample test
    discard_name = "test_discard_conf_45.jpg"
    discard_path = drift_monitor.queue_dir / discard_name
    discard_path.write_bytes(create_synthetic_image_bytes(100, 100))
    
    disc_res = client.post(
        "/active-learning/review",
        json={"filename": discard_name, "action": "discard"}
    )
    assert disc_res.status_code == 200
    assert disc_res.json()["status"] == "success"
    assert not discard_path.exists()

    # 5. Non-existent sample error handling
    err_res = client.post(
        "/active-learning/review",
        json={"filename": "non_existent_defect_file.jpg", "action": "approve"}
    )
    assert err_res.status_code == 400

def test_canary_config_endpoints():
    """Verify /canary/config GET and POST update behaviors."""
    # 1. GET initial config
    res = client.get("/canary/config")
    assert res.status_code == 200
    cfg = res.json()
    assert "enabled" in cfg
    assert "canary_percentage" in cfg
    assert "champion_name" in cfg
    assert "routing_strategy" in cfg

    # 2. Update config dynamically
    update_payload = {
        "enabled": True,
        "canary_percentage": 30.0,
        "champion_name": "Champion-Primary-v1",
        "canary_name": "Canary-Candidate-v2"
    }
    post_res = client.post("/canary/config", json=update_payload)
    assert post_res.status_code == 200
    new_cfg = post_res.json()
    assert new_cfg["enabled"] is True
    assert new_cfg["canary_percentage"] == 30.0
    assert new_cfg["champion_name"] == "Champion-Primary-v1"
    assert new_cfg["canary_name"] == "Canary-Candidate-v2"

    # Reset back to default
    client.post("/canary/config", json={"enabled": False, "canary_percentage": 20.0})

def test_canary_routing_and_metrics():
    """Verify inference routing tagged with model_variant and /canary/metrics telemetry."""
    img_bytes = create_synthetic_image_bytes()
    files = {"file": ("test_canary.jpg", img_bytes, "image/jpeg")}

    # 1. Test deterministic routing using X-FactoryEye-Model header
    headers = {"X-FactoryEye-Model": "champion"}
    res = client.post("/predict?conf=0.20", files=files, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["model_variant"] == "champion"
    assert "model_name" in data

    # 2. Check /canary/metrics
    met_res = client.get("/canary/metrics")
    assert met_res.status_code == 200
    met = met_res.json()
    assert "champion" in met
    assert "canary" in met
    assert met["champion"]["inferences_count"] > 0
    assert "mean_latency_ms" in met["champion"]

    # 3. Verify Prometheus metric exposition
    prom_res = client.get("/metrics")
    assert prom_res.status_code == 200
    assert 'factoryeye_canary_inferences_total{model_variant="champion"}' in prom_res.text

def test_canary_rollback_and_promote():
    """Verify emergency rollback kill-switch and candidate promotion."""
    # 1. Configure and enable canary
    client.post("/canary/config", json={"enabled": True, "canary_percentage": 50.0})

    # 2. Emergency rollback kill-switch
    roll_res = client.post("/canary/rollback")
    assert roll_res.status_code == 200
    roll_data = roll_res.json()
    assert roll_data["status"] == "success"
    assert roll_data["config"]["enabled"] is False
    assert roll_data["config"]["canary_percentage"] == 0.0

    # 3. Promotion without candidate returns 400
    canary_router.canary_engine = None
    prom_fail = client.post("/canary/promote")
    assert prom_fail.status_code == 400

    # 4. Attach valid candidate engine and test successful promotion
    canary_router.canary_engine = canary_router.champion_engine
    canary_router.canary_name = "NextGen-Model"
    canary_router.canary_path = "training/runs/train/weights/best.pt"
    
    prom_res = client.post("/canary/promote")
    assert prom_res.status_code == 200
    assert prom_res.json()["status"] == "success"
    current_cfg = client.get("/canary/config").json()
    assert "NextGen-Model" in current_cfg["champion_name"]
    assert current_cfg["enabled"] is False

def test_retrain_curated_summary():
    """Verify /retrain/curated-summary returns pool counts and class distribution."""
    res = client.get("/retrain/curated-summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_curated_images" in data
    assert "total_curated_labels" in data
    assert "class_distribution" in data
    assert "ready_for_retraining" in data
    assert isinstance(data["total_curated_images"], int)
    assert isinstance(data["class_distribution"], dict)

def test_retrain_trigger_and_status():
    """Verify triggering asynchronous retraining, polling progress, SLA gate check, and canary mount."""
    payload = {
        "epochs": 3,
        "batch_size": 8,
        "auto_mount_canary": True,
        "canary_split_percent": 25.0,
        "sla_max_latency_ms": 40.0,
        "dry_run": True
    }
    trigger_res = client.post("/retrain/trigger", json=payload)
    assert trigger_res.status_code == 200
    initial_status = trigger_res.json()
    assert "job_id" in initial_status
    assert initial_status["status"] in ["PENDING", "DATASET_PREP", "TRAINING", "GATE_EVALUATION", "CANARY_MOUNT", "COMPLETED"]

    job_id = initial_status["job_id"]

    # Poll until terminal state (COMPLETED or FAILED) with timeout
    final_status = None
    for _ in range(40):
        time.sleep(0.1)
        stat_res = client.get(f"/retrain/status/{job_id}")
        assert stat_res.status_code == 200
        stat_data = stat_res.json()
        if stat_data["status"] in ["COMPLETED", "FAILED"]:
            final_status = stat_data
            break

    assert final_status is not None, "Retraining job did not finish within timeout"
    assert final_status["status"] == "COMPLETED"
    assert final_status["progress_percent"] == 100.0
    assert final_status["gate_passed"] is True
    assert final_status["canary_mounted"] is True
    assert len(final_status["logs"]) > 0

    # Test 404 for invalid job ID
    err_res = client.get("/retrain/status/non_existent_retrain_job_xyz")
    assert err_res.status_code == 404

def test_retrain_jobs_list():
    """Verify /retrain/jobs lists all executed background jobs."""
    res = client.get("/retrain/jobs")
    assert res.status_code == 200
    data = res.json()
    assert "total_jobs" in data
    assert "jobs" in data
    assert data["total_jobs"] >= 1
    assert any(j["status"] == "COMPLETED" for j in data["jobs"])

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
    test_audit_defect_trends_endpoint()
    print("  ✓ test_audit_defect_trends_endpoint passed")
    test_audit_stations_endpoint()
    print("  ✓ test_audit_stations_endpoint passed")
    test_system_hardware_info_endpoint()
    print("  ✓ test_system_hardware_info_endpoint passed")
    test_defect_roi_crop_endpoint()
    print("  ✓ test_defect_roi_crop_endpoint passed")
    test_quality_certificate_endpoint()
    print("  ✓ test_quality_certificate_endpoint passed")
    test_explainability_heatmap_endpoint()
    print("  ✓ test_explainability_heatmap_endpoint passed")
    test_predict_image_success()
    print("  ✓ test_predict_image_success passed")
    test_predict_rejects_non_image()
    print("  ✓ test_predict_rejects_non_image passed")
    test_predict_rejects_empty_file()
    print("  ✓ test_predict_rejects_empty_file passed")
    test_active_learning_queue_endpoint()
    print("  ✓ test_active_learning_queue_endpoint passed")
    test_active_learning_review_endpoint()
    print("  ✓ test_active_learning_review_endpoint passed")
    test_canary_config_endpoints()
    print("  ✓ test_canary_config_endpoints passed")
    test_canary_routing_and_metrics()
    print("  ✓ test_canary_routing_and_metrics passed")
    test_canary_rollback_and_promote()
    print("  ✓ test_canary_rollback_and_promote passed")
    test_retrain_curated_summary()
    print("  ✓ test_retrain_curated_summary passed")
    test_retrain_trigger_and_status()
    print("  ✓ test_retrain_trigger_and_status passed")
    test_retrain_jobs_list()
    print("  ✓ test_retrain_jobs_list passed")
    print("\n🎉 ALL 22 API TESTS PASSED SUCCESSFULLY!")
