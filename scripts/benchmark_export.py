#!/usr/bin/env python3
"""
FactoryEye — Multi-Backend Model Benchmarker & Optimizer
Exports PyTorch weights to ONNX Runtime and benchmarks inference latency, FPS, and model size.
"""
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def benchmark_model(model_path: str, warmup: int = 10, runs: int = 50) -> dict:
    model = YOLO(model_path)
    # Synthetic test frame (640x640)
    dummy_frame = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        model.predict(dummy_frame, verbose=False)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(dummy_frame, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    fps = 1000.0 / mean_latency
    file_size_mb = Path(model_path).stat().st_size / (1024 * 1024)

    return {
        "backend": Path(model_path).suffix.upper().replace(".", "") or "PyTorch",
        "file_size_mb": round(file_size_mb, 2),
        "mean_latency_ms": round(mean_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "fps": round(fps, 1)
    }

def main():
    pt_path = Path("training/runs/train/weights/best.pt")
    if not pt_path.exists():
        pt_path = Path("yolov8n.pt")

    print(f"🚀 Benchmarking base model: {pt_path}...")
    pt_results = benchmark_model(str(pt_path))

    # Export to ONNX
    print("\n📦 Exporting to ONNX Runtime format...")
    model = YOLO(str(pt_path))
    onnx_path = model.export(format="onnx", dynamic=True, simplify=True)

    print(f"🚀 Benchmarking optimized ONNX model: {onnx_path}...")
    onnx_results = benchmark_model(str(onnx_path))

    print("\n" + "═" * 70)
    print("               FACTORYEYE EDGE INFERENCE BENCHMARK REPORT         ")
    print("═" * 70)
    print(f"{'Backend':<18} | {'Size (MB)':<10} | {'Mean Latency':<14} | {'P95 Latency':<12} | {'FPS':<8}")
    print("-" * 70)
    for res in [pt_results, onnx_results]:
        print(f"{res['backend']:<18} | {res['file_size_mb']:<10.2f} | {res['mean_latency_ms']:<11.2f} ms | {res['p95_latency_ms']:<9.2f} ms | {res['fps']:<8.1f}")
    print("═" * 70)

if __name__ == "__main__":
    main()
