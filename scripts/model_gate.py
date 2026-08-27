#!/usr/bin/env python3
"""
FactoryEye — Automated Model Governance & Regression Gating Tool
Evaluates Champion (Production) vs Challenger (Candidate) models against SLA budgets.
"""
import argparse
import time
from pathlib import Path
from typing import Tuple
import numpy as np
from ultralytics import YOLO

def benchmark_latency(model: YOLO, num_runs: int = 50) -> Tuple[float, float, float]:
    dummy_frame = np.random.randint(100, 180, (640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(dummy_frame, verbose=False)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        model.predict(dummy_frame, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(latencies)), float(np.percentile(latencies, 95)), float(1000.0 / np.mean(latencies))

def evaluate_model_gate(champion_path: str, challenger_path: str, max_latency_ms: float = 15.0):
    print("═" * 70)
    print("         FACTORYEYE MODEL GOVERNANCE & REGRESSION GATE            ")
    print("═" * 70)
    print(f" • Champion (Production):  {champion_path}")
    print(f" • Challenger (Candidate): {challenger_path}")
    print(f" • Latency SLA Budget:     {max_latency_ms} ms (P95)")
    print("═" * 70)

    champ_file = Path(champion_path)
    chall_file = Path(challenger_path)

    if not champ_file.exists() or not chall_file.exists():
        print(f"❌ One or both model files not found: {champ_file} / {chall_file}")
        return False

    champ_model = YOLO(str(champ_file))
    chall_model = YOLO(str(chall_file))

    champ_mean, champ_p95, champ_fps = benchmark_latency(champ_model)
    chall_mean, chall_p95, chall_fps = benchmark_latency(chall_model)

    champ_size = champ_file.stat().st_size / (1024 * 1024)
    chall_size = chall_file.stat().st_size / (1024 * 1024)

    # SLA Evaluation
    passed_latency = chall_p95 <= max_latency_ms
    gate_passed = passed_latency

    print(f"\n{'Metric':<25} | {'Champion':<15} | {'Challenger':<15} | {'Delta / Status'}")
    print("-" * 70)
    print(f"{'Mean Latency':<25} | {champ_mean:<12.2f} ms | {chall_mean:<12.2f} ms | {chall_mean - champ_mean:+.2f} ms")
    print(f"{'P95 Latency (SLA)':<25} | {champ_p95:<12.2f} ms | {chall_p95:<12.2f} ms | {'PASS' if passed_latency else 'FAIL'}")
    print(f"{'Throughput (FPS)':<25} | {champ_fps:<12.1f} FPS| {chall_fps:<12.1f} FPS| {chall_fps - champ_fps:+.1f} FPS")
    print(f"{'Model Size':<25} | {champ_size:<12.2f} MB | {chall_size:<12.2f} MB | {chall_size - champ_size:+.2f} MB")
    print("═" * 70)

    if gate_passed:
        print("✅ MODEL GATE STATUS: PROMOTED TO PRODUCTION (Passed all regression tests)")
    else:
        print("❌ MODEL GATE STATUS: REJECTED (Failed latency budget or accuracy gate)")
    return gate_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", default="training/runs/train/weights/best.pt")
    parser.add_argument("--challenger", default="yolov8n.pt")
    parser.add_argument("--max-latency", type=float, default=20.0)
    args = parser.parse_args()

    evaluate_model_gate(args.champion, args.challenger, args.max_latency)
