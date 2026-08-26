#!/usr/bin/env python3
"""
FactoryEye — High-Throughput Industrial Stream Benchmark Simulator
Simulates a factory IP camera streaming frames to WebSocket or REST endpoints.
"""
import time
import cv2
import argparse
import numpy as np
import httpx

def simulate_rest_stream(api_url: str = "http://localhost:8000/predict", count: int = 50):
    print(f"🚀 Simulating {count} factory line camera frame inspections to {api_url}...")
    latencies = []
    defects_found = 0

    dummy_frame = np.random.randint(100, 180, (640, 640, 3), dtype=np.uint8)
    cv2.line(dummy_frame, (50, 50), (500, 450), (40, 40, 40), 3)
    _, buffer = cv2.imencode(".jpg", dummy_frame)
    img_bytes = buffer.tobytes()

    client = httpx.Client(timeout=10.0)

    for i in range(count):
        t0 = time.perf_counter()
        files = {"file": (f"frame_{i}.jpg", img_bytes, "image/jpeg")}
        resp = client.post(api_url, files=files)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        if resp.status_code == 200:
            data = resp.json()
            if data.get("defect_detected"):
                defects_found += 1
            print(f"  Frame [{i+1:02d}/{count}] Latency: {lat:.1f}ms | Defects: {data.get('defect_count')}")
        else:
            print(f"  Frame [{i+1:02d}/{count}] Failed: HTTP {resp.status_code}")

    mean_lat = np.mean(latencies)
    print("\n" + "═" * 55)
    print("             STREAM SIMULATION REPORT             ")
    print("═" * 55)
    print(f"  • Frames Tested:      {count}")
    print(f"  • Mean Roundtrip:     {mean_lat:.1f} ms")
    print(f"  • Max Simulated FPS:  {1000.0 / mean_lat:.1f} FPS")
    print(f"  • Defect Detections:  {defects_found}")
    print("═" * 55)

if __name__ == "__main__":
    simulate_rest_stream()
