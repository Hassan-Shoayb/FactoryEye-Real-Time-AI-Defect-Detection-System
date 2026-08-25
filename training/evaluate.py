#!/usr/bin/env python3
"""
FactoryEye — Model Validation & Evaluation CLI
Evaluates fine-tuned weights on validation/test splits with per-class AP50 breakdown.
"""
import argparse
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="FactoryEye Model Evaluation CLI")
    parser.add_argument("--weights", type=str, default="training/runs/train/weights/best.pt", help="Path to weights")
    parser.add_argument("--data", type=str, default="data/processed/data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌ Weights not found at {weights_path}")
        return

    print(f"Evaluating {weights_path} on {args.data}...")
    model = YOLO(str(weights_path))
    val_metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)

    print("═" * 55)
    print("             OVERALL VALIDATION RESULTS           ")
    print("═" * 55)
    print(f"  • mAP50:      {val_metrics.box.map50:.4f}")
    print(f"  • mAP50-95:   {val_metrics.box.map:.4f}")
    print(f"  • Precision:  {val_metrics.box.mp:.4f}")
    print(f"  • Recall:     {val_metrics.box.mr:.4f}")
    print("═" * 55)

if __name__ == "__main__":
    main()
