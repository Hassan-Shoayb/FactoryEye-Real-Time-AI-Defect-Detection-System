#!/usr/bin/env python3
"""
FactoryEye — Standalone Headless Training CLI
Fine-tunes YOLO backends with MLflow tracking, RAM caching, and Model Registry promotion.
"""
import argparse
import time
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import mlflow

def parse_args():
    parser = argparse.ArgumentParser(description="FactoryEye YOLO Model Training CLI")
    parser.add_argument("--data", type=str, default="data/processed/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model backbone")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--cache", action="store_true", default=True, help="Enable RAM caching for fast loading")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device (0, cpu, mps)")
    parser.add_argument("--project", type=str, default="training/runs", help="Output project directory")
    parser.add_argument("--name", type=str, default="train", help="Run name")
    parser.add_argument("--mlflow-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    return parser.parse_args()

def main():
    args = parse_args()
    print("═" * 60)
    print("         FACTORYEYE STANDALONE MODEL TRAINING CLI         ")
    print("═" * 60)
    print(f" • Base Model:      {args.model}")
    print(f" • Device:          {args.device}")
    print(f" • Epochs:          {args.epochs}")
    print(f" • Batch Size:      {args.batch}")
    print(f" • Image Size:      {args.imgsz}")
    print(f" • Dataset Config:  {args.data}")
    print("═" * 60)

    # MLflow Setup
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("defect-detection")

    with mlflow.start_run() as run:
        print(f"🚀 Active MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(vars(args))

        model = YOLO(args.model)
        start_time = time.time()
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            cache=args.cache,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            verbose=True
        )
        duration_sec = round(time.time() - start_time, 2)

        # Log metrics
        metrics = getattr(results, "results_dict", {})
        map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
        mlflow.log_metrics({
            "mAP50": map50,
            "mAP50_95": float(metrics.get("metrics/mAP50-95(B)", 0.0)),
            "precision": float(metrics.get("metrics/precision(B)", 0.0)),
            "recall": float(metrics.get("metrics/recall(B)", 0.0)),
            "duration_sec": duration_sec
        })

        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
            print(f"✓ Best weights saved & logged: {best_weights.resolve()}")

    print(f"\n✅ Training complete in {duration_sec}s. Final mAP50: {map50:.4f}")

if __name__ == "__main__":
    main()
