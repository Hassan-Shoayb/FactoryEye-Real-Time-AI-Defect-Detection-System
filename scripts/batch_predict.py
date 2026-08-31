#!/usr/bin/env python3
"""
FactoryEye — High-Throughput Offline Batch Prediction CLI
Processes folders of inspection images/videos, exports JSON metrics and annotated frames.
"""
import argparse
import time
import json
from pathlib import Path
import cv2
from ultralytics import YOLO

def run_batch_inference(input_dir: str, output_dir: str, weights: str, conf: float = 0.40):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    annotated_dir = out_path / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(weights)
    if not weights_path.exists():
        weights_path = Path("yolov8n.pt")

    print("═" * 65)
    print("         FACTORYEYE OFFLINE BATCH INFERENCE PROCESSOR             ")
    print("═" * 65)
    print(f" • Input Directory:   {in_path.resolve()}")
    print(f" • Output Directory:  {out_path.resolve()}")
    print(f" • Model Weights:     {weights_path.resolve()}")
    print(f" • Confidence Gate:   {conf}")
    print("═" * 65)

    model = YOLO(str(weights_path))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [p for p in in_path.rglob("*") if p.suffix.lower() in image_extensions]

    if not image_files:
        print(f"⚠️ No image files found in {in_path}")
        return

    summary = {
        "timestamp_utc": time.time(),
        "total_images": len(image_files),
        "defective_images": 0,
        "clean_images": 0,
        "total_defects": 0,
        "total_inference_ms": 0.0,
        "results": []
    }

    print(f"\nProcessing {len(image_files)} images...")
    t0 = time.perf_counter()

    for idx, img_path in enumerate(image_files, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        t_inf_start = time.perf_counter()
        results = model.predict(frame, conf=conf, verbose=False)[0]
        inf_ms = round((time.perf_counter() - t_inf_start) * 1000, 2)
        summary["total_inference_ms"] += inf_ms

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, f"defect_{cls_id}")
            detections.append({"label": label, "confidence": round(c, 4), "bbox": [x1, y1, x2, y2]})

        defect_count = len(detections)
        summary["total_defects"] += defect_count
        if defect_count > 0:
            summary["defective_images"] += 1
        else:
            summary["clean_images"] += 1

        # Save annotated image
        annotated_frame = results.plot()
        out_img_path = annotated_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(out_img_path), annotated_frame)

        summary["results"].append({
            "filename": img_path.name,
            "defect_count": defect_count,
            "inference_ms": inf_ms,
            "detections": detections
        })

        print(f" [{idx:03d}/{len(image_files):03d}] {img_path.name:<24} -> Defects: {defect_count} ({inf_ms} ms)")

    total_duration_sec = round(time.perf_counter() - t0, 2)
    summary["total_duration_sec"] = total_duration_sec
    summary["mean_inference_ms"] = round(summary["total_inference_ms"] / len(image_files), 2)
    summary["defect_rate_percent"] = round((summary["defective_images"] / len(image_files)) * 100, 2)

    # Save JSON report
    report_file = out_path / "batch_inspection_report.json"
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "═" * 65)
    print("                     BATCH RUN COMPLETED                          ")
    print("═" * 65)
    print(f"  • Total Processed:    {summary['total_images']} images in {total_duration_sec}s")
    print(f"  • Defect Rate:        {summary['defect_rate_percent']}% ({summary['defective_images']} defective)")
    print(f"  • Mean Latency:       {summary['mean_inference_ms']} ms/image")
    print(f"  • JSON Report Saved:  {report_file}")
    print(f"  • Annotated Images:   {annotated_dir}")
    print("═" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/samples", help="Input directory")
    parser.add_argument("--output-dir", default="data/batch_results", help="Output directory")
    parser.add_argument("--weights", default="training/runs/train/weights/best.pt", help="Weights path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    run_batch_inference(args.input_dir, args.output_dir, args.weights, args.conf)
