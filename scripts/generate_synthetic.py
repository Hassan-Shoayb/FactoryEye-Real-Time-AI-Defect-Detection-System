#!/usr/bin/env python3
"""
FactoryEye — Procedural Synthetic Metal Defect Generator
Synthesizes realistic metal surface textures and defect patterns (scratches, inclusions, pits)
with corresponding normalized YOLO bounding box label annotations.
"""
import argparse
import random
from pathlib import Path
import numpy as np
import cv2

DEFECT_CLASSES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches"
}

def generate_steel_texture(w: int = 640, h: int = 640) -> np.ndarray:
    """Generates a brushed metallic steel surface texture."""
    base_gray = random.randint(130, 175)
    noise = np.random.normal(base_gray, 12, (h, w)).astype(np.uint8)
    
    # Apply directional brush blur
    kernel_size = 9
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
    brushed = cv2.filter2D(noise, -1, kernel)
    return cv2.cvtColor(brushed, cv2.COLOR_GRAY2BGR)

def add_synthetic_scratch(img: np.ndarray) -> list:
    """Draws a linear jagged surface scratch and returns normalized YOLO bbox."""
    h, w = img.shape[:2]
    x1, y1 = random.randint(40, w - 150), random.randint(40, h - 150)
    length = random.randint(80, 220)
    angle = random.uniform(0.2, 2.8)
    x2 = int(x1 + length * np.cos(angle))
    y2 = int(y1 + length * np.sin(angle))
    
    # Clip
    x2, y2 = min(w - 20, max(20, x2)), min(h - 20, max(20, y2))
    scratch_color = (random.randint(40, 75), random.randint(40, 75), random.randint(40, 75))
    cv2.line(img, (x1, y1), (x2, y2), scratch_color, thickness=random.randint(2, 4))
    
    # Compute normalized YOLO bbox
    min_x, max_x = min(x1, x2) - 5, max(x1, x2) + 5
    min_y, max_y = min(y1, y2) - 5, max(y1, y2) + 5
    bx = ((min_x + max_x) / 2.0) / w
    by = ((min_y + max_y) / 2.0) / h
    bw = (max_x - min_x) / w
    bh = (max_y - min_y) / h
    return [5, bx, by, bw, bh]  # class 5 = scratches

def add_synthetic_inclusion(img: np.ndarray) -> list:
    """Draws an irregular dark inclusion spot and returns normalized YOLO bbox."""
    h, w = img.shape[:2]
    cx, cy = random.randint(80, w - 80), random.randint(80, h - 80)
    radius = random.randint(15, 45)
    
    color = (random.randint(30, 60), random.randint(30, 60), random.randint(30, 60))
    cv2.circle(img, (cx, cy), radius, color, -1)
    cv2.GaussianBlur(img, (5, 5), 0, dst=img)
    
    bx = cx / w
    by = cy / h
    bw = (radius * 2 + 8) / w
    bh = (radius * 2 + 8) / h
    return [1, bx, by, bw, bh]  # class 1 = inclusion

def generate_samples(count: int, output_dir: str = "data/synthetic_samples"):
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    labels_dir = out_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"✨ Generating {count} synthetic metal defect samples in {out_path}...")
    for i in range(count):
        img = generate_steel_texture()
        bboxes = []
        
        # Add 1-3 random defects
        num_defects = random.randint(1, 3)
        for _ in range(num_defects):
            if random.random() > 0.5:
                bboxes.append(add_synthetic_scratch(img))
            else:
                bboxes.append(add_synthetic_inclusion(img))

        img_file = images_dir / f"synthetic_{i+1:04d}.jpg"
        lbl_file = labels_dir / f"synthetic_{i+1:04d}.txt"

        cv2.imwrite(str(img_file), img)
        with open(lbl_file, "w") as f:
            for b in bboxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    print(f"✓ Created {count} synthetic training images and YOLO label files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of synthetic samples to generate")
    parser.add_argument("--output-dir", default="data/synthetic_samples", help="Output directory")
    args = parser.parse_args()
    generate_samples(args.count, args.output_dir)
