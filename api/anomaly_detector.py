import os
import io
import time
import json
import base64
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import cv2
import numpy as np

from api.schemas import (
    AnomalyBoundingBox,
    AnomalyDetectResponse,
    NovelFlawCandidate,
    NovelFlawListResponse,
    NovelFlawClassifyRequest,
    AnomalyStatsSummary
)

logger = logging.getLogger("factoryeye.anomaly")

class ZeroShotAnomalyDetector:
    """
    Zero-Shot Edge Anomaly Detection & Out-of-Distribution (OOD) Novel Flaw Discovery Engine.
    Employs dual-domain (Spatial Gradient Entropy + 2D FFT Spectral Residual) texture analysis
    to detect and localize unclassified surface disruptions without requiring labeled defect priors.
    """
    def __init__(self, data_dir: Optional[str] = None):
        self._lock = threading.RLock()
        self.base_dir = Path(data_dir or Path(__file__).resolve().parent.parent / "data")
        self.quarantine_dir = self.base_dir / "novel_flaw_candidates"
        self.curated_dir = self.base_dir / "curated_training_set"
        
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        (self.curated_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.curated_dir / "labels").mkdir(parents=True, exist_ok=True)

        self.default_threshold = 0.45
        self.recent_scores: List[float] = []
        self.max_history = 200
        self.total_scans = 0
        self.total_anomalies = 0

        # Seed pre-existing sample if quarantine is empty for out-of-the-box demonstration
        self._seed_demo_candidate_if_needed()

    def _seed_demo_candidate_if_needed(self):
        """Seeds a demonstrator novel flaw in quarantine if directory is empty."""
        try:
            existing = list(self.quarantine_dir.glob("*.jpg"))
            if not existing:
                demo_name = "novel_flaw_demo_roll_chatter_01.jpg"
                demo_path = self.quarantine_dir / demo_name
                meta_path = self.quarantine_dir / "novel_flaw_demo_roll_chatter_01.json"

                # Generate synthetic striped chatter texture
                h, w = 300, 300
                img = np.full((h, w, 3), 140, dtype=np.uint8)
                for i in range(20, 280, 15):
                    cv2.line(img, (i, 40), (i + 10, 260), (70, 70, 70), 2)
                cv2.imwrite(str(demo_path), img)

                meta = {
                    "candidate_id": "NOVEL-CHG-001",
                    "filename": demo_name,
                    "timestamp_utc": time.time() - 3600,
                    "datetime_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3600)),
                    "station_id": "STATION_01",
                    "anomaly_score": 0.78,
                    "status": "PENDING_REVIEW",
                    "bbox": [20, 40, 280, 260],
                    "classified_as": None
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not seed demo anomaly candidate: {e}")

    def compute_spectral_residual_saliency(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates 2D FFT Spectral Residual saliency map and spectral entropy.
        Hou & Zhang spectral residual algorithm for visual anomaly localization.
        """
        h, w = gray.shape
        # Compute 2D DFT
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        mag = np.abs(dft_shift)
        phase = np.angle(dft_shift)

        # Log spectrum
        log_mag = np.log(mag + 1e-6)

        # Average spectrum approximation via box filter
        avg_log_mag = cv2.blur(log_mag, (5, 5))

        # Spectral residual
        residual = log_mag - avg_log_mag

        # Inverse FFT to reconstruct spatial saliency map
        exp_res = np.exp(residual)
        reconstructed = exp_res * np.cos(phase) + 1j * exp_res * np.sin(phase)
        reconstructed_shift = np.fft.ifftshift(reconstructed)
        spatial_map = np.abs(np.fft.ifft2(reconstructed_shift))

        # Smooth and square
        saliency = cv2.GaussianBlur(spatial_map, (9, 9), 2.5) ** 2
        # Normalize to 0..1
        min_v, max_v = saliency.min(), saliency.max()
        if max_v > min_v:
            saliency = (saliency - min_v) / (max_v - min_v)
        else:
            saliency = np.zeros_like(saliency)

        # Calculate Shannon spectral entropy
        p = (mag / (mag.sum() + 1e-6)).flatten()
        p = p[p > 0]
        spectral_entropy = float(-np.sum(p * np.log2(p)))

        return saliency, spectral_entropy

    def analyze_frame(
        self,
        img_bgr: np.ndarray,
        threshold: Optional[float] = None,
        quarantine_if_anomalous: bool = False,
        station_id: str = "STATION_01"
    ) -> AnomalyDetectResponse:
        """
        Performs dual-domain anomaly detection on a single RGB/BGR steel inspection frame.
        """
        start_time = time.time()
        operating_thresh = threshold if threshold is not None else self.default_threshold
        h, w = img_bgr.shape[:2]

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()

        # 1. Spatial Domain: Gradient Magnitude & Variance
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_var = float(np.var(grad_mag) / 1000.0)

        # 2. Frequency Domain: 2D FFT Spectral Residual Saliency
        saliency_map, spectral_entropy = self.compute_spectral_residual_saliency(gray)

        # 3. Anomaly Scoring: top 5% intensity + gradient dispersion
        flat_saliency = np.sort(saliency_map.flatten())
        top_5_percent = flat_saliency[int(0.95 * len(flat_saliency)):]
        peak_intensity = float(np.mean(top_5_percent))

        # Composite normalized score in range [0.0, 1.0]
        raw_score = 0.65 * peak_intensity + 0.35 * min(1.0, grad_var / 5.0)
        anomaly_score = float(np.clip(raw_score, 0.0, 1.0))
        is_anomalous = bool(anomaly_score >= operating_thresh)

        # 4. Localized Anomaly Bounding Boxes
        bboxes: List[AnomalyBoundingBox] = []
        bin_mask = (saliency_map > 0.40).astype(np.uint8) * 255
        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if 120 <= area <= int(0.80 * h * w):
                bx, by, bw, bh = cv2.boundingRect(cnt)
                roi_intensity = float(np.mean(saliency_map[by:by + bh, bx:bx + bw]))
                bboxes.append(AnomalyBoundingBox(
                    bbox=[int(bx), int(by), int(bx + bw), int(by + bh)],
                    anomaly_intensity=round(roi_intensity, 3),
                    area_px=area,
                    suggested_tag="surface_texture_perturbation" if bw > bh * 2 else "localized_novel_flaw"
                ))

        # Sort by anomaly intensity descending
        bboxes.sort(key=lambda b: b.anomaly_intensity, reverse=True)

        # 5. Jet Colormap Anomaly Overlay Heatmap
        sal_uint8 = (saliency_map * 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(img_bgr, 0.60, color_heatmap, 0.40, 0)

        # Draw detected bounding boxes
        for b in bboxes[:5]:
            x1, y1, x2, y2 = b.bbox
            cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(blended, f"OOD: {b.anomaly_intensity:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        _, enc_buf = cv2.imencode(".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        b64_heatmap = f"data:image/jpeg;base64,{base64.b64encode(enc_buf.tobytes()).decode('utf-8')}"

        # 6. Quarantining into Novel Flaw Discovery Pool if anomalous
        quarantined = False
        quarantine_fn = None

        with self._lock:
            self.total_scans += 1
            self.recent_scores.append(anomaly_score)
            if len(self.recent_scores) > self.max_history:
                self.recent_scores.pop(0)

            if is_anomalous:
                self.total_anomalies += 1
                if quarantine_if_anomalous:
                    quarantined = True
                    timestamp_now = time.time()
                    quarantine_fn = f"novel_{int(timestamp_now * 1000)}_{np.random.randint(100, 999)}.jpg"
                    img_path = self.quarantine_dir / quarantine_fn
                    meta_path = self.quarantine_dir / f"{Path(quarantine_fn).stem}.json"
                    cv2.imwrite(str(img_path), img_bgr)

                    top_bbox = bboxes[0].bbox if bboxes else [0, 0, w, h]
                    meta = {
                        "candidate_id": f"NOVEL-{int(timestamp_now)}",
                        "filename": quarantine_fn,
                        "timestamp_utc": timestamp_now,
                        "datetime_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp_now)),
                        "station_id": station_id,
                        "anomaly_score": round(anomaly_score, 3),
                        "status": "PENDING_REVIEW",
                        "bbox": top_bbox,
                        "classified_as": None
                    }
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)

        elapsed_ms = (time.time() - start_time) * 1000.0

        return AnomalyDetectResponse(
            anomaly_score=round(anomaly_score, 3),
            is_anomalous=is_anomalous,
            threshold=operating_thresh,
            spectral_entropy=round(spectral_entropy, 2),
            gradient_variance=round(grad_var, 3),
            anomaly_bboxes=bboxes[:5],
            annotated_heatmap=b64_heatmap,
            quarantined=quarantined,
            quarantine_filename=quarantine_fn,
            inference_ms=round(elapsed_ms, 2)
        )

    def list_novel_flaws(self) -> NovelFlawListResponse:
        """Lists all quarantined out-of-distribution candidate flaws."""
        with self._lock:
            candidates: List[NovelFlawCandidate] = []
            json_files = sorted(self.quarantine_dir.glob("*.json"), key=os.path.getmtime, reverse=True)

            for jf in json_files:
                try:
                    with open(jf, "r") as f:
                        meta = json.load(f)
                    fn = meta.get("filename", "")
                    img_path = self.quarantine_dir / fn
                    if img_path.exists():
                        candidates.append(NovelFlawCandidate(
                            candidate_id=meta.get("candidate_id", f"NOVEL-{jf.stem}"),
                            filename=fn,
                            timestamp_utc=meta.get("timestamp_utc", 0.0),
                            datetime_iso=meta.get("datetime_iso", ""),
                            station_id=meta.get("station_id", "STATION_01"),
                            anomaly_score=meta.get("anomaly_score", 0.70),
                            status=meta.get("status", "PENDING_REVIEW"),
                            thumbnail_url=f"/anomaly/novel-flaws/{fn}/crop",
                            bbox=meta.get("bbox", [0, 0, 100, 100]),
                            classified_as=meta.get("classified_as")
                        ))
                except Exception as e:
                    logger.warning(f"Error reading novel flaw metadata {jf}: {e}")

            pending_cnt = sum(1 for c in candidates if c.status == "PENDING_REVIEW")
            return NovelFlawListResponse(
                total_candidates=len(candidates),
                pending_review_count=pending_cnt,
                candidates=candidates
            )

    def classify_and_promote(self, req: NovelFlawClassifyRequest) -> Dict[str, Any]:
        """
        Classifies a quarantined novel flaw and promotes it into curated retraining dataset,
        or discards it.
        """
        with self._lock:
            img_path = self.quarantine_dir / req.filename
            meta_path = self.quarantine_dir / f"{Path(req.filename).stem}.json"

            if not img_path.exists():
                return {"status": "error", "message": f"Candidate image {req.filename} not found in quarantine."}

            if req.action == "discard":
                try:
                    img_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    return {"status": "discarded", "filename": req.filename}
                except Exception as e:
                    return {"status": "error", "message": str(e)}

            elif req.action == "promote":
                assigned_class = (req.assigned_class or "novel_defect").strip().lower().replace(" ", "_")
                # Load metadata
                meta = {}
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)

                # Move image to curated training set
                dest_img_path = self.curated_dir / "images" / req.filename
                cv2.imwrite(str(dest_img_path), cv2.imread(str(img_path)))

                # Create YOLO label file with normalized bounding box
                img = cv2.imread(str(img_path))
                ih, iw = img.shape[:2] if img is not None else (300, 300)
                bx1, by1, bx2, by2 = meta.get("bbox", [50, 50, 200, 200])

                xc = ((bx1 + bx2) / 2.0) / iw
                yc = ((by1 + by2) / 2.0) / ih
                bw = (bx2 - bx1) / iw
                bh = (by2 - by1) / ih

                # Class index 6 (extending 0..5 base classes)
                label_path = self.curated_dir / "labels" / f"{Path(req.filename).stem}.txt"
                with open(label_path, "w") as f:
                    f.write(f"6 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                # Update metadata status
                meta["status"] = "CLASSIFIED"
                meta["classified_as"] = assigned_class
                if meta_path.exists():
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)

                logger.info(f"✓ Novel flaw {req.filename} promoted to curated dataset as new class '{assigned_class}'")
                return {
                    "status": "promoted",
                    "filename": req.filename,
                    "assigned_class": assigned_class,
                    "curated_image_path": str(dest_img_path),
                    "curated_label_path": str(label_path)
                }

            return {"status": "error", "message": f"Unknown action: {req.action}"}

    def get_stats(self) -> AnomalyStatsSummary:
        """Returns rolling statistical summary of surface anomaly and OOD behavior."""
        with self._lock:
            mean_score = float(np.mean(self.recent_scores)) if self.recent_scores else 0.22
            ood_rate = (self.total_anomalies / max(1, self.total_scans)) * 100.0

            # Baseline texture stability
            if mean_score < 0.35:
                stability = "STABLE"
            elif mean_score < 0.55:
                stability = "DRIFTING"
            else:
                stability = "HIGH_PERTURBATION"

            q_files = list(self.quarantine_dir.glob("*.jpg"))

            return AnomalyStatsSummary(
                rolling_mean_anomaly_score=round(mean_score, 3),
                ood_event_rate_percent=round(ood_rate, 2),
                total_scans_evaluated=self.total_scans,
                total_anomalies_flagged=self.total_anomalies,
                quarantined_pool_size=len(q_files),
                texture_baseline_stability=stability
            )

anomaly_detector = ZeroShotAnomalyDetector()
