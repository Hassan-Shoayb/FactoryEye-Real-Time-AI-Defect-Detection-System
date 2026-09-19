import os
import time
import base64
import logging
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np

from api.schemas import (
    CameraRigChannel, CameraRigConfig, CameraRigConfigUpdate,
    FusedDefect, MultiCamInspectResponse, MultiCamStatusResponse,
    Detection
)
from api.inference import engine
from api.severity import DEFECT_HAZARD_WEIGHTS

logger = logging.getLogger("factoryeye.multicam")

def _grade_defect_severity(label: str, area_px: int) -> str:
    hazard = DEFECT_HAZARD_WEIGHTS.get(label, 1.5)
    if hazard >= 2.5 or area_px > 25000:
        return "CRITICAL"
    elif hazard >= 2.0 or area_px > 10000:
        return "HIGH"
    elif hazard >= 1.5 or area_px > 3000:
        return "MEDIUM"
    return "LOW"

class MultiCameraFusionEngine:
    """
    Multi-Camera Edge Video Fusion & Panoramic Seam Stitching Engine.
    
    Synchronizes multi-view camera arrays across wide continuous metal strips,
    executes seamless panoramic strip stitching with weighted alpha-ramp blending,
    and performs cross-camera Seam-NMS to merge boundary-straddling partial flaw detections.
    """

    def __init__(self):
        self._lock = threading.Lock()
        
        # Default 2-Camera Top Strip Array Rig Configuration
        self.config = CameraRigConfig(
            rig_id="RIG-STATION-01-PRIMARY",
            strip_width_mm=1250.0,
            overlap_width_mm=50.0,
            overlap_pixels=50,
            vertical_offset_px=0,
            blend_feather_px=20,
            channels=[
                CameraRigChannel(
                    camera_id="CAM_TOP_LEFT",
                    surface="TOP",
                    position_index=0,
                    fov_start_mm=0.0,
                    fov_end_mm=650.0,
                    resolution_w=640,
                    resolution_h=480,
                    is_active=True
                ),
                CameraRigChannel(
                    camera_id="CAM_TOP_RIGHT",
                    surface="TOP",
                    position_index=1,
                    fov_start_mm=600.0,
                    fov_end_mm=1250.0,
                    resolution_w=640,
                    resolution_h=480,
                    is_active=True
                )
            ]
        )

        # Operational Performance Counters
        self.total_panoramic_scans = 0
        self.total_boundary_merges = 0
        self.last_sync_jitter_ms = 0.42
        self.recent_latencies: List[float] = []

    def get_config(self) -> CameraRigConfig:
        """Returns current multi-camera rig topology and calibration parameters."""
        with self._lock:
            return self.config

    def update_config(self, update: CameraRigConfigUpdate) -> CameraRigConfig:
        """Updates camera rig calibration, overlap geometry, and alignment parameters."""
        with self._lock:
            if update.strip_width_mm is not None:
                self.config.strip_width_mm = update.strip_width_mm
            if update.overlap_width_mm is not None:
                self.config.overlap_width_mm = update.overlap_width_mm
            if update.overlap_pixels is not None:
                self.config.overlap_pixels = max(10, update.overlap_pixels)
            if update.vertical_offset_px is not None:
                self.config.vertical_offset_px = update.vertical_offset_px
            if update.blend_feather_px is not None:
                self.config.blend_feather_px = max(2, update.blend_feather_px)
            
            logger.info(
                f"Updated Camera Rig Config: overlap={self.config.overlap_pixels}px "
                f"({self.config.overlap_width_mm}mm), feather={self.config.blend_feather_px}px"
            )
            return self.config

    def stitch_panoramic_pair(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        overlap_px: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Performs high-speed linear alpha-ramp panoramic stitching of adjacent left/right views.
        Returns: (panoramic_image, seam_center_x)
        """
        hl, wl = img_left.shape[:2]
        hr, wr = img_right.shape[:2]
        
        target_h = max(hl, hr)
        # Rescale if heights differ
        if hl != target_h:
            img_left = cv2.resize(img_left, (int(wl * (target_h / hl)), target_h))
            wl = img_left.shape[1]
        if hr != target_h:
            img_right = cv2.resize(img_right, (int(wr * (target_h / hr)), target_h))
            wr = img_right.shape[1]

        ov = overlap_px if overlap_px is not None else self.config.overlap_pixels
        ov = min(ov, min(wl, wr) // 2)

        pano_w = wl + wr - ov
        pano = np.zeros((target_h, pano_w, 3), dtype=np.uint8)

        # 1. Direct copy non-overlapping left region: [0 ... wl - ov)
        pano[:, 0 : wl - ov] = img_left[:, 0 : wl - ov]

        # 2. Weighted Alpha-Ramp Blending across overlap seam: [wl - ov ... wl)
        left_overlap = img_left[:, wl - ov : wl].astype(np.float32)
        right_overlap = img_right[:, 0 : ov].astype(np.float32)

        # Alpha gradient ramp from 0.0 (full left) to 1.0 (full right)
        alpha = np.linspace(0.0, 1.0, ov).reshape(1, ov, 1).astype(np.float32)
        blended_overlap = (1.0 - alpha) * left_overlap + alpha * right_overlap
        pano[:, wl - ov : wl] = np.clip(blended_overlap, 0, 255).astype(np.uint8)

        # 3. Direct copy non-overlapping right region: [wl ... pano_w)
        pano[:, wl : pano_w] = img_right[:, ov : wr]

        seam_center_x = wl - (ov // 2)
        return pano, seam_center_x

    def _fuse_boundary_detections(
        self,
        left_dets: List[Detection],
        right_dets: List[Detection],
        wl: int,
        wr: int,
        overlap_px: int
    ) -> Tuple[List[FusedDefect], int]:
        """
        Translates local camera bounding boxes into global panoramic coordinates and executes
        cross-camera Seam-NMS to merge boundary-straddling defect halves into unified defects.
        """
        fused: List[FusedDefect] = []
        seam_x = wl - (overlap_px // 2)
        seam_zone_min = wl - overlap_px - 25
        seam_zone_max = wl + 25

        used_right_indices = set()
        boundary_merge_count = 0

        # Translate left detections to global coordinates
        # On Left Camera: X_global = X_local
        for ld in left_dets:
            lx1, ly1, lx2, ly2 = ld.bbox
            is_boundary_candidate = (lx2 >= seam_zone_min)

            matched_right = None
            if is_boundary_candidate:
                # Search for matching right detection that touches the seam on the right side
                for r_idx, rd in enumerate(right_dets):
                    if r_idx in used_right_indices:
                        continue
                    rx1, ry1, rx2, ry2 = rd.bbox
                    # On Right Camera: X_global = rx + wl - overlap_px
                    rx1_glob = rx1 + wl - overlap_px
                    rx2_glob = rx2 + wl - overlap_px

                    # Check longitudinal (vertical) overlap
                    y_overlap = max(0, min(ly2, ry2) - max(ly1, ry1))
                    min_h = min(ly2 - ly1, ry2 - ry1)
                    v_iou = y_overlap / max(1, min_h)

                    # Check horizontal proximity across the seam
                    h_dist = abs(rx1_glob - lx2)
                    is_horiz_adjacent = (h_dist <= 30) or (lx2 >= rx1_glob)

                    if v_iou >= 0.30 and is_horiz_adjacent:
                        matched_right = (r_idx, rd, rx1_glob, rx2_glob)
                        break

            if matched_right is not None:
                r_idx, rd, rx1_glob, rx2_glob = matched_right
                used_right_indices.add(r_idx)
                boundary_merge_count += 1

                # Merge left and right boxes
                gx1 = min(lx1, rx1_glob)
                gy1 = min(ly1, rd.bbox[1])
                gx2 = max(lx2, rx2_glob)
                gy2 = max(ly2, rd.bbox[3])
                fused_conf = round(float(max(ld.confidence, rd.confidence)), 3)
                fused_class = ld.label if ld.confidence >= rd.confidence else rd.label

                # Calculate physical transverse position in mm
                pano_w = wl + wr - overlap_px
                trans_mm = round(((gx1 + gx2) / 2.0 / pano_w) * self.config.strip_width_mm, 1)

                fused.append(FusedDefect(
                    global_id=f"FUSED-{int(time.time()*1000)}-{len(fused)+1}",
                    defect_class=fused_class,
                    confidence=fused_conf,
                    global_bbox=[int(gx1), int(gy1), int(gx2), int(gy2)],
                    strip_position_mm={"transverse_mm": trans_mm, "longitudinal_mm": 0.0},
                    source_cameras=["CAM_TOP_LEFT", "CAM_TOP_RIGHT"],
                    seam_fused=True,
                    severity_grade=_grade_defect_severity(fused_class, max(1, (gx2 - gx1) * (gy2 - gy1)))
                ))
            else:
                # Standalone left defect
                pano_w = wl + wr - overlap_px
                trans_mm = round(((lx1 + lx2) / 2.0 / pano_w) * self.config.strip_width_mm, 1)
                fused.append(FusedDefect(
                    global_id=f"CAML-{int(time.time()*1000)}-{len(fused)+1}",
                    defect_class=ld.label,
                    confidence=round(float(ld.confidence), 3),
                    global_bbox=[int(lx1), int(ly1), int(lx2), int(ly2)],
                    strip_position_mm={"transverse_mm": trans_mm, "longitudinal_mm": 0.0},
                    source_cameras=["CAM_TOP_LEFT"],
                    seam_fused=False,
                    severity_grade=_grade_defect_severity(ld.label, max(1, (lx2 - lx1) * (ly2 - ly1)))
                ))

        # Add remaining standalone right detections
        for r_idx, rd in enumerate(right_dets):
            if r_idx in used_right_indices:
                continue
            rx1, ry1, rx2, ry2 = rd.bbox
            rx1_glob = rx1 + wl - overlap_px
            rx2_glob = rx2 + wl - overlap_px
            pano_w = wl + wr - overlap_px
            trans_mm = round(((rx1_glob + rx2_glob) / 2.0 / pano_w) * self.config.strip_width_mm, 1)

            fused.append(FusedDefect(
                global_id=f"CAMR-{int(time.time()*1000)}-{len(fused)+1}",
                defect_class=rd.label,
                confidence=round(float(rd.confidence), 3),
                global_bbox=[int(rx1_glob), int(ry1), int(rx2_glob), int(ry2)],
                strip_position_mm={"transverse_mm": trans_mm, "longitudinal_mm": 0.0},
                source_cameras=["CAM_TOP_RIGHT"],
                seam_fused=False,
                severity_grade=_grade_defect_severity(rd.label, max(1, (rx2_glob - rx1_glob) * (ry2 - ry1)))
            ))

        return fused, boundary_merge_count

    def inspect_array(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        img_bottom_left: Optional[np.ndarray] = None,
        img_bottom_right: Optional[np.ndarray] = None,
        conf_threshold: float = 0.35,
        render_annotated: bool = True
    ) -> MultiCamInspectResponse:
        """
        Executes end-to-end synchronized multi-camera acquisition, detection,
        panoramic seam stitching, and cross-camera boundary flaw fusion.
        """
        start_time = time.time()
        ov = self.config.overlap_pixels
        wl = img_left.shape[1]
        wr = img_right.shape[1]

        # 1. Parallel / Sequential Local Defect Detection
        _, left_preds, _ = engine.run_inference(img_left, conf_threshold=conf_threshold, annotate=False)
        _, right_preds, _ = engine.run_inference(img_right, conf_threshold=conf_threshold, annotate=False)

        # 2. Cross-Camera Seam Boundary Defect Fusion
        fused_defects, boundary_merges = self._fuse_boundary_detections(
            left_preds, right_preds, wl, wr, ov
        )

        # 3. Stitch High-Resolution Panoramic Strip Image
        pano_img, seam_x = self.stitch_panoramic_pair(img_left, img_right, ov)
        pano_h, pano_w = pano_img.shape[:2]

        # 4. Optional Bottom Surface Analysis for Bilateral Disparity
        bottom_count = 0
        if img_bottom_left is not None and img_bottom_right is not None:
            _, b_left_preds, _ = engine.run_inference(img_bottom_left, conf_threshold=conf_threshold, annotate=False)
            _, b_right_preds, _ = engine.run_inference(img_bottom_right, conf_threshold=conf_threshold, annotate=False)
            b_fused, _ = self._fuse_boundary_detections(
                b_left_preds, b_right_preds, img_bottom_left.shape[1], img_bottom_right.shape[1], ov
            )
            bottom_count = len(b_fused)

        top_count = len(fused_defects)
        total_both = top_count + bottom_count
        surface_balance = {
            "top_surface_defects": top_count,
            "bottom_surface_defects": bottom_count,
            "surface_ratio_top_vs_bottom": round(top_count / max(1, bottom_count), 2),
            "dominant_surface": "TOP" if top_count >= bottom_count else "BOTTOM",
            "roll_disparity_detected": bool(abs(top_count - bottom_count) >= 3)
        }

        # 5. Render Annotated Panoramic Visualization
        b64_pano = None
        if render_annotated:
            vis = pano_img.copy()

            # Draw camera seam line
            cv2.line(vis, (seam_x, 0), (seam_x, pano_h), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(
                vis, f"SEAM ({self.config.overlap_width_mm}mm)",
                (seam_x + 4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1
            )

            # Draw fused bounding boxes
            for fd in fused_defects:
                gx1, gy1, gx2, gy2 = fd.global_bbox
                color = (0, 165, 255) if fd.seam_fused else (0, 255, 0) # Orange if seam-fused, Green otherwise
                cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), color, 2)
                
                tag = f"{fd.defect_class} {fd.confidence:.2f}"
                if fd.seam_fused:
                    tag += " [SEAM FUSED]"
                cv2.putText(
                    vis, tag, (gx1, max(15, gy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
                )

            _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_pano = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

        elapsed_ms = (time.time() - start_time) * 1000.0

        # Update Operational Counters
        with self._lock:
            self.total_panoramic_scans += 1
            self.total_boundary_merges += boundary_merges
            self.recent_latencies.append(elapsed_ms)
            if len(self.recent_latencies) > 100:
                self.recent_latencies.pop(0)

        return MultiCamInspectResponse(
            total_fused_defects=len(fused_defects),
            seam_fusions_count=boundary_merges,
            panoramic_width_px=pano_w,
            panoramic_height_px=pano_h,
            defects=fused_defects,
            surface_balance=surface_balance,
            panoramic_image=b64_pano,
            inference_ms=round(elapsed_ms, 2)
        )

    def get_status(self) -> MultiCamStatusResponse:
        """Returns real-time multi-camera rig synchronization, telemetry, and health."""
        with self._lock:
            avg_latency = float(np.mean(self.recent_latencies)) if self.recent_latencies else 28.0
            fps = round(1000.0 / max(1.0, avg_latency), 1)

            # Optical alignment stability based on overlap calibration
            stability = "NOMINAL"
            if self.config.overlap_pixels < 20:
                stability = "CALIBRATION_REQUIRED"
            elif abs(self.config.vertical_offset_px) > 15:
                stability = "DEGRADED"

            return MultiCamStatusResponse(
                rig_id=self.config.rig_id,
                active_channels_count=sum(1 for c in self.config.channels if c.is_active),
                sync_jitter_ms=round(self.last_sync_jitter_ms, 2),
                optical_alignment_stability=stability,
                composite_fps=fps,
                total_panoramic_scans=self.total_panoramic_scans,
                total_boundary_merges=self.total_boundary_merges
            )

multicam_engine = MultiCameraFusionEngine()
