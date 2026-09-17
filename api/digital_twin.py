import math
import time
import logging
import threading
from typing import List, Dict, Optional, Any, Tuple

from api.database import audit_db
from api.schemas import (
    CoilParameters,
    CoilGeometryResponse,
    CoilFlawLocation,
    CoilSegmentProfile,
    LongitudinalDefectProfileResponse,
    ShearCutRequest,
    ShearCutSegment,
    ShearCutPlanResponse,
    CoilQualityMapExport
)

logger = logging.getLogger("factoryeye.digital_twin")

class CoilDigitalTwinEngine:
    """
    Coil Digital Twin & Automated Shear-Cut Optimization Engine.
    Simulates physical continuous steel strip recoiling geometry, maps defect
    coordinates to longitudinal meter marks and nested coil wraps, and computes
    optimal flying shear cut plans to maximize Grade-A prime continuous yield.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self.default_params = CoilParameters()

    def compute_coil_geometry(self, params: Optional[CoilParameters] = None) -> CoilGeometryResponse:
        """
        Computes analytical coiler physics: outer diameter expansion, total volume,
        mass in tonnes, and wrap layer count based on Archimedean spiral wrap theory.
        """
        p = params or self.default_params
        
        # Dimensions in meters
        length_m = p.strip_length_m
        width_m = p.strip_width_mm / 1000.0
        thickness_m = p.strip_thickness_mm / 1000.0
        r_inner_m = (p.inner_diameter_mm / 2.0) / 1000.0

        # Physical volume & weight
        volume_m3 = length_m * width_m * thickness_m
        weight_kg = volume_m3 * p.steel_density_kg_m3
        weight_tonnes = weight_kg / 1000.0

        # Spiral cross-sectional area of wound strip
        strip_cross_area_m2 = length_m * thickness_m
        # R_outer = sqrt(R_inner^2 + Area_strip / pi)
        r_outer_m = math.sqrt(r_inner_m ** 2 + (strip_cross_area_m2 / math.pi))
        outer_diameter_mm = r_outer_m * 2.0 * 1000.0

        # Number of wraps (laps around mandrel)
        total_wraps = max(1, int(round((r_outer_m - r_inner_m) / thickness_m)))
        bu_ratio = outer_diameter_mm / p.inner_diameter_mm

        return CoilGeometryResponse(
            strip_length_m=round(length_m, 2),
            strip_width_mm=round(p.strip_width_mm, 2),
            strip_thickness_mm=round(p.strip_thickness_mm, 3),
            inner_diameter_mm=round(p.inner_diameter_mm, 1),
            outer_diameter_mm=round(outer_diameter_mm, 1),
            coil_volume_m3=round(volume_m3, 4),
            coil_weight_kg=round(weight_kg, 2),
            coil_weight_tonnes=round(weight_tonnes, 3),
            total_wraps=total_wraps,
            coil_build_up_ratio=round(bu_ratio, 2)
        )

    def _map_flaws(
        self,
        records: List[Dict[str, Any]],
        strip_length_m: float,
        strip_width_mm: float,
        inner_dia_mm: float,
        thickness_mm: float
    ) -> List[CoilFlawLocation]:
        """
        Maps raw defect records to longitudinal meters and nested coil wrap layers.
        """
        flaws: List[CoilFlawLocation] = []
        r_in_m = (inner_dia_mm / 2.0) / 1000.0
        t_m = thickness_mm / 1000.0

        if not records:
            return flaws

        # Sort chronologically or by ID
        sorted_records = sorted(records, key=lambda r: (r.get("timestamp_utc", 0.0), r.get("id", 0)))
        t0 = sorted_records[0].get("timestamp_utc", 0.0)

        for idx, r in enumerate(sorted_records):
            rec_id = r.get("id", idx + 1)
            cls_name = r.get("defect_class", "defect").lower()
            conf = float(r.get("confidence", 0.85))

            # Longitudinal position along strip (meters)
            ts = r.get("timestamp_utc", 0.0)
            if ts > t0:
                # 2.0 m/s nominal line speed
                long_m = ((ts - t0) * 2.0) % strip_length_m
            else:
                # Deterministic golden-ratio distribution along strip
                long_m = ((rec_id * 161.803) + (idx * 37.0)) % (strip_length_m - 20.0) + 10.0

            # Transverse position (mm across width)
            bx1 = r.get("bbox_x1", 100)
            bx2 = r.get("bbox_x2", 200)
            mid_x_norm = max(0.05, min(0.95, (bx1 + bx2) / (2.0 * 640.0)))
            trans_mm = mid_x_norm * strip_width_mm

            # Spiral radius at this meter: r(L) = sqrt(r_in^2 + (L * t) / pi)
            r_loc_m = math.sqrt(r_in_m ** 2 + ((long_m * t_m) / math.pi))
            r_loc_mm = r_loc_m * 1000.0
            wrap_idx = max(0, int((r_loc_m - r_in_m) / t_m))

            # Severity classification
            if cls_name in ["inclusion", "crazing", "delamination"]:
                sev_grade = "CRITICAL"
                sev_weight = 3.0
            elif cls_name in ["scratches", "pitted_surface", "pitted"]:
                sev_grade = "MODERATE"
                sev_weight = 1.5
            else:
                sev_grade = "MINOR"
                sev_weight = 1.0

            flaws.append(CoilFlawLocation(
                flaw_id=rec_id,
                defect_class=cls_name,
                confidence=round(conf, 3),
                longitudinal_meter=round(long_m, 2),
                transverse_mm=round(trans_mm, 1),
                wrap_index=wrap_idx,
                wrap_radius_mm=round(r_loc_mm, 1),
                severity_grade=sev_grade,
                severity_weight=sev_weight
            ))

        return sorted(flaws, key=lambda f: f.longitudinal_meter)

    def get_longitudinal_profile(
        self,
        batch_id: str = "BATCH-2026-COIL-A",
        coil_params: Optional[CoilParameters] = None
    ) -> LongitudinalDefectProfileResponse:
        """
        Discretizes coil into 10-meter segments and produces density and quality profile.
        """
        with self._lock:
            p = coil_params or self.default_params
            records, _ = audit_db.query_defects(limit=200)
            
            flaws = self._map_flaws(
                records=records,
                strip_length_m=p.strip_length_m,
                strip_width_mm=p.strip_width_mm,
                inner_dia_mm=p.inner_diameter_mm,
                thickness_mm=p.strip_thickness_mm
            )

            segment_len = 10.0
            total_segments = max(1, int(math.ceil(p.strip_length_m / segment_len)))
            segments: List[CoilSegmentProfile] = []

            # Segment bins
            for i in range(total_segments):
                s_start = i * segment_len
                s_end = min(p.strip_length_m, (i + 1) * segment_len)
                seg_flaws = [f for f in flaws if s_start <= f.longitudinal_meter < s_end]
                f_count = len(seg_flaws)
                sev_score = sum(f.severity_weight for f in seg_flaws)

                dominant = None
                if seg_flaws:
                    dominant = max(set(f.defect_class for f in seg_flaws), key=lambda c: sum(1 for f in seg_flaws if f.defect_class == c))

                # Quality Grade logic:
                # Grade A Prime: 0 flaws or 1 minor flaw (sev_score <= 1.0)
                # Grade B Commercial: 1-2 flaws with sev_score <= 2.5
                # Scrap: any critical flaw (sev_score >= 3.0) or f_count >= 3
                if f_count == 0 or (f_count == 1 and sev_score <= 1.0):
                    grade = "GRADE_A_PRIME"
                elif f_count <= 2 and sev_score <= 2.5:
                    grade = "GRADE_B_COMMERCIAL"
                else:
                    grade = "SCRAP_REJECT"

                segments.append(CoilSegmentProfile(
                    segment_index=i,
                    start_meter=round(s_start, 1),
                    end_meter=round(s_end, 1),
                    flaw_count=f_count,
                    severity_score=round(sev_score, 1),
                    dominant_defect=dominant,
                    grade=grade
                ))

            prime_cnt = sum(1 for s in segments if s.grade == "GRADE_A_PRIME")
            comm_cnt = sum(1 for s in segments if s.grade == "GRADE_B_COMMERCIAL")
            scrap_cnt = sum(1 for s in segments if s.grade == "SCRAP_REJECT")

            prime_ratio = (prime_cnt / total_segments) * 100.0
            if prime_ratio >= 85.0:
                overall = "GRADE_A_PRIME_CERTIFIED"
            elif (prime_cnt + comm_cnt) / total_segments >= 0.75:
                overall = "GRADE_B_COMMERCIAL_ACCEPTABLE"
            else:
                overall = "SUBSTANDARD_DOWNGRADE"

            density_100m = (len(flaws) / p.strip_length_m) * 100.0

            return LongitudinalDefectProfileResponse(
                batch_id=batch_id,
                total_strip_length_m=p.strip_length_m,
                segment_length_m=segment_len,
                total_segments=total_segments,
                prime_segments_count=prime_cnt,
                commercial_segments_count=comm_cnt,
                scrap_segments_count=scrap_cnt,
                overall_coil_grade=overall,
                defect_density_per_100m=round(density_100m, 2),
                flaws=flaws,
                segments=segments
            )

    def compute_shear_cut_plan(self, req: ShearCutRequest) -> ShearCutPlanResponse:
        """
        Optimizes flying shear cutting points to excise defect clusters while
        maximizing continuous Prime Grade-A mother coil segments >= min_prime_length_m.
        """
        with self._lock:
            p = req.coil_params or self.default_params
            profile = self.get_longitudinal_profile(req.batch_id or "BATCH-2026-COIL-A", p)
            segments = profile.segments
            min_prime = req.min_prime_length_m

            # Identify contiguous homogeneous runs
            runs: List[Dict[str, Any]] = []
            if not segments:
                return ShearCutPlanResponse(
                    batch_id=req.batch_id or "BATCH-2026-COIL-A",
                    total_strip_length_m=p.strip_length_m,
                    total_cuts_required=0,
                    prime_yield_percent=100.0,
                    secondary_yield_percent=0.0,
                    scrap_loss_percent=0.0,
                    prime_weight_tonnes=0.0,
                    scrap_weight_tonnes=0.0,
                    cut_schedule=[],
                    execution_status="OPTIMAL"
                )

            current_grade = segments[0].grade
            start_m = segments[0].start_meter
            flaw_sum = segments[0].flaw_count

            for s in segments[1:]:
                if s.grade == current_grade:
                    flaw_sum += s.flaw_count
                else:
                    runs.append({
                        "start_m": start_m,
                        "end_m": s.start_meter,
                        "grade": current_grade,
                        "flaws": flaw_sum
                    })
                    current_grade = s.grade
                    start_m = s.start_meter
                    flaw_sum = s.flaw_count

            runs.append({
                "start_m": start_m,
                "end_m": segments[-1].end_meter,
                "grade": current_grade,
                "flaws": flaw_sum
            })

            # Formulate executable cut sections
            total_weight_tonnes = self.compute_coil_geometry(p).coil_weight_tonnes
            tonnes_per_meter = total_weight_tonnes / p.strip_length_m

            cut_schedule: List[ShearCutSegment] = []
            cut_idx = 1
            prime_len = 0.0
            sec_len = 0.0
            scrap_len = 0.0

            for idx, r in enumerate(runs):
                length = r["end_m"] - r["start_m"]
                w_tonnes = round(length * tonnes_per_meter, 3)

                if r["grade"] == "GRADE_A_PRIME":
                    if length >= min_prime:
                        action = "PRIME_SHIPMENT"
                        assigned_grade = "GRADE_A_PRIME"
                        prime_len += length
                    else:
                        # Short prime sections that do not meet min coil length are sold as secondary
                        action = "SECONDARY_OFFGRADE"
                        assigned_grade = "GRADE_B_COMMERCIAL"
                        sec_len += length
                elif r["grade"] == "GRADE_B_COMMERCIAL":
                    action = "SECONDARY_OFFGRADE"
                    assigned_grade = "GRADE_B_COMMERCIAL"
                    sec_len += length
                else:
                    action = "SCRAP_EXCISE"
                    assigned_grade = "SCRAP_REJECT"
                    scrap_len += length

                cut_schedule.append(ShearCutSegment(
                    section_id=f"COIL-SEC-{cut_idx:02d}",
                    cut_index=cut_idx,
                    start_meter=round(r["start_m"], 1),
                    end_meter=round(r["end_m"], 1),
                    length_m=round(length, 1),
                    weight_tonnes=w_tonnes,
                    grade=assigned_grade,
                    action=action,
                    flaw_count=r["flaws"]
                ))
                cut_idx += 1

            cuts_count = max(0, len(cut_schedule) - 1)
            total_len = max(1.0, p.strip_length_m)

            prime_yield_pct = round((prime_len / total_len) * 100.0, 1)
            sec_yield_pct = round((sec_len / total_len) * 100.0, 1)
            scrap_pct = round((scrap_len / total_len) * 100.0, 1)

            prime_wt = round(prime_len * tonnes_per_meter, 2)
            scrap_wt = round(scrap_len * tonnes_per_meter, 2)

            return ShearCutPlanResponse(
                batch_id=req.batch_id or "BATCH-2026-COIL-A",
                total_strip_length_m=round(total_len, 1),
                total_cuts_required=cuts_count,
                prime_yield_percent=prime_yield_pct,
                secondary_yield_percent=sec_yield_pct,
                scrap_loss_percent=scrap_pct,
                prime_weight_tonnes=prime_wt,
                scrap_weight_tonnes=scrap_wt,
                cut_schedule=cut_schedule,
                execution_status="OPTIMAL" if prime_yield_pct >= 70.0 else "SUBOPTIMAL_HIGH_SCRAP"
            )

    def export_coil_quality_map(
        self,
        batch_id: str = "BATCH-2026-COIL-A",
        coil_params: Optional[CoilParameters] = None
    ) -> CoilQualityMapExport:
        """
        Generates standard Coil Quality Map (CQM) export file for MES/ERP and shear PLCs.
        """
        p = coil_params or self.default_params
        geo = self.compute_coil_geometry(p)
        prof = self.get_longitudinal_profile(batch_id, p)
        req = ShearCutRequest(batch_id=batch_id, coil_params=p)
        plan = self.compute_shear_cut_plan(req)

        summary = {
            "total_flaws": len(prof.flaws),
            "defect_density_per_100m": prof.defect_density_per_100m,
            "overall_grade": prof.overall_coil_grade,
            "prime_segments": prof.prime_segments_count,
            "scrap_segments": prof.scrap_segments_count
        }

        return CoilQualityMapExport(
            format_version="CQM-2026-V1.0",
            generated_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            batch_id=batch_id,
            geometry=geo,
            profile_summary=summary,
            shear_cut_plan=plan
        )

digital_twin_engine = CoilDigitalTwinEngine()
