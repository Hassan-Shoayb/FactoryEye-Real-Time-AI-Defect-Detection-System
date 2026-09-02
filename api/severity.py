import numpy as np
from typing import List, Dict, Tuple
from api.schemas import Detection

# Hazard weights for structural integrity (1.0 = superficial, 3.0 = severe structural flaw)
DEFECT_HAZARD_WEIGHTS = {
    "crazing": 3.0,          # Micro-crack networks (high failure risk)
    "inclusion": 2.5,        # Foreign material embedded (structural defect)
    "pitted_surface": 2.2,   # Surface cavities/holes
    "rolled-in_scale": 1.8,  # Pressed scale
    "scratches": 1.2,        # Linear abrasions (often reworkable)
    "patches": 1.0           # Surface oxidation/texture variation
}

class DefectSeverityEngine:
    """
    Evaluates defect severity, surface area coverage density, and assigns
    deterministic manufacturing action recommendations (PASS, REWORK, SCRAP).
    """
    def evaluate_severity(
        self,
        frame_shape: Tuple[int, int],
        detections: List[Detection]
    ) -> Dict[str, any]:
        if not detections:
            return {
                "severity_grade": "NONE",
                "severity_score": 0.0,
                "defect_coverage_percent": 0.0,
                "action_recommendation": "PASS: Surface meets quality specification"
            }

        img_h, img_w = frame_shape[:2]
        total_surface_pixels = max(1, img_h * img_w)
        total_defect_pixels = 0
        weighted_hazard_sum = 0.0

        for d in detections:
            x1, y1, x2, y2 = d.bbox
            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            total_defect_pixels += box_area
            
            hazard = DEFECT_HAZARD_WEIGHTS.get(d.label, 1.5)
            weighted_hazard_sum += (box_area / total_surface_pixels) * hazard * d.confidence * 100.0

        coverage_pct = round((total_defect_pixels / total_surface_pixels) * 100.0, 2)
        # Bounded severity score (0 to 100)
        severity_score = round(min(100.0, weighted_hazard_sum * 15.0 + coverage_pct * 2.0), 1)

        # Categorize Severity Grade
        if severity_score >= 60.0 or any(d.label == "crazing" and d.confidence > 0.6 for d in detections):
            severity_grade = "CRITICAL"
            action_recommendation = "SCRAP: Critical structural defect detected. Reject item from line."
        elif severity_score >= 25.0 or coverage_pct > 3.0:
            severity_grade = "MAJOR"
            action_recommendation = "REWORK: Major surface defect. Route to precision polishing station."
        elif severity_score > 0.0:
            severity_grade = "MINOR"
            action_recommendation = "REWORK: Minor superficial marks. Buff surface and re-inspect."
        else:
            severity_grade = "NONE"
            action_recommendation = "PASS: Surface meets quality specification"

        return {
            "severity_grade": severity_grade,
            "severity_score": severity_score,
            "defect_coverage_percent": coverage_pct,
            "action_recommendation": action_recommendation
        }

severity_engine = DefectSeverityEngine()
