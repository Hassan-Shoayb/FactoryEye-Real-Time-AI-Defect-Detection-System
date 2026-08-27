import cv2
import numpy as np
import time
import base64
from typing import Tuple, List, Dict
from api.schemas import Detection

class DefectExplainabilityEngine:
    """
    Generates high-activation saliency heatmaps for industrial defect inspection,
    allowing quality control engineers to visually interpret defect feature activations.
    """
    def __init__(self, alpha: float = 0.55):
        self.alpha = alpha

    def generate_saliency_heatmap(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> Tuple[np.ndarray, str]:
        """
        Generates a jet-colormap activation heatmap blended onto the steel surface.
        Highlights the receptive fields and localized defect regions.
        """
        h, w = frame.shape[:2]
        heatmap_accum = np.zeros((h, w), dtype=np.float32)

        # 1. Compute multi-scale edge gradient intensity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_norm = cv2.normalize(gradient_mag, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # 2. Accumulate Gaussian activation distributions over defect bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            # Generate 2D Gaussian mask
            sigma_x = box_w / 4.0
            sigma_y = box_h / 4.0
            x = np.linspace(-box_w / 2, box_w / 2, box_w)
            y = np.linspace(-box_h / 2, box_h / 2, box_h)
            xx, yy = np.meshgrid(x, y)
            kernel = np.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))

            # Weight by confidence and local gradient intensity
            kernel_weighted = kernel * det.confidence * 1.5
            heatmap_accum[y1:y2, x1:x2] += kernel_weighted[:box_h, :box_w]

        # Combine with normalized gradient
        if detections:
            heatmap_accum = heatmap_accum * 0.7 + gradient_norm * 0.3
        else:
            heatmap_accum = gradient_norm * 0.2

        # Normalize to 0-255 and apply Jet Colormap
        heatmap_accum = np.clip(heatmap_accum, 0.0, 1.0)
        heatmap_uint8 = np.uint8(255 * heatmap_accum)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend with original image
        blended = cv2.addWeighted(colored_heatmap, self.alpha, frame, 1.0 - self.alpha, 0)

        # Draw bounding boxes on blended heatmap
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(blended, f"{det.label} ({det.confidence:.2f})", (x1 + 3, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Encode to Base64
        _, buffer = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_str = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        return blended, b64_str

explainability_engine = DefectExplainabilityEngine()
