import cv2
import time
import threading
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("factoryeye.rtsp")

class RTSPCameraWorker:
    """
    Threaded RTSP IP camera frame grabber.
    Uses a zero-lag latest-frame buffer to prevent video queue backlog
    when processing high-FPS industrial camera feeds.
    """
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.running: bool = False
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.fps: float = 0.0
        self.frames_received: int = 0

    def start(self) -> bool:
        if self.running:
            return True

        logger.info(f"Connecting to RTSP camera stream: {self.rtsp_url}...")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            logger.warning(f"Failed to connect to RTSP stream: {self.rtsp_url}. (Simulating fallback)")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("✓ RTSP capture thread running.")
        return True

    def _capture_loop(self):
        last_time = time.time()
        count = 0

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.latest_frame = frame
                self.frames_received += 1

            count += 1
            now = time.time()
            if now - last_time >= 1.0:
                self.fps = count / (now - last_time)
                count = 0
                last_time = now

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Returns the newest available frame without waiting or buffering."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        logger.info("RTSP camera worker stopped.")

    def get_status(self) -> dict:
        return {
            "rtsp_url": self.rtsp_url,
            "connected": self.running and (self.cap is not None and self.cap.isOpened()),
            "fps": round(self.fps, 1),
            "total_frames_received": self.frames_received
        }

active_rtsp_workers: dict = {}
