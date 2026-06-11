import cv2
import base64
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import asyncio

app = FastAPI()
model = YOLO("yolov8n.pt")  # swap with your fine-tuned model path

def decode_frame(data: str) -> np.ndarray:
    """Base64-encoded JPEG → OpenCV frame."""
    img_bytes = base64.b64decode(data)
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def encode_frame(frame: np.ndarray) -> str:
    """OpenCV frame → base64-encoded JPEG."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode("utf-8")

def run_inference(frame: np.ndarray, conf_threshold: float = 0.5):
    """Run YOLOv8 and draw bounding boxes. Returns annotated frame + detections."""
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # Draw bounding box
        color = (0, 0, 255) if conf > 0.7 else (0, 165, 255)  # red=high, orange=medium
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        detections.append({"label": label, "confidence": conf, "bbox": [x1, y1, x2, y2]})

    return frame, detections

@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # Receive raw base64 frame from client
            data = await websocket.receive_text()
            frame = decode_frame(data)

            if frame is None:
                continue

            # Run inference (offload to thread so we don't block the event loop)
            annotated, detections = await asyncio.to_thread(run_inference, frame)

            # Send back annotated frame + metadata
            await websocket.send_json({
                "frame": encode_frame(annotated),
                "detections": detections,
                "defect_detected": len(detections) > 0
            })

    except WebSocketDisconnect:
        print("Client disconnected")