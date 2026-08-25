import json
import time
import logging
from typing import List, Dict

logger = logging.getLogger("factoryeye.mqtt")

class IndustrialMQTTPublisher:
    """
    Industrial Telemetry Publisher for factory line PLCs, SCADA, and pneumatic reject actuators.
    Emulates MQTT publishing over topic 'factory/line1/defects'.
    """
    def __init__(self, broker_host: str = "localhost", topic: str = "factory/line1/defects"):
        self.broker_host = broker_host
        self.topic = topic
        self.enabled = False  # Enabled when an external MQTT broker (e.g. Mosquitto) is connected

    def publish_defect_event(self, defect_count: int, detections: List[Dict], source: str):
        if defect_count == 0:
            return

        payload = {
            "timestamp_utc": time.time(),
            "station_id": "SURFACE_INSPECTION_STATION_01",
            "source": source,
            "defect_count": defect_count,
            "action_required": "TRIGGER_PNEUMATIC_REJECT_ARM" if defect_count > 0 else "PASS",
            "defects": [
                {
                    "class": d.get("label", "") if isinstance(d, dict) else getattr(d, "label", ""),
                    "confidence": d.get("confidence", 0.0) if isinstance(d, dict) else getattr(d, "confidence", 0.0),
                    "bbox": d.get("bbox", []) if isinstance(d, dict) else getattr(d, "bbox", [])
                }
                for d in detections
            ]
        }
        
        # Log industrial payload
        logger.info(f"📡 [MQTT -> {self.topic}] Event: {payload['action_required']} ({defect_count} defects)")

mqtt_publisher = IndustrialMQTTPublisher()
