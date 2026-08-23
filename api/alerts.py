import time
import logging
from typing import List, Dict, Optional
import httpx
from api.config import SLACK_WEBHOOK_URL, ALERT_COOLDOWN_SECONDS

logger = logging.getLogger("factoryeye.alerts")

class AlertManager:
    """
    Manages notifications to external webhooks (e.g., Slack) with debouncing / cooldown timer
    to prevent flooding operators with redundant alerts during continuous video streams.
    """
    def __init__(self, webhook_url: Optional[str] = SLACK_WEBHOOK_URL, cooldown_sec: int = ALERT_COOLDOWN_SECONDS):
        self.webhook_url = webhook_url
        self.cooldown_sec = cooldown_sec
        self.last_alert_time: float = 0.0

    def can_alert(self) -> bool:
        if not self.webhook_url:
            return False
        return (time.time() - self.last_alert_time) >= self.cooldown_sec

    async def send_defect_alert(self, defect_count: int, detections: List[Dict], source: str = "Live Camera"):
        """Asynchronously sends a formatted defect alert to the configured webhook."""
        if not self.can_alert() or defect_count == 0:
            return

        self.last_alert_time = time.time()
        
        defect_labels = [f"• *{d.get('label', 'Defect')}* (conf: {d.get('confidence', 0):.2f})" for d in detections[:5]]
        defect_summary = "\n".join(defect_labels)
        if len(detections) > 5:
            defect_summary += f"\n... and {len(detections) - 5} more."

        payload = {
            "text": f"🚨 *FactoryEye Defect Alert!* ({defect_count} defects detected)",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "🚨 FactoryEye Defect Alert", "emoji": True}
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Source:* {source}"},
                        {"type": "mrkdwn", "text": f"*Defect Count:* {defect_count}"}
                    ]
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Detections:*\n{defect_summary}"}
                }
            ]
        }

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.post(self.webhook_url, json=payload)
                if resp.status_code == 200:
                    logger.info("✓ Defect alert sent successfully.")
                else:
                    logger.warning(f"Failed to post alert: HTTP {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Error sending defect alert: {e}")

alert_manager = AlertManager()
