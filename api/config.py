import os
from pathlib import Path
from typing import Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip()


BASE_DIR = Path(__file__).resolve().parent.parent

# Model Configuration
DEFAULT_MODEL_PATH = str(BASE_DIR / "training" / "runs" / "train" / "weights" / "best.pt")
MODEL_PATH: str = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
DEVICE: str = os.getenv("DEVICE", "cpu")

# MLflow Tracking
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "defect-detection")

# Alerting
SLACK_WEBHOOK_URL: Optional[str] = os.getenv("SLACK_WEBHOOK_URL") or None
ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))

# Server
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_VERSION: str = "1.0.0"

# Defect Classes Reference (NEU Surface Defect Database)
DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]
