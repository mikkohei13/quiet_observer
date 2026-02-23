from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_DIR}/quiet_observer.db"

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = BASE_DIR / "static"

# Uncertainty threshold for review queue
UNCERTAINTY_THRESHOLD = 0.5

# YOLO base model for fine-tuning
YOLO_BASE_MODEL = "yolo11n.pt"

# YOLO prediction confidence threshold (detections below this are discarded)
YOLO_INFERENCE_CONF = 0.1

# During inference, sample a frame for labeling every N seconds
AUTO_SAMPLE_INTERVAL_SECONDS = 600

# During inference, sample a frame when any detection confidence is below this
LOW_CONFIDENCE_SAMPLE_THRESHOLD = 0.3
