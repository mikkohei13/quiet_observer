from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_DIR}/quiet_observer.db"

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = BASE_DIR / "static"

# YOLO base model for fine-tuning
YOLO_BASE_MODEL = "yolo11n.pt"

# YOLO prediction confidence threshold (detections below this are discarded)
YOLO_INFERENCE_CONF = 0.1

# During inference post-processing, suppress lower-confidence detections that
# overlap with higher-confidence detections above this IoU value.
DETECTION_SUPPRESSION_IOU_THRESHOLD = 0.9

# During inference, sample a frame for labeling every N seconds
AUTO_SAMPLE_INTERVAL_SECONDS = 600

# During inference, sample a frame when any detection confidence is within
# [LOW, HIGH] range. Detections below LOW are treated as noise and ignored.
LOW_CONFIDENCE_SAMPLE_THRESHOLD = 0.3
HIGH_CONFIDENCE_SAMPLE_THRESHOLD = 0.7
