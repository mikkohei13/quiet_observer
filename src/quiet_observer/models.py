from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text,
)
from .database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    youtube_url = Column(String, nullable=False)
    sample_interval_seconds = Column(Integer, default=60)
    inference_interval_seconds = Column(Integer, default=30)
    created_at = Column(DateTime, default=datetime.utcnow)
    sampling_active = Column(Boolean, default=False)
    inference_active = Column(Boolean, default=False)
    last_sample_at = Column(DateTime, nullable=True)
    last_inference_at = Column(DateTime, nullable=True)
    last_inferred_frame_id = Column(Integer, nullable=True)


class Frame(Base):
    __tablename__ = "frames"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    captured_at = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String, nullable=False)  # relative to DATA_DIR
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    source = Column(String, default="sampler")  # "sampler" or "inference"
    label_status = Column(String, default="unlabeled")  # "unlabeled", "annotated", "negative"
    in_review_queue = Column(Boolean, default=False)


class Class(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    name = Column(String, nullable=False)
    color = Column(String, default="#e74c3c")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey("frames.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    # Normalized coordinates: x_center, y_center, width, height (0.0–1.0)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    name = Column(String, nullable=False)
    frame_count = Column(Integer, default=0)


class DatasetVersionFrame(Base):
    __tablename__ = "dataset_version_frames"

    dataset_version_id = Column(Integer, ForeignKey("dataset_versions.id"), primary_key=True)
    frame_id = Column(Integer, ForeignKey("frames.id"), primary_key=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    dataset_version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, default="pending")  # pending, running, done, failed
    config_json = Column(Text, nullable=True)
    log_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    weights_path = Column(String, nullable=False)
    metrics_json = Column(Text, nullable=True)
    class_map_json = Column(Text, nullable=True)


class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    deployed_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey("frames.id"), nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    class_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    # Normalized: x_center, y_center, width, height (0.0–1.0)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)


class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer, ForeignKey("frames.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)
    reason = Column(String, nullable=True)
    is_labeled = Column(Boolean, default=False)


class InferenceSession(Base):
    __tablename__ = "inference_sessions"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    # Most recently active model version; updated when deployment changes mid-session
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)
    started_at = Column(DateTime, nullable=False)
    stopped_at = Column(DateTime, nullable=True)   # None = still running / interrupted
    frames_processed = Column(Integer, default=0)
