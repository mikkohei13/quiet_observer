"""Inference worker: captures frames from YouTube and runs YOLO detection."""
import asyncio
import functools
import logging
from datetime import datetime
from pathlib import Path

from ..config import DATA_DIR, UNCERTAINTY_THRESHOLD, YOLO_INFERENCE_CONF
from ..database import SessionLocal
from ..models import (
    Deployment, Detection, Frame, InferenceSession,
    ModelVersion, Project, ReviewQueue,
)
from .capture import resolve_stream_url, capture_frame, get_image_dimensions

logger = logging.getLogger(__name__)


def _run_model_sync(model, frame_path_str: str):
    """Run YOLO model synchronously (called via run_in_executor)."""
    return model(frame_path_str, verbose=False, conf=YOLO_INFERENCE_CONF)


async def run_inference_on_frame(frame: Frame, model, db) -> list[dict]:
    """Run YOLO inference on a frame using a pre-loaded model object."""
    try:
        frame_path = DATA_DIR / frame.file_path
        if not frame_path.exists():
            logger.warning("Frame file not found: %s", frame_path)
            return []

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, functools.partial(_run_model_sync, model, str(frame_path))
        )
        detections = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                h, w = result.orig_shape
                for box in result.boxes:
                    cls_idx = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = result.names.get(cls_idx, str(cls_idx))

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    detections.append({
                        "class_name": cls_name,
                        "confidence": conf,
                        "x": x_center,
                        "y": y_center,
                        "width": bw,
                        "height": bh,
                    })

        return detections

    except Exception as e:
        logger.exception("Inference error on frame %d: %s", frame.id, e)
        return []


def should_add_to_review_queue(detections: list[dict]) -> tuple[bool, str]:
    """Decide whether to push frame to review queue."""
    if not detections:
        return True, "no_detection"

    max_conf = max(d["confidence"] for d in detections)
    if max_conf < UNCERTAINTY_THRESHOLD:
        return True, f"low_confidence:{max_conf:.2f}"

    return False, ""


def _close_orphaned_sessions(project_id: int) -> None:
    """Mark any still-open sessions from a previous run as interrupted."""
    db = SessionLocal()
    try:
        db.query(InferenceSession).filter(
            InferenceSession.project_id == project_id,
            InferenceSession.stopped_at.is_(None),
        ).update({"stopped_at": datetime.utcnow()})
        db.commit()
    finally:
        db.close()


def _open_session(project_id: int) -> int:
    """Create a new InferenceSession row and return its id."""
    db = SessionLocal()
    try:
        sess = InferenceSession(
            project_id=project_id,
            started_at=datetime.utcnow(),
        )
        db.add(sess)
        db.commit()
        db.refresh(sess)
        return sess.id
    finally:
        db.close()


def _close_session(session_id: int) -> None:
    """Set stopped_at on the session row."""
    db = SessionLocal()
    try:
        sess = db.query(InferenceSession).filter(InferenceSession.id == session_id).first()
        if sess:
            sess.stopped_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


async def inference_loop(project_id: int) -> None:
    """Main inference loop. Runs until cancelled.

    Each iteration captures a frame from the YouTube stream and runs YOLO on it.
    Independent of the capture worker.
    """
    logger.info("Inference loop starting for project %d", project_id)

    _close_orphaned_sessions(project_id)
    session_id = _open_session(project_id)

    _model = None
    _model_version_id = None
    _stream_url = None
    frames_processed = 0

    try:
        while True:
            db = SessionLocal()
            try:
                project = db.query(Project).filter(Project.id == project_id).first()
                if not project:
                    logger.error("Project %d not found, stopping inference", project_id)
                    return

                interval = project.inference_interval_seconds

                deployment = db.query(Deployment).filter(
                    Deployment.project_id == project_id,
                    Deployment.is_active == True,
                ).first()

                if not deployment:
                    logger.info("No active deployment for project %d, waiting...", project_id)
                else:
                    model_version = db.query(ModelVersion).filter(
                        ModelVersion.id == deployment.model_version_id
                    ).first()

                    if not model_version:
                        logger.warning("Model version not found for deployment %d", deployment.id)
                    else:
                        if _model_version_id != model_version.id:
                            weights_path = Path(model_version.weights_path)
                            if weights_path.exists():
                                from ultralytics import YOLO
                                logger.info(
                                    "Loading model v%d from %s", model_version.id, weights_path
                                )
                                _model = YOLO(str(weights_path))
                                _model_version_id = model_version.id

                                sess = db.query(InferenceSession).filter(
                                    InferenceSession.id == session_id
                                ).first()
                                if sess:
                                    sess.model_version_id = model_version.id
                                    db.commit()
                            else:
                                logger.error("Weights not found: %s", weights_path)
                                _model = None

                        if _model is not None:
                            # Resolve stream URL (re-resolve periodically as URLs expire)
                            if not _stream_url:
                                _stream_url = await resolve_stream_url(project.youtube_url)
                                if not _stream_url:
                                    logger.warning("Could not resolve stream for project %d", project_id)

                            if _stream_url:
                                timestamp = datetime.utcnow()
                                rel_path = Path(
                                    f"projects/{project_id}/frames/{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                                )
                                abs_path = DATA_DIR / rel_path

                                success = await capture_frame(_stream_url, abs_path)
                                if not success:
                                    logger.warning("Inference frame capture failed, re-resolving stream URL")
                                    _stream_url = None
                                else:
                                    width, height = get_image_dimensions(abs_path)
                                    frame = Frame(
                                        project_id=project_id,
                                        captured_at=timestamp,
                                        file_path=str(rel_path),
                                        width=width,
                                        height=height,
                                    )
                                    db.add(frame)
                                    db.flush()

                                    detections = await run_inference_on_frame(frame, _model, db)

                                    for det in detections:
                                        db.add(Detection(
                                            frame_id=frame.id,
                                            model_version_id=model_version.id,
                                            class_name=det["class_name"],
                                            confidence=det["confidence"],
                                            x=det["x"],
                                            y=det["y"],
                                            width=det["width"],
                                            height=det["height"],
                                            detected_at=datetime.utcnow(),
                                        ))

                                    add_to_queue, reason = should_add_to_review_queue(detections)
                                    if add_to_queue:
                                        db.add(ReviewQueue(
                                            frame_id=frame.id,
                                            project_id=project_id,
                                            reason=reason,
                                        ))
                                        frame.in_review_queue = True

                                    project.last_inference_at = datetime.utcnow()
                                    project.last_inferred_frame_id = frame.id
                                    db.commit()

                                    frames_processed += 1
                                    if frames_processed % 10 == 1 or frames_processed <= 3:
                                        logger.info(
                                            "Project %d: inference frame %d processed (%d detections)",
                                            project_id, frame.id, len(detections),
                                        )

                                    sess = db.query(InferenceSession).filter(
                                        InferenceSession.id == session_id
                                    ).first()
                                    if sess:
                                        sess.frames_processed = frames_processed
                                        db.commit()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Error in inference loop for project %d: %s", project_id, e)
            finally:
                db.close()

            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise

    finally:
        _close_session(session_id)
        logger.info("Inference session %d closed for project %d", session_id, project_id)
