"""Inference worker: runs YOLO detection on captured frames."""
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from ..config import DATA_DIR, UNCERTAINTY_THRESHOLD
from ..database import SessionLocal
from ..models import (
    Deployment, Detection, Frame, InferenceSession,
    ModelVersion, Project, ReviewQueue,
)

logger = logging.getLogger(__name__)


async def run_inference_on_frame(frame: Frame, model, db) -> list[dict]:
    """Run YOLO inference on a frame using a pre-loaded model object."""
    try:
        frame_path = DATA_DIR / frame.file_path
        if not frame_path.exists():
            logger.warning("Frame file not found: %s", frame_path)
            return []

        results = model(str(frame_path), verbose=False)
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

    Per-run behaviour:
    - Orphaned open sessions (from a previous server crash) are closed first.
    - A new InferenceSession row is created on entry and closed on exit.
    - The YOLO model is loaded once and cached; only reloaded when the
      deployed model version changes.
    - Every frame newer than the last processed one is run through the model,
      so no captured frames are silently skipped.
    - A uniqueness guard prevents duplicate Detection rows for the same
      (frame, model_version) pair.
    """
    logger.info("Inference loop starting for project %d", project_id)

    _close_orphaned_sessions(project_id)
    session_id = _open_session(project_id)

    _model = None             # cached YOLO model instance
    _model_version_id = None  # version currently loaded in _model
    last_frame_id = 0         # highest frame.id fully processed

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
                        # Load (or reload) model only when version changes
                        if _model_version_id != model_version.id:
                            weights_path = Path(model_version.weights_path)
                            if weights_path.exists():
                                from ultralytics import YOLO
                                logger.info(
                                    "Loading model v%d from %s", model_version.id, weights_path
                                )
                                _model = YOLO(str(weights_path))
                                _model_version_id = model_version.id

                                # Record which model version this session is using
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
                            # Process every new frame since the last tick, in order
                            new_frames = (
                                db.query(Frame)
                                .filter(
                                    Frame.project_id == project_id,
                                    Frame.id > last_frame_id,
                                )
                                .order_by(Frame.id.asc())
                                .all()
                            )

                            processed = 0
                            for frame in new_frames:
                                # Safety guard: skip if already processed by this model version
                                already_done = db.query(Detection).filter(
                                    Detection.frame_id == frame.id,
                                    Detection.model_version_id == model_version.id,
                                ).first()
                                if already_done:
                                    last_frame_id = frame.id
                                    continue

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
                                    already_queued = db.query(ReviewQueue).filter(
                                        ReviewQueue.frame_id == frame.id,
                                    ).first()
                                    if not already_queued:
                                        db.add(ReviewQueue(
                                            frame_id=frame.id,
                                            project_id=project_id,
                                            reason=reason,
                                        ))
                                        frame.in_review_queue = True

                                last_frame_id = frame.id
                                processed += 1

                            if processed > 0:
                                project.last_inference_at = datetime.utcnow()
                                # Update frames_processed count on the session
                                sess = db.query(InferenceSession).filter(
                                    InferenceSession.id == session_id
                                ).first()
                                if sess:
                                    sess.frames_processed += processed
                                db.commit()
                                logger.info(
                                    "Project %d: processed %d new frame(s)",
                                    project_id, processed,
                                )

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
