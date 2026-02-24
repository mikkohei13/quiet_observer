"""Inference worker: captures frames from YouTube and runs YOLO detection."""
import asyncio
import functools
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

from ..config import (
    AUTO_SAMPLE_INTERVAL_SECONDS,
    DATA_DIR,
    DETECTION_SUPPRESSION_IOU_THRESHOLD,
    HIGH_CONFIDENCE_SAMPLE_THRESHOLD,
    LOW_CONFIDENCE_SAMPLE_THRESHOLD,
    YOLO_INFERENCE_CONF,
)
from ..database import SessionLocal
from ..models import (
    Deployment, Detection, Frame, InferenceSession,
    ModelVersion, Project,
)
from .capture import resolve_stream_url, capture_frame, get_image_dimensions
from .manager import worker_manager

logger = logging.getLogger(__name__)


def _xywh_to_xyxy(box: dict) -> tuple[float, float, float, float]:
    """Convert normalized center box format to corner format."""
    half_w = box["width"] / 2
    half_h = box["height"] / 2
    x1 = box["x"] - half_w
    y1 = box["y"] - half_h
    x2 = box["x"] + half_w
    y2 = box["y"] + half_h
    return x1, y1, x2, y2


def _iou_xywh(first: dict, second: dict) -> float:
    """Compute IoU between two normalized boxes in x,y,width,height format."""
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(second)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _suppress_overlapping_detections(
    detections: list[dict], iou_threshold: float = DETECTION_SUPPRESSION_IOU_THRESHOLD
) -> list[dict]:
    """Keep highest-confidence detection when boxes overlap heavily."""
    if len(detections) <= 1:
        return detections

    sorted_detections = sorted(
        detections, key=lambda det: det["confidence"], reverse=True
    )
    kept: list[dict] = []
    for candidate in sorted_detections:
        if all(_iou_xywh(candidate, existing) <= iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def _run_model_sync(model, frame_path_str: str):
    """Run YOLO model synchronously (called via run_in_executor)."""
    return model(frame_path_str, verbose=False, conf=YOLO_INFERENCE_CONF)


async def run_inference_on_frame(frame_path: Path, model) -> list[dict]:
    """Run YOLO inference on an image path using a pre-loaded model object."""
    try:
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

        return _suppress_overlapping_detections(detections)

    except Exception as e:
        logger.exception("Inference error on frame path %s: %s", frame_path, e)
        return []


def should_sample_frame(
    detections: list[dict],
    last_sampled_at: float,
    now: float,
    auto_sample_interval: float = AUTO_SAMPLE_INTERVAL_SECONDS,
    low_threshold: float = LOW_CONFIDENCE_SAMPLE_THRESHOLD,
    high_threshold: float = HIGH_CONFIDENCE_SAMPLE_THRESHOLD,
) -> tuple[bool, str]:
    """Decide whether to sample this frame for the labeling pipeline.

    A frame is sampled when:
    - Enough time has elapsed since the last sample (time-based), or
    - Any detection has confidence in [low_threshold, high_threshold] — the
      uncertain range worth human review.  Detections below low_threshold are
      treated as noise and do NOT trigger sampling.
    """
    if detections:
        for d in detections:
            conf = d["confidence"]
            if low_threshold <= conf <= high_threshold:
                return True, f"uncertain_confidence:{conf:.2f}"

    if now - last_sampled_at >= auto_sample_interval:
        return True, "auto_sample"

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
    Independent of the sampling worker.
    """
    logger.info("Inference loop starting for project %d", project_id)

    _close_orphaned_sessions(project_id)
    session_id = _open_session(project_id)

    _model = None
    _model_version_id = None
    _stream_url = None
    frames_processed = 0
    _last_sampled_at = 0.0
    _live_tick_id = 0
    _latest_live_file: Path | None = None

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
                                live_rel_path = Path(
                                    f"projects/{project_id}/live/{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                                )
                                live_abs_path = DATA_DIR / live_rel_path

                                success = await capture_frame(_stream_url, live_abs_path)
                                if not success:
                                    logger.warning("Inference frame capture failed, re-resolving stream URL")
                                    _stream_url = None
                                else:
                                    width, height = get_image_dimensions(live_abs_path)
                                    detections = await run_inference_on_frame(live_abs_path, _model)

                                    _live_tick_id += 1
                                    worker_manager.set_latest_inference_live(project_id, {
                                        "tick_id": _live_tick_id,
                                        "captured_at": timestamp.isoformat(),
                                        "width": width,
                                        "height": height,
                                        "file_path": str(live_rel_path),
                                        "detections": detections,
                                    })

                                    should_sample, _reason = should_sample_frame(
                                        detections, _last_sampled_at, time.monotonic(),
                                        auto_sample_interval=project.auto_sample_interval_seconds or AUTO_SAMPLE_INTERVAL_SECONDS,
                                        low_threshold=project.low_confidence_threshold if project.low_confidence_threshold is not None else LOW_CONFIDENCE_SAMPLE_THRESHOLD,
                                        high_threshold=project.high_confidence_threshold if project.high_confidence_threshold is not None else HIGH_CONFIDENCE_SAMPLE_THRESHOLD,
                                    )
                                    if should_sample:
                                        sampled_rel_path = Path(
                                            f"projects/{project_id}/frames/{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                                        )
                                        sampled_abs_path = DATA_DIR / sampled_rel_path
                                        sampled_abs_path.parent.mkdir(parents=True, exist_ok=True)
                                        shutil.copy2(live_abs_path, sampled_abs_path)

                                        frame = Frame(
                                            project_id=project_id,
                                            captured_at=timestamp,
                                            file_path=str(sampled_rel_path),
                                            width=width,
                                            height=height,
                                            source="inference",
                                        )
                                        db.add(frame)
                                        db.flush()

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

                                        _last_sampled_at = time.monotonic()
                                        project.last_inferred_frame_id = frame.id

                                    project.last_inference_at = datetime.utcnow()
                                    db.commit()

                                    frames_processed += 1
                                    if frames_processed % 10 == 1 or frames_processed <= 3:
                                        logger.info(
                                            "Project %d: live inference tick %d processed (%d detections)",
                                            project_id, _live_tick_id, len(detections),
                                        )

                                    sess = db.query(InferenceSession).filter(
                                        InferenceSession.id == session_id
                                    ).first()
                                    if sess:
                                        sess.frames_processed = frames_processed
                                        db.commit()

                                    if _latest_live_file and _latest_live_file.exists():
                                        try:
                                            _latest_live_file.unlink()
                                        except Exception:
                                            logger.debug("Failed to delete previous live frame: %s", _latest_live_file)
                                    _latest_live_file = live_abs_path

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
        worker_manager.set_latest_inference_live(project_id, None)
        if _latest_live_file and _latest_live_file.exists():
            try:
                _latest_live_file.unlink()
            except Exception:
                logger.debug("Failed to delete latest live frame on shutdown: %s", _latest_live_file)
        _close_session(session_id)
        logger.info("Inference session %d closed for project %d", session_id, project_id)
