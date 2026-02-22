"""Inference worker: runs YOLO detection on captured frames."""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from ..config import DATA_DIR, UNCERTAINTY_THRESHOLD
from ..database import SessionLocal
from ..models import Deployment, Detection, Frame, ModelVersion, Project, ReviewQueue

logger = logging.getLogger(__name__)


async def run_inference_on_frame(frame: Frame, model_version: ModelVersion, db) -> list[dict]:
    """Run YOLO inference on a frame. Returns list of detection dicts."""
    try:
        from ultralytics import YOLO
        weights_path = Path(model_version.weights_path)
        if not weights_path.exists():
            logger.error("Weights not found: %s", weights_path)
            return []

        model = YOLO(str(weights_path))
        frame_path = DATA_DIR / frame.file_path

        if not frame_path.exists():
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

                    # Convert xyxy to normalized xywh
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
        logger.exception("Inference error: %s", e)
        return []


def should_add_to_review_queue(detections: list[dict]) -> tuple[bool, str]:
    """Decide whether to push frame to review queue."""
    if not detections:
        return True, "no_detection"

    max_conf = max(d["confidence"] for d in detections)
    if max_conf < UNCERTAINTY_THRESHOLD:
        return True, f"low_confidence:{max_conf:.2f}"

    return False, ""


async def inference_loop(project_id: int) -> None:
    """Main inference loop. Runs until cancelled."""
    logger.info("Inference loop starting for project %d", project_id)

    while True:
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                logger.error("Project %d not found, stopping inference", project_id)
                return

            interval = project.inference_interval_seconds

            # Find active deployment
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
                    # Get latest frame
                    frame = (
                        db.query(Frame)
                        .filter(Frame.project_id == project_id)
                        .order_by(Frame.captured_at.desc())
                        .first()
                    )

                    if frame:
                        detections = await run_inference_on_frame(frame, model_version, db)

                        # Store detections
                        for det in detections:
                            db_det = Detection(
                                frame_id=frame.id,
                                model_version_id=model_version.id,
                                class_name=det["class_name"],
                                confidence=det["confidence"],
                                x=det["x"],
                                y=det["y"],
                                width=det["width"],
                                height=det["height"],
                                detected_at=datetime.utcnow(),
                            )
                            db.add(db_det)

                        # Review queue logic
                        add_to_queue, reason = should_add_to_review_queue(detections)
                        if add_to_queue and not frame.in_review_queue:
                            already_in_queue = db.query(ReviewQueue).filter(
                                ReviewQueue.frame_id == frame.id,
                            ).first()
                            if not already_in_queue:
                                rq = ReviewQueue(
                                    frame_id=frame.id,
                                    project_id=project_id,
                                    reason=reason,
                                )
                                db.add(rq)
                                frame.in_review_queue = True

                        project.last_inference_at = datetime.utcnow()
                        db.commit()
                        logger.info(
                            "Inference done for project %d: %d detections", project_id, len(detections)
                        )
                    else:
                        logger.info("No frames yet for project %d", project_id)

        except asyncio.CancelledError:
            logger.info("Inference loop cancelled for project %d", project_id)
            raise
        except Exception as e:
            logger.exception("Error in inference loop for project %d: %s", project_id, e)
        finally:
            db.close()

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Inference loop cancelled during sleep for project %d", project_id)
            raise
