from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import TEMPLATES_DIR
from ..database import get_db
from ..models import (
    Deployment, Detection, Frame, InferenceSession,
    ModelVersion, Project, ReviewQueue,
)
from ..workers.manager import worker_manager

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m"


@router.get("/projects/{project_id}/monitor", response_class=HTMLResponse)
async def monitor_page(
    request: Request,
    project_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    active_deployment = (
        db.query(Deployment)
        .filter(Deployment.project_id == project_id, Deployment.is_active == True)
        .first()
    )
    deployed_model = None
    if active_deployment:
        deployed_model = db.query(ModelVersion).filter(
            ModelVersion.id == active_deployment.model_version_id
        ).first()

    inference_running = worker_manager.is_inference_running(project_id)

    # ── Inference session history ─────────────────────────────────────────────
    sessions = (
        db.query(InferenceSession)
        .filter(InferenceSession.project_id == project_id)
        .order_by(InferenceSession.started_at.desc())
        .limit(10)
        .all()
    )

    # Batch-load model versions referenced by sessions
    mv_ids = {s.model_version_id for s in sessions if s.model_version_id}
    model_versions_map = (
        {mv.id: mv for mv in db.query(ModelVersion).filter(ModelVersion.id.in_(mv_ids)).all()}
        if mv_ids else {}
    )

    now = datetime.utcnow()
    session_data = []
    for i, sess in enumerate(sessions):
        end_time = sess.stopped_at or now
        delta_secs = max(0, int((end_time - sess.started_at).total_seconds()))

        # Detection class summary for this session's time window
        det_summary = (
            db.query(Detection.class_name, func.count(Detection.id).label("count"))
            .join(Frame, Frame.id == Detection.frame_id)
            .filter(
                Frame.project_id == project_id,
                Detection.detected_at >= sess.started_at,
                Detection.detected_at <= end_time,
            )
            .group_by(Detection.class_name)
            .order_by(func.count(Detection.id).desc())
            .all()
        )

        # Status: running only for the most-recent open session while worker is active
        if sess.stopped_at is not None:
            status = "stopped"
        elif i == 0 and inference_running:
            status = "running"
        else:
            status = "interrupted"

        session_data.append({
            "session": sess,
            "model_version": model_versions_map.get(sess.model_version_id),
            "duration_str": _format_duration(delta_secs),
            "detection_summary": [
                {"class_name": r.class_name, "count": r.count} for r in det_summary
            ],
            "status": status,
        })

    # ── Recent detections — batch-load frames to avoid N+1 ───────────────────
    recent_detections = (
        db.query(Detection)
        .join(Frame, Frame.id == Detection.frame_id)
        .filter(Frame.project_id == project_id)
        .order_by(Detection.detected_at.desc())
        .limit(limit)
        .all()
    )

    det_frame_ids = {det.frame_id for det in recent_detections}
    det_frames = (
        {f.id: f for f in db.query(Frame).filter(Frame.id.in_(det_frame_ids)).all()}
        if det_frame_ids else {}
    )
    detection_data = [
        {"detection": det, "frame": det_frames.get(det.frame_id)}
        for det in recent_detections
    ]

    # ── Review queue — batch-load frames to avoid N+1 ────────────────────────
    review_items = (
        db.query(ReviewQueue)
        .filter(ReviewQueue.project_id == project_id, ReviewQueue.is_labeled == False)
        .order_by(ReviewQueue.added_at.desc())
        .limit(10)
        .all()
    )
    rq_frame_ids = {item.frame_id for item in review_items}
    rq_frames = (
        {f.id: f for f in db.query(Frame).filter(Frame.id.in_(rq_frame_ids)).all()}
        if rq_frame_ids else {}
    )
    review_with_frames = [
        {"item": item, "frame": rq_frames.get(item.frame_id)}
        for item in review_items
    ]

    return templates.TemplateResponse(
        "monitor.html",
        {
            "request": request,
            "project": project,
            "deployed_model": deployed_model,
            "session_data": session_data,
            "detection_data": detection_data,
            "review_with_frames": review_with_frames,
            "inference_running": inference_running,
        },
    )


@router.get("/status", response_class=HTMLResponse)
async def status_page(request: Request, db: Session = Depends(get_db)):
    projects = db.query(Project).all()
    status_data = []
    for project in projects:
        status_data.append({
            "project": project,
            "sampling_running": worker_manager.is_sampling_running(project.id),
            "inference_running": worker_manager.is_inference_running(project.id),
        })

    return templates.TemplateResponse(
        "status.html",
        {"request": request, "status_data": status_data},
    )
