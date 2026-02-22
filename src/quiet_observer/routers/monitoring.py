import json
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..config import TEMPLATES_DIR
from ..database import get_db
from ..models import (
    Deployment, Detection, Frame, ModelVersion, Project, ReviewQueue,
)
from ..workers.manager import worker_manager

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


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

    # Recent detections with frame info
    recent_detections = (
        db.query(Detection)
        .join(Frame, Frame.id == Detection.frame_id)
        .filter(Frame.project_id == project_id)
        .order_by(Detection.detected_at.desc())
        .limit(limit)
        .all()
    )

    detection_data = []
    for det in recent_detections:
        frame = db.query(Frame).filter(Frame.id == det.frame_id).first()
        detection_data.append({
            "detection": det,
            "frame": frame,
        })

    # Review queue
    review_items = (
        db.query(ReviewQueue)
        .filter(ReviewQueue.project_id == project_id, ReviewQueue.is_labeled == False)
        .order_by(ReviewQueue.added_at.desc())
        .limit(10)
        .all()
    )
    review_with_frames = []
    for item in review_items:
        frame = db.query(Frame).filter(Frame.id == item.frame_id).first()
        review_with_frames.append({"item": item, "frame": frame})

    inference_running = worker_manager.is_inference_running(project_id)

    return templates.TemplateResponse(
        "monitor.html",
        {
            "request": request,
            "project": project,
            "deployed_model": deployed_model,
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
            "capture_running": worker_manager.is_capture_running(project.id),
            "inference_running": worker_manager.is_inference_running(project.id),
        })

    return templates.TemplateResponse(
        "status.html",
        {"request": request, "status_data": status_data},
    )
