import subprocess
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..config import TEMPLATES_DIR
from ..database import get_db
from ..models import Project
from ..workers.manager import worker_manager

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def validate_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


@router.get("/", response_class=HTMLResponse)
async def list_projects(request: Request, db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return templates.TemplateResponse(
        "projects.html", {"request": request, "projects": projects}
    )


@router.get("/projects/new", response_class=HTMLResponse)
async def new_project_form(request: Request):
    return templates.TemplateResponse("project_new.html", {"request": request})


@router.post("/projects")
async def create_project(
    request: Request,
    name: str = Form(...),
    youtube_url: str = Form(...),
    capture_interval_seconds: int = Form(60),
    inference_interval_seconds: int = Form(30),
    db: Session = Depends(get_db),
):
    if not validate_youtube_url(youtube_url):
        return templates.TemplateResponse(
            "project_new.html",
            {
                "request": request,
                "error": "Invalid YouTube URL. Must contain youtube.com or youtu.be",
                "form": {
                    "name": name,
                    "youtube_url": youtube_url,
                    "capture_interval_seconds": capture_interval_seconds,
                    "inference_interval_seconds": inference_interval_seconds,
                },
            },
        )

    project = Project(
        name=name,
        youtube_url=youtube_url,
        capture_interval_seconds=capture_interval_seconds,
        inference_interval_seconds=inference_interval_seconds,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    return RedirectResponse(f"/projects/{project.id}", status_code=303)


@router.get("/projects/{project_id}", response_class=HTMLResponse)
async def project_detail(request: Request, project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    from ..models import Frame, Class, ModelVersion, Deployment, ReviewQueue
    from sqlalchemy import func

    frame_count = db.query(func.count(Frame.id)).filter(Frame.project_id == project_id).scalar()
    labeled_count = (
        db.query(func.count(Frame.id))
        .filter(Frame.project_id == project_id, Frame.is_labeled == True)
        .scalar()
    )
    classes = db.query(Class).filter(Class.project_id == project_id).all()
    review_count = (
        db.query(func.count(ReviewQueue.id))
        .filter(ReviewQueue.project_id == project_id, ReviewQueue.is_labeled == False)
        .scalar()
    )
    recent_frames = (
        db.query(Frame)
        .filter(Frame.project_id == project_id)
        .order_by(Frame.captured_at.desc())
        .limit(12)
        .all()
    )

    active_deployment = (
        db.query(Deployment)
        .filter(Deployment.project_id == project_id, Deployment.is_active == True)
        .first()
    )
    deployed_model = None
    if active_deployment:
        deployed_model = (
            db.query(ModelVersion)
            .filter(ModelVersion.id == active_deployment.model_version_id)
            .first()
        )

    latest_frame = recent_frames[0] if recent_frames else None

    capture_running = worker_manager.is_capture_running(project_id)
    inference_running = worker_manager.is_inference_running(project_id)

    return templates.TemplateResponse(
        "project_detail.html",
        {
            "request": request,
            "project": project,
            "frame_count": frame_count,
            "labeled_count": labeled_count,
            "classes": classes,
            "review_count": review_count,
            "recent_frames": recent_frames,
            "latest_frame": latest_frame,
            "deployed_model": deployed_model,
            "capture_running": capture_running,
            "inference_running": inference_running,
        },
    )


@router.get("/projects/{project_id}/edit", response_class=HTMLResponse)
async def edit_project_form(request: Request, project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return templates.TemplateResponse("project_edit.html", {"request": request, "project": project})


@router.post("/projects/{project_id}/edit")
async def edit_project(
    request: Request,
    project_id: int,
    name: str = Form(...),
    youtube_url: str = Form(...),
    capture_interval_seconds: int = Form(...),
    inference_interval_seconds: int = Form(...),
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not validate_youtube_url(youtube_url):
        return templates.TemplateResponse(
            "project_edit.html",
            {
                "request": request,
                "project": project,
                "error": "Invalid YouTube URL. Must contain youtube.com or youtu.be",
                "form": {
                    "name": name,
                    "youtube_url": youtube_url,
                    "capture_interval_seconds": capture_interval_seconds,
                    "inference_interval_seconds": inference_interval_seconds,
                },
            },
        )

    project.name = name
    project.youtube_url = youtube_url
    project.capture_interval_seconds = capture_interval_seconds
    project.inference_interval_seconds = inference_interval_seconds
    db.commit()

    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/capture/start")
async def start_capture(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    await worker_manager.start_capture(project_id, db)
    project.capture_active = True
    db.commit()

    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/capture/stop")
async def stop_capture(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    await worker_manager.stop_capture(project_id)
    project.capture_active = False
    db.commit()

    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/inference/start")
async def start_inference(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    await worker_manager.start_inference(project_id, db)
    project.inference_active = True
    db.commit()

    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/inference/stop")
async def stop_inference(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    await worker_manager.stop_inference(project_id)
    project.inference_active = False
    db.commit()

    return RedirectResponse(f"/projects/{project_id}", status_code=303)
