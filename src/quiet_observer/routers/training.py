import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..config import TEMPLATES_DIR
from ..database import get_db
from ..models import (
    DatasetVersion, DatasetVersionFrame, Deployment,
    Frame, ModelVersion, Project, TrainingRun,
)

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/projects/{project_id}/train", response_class=HTMLResponse)
async def train_page(request: Request, project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    training_runs = (
        db.query(TrainingRun)
        .filter(TrainingRun.project_id == project_id)
        .order_by(TrainingRun.started_at.desc())
        .all()
    )

    model_versions = (
        db.query(ModelVersion)
        .filter(ModelVersion.project_id == project_id)
        .order_by(ModelVersion.created_at.desc())
        .all()
    )

    active_deployment = (
        db.query(Deployment)
        .filter(Deployment.project_id == project_id, Deployment.is_active == True)
        .first()
    )

    labeled_count = (
        db.query(Frame)
        .filter(Frame.project_id == project_id, Frame.is_labeled == True)
        .count()
    )

    # Batch-load dataset versions for all runs
    dv_ids = {run.dataset_version_id for run in training_runs}
    dataset_versions = (
        {dv.id: dv for dv in db.query(DatasetVersion).filter(DatasetVersion.id.in_(dv_ids)).all()}
        if dv_ids else {}
    )

    runs_with_metrics = []
    for run in training_runs:
        mv = db.query(ModelVersion).filter(ModelVersion.training_run_id == run.id).first()
        metrics = None
        if mv and mv.metrics_json:
            try:
                metrics = json.loads(mv.metrics_json)
            except Exception:
                pass
        dv = dataset_versions.get(run.dataset_version_id)
        runs_with_metrics.append({
            "run": run,
            "model_version": mv,
            "metrics": metrics,
            "frame_count": dv.frame_count if dv else "—",
        })

    mv_with_deploy = []
    for mv in model_versions:
        dep = db.query(Deployment).filter(
            Deployment.model_version_id == mv.id,
            Deployment.project_id == project_id,
        ).first()
        mv_with_deploy.append({"mv": mv, "is_deployed": dep is not None and dep.is_active})

    return templates.TemplateResponse(
        "train.html",
        {
            "request": request,
            "project": project,
            "runs_with_metrics": runs_with_metrics,
            "mv_with_deploy": mv_with_deploy,
            "active_deployment": active_deployment,
            "labeled_count": labeled_count,
        },
    )


@router.post("/projects/{project_id}/train/start")
async def start_training(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    labeled_frames = (
        db.query(Frame)
        .filter(Frame.project_id == project_id, Frame.is_labeled == True)
        .all()
    )
    if not labeled_frames:
        raise HTTPException(status_code=400, detail="No labeled frames to train on")

    # Create dataset version snapshot
    dv = DatasetVersion(
        project_id=project_id,
        name=f"v{db.query(DatasetVersion).filter(DatasetVersion.project_id == project_id).count() + 1}",
        frame_count=len(labeled_frames),
    )
    db.add(dv)
    db.flush()

    for frame in labeled_frames:
        dvf = DatasetVersionFrame(dataset_version_id=dv.id, frame_id=frame.id)
        db.add(dvf)

    # Create training run
    run = TrainingRun(
        project_id=project_id,
        dataset_version_id=dv.id,
        status="pending",
        config_json=json.dumps({"epochs": 30, "imgsz": 640}),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Launch training as background task
    from ..ml.trainer import run_training
    asyncio.create_task(run_training(run.id))

    return RedirectResponse(f"/projects/{project_id}/train", status_code=303)


@router.post("/model_versions/{mv_id}/deploy")
async def deploy_model(mv_id: int, db: Session = Depends(get_db)):
    mv = db.query(ModelVersion).filter(ModelVersion.id == mv_id).first()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")

    # Deactivate existing deployments for this project
    db.query(Deployment).filter(
        Deployment.project_id == mv.project_id,
        Deployment.is_active == True,
    ).update({"is_active": False})

    dep = Deployment(project_id=mv.project_id, model_version_id=mv_id, is_active=True)
    db.add(dep)
    db.commit()

    return RedirectResponse(f"/projects/{mv.project_id}/train", status_code=303)


@router.get("/training_runs/{run_id}/log", response_class=HTMLResponse)
async def training_log(request: Request, run_id: int, db: Session = Depends(get_db)):
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    log_content = ""
    if run.log_path:
        from pathlib import Path
        log_path = Path(run.log_path)
        if log_path.exists():
            log_content = log_path.read_text()

    return templates.TemplateResponse(
        "training_log.html",
        {"request": request, "run": run, "log_content": log_content},
    )
