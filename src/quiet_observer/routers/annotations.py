import json
from fastapi import APIRouter, Depends, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..config import TEMPLATES_DIR
from ..database import get_db
from ..models import Annotation, Class, Frame, Project, ReviewQueue

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

CLASS_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


@router.get("/projects/{project_id}/label", response_class=HTMLResponse)
async def label_index(request: Request, project_id: int, db: Session = Depends(get_db)):
    """Show first unlabeled frame, or the review queue."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check review queue first
    review = (
        db.query(ReviewQueue)
        .filter(ReviewQueue.project_id == project_id, ReviewQueue.is_labeled == False)
        .first()
    )
    if review:
        return RedirectResponse(
            f"/projects/{project_id}/label/{review.frame_id}?from_queue=1", status_code=303
        )

    # Otherwise, first unlabeled capture frame (inference frames come through the queue)
    frame = (
        db.query(Frame)
        .filter(
            Frame.project_id == project_id,
            Frame.is_labeled == False,
            Frame.source == "capture",
        )
        .order_by(Frame.captured_at.asc())
        .first()
    )
    if frame:
        return RedirectResponse(
            f"/projects/{project_id}/label/{frame.id}", status_code=303
        )

    # All labeled
    frames = (
        db.query(Frame)
        .filter(Frame.project_id == project_id)
        .order_by(Frame.captured_at.desc())
        .limit(20)
        .all()
    )
    classes = db.query(Class).filter(Class.project_id == project_id).all()
    return templates.TemplateResponse(
        "label_done.html",
        {"request": request, "project": project, "frames": frames, "classes": classes},
    )


@router.get("/projects/{project_id}/label/{frame_id}", response_class=HTMLResponse)
async def label_frame(
    request: Request,
    project_id: int,
    frame_id: int,
    from_queue: int = 0,
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    frame = db.query(Frame).filter(Frame.id == frame_id, Frame.project_id == project_id).first()
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")

    classes = db.query(Class).filter(Class.project_id == project_id).all()
    annotations = db.query(Annotation).filter(Annotation.frame_id == frame_id).all()

    ann_data = []
    for ann in annotations:
        cls = db.query(Class).filter(Class.id == ann.class_id).first()
        ann_data.append({
            "id": ann.id,
            "class_id": ann.class_id,
            "class_name": cls.name if cls else "unknown",
            "color": cls.color if cls else "#666",
            "x": ann.x,
            "y": ann.y,
            "width": ann.width,
            "height": ann.height,
        })

    # Prev / next frames for navigation
    prev_frame = (
        db.query(Frame)
        .filter(Frame.project_id == project_id, Frame.id < frame_id)
        .order_by(Frame.id.desc())
        .first()
    )
    next_frame = (
        db.query(Frame)
        .filter(Frame.project_id == project_id, Frame.id > frame_id)
        .order_by(Frame.id.asc())
        .first()
    )

    total_frames = db.query(Frame).filter(Frame.project_id == project_id).count()
    frame_index = db.query(Frame).filter(Frame.project_id == project_id, Frame.id <= frame_id).count()

    classes_data = [{"id": c.id, "name": c.name, "color": c.color} for c in classes]

    return templates.TemplateResponse(
        "label.html",
        {
            "request": request,
            "project": project,
            "frame": frame,
            "classes_json": json.dumps(classes_data),
            "annotations_json": json.dumps(ann_data),
            "prev_frame": prev_frame,
            "next_frame": next_frame,
            "frame_index": frame_index,
            "total_frames": total_frames,
            "from_queue": from_queue,
        },
    )


@router.post("/projects/{project_id}/frames/{frame_id}/annotations")
async def save_annotations(
    project_id: int,
    frame_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """Receive JSON body with list of annotations, replace existing ones."""
    body = await request.json()
    annotations = body.get("annotations", [])

    frame = db.query(Frame).filter(Frame.id == frame_id, Frame.project_id == project_id).first()
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")

    # Delete existing annotations
    db.query(Annotation).filter(Annotation.frame_id == frame_id).delete()

    for ann in annotations:
        db_ann = Annotation(
            frame_id=frame_id,
            class_id=ann["class_id"],
            x=ann["x"],
            y=ann["y"],
            width=ann["width"],
            height=ann["height"],
        )
        db.add(db_ann)

    frame.is_labeled = len(annotations) > 0

    # Mark as labeled in review queue if applicable
    review = db.query(ReviewQueue).filter(
        ReviewQueue.frame_id == frame_id,
        ReviewQueue.project_id == project_id,
    ).first()
    if review:
        review.is_labeled = True

    db.commit()
    return JSONResponse({"status": "ok", "count": len(annotations)})


@router.post("/projects/{project_id}/classes")
async def create_class(
    project_id: int,
    name: str = Form(...),
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    existing_count = db.query(Class).filter(Class.project_id == project_id).count()
    color = CLASS_COLORS[existing_count % len(CLASS_COLORS)]

    cls = Class(project_id=project_id, name=name, color=color)
    db.add(cls)
    db.commit()
    db.refresh(cls)

    return JSONResponse({"id": cls.id, "name": cls.name, "color": cls.color})


@router.post("/classes/{class_id}/rename")
async def rename_class(
    class_id: int,
    name: str = Form(...),
    db: Session = Depends(get_db),
):
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    cls.name = name.strip()
    db.commit()

    return JSONResponse({"id": cls.id, "name": cls.name, "color": cls.color})


@router.post("/classes/{class_id}/delete")
async def delete_class(class_id: int, db: Session = Depends(get_db)):
    cls = db.query(Class).filter(Class.id == class_id).first()
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")

    db.query(Annotation).filter(Annotation.class_id == class_id).delete()
    db.delete(cls)
    db.commit()

    return JSONResponse({"status": "ok"})
