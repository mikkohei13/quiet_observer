from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import DATA_DIR
from ..database import get_db
from ..models import Frame

router = APIRouter()


@router.get("/frames/{frame_id}/image")
async def serve_frame_image(frame_id: int, db: Session = Depends(get_db)):
    frame = db.query(Frame).filter(Frame.id == frame_id).first()
    if not frame:
        raise HTTPException(status_code=404, detail="Frame not found")

    file_path = DATA_DIR / frame.file_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Frame image not found on disk")

    return FileResponse(str(file_path), media_type="image/jpeg")
