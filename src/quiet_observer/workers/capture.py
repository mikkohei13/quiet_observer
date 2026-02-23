"""Frame capture worker: resolves YouTube stream and grabs frames at a fixed interval."""
import asyncio
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image

from ..config import DATA_DIR
from ..database import SessionLocal
from ..models import Frame, Project

logger = logging.getLogger(__name__)


async def resolve_stream_url(youtube_url: str) -> str | None:
    """Use yt-dlp to resolve a YouTube URL to a direct stream URL."""
    try:
        result = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--no-warnings",
            "-f", "best[height<=720]/best",
            "-g",
            youtube_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30)
        if result.returncode == 0:
            url = stdout.decode().strip().splitlines()[0]
            return url
        logger.warning("yt-dlp failed for %s: %s", youtube_url, stderr.decode())
        return None
    except asyncio.TimeoutError:
        logger.error("yt-dlp timed out for %s", youtube_url)
        return None
    except FileNotFoundError:
        logger.error("yt-dlp not found. Install with: brew install yt-dlp")
        return None


async def capture_frame(stream_url: str, output_path: Path) -> bool:
    """Use ffmpeg to capture a single frame from the stream."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-i", stream_url,
            "-vframes", "1",
            "-q:v", "2",
            str(output_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(result.communicate(), timeout=60)
        if result.returncode == 0 and output_path.exists():
            return True
        logger.warning("ffmpeg capture failed: %s", stderr.decode()[-500:])
        return False
    except asyncio.TimeoutError:
        logger.error("ffmpeg timed out capturing frame")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found. Install with: brew install ffmpeg")
        return False


def get_image_dimensions(path: Path) -> tuple[int, int] | tuple[None, None]:
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return None, None


async def capture_loop(project_id: int) -> None:
    """Main capture loop. Runs until cancelled."""
    logger.info("Capture loop starting for project %d", project_id)

    while True:
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                logger.error("Project %d not found, stopping capture", project_id)
                return

            interval = project.capture_interval_seconds
            youtube_url = project.youtube_url

            logger.info("Resolving stream URL for project %d...", project_id)
            stream_url = await resolve_stream_url(youtube_url)

            if not stream_url:
                logger.warning("Could not resolve stream for project %d, retrying later", project_id)
            else:
                timestamp = datetime.utcnow()
                rel_path = Path(f"projects/{project_id}/frames/{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg")
                abs_path = DATA_DIR / rel_path

                success = await capture_frame(stream_url, abs_path)
                if success:
                    width, height = get_image_dimensions(abs_path)
                    frame = Frame(
                        project_id=project_id,
                        captured_at=timestamp,
                        file_path=str(rel_path),
                        width=width,
                        height=height,
                        source="capture",
                    )
                    db.add(frame)
                    project.last_capture_at = timestamp
                    db.commit()
                    logger.info("Captured frame for project %d: %s", project_id, rel_path)
                else:
                    logger.warning("Frame capture failed for project %d", project_id)

        except asyncio.CancelledError:
            logger.info("Capture loop cancelled for project %d", project_id)
            raise
        except Exception as e:
            logger.exception("Error in capture loop for project %d: %s", project_id, e)
        finally:
            db.close()

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("Capture loop cancelled during sleep for project %d", project_id)
            raise
