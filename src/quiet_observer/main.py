import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .config import STATIC_DIR
from .database import init_db
from .ml.trainer import reconcile_stale_training_runs
from .routers import annotations, frames, monitoring, projects, training
from .workers.manager import worker_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    reconcile_stale_training_runs()
    yield
    await worker_manager.stop_all()


app = FastAPI(title="Quiet Observer", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(projects.router)
app.include_router(frames.router)
app.include_router(annotations.router)
app.include_router(training.router)
app.include_router(monitoring.router)


def run():
    uvicorn.run("quiet_observer.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()
