import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages asyncio background tasks for capture and inference workers."""

    def __init__(self):
        self._capture_tasks: dict[int, asyncio.Task] = {}
        self._inference_tasks: dict[int, asyncio.Task] = {}

    def is_capture_running(self, project_id: int) -> bool:
        task = self._capture_tasks.get(project_id)
        return task is not None and not task.done()

    def is_inference_running(self, project_id: int) -> bool:
        task = self._inference_tasks.get(project_id)
        return task is not None and not task.done()

    async def start_capture(self, project_id: int, db=None) -> None:
        if self.is_capture_running(project_id):
            return

        from .capture import capture_loop
        task = asyncio.create_task(capture_loop(project_id), name=f"capture-{project_id}")
        self._capture_tasks[project_id] = task
        logger.info("Started capture worker for project %d", project_id)

    async def stop_capture(self, project_id: int) -> None:
        task = self._capture_tasks.get(project_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._capture_tasks.pop(project_id, None)
        logger.info("Stopped capture worker for project %d", project_id)

    async def start_inference(self, project_id: int, db=None) -> None:
        if self.is_inference_running(project_id):
            return

        from .inference import inference_loop
        task = asyncio.create_task(inference_loop(project_id), name=f"inference-{project_id}")
        self._inference_tasks[project_id] = task
        logger.info("Started inference worker for project %d", project_id)

    async def stop_inference(self, project_id: int) -> None:
        task = self._inference_tasks.get(project_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._inference_tasks.pop(project_id, None)
        logger.info("Stopped inference worker for project %d", project_id)

    async def stop_all(self) -> None:
        for project_id in list(self._capture_tasks.keys()):
            await self.stop_capture(project_id)
        for project_id in list(self._inference_tasks.keys()):
            await self.stop_inference(project_id)


# Singleton used across the app
worker_manager = WorkerManager()
