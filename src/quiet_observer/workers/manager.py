import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WorkerManager:
    """Manages asyncio background tasks for sampling and inference workers."""

    def __init__(self):
        self._sampling_tasks: dict[int, asyncio.Task] = {}
        self._inference_tasks: dict[int, asyncio.Task] = {}
        self._latest_inference_live: dict[int, dict] = {}

    def is_sampling_running(self, project_id: int) -> bool:
        task = self._sampling_tasks.get(project_id)
        return task is not None and not task.done()

    def is_inference_running(self, project_id: int) -> bool:
        task = self._inference_tasks.get(project_id)
        return task is not None and not task.done()

    def set_latest_inference_live(self, project_id: int, snapshot: dict | None) -> None:
        if snapshot is None:
            self._latest_inference_live.pop(project_id, None)
        else:
            self._latest_inference_live[project_id] = snapshot

    def get_latest_inference_live(self, project_id: int) -> Optional[dict]:
        snap = self._latest_inference_live.get(project_id)
        return dict(snap) if snap else None

    async def start_sampling(self, project_id: int, db=None) -> None:
        if self.is_sampling_running(project_id):
            return

        from .capture import sample_loop
        task = asyncio.create_task(sample_loop(project_id), name=f"sampling-{project_id}")
        self._sampling_tasks[project_id] = task
        logger.info("Started sampling worker for project %d", project_id)

    async def stop_sampling(self, project_id: int) -> None:
        task = self._sampling_tasks.get(project_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._sampling_tasks.pop(project_id, None)
        logger.info("Stopped sampling worker for project %d", project_id)

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
        self._latest_inference_live.pop(project_id, None)
        logger.info("Stopped inference worker for project %d", project_id)

    async def stop_all(self) -> None:
        for project_id in list(self._sampling_tasks.keys()):
            await self.stop_sampling(project_id)
        for project_id in list(self._inference_tasks.keys()):
            await self.stop_inference(project_id)


# Singleton used across the app
worker_manager = WorkerManager()
