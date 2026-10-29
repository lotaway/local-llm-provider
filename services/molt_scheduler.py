import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from constants import (
    LEARNING,
    MOLT_ABSTRACTION_INTERVAL_SECONDS,
    MOLT_DECAY_INTERVAL_SECONDS,
    MOLT_EVALUATION_INTERVAL_SECONDS,
    MOLT_TASK_MAX_CONCURRENCY,
    MOLT_TASK_MAX_RETRIES,
    MOLT_TASK_RETRY_BACKOFF_SECONDS,
)
from interfaces.molt_types import IMoltController, IMoltScheduler, MoltJob, MoltScheduleConfig

logger = logging.getLogger(__name__)


class MoltCycleError(RuntimeError):
    pass


def default_schedule_config() -> MoltScheduleConfig:
    return MoltScheduleConfig(
        decay_interval_seconds=MOLT_DECAY_INTERVAL_SECONDS,
        abstraction_interval_seconds=MOLT_ABSTRACTION_INTERVAL_SECONDS,
        evaluation_interval_seconds=MOLT_EVALUATION_INTERVAL_SECONDS,
        max_retries=MOLT_TASK_MAX_RETRIES,
        retry_backoff_seconds=MOLT_TASK_RETRY_BACKOFF_SECONDS,
        max_concurrency=MOLT_TASK_MAX_CONCURRENCY,
    )


class MoltScheduler(IMoltScheduler):
    def __init__(
        self,
        controller: IMoltController,
        task_scheduler=None,
        config: Optional[MoltScheduleConfig] = None,
    ):
        self._controller = controller
        self._task_scheduler = task_scheduler
        self._config = config or default_schedule_config()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_loop = None
        self._tasks: List[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        if not LEARNING or self._running:
            return
        self._running = True
        self._tasks = [asyncio.create_task(self._loop(job)) for job in MoltJob]
        logger.info("molt scheduler started", extra={"jobs": [job.value for job in MoltJob]})

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        logger.info("molt scheduler stopped")

    async def run_once(self, job: MoltJob) -> Dict[str, Any]:
        return await self._execute(job, self._handler(job))

    async def _loop(self, job: MoltJob) -> None:
        while self._running:
            await asyncio.sleep(self._config.interval_for(job))
            await self.run_once(job)

    def _handler(self, job: MoltJob):
        return {
            MoltJob.DECAY: self._controller.run_decay_cycle,
            MoltJob.ABSTRACTION: self._controller.run_abstraction_cycle,
            MoltJob.EVALUATION: self._controller.evaluate_and_adjust,
        }[job]

    async def _execute(self, job: MoltJob, handler) -> Dict[str, Any]:
        async with self._concurrency_guard():
            return await self._retry(job, handler)

    def _concurrency_guard(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self._config.max_concurrency)
            self._semaphore_loop = loop
        return self._semaphore

    async def _retry(self, job: MoltJob, handler) -> Dict[str, Any]:
        for attempt in range(self._config.max_retries + 1):
            try:
                return await self._dispatch(job, handler)
            except Exception as error:
                if attempt == self._config.max_retries:
                    logger.error(
                        "molt cycle failed",
                        extra={"job": job.value, "attempt": attempt + 1, "error": str(error)},
                        exc_info=True,
                    )
                    raise
                await asyncio.sleep(self._config.retry_backoff_seconds * (attempt + 1))
        raise MoltCycleError(f"molt job {job.value} exhausted retries")

    async def _dispatch(self, job: MoltJob, handler) -> Dict[str, Any]:
        if self._task_scheduler is None:
            return await self._run_handler(handler)
        job_id = f"molt_{job.value}_{uuid4().hex}"
        result_queue = await self._task_scheduler.submit(job_id, lambda: self._run_handler(handler))
        result = await result_queue.get()
        if not result or not result.get("success"):
            raise MoltCycleError(f"molt job {job.value} failed in task scheduler")
        return result

    async def _run_handler(self, handler) -> Dict[str, Any]:
        try:
            value = await asyncio.to_thread(handler)
            return {"success": True, "value": value}
        except Exception as error:
            return {"success": False, "error": str(error)}
