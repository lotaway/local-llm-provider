import asyncio

import pytest

from interfaces.molt_types import MoltJob, MoltScheduleConfig
from services import molt_scheduler as scheduler_module
from services.molt_scheduler import MoltScheduler


class FakeController:
    def __init__(self):
        self.calls = []
        self.failures = 0

    def run_decay_cycle(self):
        self.calls.append("decay")
        if self.failures:
            self.failures -= 1
            raise RuntimeError("decay failure")
        return {"updated": 1}

    def run_abstraction_cycle(self):
        self.calls.append("abstraction")
        return []

    def evaluate_and_adjust(self):
        self.calls.append("evaluation")
        return {"adjustments": []}


class FakeTaskScheduler:
    def __init__(self):
        self.task_ids = []

    async def submit(self, task_id, handler):
        self.task_ids.append(task_id)
        queue = asyncio.Queue()
        await queue.put(await handler())
        return queue


def config(**overrides):
    values = {
        "decay_interval_seconds": 1,
        "abstraction_interval_seconds": 1,
        "evaluation_interval_seconds": 1,
        "max_retries": 2,
        "retry_backoff_seconds": 0,
        "max_concurrency": 1,
    }
    values.update(overrides)
    return MoltScheduleConfig(**values)


class TestMoltScheduler:
    def test_start_is_disabled_without_learning(self, monkeypatch):
        monkeypatch.setattr(scheduler_module, "LEARNING", False)
        controller = FakeController()
        scheduler = MoltScheduler(controller, config=config())

        asyncio.run(scheduler.start())

        assert scheduler._tasks == []

    def test_run_once_delegates_to_controller(self):
        controller = FakeController()
        scheduler = MoltScheduler(controller, config=config())

        result = asyncio.run(scheduler.run_once(MoltJob.DECAY))

        assert result["success"] is True
        assert controller.calls == ["decay"]

    def test_retry_succeeds_after_transient_failures(self):
        controller = FakeController()
        controller.failures = 2
        scheduler = MoltScheduler(controller, config=config(max_retries=2))

        result = asyncio.run(scheduler.run_once(MoltJob.DECAY))

        assert result["success"] is True
        assert len(controller.calls) == 3

    def test_retry_raises_after_exhaustion(self):
        controller = FakeController()
        controller.failures = 5
        scheduler = MoltScheduler(controller, config=config(max_retries=1))

        with pytest.raises(RuntimeError):
            asyncio.run(scheduler.run_once(MoltJob.DECAY))

        assert len(controller.calls) == 2

    def test_dispatch_uses_unified_task_scheduler(self):
        controller = FakeController()
        task_scheduler = FakeTaskScheduler()
        scheduler = MoltScheduler(controller, task_scheduler=task_scheduler, config=config())

        result = asyncio.run(scheduler.run_once(MoltJob.EVALUATION))

        assert result["success"] is True
        assert task_scheduler.task_ids[0].startswith("molt_evaluation_")
