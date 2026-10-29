from interfaces.molt_types import IMoltScheduler
from constants import LEARNING


class MoltScheduler(IMoltScheduler):
    def __init__(self, molt_controller=None, decay_scheduler=None, abstraction_engine=None, version_manager=None, task_scheduler=None):
        self._controller = molt_controller
        self._decay = decay_scheduler
        self._abstraction = abstraction_engine
        self._version = version_manager
        self._task_scheduler = task_scheduler
        self._running = False

    def start(self) -> None:
        if not LEARNING:
            return
        self._running = True
        if self._task_scheduler and hasattr(self._task_scheduler, "schedule"):
            try:
                self._task_scheduler.schedule("molt_decay", self._run_decay)
                self._task_scheduler.schedule("molt_abstraction", self._run_abstraction)
                self._task_scheduler.schedule("molt_evaluation", self._run_evaluation)
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        if self._task_scheduler and hasattr(self._task_scheduler, "unschedule"):
            try:
                self._task_scheduler.unschedule("molt_decay")
                self._task_scheduler.unschedule("molt_abstraction")
                self._task_scheduler.unschedule("molt_evaluation")
            except Exception:
                pass

    def _run_decay(self):
        if self._decay and hasattr(self._decay, "apply_decay_all"):
            return self._decay.apply_decay_all()
        if self._controller and hasattr(self._controller, "run_decay_cycle"):
            return self._controller.run_decay_cycle()
        return {"skipped": True}

    def _run_abstraction(self):
        if self._abstraction and hasattr(self._abstraction, "auto_discover_and_abstract"):
            return self._abstraction.auto_discover_and_abstract()
        if self._controller and hasattr(self._controller, "run_abstraction_cycle"):
            return self._controller.run_abstraction_cycle()
        return []

    def _run_evaluation(self):
        if self._controller and hasattr(self._controller, "evaluate_and_adjust"):
            return self._controller.evaluate_and_adjust()
        return {"skipped": True}
