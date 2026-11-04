"""Composition of the MOLT memory subsystem for the application lifespan."""

from typing import Any, Callable, Optional, TypeVar

from services.abstraction_engine import AbstractionEngine
from services.decay_calculator import ExponentialDecayCalculator
from services.decay_scheduler import DecayScheduler
from services.feedback_service import FeedbackService
from services.molt_controller import MoltController
from services.version_manager import VersionManager

T = TypeVar("T")

import globals as backend_globals


def _optional_repo(repo: Any, factory: Callable[[Any], T]) -> Optional[T]:
    return factory(repo) if repo else None


def current_memory_repo() -> Optional[Any]:
    return getattr(backend_globals.local_rag, "mongo_repo", None) if backend_globals.local_rag else None


def build_molt_controller(molt_memory_repo) -> MoltController:
    neo4j_repo = (
        getattr(backend_globals.local_rag, "neo4j_repo", None)
        if backend_globals.local_rag
        else None
    )
    return MoltController(
        mongo_repo=molt_memory_repo,
        neo4j_repo=neo4j_repo,
        decay_scheduler=_optional_repo(molt_memory_repo, lambda repo: DecayScheduler(repo, ExponentialDecayCalculator())),
        abstraction_engine=_optional_repo(molt_memory_repo, lambda repo: AbstractionEngine(mongo_repo=repo)),
        version_manager=VersionManager(neo4j_repo=neo4j_repo, mongo_repo=molt_memory_repo),
        feedback_service=_optional_repo(molt_memory_repo, lambda repo: FeedbackService(repo)),
    )
