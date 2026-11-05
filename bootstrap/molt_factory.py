from typing import Optional

import globals as backend_globals
from bootstrap.molt_dependencies import MoltDependencies, build_molt_dependencies
from services.abstraction_engine import AbstractionEngine
from services.decay_calculator import ExponentialDecayCalculator
from services.decay_scheduler import DecayScheduler
from services.feedback_service import FeedbackService
from services.molt_controller import MoltController
from services.version_manager import VersionManager


def current_memory_repo():
    if backend_globals.local_rag is None:
        return None
    return getattr(backend_globals.local_rag, "mongo_repo", None)


def current_dependencies() -> MoltDependencies:
    return build_molt_dependencies(backend_globals.local_rag)


def build_molt_controller(molt_memory_repo, dependencies: Optional[MoltDependencies] = None) -> MoltController:
    deps = dependencies or current_dependencies()
    return MoltController(
        mongo_repo=molt_memory_repo,
        neo4j_repo=deps.neo4j_repo,
        decay_scheduler=_decay_scheduler(molt_memory_repo),
        abstraction_engine=_abstraction_engine(molt_memory_repo, deps),
        version_manager=_version_manager(molt_memory_repo, deps),
        feedback_service=_feedback_service(molt_memory_repo),
        sensory_cache=deps.sensory_cache,
        working_memory=deps.working_memory,
        history_store=deps.history_store,
    )


def _decay_scheduler(repo):
    if repo is None:
        return None
    return DecayScheduler(repo, ExponentialDecayCalculator())


def _feedback_service(repo):
    if repo is None:
        return None
    return FeedbackService(repo)


def _version_manager(repo, deps: MoltDependencies) -> VersionManager:
    return VersionManager(neo4j_repo=deps.neo4j_repo, mongo_repo=repo, ltm_index=deps.ltm_index)


def _abstraction_engine(repo, deps: MoltDependencies) -> Optional[AbstractionEngine]:
    if repo is None:
        return None
    return AbstractionEngine(mongo_repo=repo, neo4j_repo=deps.neo4j_repo, ltm_index=deps.ltm_index)
