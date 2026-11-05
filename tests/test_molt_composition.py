from bootstrap.molt_dependencies import MoltDependencies, build_memory_tiers
from bootstrap.molt_factory import build_molt_controller


class FakeLtmIndex:
    def __init__(self):
        self.saved = []

    def save_ltm_vector(self, ltm):
        self.saved.append(("vector", ltm))
        return "id"

    def index_ltm_keywords(self, ltm):
        self.saved.append(("keywords", ltm))
        return "id"

    def nearest_similarity(self, ltm):
        return 0.85


class FakeHistoryStore:
    def append(self, record):
        pass

    def recent(self, limit):
        return []

    def count(self):
        return 0


class FakeSensoryCache:
    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def stats(self):
        return {"entries": 0}


class FakeWorkingMemory:
    def clear(self):
        self.cleared = True

    def stats(self):
        return {"size": 0}


class FakeMemoryRepo:
    def get_by_type(self, memory_type):
        return []


def _dependencies():
    return MoltDependencies(
        neo4j_repo=object(),
        ltm_index=FakeLtmIndex(),
        sensory_cache=FakeSensoryCache(),
        working_memory=FakeWorkingMemory(),
        history_store=FakeHistoryStore(),
    )


def test_controller_receives_every_injected_dependency():
    deps = _dependencies()
    controller = build_molt_controller(None, deps)
    assert controller._neo4j_repo is deps.neo4j_repo
    assert controller._sensory_cache is deps.sensory_cache
    assert controller._working_memory is deps.working_memory
    assert controller._history_store is deps.history_store
    assert controller._version_manager._ltm_index is deps.ltm_index


def test_vector_similarity_is_no_longer_always_none():
    deps = _dependencies()
    controller = build_molt_controller(None, deps)
    conflict = controller._version_manager.detect_conflict("topic", "conclusion")
    assert conflict["vector_similarity"] == 0.85


def test_abstraction_engine_receives_graph_and_index_repositories():
    deps = _dependencies()
    controller = build_molt_controller(FakeMemoryRepo(), deps)
    engine = controller._abstraction_engine
    assert engine._neo4j_repo is deps.neo4j_repo
    assert engine._ltm_index is deps.ltm_index


def test_memory_tiers_are_built_without_ltm_index_clients():
    tiers = build_memory_tiers(None)
    assert tiers.working_memory is not None
    assert tiers.sensory_cache is None
    assert tiers.ltm_index is None
