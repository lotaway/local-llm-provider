from datetime import datetime

import pytest

from interfaces.molt_types import MoltJob, MoltScheduleConfig, MoltStrategyConfig
from schemas.graph import LTMNode
from services import molt_controller as controller_module
from services import version_manager as version_module
from services import decay_scheduler as decay_module
from services import abstraction_engine as abstraction_module
from services.abstraction_engine import AbstractionEngine
from services.decay_scheduler import DecayScheduler
from services.feedback_service import FeedbackService
from services.molt_controller import MoltController
from services.molt_scheduler import MoltScheduler
from services.strategy_history import InMemoryStrategyHistoryStore
from services.version_manager import VersionManager
from services.working_memory import RingWorkingMemory


class InMemoryMemoryRepository:
    def __init__(self, episodes=None):
        self.episodes = episodes or []
        self.batch_calls = []

    def get_by_id(self, chunk_id):
        return next((item for item in self.episodes if item["chunk_id"] == chunk_id), None)

    def get_by_type(self, memory_type):
        return [item for item in self.episodes if item.get("memory_type", "episodic") == memory_type]

    def update_importance(self, chunk_id, score):
        self.get_by_id(chunk_id)["importance_score"] = score

    def increment_importance(self, chunk_id, delta):
        memory = self.get_by_id(chunk_id)
        memory["importance_score"] = max(0.0, min(1.0, memory["importance_score"] + delta))

    def mark_reviewed(self, chunk_id):
        self.get_by_id(chunk_id)["last_reviewed"] = datetime.utcnow()

    def mark_discarded(self, chunk_id):
        self.get_by_id(chunk_id)["discarded"] = True

    def get_low_importance(self, threshold, limit):
        return [item for item in self.episodes if item["importance_score"] < threshold][:limit]


class InMemoryGraphRepository:
    def __init__(self):
        self.versions = {}
        self.relations = []

    def save_ltm(self, node):
        record = node.model_dump()
        topic_versions = self.versions.setdefault(node.topic, [])
        topic_versions[:] = [item for item in topic_versions if item["version"] != node.version]
        topic_versions.append(record)
        return f"{node.topic}_v{node.version}"

    def get_ltm_by_topic(self, topic):
        return list(self.versions.get(topic, []))

    def get_ltm_versions_chain(self, topic):
        return sorted(self.versions.get(topic, []), key=lambda item: item["version"], reverse=True)

    def create_ltm_version_relation(self, old_key, new_key, relation_types=None):
        self.relations.append((old_key, new_key, tuple(relation_types or ())))

    def get_episodic_for_ltm(self, topic_version):
        return []

    def count_ltm_nodes(self):
        return sum(len(items) for items in self.versions.values())


class RecordingDecayScheduler:
    def __init__(self):
        self.calls = 0
        self.result = {"updated": 2}

    def apply_decay_all(self):
        self.calls += 1
        return self.result


class RecordingAbstractionEngine:
    def __init__(self):
        self.calls = 0
        self.result = [{"topic": "vector-search"}]

    def auto_discover_and_abstract(self):
        self.calls += 1
        return self.result


class RecordingLtmIndex:
    def __init__(self):
        self.vectors = []
        self.keywords = []

    def save_ltm_vector(self, ltm):
        self.vectors.append(ltm)
        return ltm["topic"]

    def index_ltm_keywords(self, ltm):
        self.keywords.append(ltm)
        return ltm["topic"]

    def nearest_similarity(self, ltm):
        return None


class FixedTopicExtractor:
    def extract(self, episodes):
        return ["vector-search"]


class FixedCalculator:
    def calculate(self, memory):
        return 0.4


def episode(chunk_id, importance=0.8):
    return {
        "chunk_id": chunk_id,
        "text": "vector-search grounding improves retrieval",
        "memory_type": "episodic",
        "importance_score": importance,
        "review_count": 0,
        "decay_rate": 0.01,
        "last_reviewed": datetime.utcnow(),
    }


@pytest.fixture
def learning_enabled(monkeypatch):
    monkeypatch.setattr(controller_module, "LEARNING", True)
    monkeypatch.setattr(version_module, "LEARNING", True)
    monkeypatch.setattr(decay_module, "LEARNING", True)
    monkeypatch.setattr(abstraction_module, "LEARNING", True)


class TestFeedbackAndDecay:
    def test_feedback_clamps_importance_at_upper_bound(self):
        repo = InMemoryMemoryRepository([episode("chunk-a", importance=0.95)])
        service = FeedbackService(repo)

        result = service.on_recall("chunk-a")

        assert result["importance_score"] == 1.0
        assert result["last_reviewed"] is not None

    def test_decay_scheduler_updates_every_episode(self, learning_enabled):
        repo = InMemoryMemoryRepository([episode("chunk-a"), episode("chunk-b")])
        scheduler = DecayScheduler(repo, FixedCalculator())

        result = scheduler.apply_decay_all()

        assert result == {"updated": 2}
        assert [item["importance_score"] for item in repo.episodes] == [0.4, 0.4]


class TestAbstractionLifecycle:
    def test_abstraction_writes_ltm_and_indexes(self, learning_enabled):
        repo = InMemoryMemoryRepository([episode(f"chunk-{index}") for index in range(5)])
        graph = InMemoryGraphRepository()
        ltm_index = RecordingLtmIndex()
        engine = AbstractionEngine(
            mongo_repo=repo,
            neo4j_repo=graph,
            topic_extractor=FixedTopicExtractor(),
            ltm_index=ltm_index,
        )

        result = engine.abstract_to_ltm("vector-search")

        assert result["episodes_count"] == 5
        assert graph.versions["vector-search"][0]["source_chunk_ids"] == [
            f"chunk-{index}" for index in range(5)
        ]
        assert len(ltm_index.vectors) == 1
        assert len(ltm_index.keywords) == 1

    def test_abstraction_does_not_trigger_without_enough_episodes(self, learning_enabled):
        engine = AbstractionEngine(
            mongo_repo=InMemoryMemoryRepository([episode("chunk-a")]),
            topic_extractor=FixedTopicExtractor(),
        )

        assert engine.abstract_to_ltm("vector-search") is None


class TestMoltStrategyAdjustment:
    def test_low_metrics_adjust_all_thresholds(self, learning_enabled):
        repo = InMemoryMemoryRepository([episode("chunk-a", importance=0.2)])
        controller = MoltController(mongo_repo=repo, neo4j_repo=InMemoryGraphRepository())
        controller._recall_hit_rate = 0.1

        result = controller.evaluate_and_adjust()

        assert {item["type"] for item in result["adjustments"]} == {
            "recall_threshold",
            "decay_rate",
            "abstraction_trigger_threshold",
        }
        assert result["strategy"]["recall_threshold"] < 0.7
        assert result["strategy"]["decay_rate"] > 0.01
        assert result["strategy"]["abstraction_trigger_threshold"] < 0.6

    def test_healthy_metrics_keep_strategy_unchanged(self, learning_enabled):
        repo = InMemoryMemoryRepository([episode("chunk-a", importance=0.9)])
        graph = InMemoryGraphRepository()
        graph.versions["vector-search"] = [{"version": 1, "topic": "vector-search"}]
        controller = MoltController(mongo_repo=repo, neo4j_repo=graph)
        controller._recall_hit_rate = 0.9

        result = controller.evaluate_and_adjust()

        assert result["adjustments"] == []
        assert result["strategy"] == {
            "recall_threshold": 0.7,
            "decay_rate": 0.01,
            "abstraction_trigger_threshold": 0.6,
        }

    def test_injected_config_controls_trigger_threshold(self, learning_enabled):
        config = MoltStrategyConfig(recall_hit_rate_threshold=0.95, recall_step=0.1)
        repo = InMemoryMemoryRepository([episode("chunk-a", importance=0.9)])
        graph = InMemoryGraphRepository()
        graph.versions["vector-search"] = [{"version": 1, "topic": "vector-search"}]
        controller = MoltController(
            mongo_repo=repo, neo4j_repo=graph, strategy_config=config
        )
        controller._recall_hit_rate = 0.9

        result = controller.evaluate_and_adjust()

        assert [item["type"] for item in result["adjustments"]] == ["recall_threshold"]
        assert result["strategy"]["recall_threshold"] == pytest.approx(0.6)

    def test_adjustment_history_is_recorded_and_exposed(self, learning_enabled):
        repo = InMemoryMemoryRepository([episode("chunk-a", importance=0.2)])
        history = InMemoryStrategyHistoryStore(limit=2)
        controller = MoltController(
            mongo_repo=repo,
            neo4j_repo=InMemoryGraphRepository(),
            history_store=history,
        )
        controller._recall_hit_rate = 0.1

        first = controller.evaluate_and_adjust()
        second = controller.evaluate_and_adjust()
        metrics = controller.collect_metrics()

        assert first["history"][-1]["adjustments"]
        assert second["history"][-1]["adjustment_id"] != first["history"][-1]["adjustment_id"]
        assert metrics["strategy_history"]["total"] == 2
        assert len(metrics["strategy_history"]["recent"]) == 2
        assert metrics["strategy_history"]["recent"][-1]["duration_ms"] >= 0

    def test_invalid_strategy_config_is_rejected(self):
        with pytest.raises(ValueError):
            MoltStrategyConfig(recall_step=0)
        with pytest.raises(ValueError):
            MoltStrategyConfig(decay_ceiling=1.5)
        with pytest.raises(ValueError):
            MoltStrategyConfig(history_limit=0)

    def test_full_cycle_uses_injected_components_once(self, learning_enabled):
        decay = RecordingDecayScheduler()
        abstraction = RecordingAbstractionEngine()
        controller = MoltController(
            mongo_repo=InMemoryMemoryRepository([episode("chunk-a")]),
            neo4j_repo=InMemoryGraphRepository(),
            decay_scheduler=decay,
            abstraction_engine=abstraction,
            working_memory=RingWorkingMemory(capacity=7),
        )

        result = controller.run_full_cycle()

        assert set(result) == {"timestamp", "decay", "abstraction", "metrics"}
        assert result["decay"] == {"updated": 2}
        assert result["abstraction"] == [{"topic": "vector-search"}]
        assert decay.calls == 1
        assert abstraction.calls == 1


class TestVersionLifecycle:
    def _manager(self, graph):
        return VersionManager(neo4j_repo=graph)

    def test_same_source_merge_preserves_single_version(self, learning_enabled):
        graph = InMemoryGraphRepository()
        graph.save_ltm(LTMNode(topic="t", conclusion="same", confidence=0.5, sources=["a"], source_chunk_ids=["chunk-a"]))
        manager = self._manager(graph)

        saved = manager.upsert_ltm(
            {
                "topic": "t",
                "conclusion": "same",
                "confidence": 0.9,
                "sources": ["a", "b"],
                "source_chunk_ids": ["chunk-a", "chunk-b"],
                "conditions": ["new"],
            }
        )

        assert saved == "t_v1"
        assert len(graph.versions["t"]) == 1
        assert graph.versions["t"][0]["conditions"] == ["new"]

    def test_different_source_creates_version_chain(self, learning_enabled):
        graph = InMemoryGraphRepository()
        graph.save_ltm(LTMNode(topic="t", conclusion="same", confidence=0.5, sources=["a"], source_chunk_ids=["chunk-a"]))
        manager = self._manager(graph)

        saved = manager.upsert_ltm(
            {
                "topic": "t",
                "conclusion": "same",
                "confidence": 0.7,
                "sources": ["b"],
                "source_chunk_ids": ["chunk-b"],
            }
        )

        assert saved == "t_v2"
        assert graph.relations[0][0] == "t_v1"
        assert "NEXT_VERSION" in graph.relations[0][2]

    def test_rollback_restores_target_as_new_version(self, learning_enabled):
        graph = InMemoryGraphRepository()
        graph.save_ltm(LTMNode(topic="t", version=1, conclusion="old", confidence=0.5))
        graph.save_ltm(LTMNode(topic="t", version=2, conclusion="new", confidence=0.8))
        manager = self._manager(graph)

        result = manager.rollback_to("t", 1)

        assert result["success"] is True
        assert result["new_version"] == 3
        assert graph.get_ltm_versions_chain("t")[0]["conclusion"] == "old"


class TestMoltSchedulerLifecycle:
    def test_scheduler_runs_each_job_once(self, learning_enabled):
        class Controller:
            def __init__(self):
                self.jobs = []

            def run_decay_cycle(self):
                self.jobs.append("decay")
                return {}

            def run_abstraction_cycle(self):
                self.jobs.append("abstraction")
                return []

            def evaluate_and_adjust(self):
                self.jobs.append("evaluation")
                return {}

        import asyncio

        controller = Controller()
        scheduler = MoltScheduler(
            controller,
            config=MoltScheduleConfig(
                decay_interval_seconds=1,
                abstraction_interval_seconds=1,
                evaluation_interval_seconds=1,
                max_retries=0,
                retry_backoff_seconds=0,
                max_concurrency=1,
            ),
        )
        for job in MoltJob:
            asyncio.run(scheduler.run_once(job))

        assert controller.jobs == ["decay", "abstraction", "evaluation"]
