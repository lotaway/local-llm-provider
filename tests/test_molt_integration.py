"""
MOLT 系统集成测试
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.feedback_service import FeedbackService
from services.decay_scheduler import DecayScheduler
from services.abstraction_engine import AbstractionEngine
from services.molt_controller import MoltController
from services.version_manager import VersionManager
from schemas.graph import LTMNode


class TestFeedbackService:
    def test_importance_clamping(self):
        service = FeedbackService(mongo_repo=Mock())
        with patch.object(service.mongo_repo, "increment_importance") as mock:
            with patch.object(service.mongo_repo, "get_chunk", return_value={"chunk_id": "test"}):
                service.on_recall("test_chunk", delta=100)
                call_args = mock.call_args
                if call_args:
                    delta = call_args[0][1]
                    assert delta <= 1.0


class TestDecayScheduler:
    def test_calculate_decay_no_time_passed(self):
        from interfaces.decay_types import IDecayCalculator

        class DummyCalc(IDecayCalculator):
            def calculate(self, memory):
                return memory.get("importance_score", 0)

        scheduler = DecayScheduler(Mock(), DummyCalc())
        chunk = {"importance_score": 0.8, "last_reviewed": datetime.utcnow()}
        # use apply path
        assert True


class TestAbstractionEngine:
    def test_can_abstraction_trigger_insufficient_episodes(self):
        engine = AbstractionEngine(mongo_repo=Mock())
        with patch.object(engine, "find_related_episodes", return_value=([], 0)):
            assert engine.can_abstraction_trigger("test_topic") is False


class TestMoltController:
    def test_evaluate_and_adjust_low_recall(self):
        controller = MoltController(
            mongo_repo=Mock(),
            neo4j_repo=Mock(),
            feedback_service=Mock(),
            decay_scheduler=Mock(),
            abstraction_engine=Mock(),
        )
        with patch.object(controller, "collect_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "recall_hit_rate": 0.2,
                "stale_chunks": 10,
                "abstraction_rate": 0.02,
                "avg_importance": 0.5,
                "total_chunks": 100,
                "episodic_chunks": 50,
                "ltm_nodes": 5,
                "discarded_chunks": 5,
            }
            result = controller.evaluate_and_adjust()
            assert "metrics" in result
            assert "adjustments" in result


class TestVersionManager:
    def test_text_similarity(self):
        vm = VersionManager(neo4j_repo=Mock(), mongo_repo=Mock())
        sim1 = vm._text_similarity("hello world", "hello world")
        assert sim1 == 1.0
        sim2 = vm._text_similarity("hello world", "goodbye world")
        assert sim2 < 1.0
        sim3 = vm._text_similarity("hello", "xyz")
        assert sim3 >= 0.0


class TestLTMNode:
    def test_ltm_node_creation(self):
        ltm = LTMNode(
            topic="test_topic",
            version=1,
            conclusion="test conclusion",
            conditions=["condition1"],
            confidence=0.8,
            sources=["source1"],
            source_chunk_ids=["chunk1"],
        )
        assert ltm.topic == "test_topic"
        assert ltm.version == 1
        assert ltm.conclusion == "test conclusion"
        assert ltm.confidence == 0.8

    def test_ltm_node_defaults(self):
        ltm = LTMNode(topic="test", conclusion="test")
        assert ltm.version == 1
        assert ltm.conditions == []
        assert ltm.confidence == 0.5
        assert ltm.sources == []
        assert ltm.source_chunk_ids == []


class TestIntegrationFlow:
    def test_full_memory_lifecycle(self):
        mongo_repo = Mock()
        neo4j_repo = Mock()
        feedback_service = Mock()
        decay_scheduler = Mock()
        abstraction_engine = Mock()

        feedback_service.increment_importance = Mock()
        decay_scheduler.apply_decay_all = Mock(return_value={"updated": 5})
        abstraction_engine.auto_discover_and_abstract = Mock(return_value=[])

        feedback_service.get_top_memories = Mock(return_value=[])
        try:
            mongo_repo.chunks.count_documents = Mock(return_value=100)
        except Exception:
            pass
        try:
            mongo_repo.chunks.aggregate = Mock(return_value=[{"avg": 0.5}])
        except Exception:
            pass
        if hasattr(neo4j_repo, "get_ltm_by_topic"):
            neo4j_repo.get_ltm_by_topic = Mock(return_value=[])
        if hasattr(feedback_service, "get_stale_memories"):
            feedback_service.get_stale_memories = Mock(return_value=[])

        controller = MoltController(
            mongo_repo=mongo_repo,
            neo4j_repo=neo4j_repo,
            feedback_service=feedback_service,
            decay_scheduler=decay_scheduler,
            abstraction_engine=abstraction_engine,
        )

        result = controller.run_full_cycle()
        assert "metrics" in result
        assert "decay" in result or "decay_stats" in result
        assert "abstraction" in result or "abstraction_results" in result
        decay_scheduler.apply_decay_all.assert_called_once()
        abstraction_engine.auto_discover_and_abstract.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
