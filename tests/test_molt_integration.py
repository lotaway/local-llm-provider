"""
MOLT 系统集成测试

测试用例：
1. 记忆召回反馈
2. 衰减生效
3. 抽象触发
4. 版本控制
5. Molt调度
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.feedback_service import FeedbackService, FeedbackConfig
from services.decay_scheduler import DecayScheduler, DecayConfig
from services.abstraction_engine import AbstractionEngine, AbstractionConfig
from services.molt_controller import MoltController, MoltConfig
from services.version_manager import VersionManager, VersionConfig
from schemas.graph import LTMNode


class TestFeedbackService:
    """反馈服务测试"""

    def test_config_defaults(self):
        """测试默认配置"""
        assert FeedbackConfig.RECALL_DELTA == 0.05
        assert FeedbackConfig.USAGE_DELTA == 0.15
        assert FeedbackConfig.MAX_IMPORTANCE == 1.0
        assert FeedbackConfig.MIN_IMPORTANCE == 0.0

    def test_get_config(self):
        """测试配置获取"""
        config = FeedbackConfig.get_config()
        assert "recall_delta" in config
        assert "usage_delta" in config
        assert "max_importance" in config

    def test_importance_clamping(self):
        """测试重要性边界限制"""
        service = FeedbackService(mongo_repo=Mock())

        with patch.object(service.mongo_repo, "increment_importance") as mock:
            with patch.object(
                service.mongo_repo, "get_chunk", return_value={"chunk_id": "test"}
            ):
                service.on_recall("test_chunk", delta=100)
                call_args = mock.call_args
                if call_args:
                    delta = call_args[0][1]
                    assert delta <= 1.0


class TestDecayScheduler:
    """衰减调度器测试"""

    def test_config_defaults(self):
        """测试默认配置"""
        assert DecayConfig.DECAY_RATE == 0.01
        assert DecayConfig.REINFORCEMENT_FACTOR == 0.05
        assert DecayConfig.IMPORTANCE_THRESHOLD == 0.1

    def test_calculate_decay_no_time_passed(self):
        """测试无时间流逝时的衰减"""
        scheduler = DecayScheduler(
            mongo_repo=Mock(),
            feedback_service=Mock(),
        )

        chunk = {
            "importance_score": 0.8,
            "last_reviewed": datetime.utcnow(),
            "decay_rate": 0.01,
            "review_count": 0,
        }

        new_score = scheduler.calculate_decay(chunk)
        assert new_score == 0.8

    def test_calculate_decay_with_time_passed(self):
        """测试时间流逝后的衰减"""
        scheduler = DecayScheduler(
            mongo_repo=Mock(),
            feedback_service=Mock(),
            decay_rate=0.01,
        )

        old_time = datetime.utcnow() - timedelta(days=7)
        chunk = {
            "importance_score": 0.8,
            "last_reviewed": old_time,
            "decay_rate": 0.01,
            "review_count": 0,
        }

        new_score = scheduler.calculate_decay(chunk)
        assert new_score < 0.8
        assert new_score > 0.0

    def test_calculate_decay_with_reinforcement(self):
        """测试复习强化"""
        scheduler = DecayScheduler(
            mongo_repo=Mock(),
            feedback_service=Mock(),
            decay_rate=0.01,
            reinforcement_factor=0.05,
        )

        old_time = datetime.utcnow() - timedelta(days=7)
        chunk = {
            "importance_score": 0.5,
            "last_reviewed": old_time,
            "decay_rate": 0.01,
            "review_count": 10,
        }

        new_score = scheduler.calculate_decay(chunk)
        assert new_score > 0.5


class TestAbstractionEngine:
    """抽象引擎测试"""

    def test_config_defaults(self):
        """测试默认配置"""
        assert AbstractionConfig.MIN_EPISODES == 5
        assert AbstractionConfig.MIN_AVG_IMPORTANCE == 0.6

    def test_can_abstraction_trigger_insufficient_episodes(self):
        """测试不满足抽象条件：片段不足"""
        engine = AbstractionEngine(
            mongo_repo=Mock(),
            neo4j_repo=Mock(),
            feedback_service=Mock(),
        )

        with patch.object(
            engine, "find_related_episodes", return_value=([{"importance": 0.7}], 0.7)
        ):
            with patch.object(
                engine.mongo_repo, "get_chunks_by_memory_type", return_value=[]
            ):
                result = engine.can_abstraction_trigger("test_topic")
                assert result is False

    def test_can_abstraction_trigger_insufficient_importance(self):
        """测试不满足抽象条件：重要性不足"""
        engine = AbstractionEngine(
            mongo_repo=Mock(),
            neo4j_repo=Mock(),
            feedback_service=Mock(),
        )

        with patch.object(
            engine, "find_related_episodes", return_value=([{"importance": 0.3}], 0.3)
        ):
            with patch.object(
                engine.mongo_repo, "get_chunks_by_memory_type", return_value=[]
            ):
                result = engine.can_abstraction_trigger("test_topic")
                assert result is False


class TestMoltController:
    """Molt Controller 测试"""

    def test_config_defaults(self):
        """测试默认配置"""
        assert MoltConfig.RECALL_HIT_RATE_THRESHOLD == 0.3
        assert MoltConfig.MIN_ABSTRACTION_RATE == 0.01

    def test_evaluate_and_adjust_low_recall(self):
        """测试低召回率时的策略调整"""
        controller = MoltController(
            mongo_repo=Mock(),
            neo4j_repo=Mock(),
            feedback_service=Mock(),
            decay_scheduler=Mock(),
            abstraction_engine=Mock(),
        )

        with patch.object(controller, "collect_metrics") as mock_metrics:
            mock_metrics.return_value = MagicMock(
                recall_hit_rate=0.2,
                stale_chunks=10,
                abstraction_rate=0.02,
                avg_importance=0.5,
                total_chunks=100,
                episodic_chunks=50,
                ltm_nodes=5,
                discarded_chunks=5,
            )

            with patch.object(
                controller.feedback_service, "get_top_memories", return_value=[]
            ):
                with patch.object(
                    controller.mongo_repo,
                    "chunks",
                    MagicMock(count_documents=Mock(return_value=100)),
                ):
                    with patch.object(
                        controller.mongo_repo.chunks,
                        "aggregate",
                        return_value=[{"avg": 0.5}],
                    ):
                        with patch.object(
                            controller.neo4j_repo, "get_ltm_by_topic", return_value=[]
                        ):
                            with patch.object(
                                controller.feedback_service,
                                "get_stale_memories",
                                return_value=[],
                            ):
                                result = controller.evaluate_and_adjust()

                                assert "metrics" in result
                                assert "adjustments" in result


class TestVersionManager:
    """版本管理器测试"""

    def test_config_defaults(self):
        """测试默认配置"""
        assert VersionConfig.CONFLICT_SIMILARITY_THRESHOLD == 0.7
        assert VersionConfig.MERGE_SIMILARITY_THRESHOLD == 0.9

    def test_text_similarity(self):
        """测试文本相似度计算"""
        vm = VersionManager(
            neo4j_repo=Mock(),
            mongo_repo=Mock(),
        )

        sim1 = vm._text_similarity("hello world", "hello world")
        assert sim1 == 1.0

        sim2 = vm._text_similarity("hello world", "goodbye world")
        assert sim2 < 1.0
        assert sim2 > 0.0

        sim3 = vm._text_similarity("hello", "xyz")
        assert sim3 == 0.0

    def test_detect_conflict_no_existing(self):
        """测试无现有版本时的冲突检测"""
        vm = VersionManager(
            neo4j_repo=Mock(),
            mongo_repo=Mock(),
        )

        with patch.object(vm.neo4j_repo, "get_ltm_by_topic", return_value=[]):
            result = vm.detect_conflict("test_topic", "new conclusion")

            assert result.has_conflict is False
            assert result.recommended_action == "create"


class TestLTMNode:
    """LTM节点模型测试"""

    def test_ltm_node_creation(self):
        """测试LTM节点创建"""
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
        """测试LTM节点默认值"""
        ltm = LTMNode(topic="test", conclusion="test")

        assert ltm.version == 1
        assert ltm.conditions == []
        assert ltm.confidence == 0.5
        assert ltm.sources == []
        assert ltm.source_chunk_ids == []


class TestIntegrationFlow:
    """集成流程测试"""

    def test_full_memory_lifecycle(self):
        """测试完整记忆生命周期"""
        mongo_repo = Mock()
        neo4j_repo = Mock()
        feedback_service = Mock()
        decay_scheduler = Mock()
        abstraction_engine = Mock()
        version_manager = Mock()

        feedback_service.increment_importance = Mock()
        decay_scheduler.apply_decay_all = Mock(return_value={"updated": 5})
        abstraction_engine.get_abstraction_candidates = Mock(return_value=[])

        feedback_service.get_top_memories = Mock(return_value=[])
        mongo_repo.chunks.count_documents = Mock(return_value=100)
        mongo_repo.chunks.aggregate = Mock(return_value=[{"avg": 0.5}])
        neo4j_repo.get_ltm_by_topic = Mock(return_value=[])
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
        assert "decay_stats" in result
        assert "abstraction_results" in result

        decay_scheduler.apply_decay_all.assert_called_once()
        abstraction_engine.get_abstraction_candidates.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
