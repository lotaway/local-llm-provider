"""
Phase 1 验收测试：基础数据模型扩展
"""

import pytest
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repositories.mongodb_repository import MongoDBRepository
from repositories.neo4j_repository import Neo4jRepository
from schemas.graph import LTMNode


class TestMongoDBMemoryFields:
    """测试MongoDB记忆字段扩展"""

    @pytest.fixture
    def mongo_repo(self):
        repo = MongoDBRepository()
        yield repo
        repo.close()

    def test_chunk_has_memory_fields(self, mongo_repo):
        """验证新创建的chunk包含记忆生命周期字段"""
        chunk = mongo_repo.save_chunk(
            chunk_id="test_chunk_phase1",
            doc_id="test_doc_001",
            index=0,
            text="This is a test chunk for phase 1 validation.",
        )

        assert "importance_score" in chunk
        assert "decay_rate" in chunk
        assert "review_count" in chunk
        assert "memory_type" in chunk
        assert "created_at" in chunk
        assert "updated_at" in chunk

        assert chunk["importance_score"] == 0.5
        assert chunk["decay_rate"] == 0.01
        assert chunk["review_count"] == 0
        assert chunk["memory_type"] == "episodic"

    def test_update_importance_score(self, mongo_repo):
        """测试重要性评分更新"""
        chunk_id = "test_chunk_phase1"

        mongo_repo.update_importance_score(chunk_id, 0.8)
        updated = mongo_repo.get_chunk(chunk_id)

        assert updated["importance_score"] == 0.8

    def test_increment_importance(self, mongo_repo):
        """测试重要性递增加权"""
        chunk_id = "test_chunk_phase1"

        mongo_repo.increment_importance(chunk_id, 0.1)
        updated = mongo_repo.get_chunk(chunk_id)

        assert updated["importance_score"] == 0.9
        assert updated["review_count"] == 1
        assert updated["last_reviewed"] is not None

    def test_get_chunks_by_importance(self, mongo_repo):
        """测试按重要性排序查询"""
        chunks = mongo_repo.get_chunks_by_importance(memory_type="episodic", limit=10)

        assert isinstance(chunks, list)
        if len(chunks) >= 2:
            for i in range(len(chunks) - 1):
                assert (
                    chunks[i]["importance_score"] >= chunks[i + 1]["importance_score"]
                )

    def test_get_low_importance_chunks(self, mongo_repo):
        """测试低重要性记忆查询"""
        low_importance = mongo_repo.get_low_importance_chunks(
            threshold=0.3, memory_type="episodic", limit=10
        )

        assert isinstance(low_importance, list)


class TestNeo4jLTMSchema:
    """测试Neo4j LTM模式扩展"""

    @pytest.fixture
    def neo4j_repo(self):
        repo = Neo4jRepository()
        yield repo
        repo.close()

    def test_save_ltm_node(self, neo4j_repo):
        """测试保存LTM节点"""
        ltm = LTMNode(
            topic="test_topic_phase1",
            version=1,
            conclusion="This is a test LTM conclusion for phase 1.",
            conditions=["condition1", "condition2"],
            confidence=0.85,
            sources=["source1", "source2"],
            source_chunk_ids=["chunk1", "chunk2"],
        )

        topic_version = neo4j_repo.save_ltm(ltm)

        assert topic_version == "test_topic_phase1_v1"

    def test_get_ltm_by_topic(self, neo4j_repo):
        """测试按主题查询LTM"""
        ltms = neo4j_repo.get_ltm_by_topic("test_topic_phase1")

        assert isinstance(ltms, list)
        assert len(ltms) >= 1

        latest = ltms[0]
        assert latest["topic"] == "test_topic_phase1"
        assert latest["version"] >= 1
        assert "conclusion" in latest
        assert "confidence" in latest

    def test_get_latest_ltm(self, neo4j_repo):
        """测试获取最新版本LTM"""
        latest = neo4j_repo.get_latest_ltm("test_topic_phase1")

        assert latest is not None
        assert latest["topic"] == "test_topic_phase1"

    def test_link_episodic_to_ltm(self, neo4j_repo):
        """测试建立 Episodic 到 LTM 的关系"""
        neo4j_repo.link_episodic_to_ltm(
            chunk_id="test_chunk_phase1", ltm_topic_version="test_topic_phase1_v1"
        )

        episodic_chunks = neo4j_repo.get_episodic_for_ltm("test_topic_phase1_v1")

        assert isinstance(episodic_chunks, list)

    def test_ltm_version_chain(self, neo4j_repo):
        """测试LTM版本链"""
        chain = neo4j_repo.get_ltm_versions_chain("test_topic_phase1")

        assert isinstance(chain, list)
        assert len(chain) >= 1

    def test_search_ltm_by_conclusion(self, neo4j_repo):
        """测试按结论内容搜索LTM"""
        results = neo4j_repo.search_ltm_by_conclusion("test LTM conclusion", limit=5)

        assert isinstance(results, list)


class TestBackwardCompatibility:
    """测试向后兼容性"""

    @pytest.fixture
    def mongo_repo(self):
        repo = MongoDBRepository()
        yield repo
        repo.close()

    def test_old_chunks_get_defaults(self, mongo_repo):
        """验证老数据有默认值"""
        old_chunk = mongo_repo.get_chunk("test_chunk_phase1")

        if old_chunk:
            assert "importance_score" in old_chunk
            assert "decay_rate" in old_chunk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
