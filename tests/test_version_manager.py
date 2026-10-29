import pytest

from services import version_manager as version_module
from services.version_manager import VersionManager


class FakeNeo4jRepository:
    def __init__(self, versions=None):
        self.versions = {topic: list(items) for topic, items in (versions or {}).items()}
        self.relations = []
        self.episodic = {}

    def get_ltm_versions_chain(self, topic):
        return sorted(self.versions.get(topic, []), key=lambda item: item["version"], reverse=True)

    def save_ltm(self, node):
        record = node.model_dump()
        existing = [item for item in self.versions.setdefault(node.topic, []) if item["version"] != node.version]
        existing.append(record)
        self.versions[node.topic] = sorted(existing, key=lambda item: item["version"], reverse=True)
        return f"{node.topic}_v{node.version}"

    def create_ltm_version_relation(self, old_key, new_key, relation_types=None):
        self.relations.append((old_key, new_key, tuple(relation_types or ())))

    def get_episodic_for_ltm(self, topic_version):
        return self.episodic.get(topic_version, [])


class FakeLtmIndex:
    def __init__(self, similarity=None):
        self.similarity = similarity

    def nearest_similarity(self, ltm):
        return self.similarity


def version(version_number, conclusion, confidence=0.6, sources=None, chunks=None):
    return {
        "topic": "vector-search",
        "version": version_number,
        "topic_version": f"vector-search_v{version_number}",
        "conclusion": conclusion,
        "conditions": [f"condition-{version_number}"],
        "confidence": confidence,
        "sources": sources or [f"source-{version_number}"],
        "source_chunk_ids": chunks or [f"chunk-{version_number}"],
    }


@pytest.fixture(autouse=True)
def enable_learning(monkeypatch):
    monkeypatch.setattr(version_module, "LEARNING", True)


class TestConflictDetection:
    def test_vector_similarity_participates_in_conflict(self):
        repo = FakeNeo4jRepository({"vector-search": [version(1, "completely different")]})
        manager = VersionManager(neo4j_repo=repo, ltm_index=FakeLtmIndex(similarity=0.95))

        result = manager.detect_conflict("vector-search", "unrelated text")

        assert result["vector_similarity"] == 0.95
        assert result["action"] == "merge"

    def test_different_conclusion_creates_new_version(self):
        repo = FakeNeo4jRepository({"vector-search": [version(1, "alpha beta")]})
        manager = VersionManager(neo4j_repo=repo)

        result = manager.detect_conflict("vector-search", "omega omega omega")

        assert result["action"] == "create"


class TestVersionMerge:
    def test_same_source_merges_conditions_sources_and_confidence(self):
        repo = FakeNeo4jRepository(
            {"vector-search": [version(1, "same conclusion", 0.6, ["a"], ["chunk-a"])]}
        )
        manager = VersionManager(neo4j_repo=repo)

        saved = manager.upsert_ltm(
            {
                "topic": "vector-search",
                "conclusion": "same conclusion",
                "conditions": ["condition-1", "new-condition"],
                "confidence": 0.8,
                "sources": ["a", "b"],
                "source_chunk_ids": ["chunk-a", "chunk-b"],
            }
        )

        merged = repo.versions["vector-search"][0]
        assert saved == "vector-search_v1"
        assert merged["conditions"] == ["condition-1", "new-condition"]
        assert merged["sources"] == ["a", "b"]
        assert merged["confidence"] == pytest.approx(0.7333)

    def test_different_source_keeps_version_chain(self):
        repo = FakeNeo4jRepository(
            {"vector-search": [version(1, "same conclusion", 0.6, ["a"], ["chunk-a"])]}
        )
        manager = VersionManager(neo4j_repo=repo)

        saved = manager.upsert_ltm(
            {
                "topic": "vector-search",
                "conclusion": "same conclusion",
                "confidence": 0.9,
                "sources": ["b"],
                "source_chunk_ids": ["chunk-b"],
            }
        )

        assert saved == "vector-search_v2"
        assert repo.relations[0][0] == "vector-search_v1"
        assert repo.relations[0][1] == "vector-search_v2"
        assert "PREV_VERSION" in repo.relations[0][2]
        assert "DERIVED_FROM" in repo.relations[0][2]


class TestRollback:
    def test_rollback_creates_restored_version_and_reports_impact(self):
        repo = FakeNeo4jRepository(
            {
                "vector-search": [
                    version(2, "newer conclusion", 0.9),
                    version(1, "older conclusion", 0.5),
                ]
            }
        )
        repo.episodic["vector-search_v1"] = [{"chunk_id": "chunk-a"}]
        manager = VersionManager(neo4j_repo=repo)

        result = manager.rollback_to("vector-search", 1)

        assert result["success"] is True
        assert result["new_version"] == 3
        assert result["affected_versions"] == [2]
        assert result["dependencies"] == ["chunk-a"]
        assert repo.versions["vector-search"][0]["conclusion"] == "older conclusion"

    def test_rollback_rejects_unknown_version(self):
        repo = FakeNeo4jRepository({"vector-search": [version(1, "only", 0.5)]})
        manager = VersionManager(neo4j_repo=repo)

        result = manager.rollback_to("vector-search", 9)

        assert result["success"] is False
        assert result["reason"] == "version_not_found"
