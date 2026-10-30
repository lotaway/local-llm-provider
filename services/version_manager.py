from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from constants import LEARNING
from interfaces.version_types import IVersionManager
from schemas.graph import LTMNode


class VersionManager(IVersionManager):
    DEFAULT_CONFLICT_THRESHOLD = 0.7
    DEFAULT_MERGE_THRESHOLD = 0.9
    VERSION_RELATIONS = (
        "PREV_VERSION",
        "NEXT_VERSION",
        "SUPERSEDES",
        "DERIVED_FROM",
        "SUPERSEDED_BY",
    )

    def __init__(
        self,
        neo4j_repo=None,
        mongo_repo=None,
        ltm_index=None,
        conflict_threshold: float = DEFAULT_CONFLICT_THRESHOLD,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    ):
        self._neo4j_repo = neo4j_repo
        self._mongo_repo = mongo_repo
        self._ltm_index = ltm_index
        self._conflict_threshold = conflict_threshold
        self._merge_threshold = merge_threshold

    def detect_conflict(
        self, topic: str, conclusion: str, ltm: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        existing = self._existing_versions(topic)
        best, text_similarity = self._best_match(existing, conclusion)
        payload = ltm if ltm is not None else {"topic": topic, "conclusion": conclusion}
        vector_similarity = self._vector_similarity(payload)
        similarity = max(text_similarity, vector_similarity or 0.0)
        return {
            "has_conflict": similarity >= self._conflict_threshold,
            "similarity": similarity,
            "text_similarity": text_similarity,
            "vector_similarity": vector_similarity,
            "action": self._conflict_action(similarity),
            "matched_version": best.get("version") if best else None,
            "existing": existing,
        }

    def upsert_ltm(self, ltm: Dict[str, Any]) -> str:
        if not LEARNING:
            return ""
        record = dict(ltm)
        conflict = self.detect_conflict(
            record.get("topic", "unknown"), record.get("conclusion", ""), record
        )
        if self._should_merge(conflict, record):
            return self._save(self._merge(record, conflict["matched_version"], conflict["existing"]))
        return self._create_version(record, conflict)

    def get_version_chain(self, topic: str) -> List[Dict[str, Any]]:
        if self._neo4j_repo is None:
            return []
        return self._neo4j_repo.get_ltm_versions_chain(topic) or []

    def rollback_to(self, topic: str, version: int) -> Dict[str, Any]:
        if not LEARNING:
            return self._rollback_result(False, topic, version, reason="learning_disabled")
        chain = self.get_version_chain(topic)
        target = self._find_version(chain, version)
        if target is None:
            return self._rollback_result(False, topic, version, reason="version_not_found")
        latest = max(chain, key=lambda item: int(item.get("version", 0)))
        if int(latest.get("version", 0)) == version:
            return self._rollback_result(True, topic, version, reason="already_current")
        restored = self._restored_record(target, latest)
        saved = self._save(restored)
        self._link_versions(latest, restored)
        return self._rollback_detail(topic, version, restored, chain, saved)

    def _existing_versions(self, topic: str) -> List[Dict[str, Any]]:
        if self._neo4j_repo is None:
            return []
        return self._neo4j_repo.get_ltm_versions_chain(topic) or []

    def _best_match(self, existing: List[Dict[str, Any]], conclusion: str) -> tuple:
        best = None
        best_similarity = 0.0
        for version in existing:
            similarity = self._text_similarity(conclusion, self._conclusion(version))
            if similarity >= best_similarity:
                best, best_similarity = version, similarity
        return best, best_similarity

    def _conclusion(self, version: Dict[str, Any]) -> str:
        return version.get("conclusion") or version.get("conclusion_text") or ""

    def _vector_similarity(self, ltm: Optional[Dict[str, Any]]) -> Optional[float]:
        if ltm is None or self._ltm_index is None:
            return None
        return self._ltm_index.nearest_similarity(ltm)

    def _conflict_action(self, similarity: float) -> str:
        if similarity >= self._merge_threshold:
            return "merge"
        if similarity >= self._conflict_threshold:
            return "keep_both"
        return "create"

    def _should_merge(self, conflict: Dict[str, Any], record: Dict[str, Any]) -> bool:
        if conflict["action"] != "merge" or not conflict["existing"]:
            return False
        return self._same_source(conflict["existing"], record)

    def _same_source(self, existing: List[Dict[str, Any]], record: Dict[str, Any]) -> bool:
        target = set(record.get("source_chunk_ids") or record.get("sources") or [])
        if not target:
            return True
        for version in existing:
            source = set(version.get("source_chunk_ids") or version.get("sources") or [])
            if target & source:
                return True
        return False

    def _merge(
        self, record: Dict[str, Any], matched_version: Any, existing: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        base = self._find_version(existing, matched_version) or existing[0]
        merged = dict(base)
        merged["conditions"] = self._union(base.get("conditions"), record.get("conditions"))
        merged["sources"] = self._union(base.get("sources"), record.get("sources"))
        merged["source_chunk_ids"] = self._union(
            base.get("source_chunk_ids"), record.get("source_chunk_ids")
        )
        merged["confidence"] = self._weighted_confidence(base, record)
        merged["conclusion"] = self._stronger_conclusion(base, record)
        merged["updated_at"] = self._now()
        return merged

    def _union(self, left: Optional[List[Any]], right: Optional[List[Any]]) -> List[Any]:
        return list(dict.fromkeys((left or []) + (right or [])))

    def _weighted_confidence(self, base: Dict[str, Any], record: Dict[str, Any]) -> float:
        base_weight = max(1, len(base.get("source_chunk_ids") or base.get("sources") or []))
        new_weight = max(1, len(record.get("source_chunk_ids") or record.get("sources") or []))
        total = base_weight + new_weight
        weighted = (
            float(base.get("confidence", 0.0)) * base_weight
            + float(record.get("confidence", 0.0)) * new_weight
        )
        return round(weighted / total, 4)

    def _stronger_conclusion(self, base: Dict[str, Any], record: Dict[str, Any]) -> str:
        if float(record.get("confidence", 0.0)) > float(base.get("confidence", 0.0)):
            return record.get("conclusion", "")
        return self._conclusion(base)

    def _create_version(self, record: Dict[str, Any], conflict: Dict[str, Any]) -> str:
        existing = conflict["existing"]
        record["version"] = self._next_version(existing)
        saved = self._save(record)
        if existing:
            self._link_versions(existing[0], record)
        return saved

    def _next_version(self, existing: List[Dict[str, Any]]) -> int:
        versions = [int(item.get("version", 0)) for item in existing]
        return max(versions, default=0) + 1

    def _save(self, record: Dict[str, Any]) -> str:
        if self._neo4j_repo is None:
            return ""
        node = self._to_node(record)
        return self._neo4j_repo.save_ltm(node)

    def _to_node(self, record: Dict[str, Any]) -> LTMNode:
        payload = {name: record[name] for name in LTMNode.model_fields if name in record}
        return LTMNode(**payload)

    def _link_versions(self, previous: Dict[str, Any], current: Dict[str, Any]) -> None:
        if self._neo4j_repo is None:
            return
        old_key = previous.get("topic_version") or self._topic_version(previous)
        new_key = current.get("topic_version") or self._topic_version(current)
        self._neo4j_repo.create_ltm_version_relation(
            old_key, new_key, relation_types=self.VERSION_RELATIONS
        )

    def _topic_version(self, record: Dict[str, Any]) -> str:
        return f"{record.get('topic', '')}_v{int(record.get('version', 1))}"

    def _find_version(self, chain: List[Dict[str, Any]], version: Any) -> Optional[Dict[str, Any]]:
        return next((item for item in chain if int(item.get("version", 0)) == int(version)), None)

    def _restored_record(self, target: Dict[str, Any], latest: Dict[str, Any]) -> Dict[str, Any]:
        restored = dict(target)
        restored.pop("topic_version", None)
        restored.pop("superseded_by", None)
        restored["version"] = int(latest.get("version", 0)) + 1
        restored["updated_at"] = self._now()
        return restored

    def _rollback_detail(
        self,
        topic: str,
        version: int,
        restored: Dict[str, Any],
        chain: List[Dict[str, Any]],
        saved: str,
    ) -> Dict[str, Any]:
        affected = sorted(
            int(item.get("version", 0))
            for item in chain
            if int(item.get("version", 0)) > version
        )
        return {
            "success": True,
            "topic": topic,
            "target_version": version,
            "new_version": restored["version"],
            "saved": saved,
            "affected_versions": affected,
            "dependencies": self._dependencies(topic, version),
            "timestamp": self._now(),
        }

    def _dependencies(self, topic: str, version: int) -> List[str]:
        if self._neo4j_repo is None or not hasattr(self._neo4j_repo, "get_episodic_for_ltm"):
            return []
        chunks = self._neo4j_repo.get_episodic_for_ltm(f"{topic}_v{version}")
        return [str(chunk.get("chunk_id")) for chunk in chunks if chunk.get("chunk_id")]

    def _rollback_result(
        self, success: bool, topic: str, version: int, reason: str
    ) -> Dict[str, Any]:
        return {
            "success": success,
            "topic": topic,
            "target_version": version,
            "reason": reason,
            "timestamp": self._now(),
        }

    def _text_similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        return SequenceMatcher(None, left, right).ratio()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
