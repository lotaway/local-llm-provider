from typing import List, Dict
from difflib import SequenceMatcher
from constants import LEARNING
from interfaces.version_types import IVersionManager


class VersionManager(IVersionManager):
    CONFLICT_THRESHOLD = 0.7
    MERGE_THRESHOLD = 0.9
    CONFLICT_SIMILARITY_THRESHOLD = 0.7
    MERGE_SIMILARITY_THRESHOLD = 0.9

    def __init__(self, neo4j_repo=None, mongo_repo=None):
        self._neo4j_repo = neo4j_repo
        self._mongo_repo = mongo_repo

    def _text_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()

    def detect_conflict(self, topic: str, conclusion: str) -> Dict:
        existing = []
        if self._neo4j_repo and hasattr(self._neo4j_repo, "get_ltm_by_topic"):
            try:
                existing = self._neo4j_repo.get_ltm_by_topic(topic) or []
            except Exception:
                existing = []
        if not existing and self._neo4j_repo and hasattr(self._neo4j_repo, "get_ltm_versions_chain"):
            try:
                existing = self._neo4j_repo.get_ltm_versions_chain(topic) or []
            except Exception:
                existing = []

        max_sim = 0.0
        for e in existing:
            c = e.get("conclusion") or e.get("conclusion_text") or ""
            sim = self._text_similarity(conclusion, c)
            if sim > max_sim:
                max_sim = sim

        has_conflict = max_sim >= self.CONFLICT_SIMILARITY_THRESHOLD
        if max_sim >= self.MERGE_SIMILARITY_THRESHOLD:
            return {"has_conflict": has_conflict, "similarity": max_sim, "action": "merge", "recommended_action": "merge"}
        if has_conflict:
            return {"has_conflict": True, "similarity": max_sim, "action": "keep_both", "recommended_action": "keep_both"}
        return {"has_conflict": False, "similarity": max_sim, "action": "create", "recommended_action": "create"}

    def upsert_ltm(self, ltm: Dict) -> str:
        if not LEARNING:
            return ""
        topic = ltm.get("topic", "unknown")
        conclusion = ltm.get("conclusion", "")
        conflict = self.detect_conflict(topic, conclusion)
        version = int(ltm.get("version", 1))
        if conflict.get("action") == "merge":
            version = version
        elif conflict.get("action") == "keep_both":
            version = version + 1
        else:
            version = version

        ltm_obj = ltm
        if hasattr(ltm_obj, "topic"):  # LTMNode
            try:
                from schemas.graph import LTMNode
                if not isinstance(ltm_obj, LTMNode):
                    ltm_obj = LTMNode(**ltm)
            except Exception:
                pass

        if self._neo4j_repo and hasattr(self._neo4j_repo, "save_ltm"):
            try:
                res = self._neo4j_repo.save_ltm(ltm_obj)
                return res
            except Exception:
                pass

        return f"{topic}_v{version}"

    def get_version_chain(self, topic: str) -> List[Dict]:
        if self._neo4j_repo and hasattr(self._neo4j_repo, "get_ltm_versions_chain"):
            try:
                return self._neo4j_repo.get_ltm_versions_chain(topic) or []
            except Exception:
                return []
        return []

    def rollback_to(self, topic: str, version: int) -> bool:
        if not LEARNING:
            return False
        if self._neo4j_repo and hasattr(self._neo4j_repo, "rollback_ltm"):
            try:
                return bool(self._neo4j_repo.rollback_ltm(topic, version))
            except Exception:
                return False
        return True
