from typing import List, Dict
from constants import LEARNING
from interfaces.version_types import IVersionManager


class VersionManager(IVersionManager):
    CONFLICT_THRESHOLD = 0.7
    MERGE_THRESHOLD = 0.9

    def detect_conflict(self, topic: str, conclusion: str) -> Dict:
        return {"has_conflict": False, "similarity": 0.0, "action": "create"}

    def upsert_ltm(self, ltm: Dict) -> str:
        if not LEARNING:
            return ""
        return f"{ltm.get('topic', 'unknown')}_v{ltm.get('version', 1)}"

    def get_version_chain(self, topic: str) -> List[Dict]:
        return []

    def rollback_to(self, topic: str, version: int) -> bool:
        if not LEARNING:
            return False
        return True
