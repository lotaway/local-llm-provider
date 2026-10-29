from datetime import datetime, timezone
from typing import Any, Dict, List

from pymongo.collection import Collection

from constants import MOLT_STRATEGY_HISTORY_COLLECTION
from interfaces.molt_types import IStrategyHistoryStore


class MongoStrategyHistoryStore(IStrategyHistoryStore):
    def __init__(self, db: Any, collection_name: str = MOLT_STRATEGY_HISTORY_COLLECTION):
        self._collection: Collection = db[collection_name]
        self._collection.create_index("created_at")

    def append(self, record: Dict[str, Any]) -> None:
        document = dict(record)
        document["created_at"] = datetime.now(timezone.utc)
        self._collection.insert_one(document)

    def recent(self, limit: int) -> List[Dict[str, Any]]:
        cursor = (
            self._collection.find({}, {"_id": 0})
            .sort("created_at", -1)
            .limit(max(1, limit))
        )
        return list(reversed(list(cursor)))

    def count(self) -> int:
        return self._collection.count_documents({})
