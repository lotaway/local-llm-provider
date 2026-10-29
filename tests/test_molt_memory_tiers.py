from datetime import datetime, timedelta

import pytest

from repositories.sensory_cache import MongoSensoryCache
from services.evolution_dispatcher import EvolutionDispatcher
from services.working_memory import RingWorkingMemory


class FakeCollection:
    def __init__(self):
        self.documents = {}
        self.indexes = []

    def create_index(self, field, expireAfterSeconds):
        self.indexes.append((field, expireAfterSeconds))

    def update_one(self, query, update, upsert=False):
        entry_id = query["_id"]
        document = self.documents.setdefault(entry_id, {"_id": entry_id}) if upsert else self.documents.get(entry_id)
        if document is None:
            return
        document.update(update.get("$set", {}))
        for key, value in update.get("$setOnInsert", {}).items():
            document.setdefault(key, value)
        for key, value in update.get("$inc", {}).items():
            document[key] = document.get(key, 0) + value

    def find_one(self, query):
        return self.documents.get(query["_id"])

    def delete_one(self, query):
        self.documents.pop(query["_id"], None)

    def count_documents(self, query):
        return len(self.documents)

    def aggregate(self, pipeline):
        return [{"total": sum(d.get("hits", 0) for d in self.documents.values())}]


class FakeDatabase:
    def __init__(self):
        self.collection = FakeCollection()

    def __getitem__(self, name):
        return self.collection


class FakeSensoryCache:
    def __init__(self):
        self.entries = {}
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def remember(self, entry_id, payload, ttl_seconds=None):
        self.entries[entry_id] = payload

    def recall(self, entry_id):
        return self.entries.get(entry_id)

    def forget(self, entry_id):
        self.entries.pop(entry_id, None)

    def stats(self):
        return {"entries": len(self.entries)}


class TestWorkingMemory:
    def test_capacity_is_bounded(self):
        with pytest.raises(ValueError):
            RingWorkingMemory(capacity=4)
        with pytest.raises(ValueError):
            RingWorkingMemory(capacity=10)

    def test_rehearsal_and_eviction(self):
        memory = RingWorkingMemory(capacity=5)
        for index in range(6):
            memory.focus(f"item-{index}")
        memory.focus("item-5")

        assert memory.recall("item-0") is None
        assert memory.recall("item-5")["rehearsal_count"] == 2
        assert memory.stats()["evictions"] == 1
        assert memory.stats()["size"] == 5


class TestSensoryCache:
    def test_ttl_index_and_hit_stats(self):
        database = FakeDatabase()
        cache = MongoSensoryCache(database, "sensory", default_ttl_seconds=60)
        cache.start()
        cache.remember("entry", {"value": 1})

        assert cache.recall("entry") == {"value": 1}
        assert cache.stats()["hits"] == 1
        assert database.collection.indexes == [("expires_at", 0)]

    def test_expired_entry_is_miss(self):
        database = FakeDatabase()
        cache = MongoSensoryCache(database, "sensory", default_ttl_seconds=60)
        cache.remember("entry", {"value": 1})
        database.collection.documents["entry"]["expires_at"] = datetime.utcnow() - timedelta(seconds=1)

        assert cache.recall("entry") is None
        assert cache.stats()["misses"] == 1


class TestEvolutionDispatcherMemoryTiers:
    def test_signal_is_admitted_to_both_tiers(self):
        sensory = FakeSensoryCache()
        working = RingWorkingMemory(capacity=7)
        dispatcher = EvolutionDispatcher(sensory_cache=sensory, working_memory=working)

        dispatcher.enqueue(
            {
                "feedback_detected": True,
                "type": "negative",
                "confidence": 0.9,
                "targets": ["tool_choice"],
            }
        )

        assert len(sensory.entries) == 1
        assert working.recall("tool_choice")["rehearsal_count"] == 1
