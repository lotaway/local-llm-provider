import logging
from dataclasses import dataclass
from typing import Any, Optional

from constants import (
    DB_HOST,
    DB_PORT,
    MOLT_LTM_COLLECTION,
    MOLT_LTM_INDEX,
    MOLT_LTM_TTL_DAYS,
    MOLT_SENSORY_COLLECTION,
    MOLT_SENSORY_TTL_SECONDS,
    MOLT_STRATEGY_HISTORY_COLLECTION,
    MOLT_WORKING_MEMORY_CAPACITY,
)
from infrastructure.rag.ltm_index import MilvusEsLtmIndex
from repositories.sensory_cache import MongoSensoryCache
from services.working_memory import RingWorkingMemory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoltDependencies:
    neo4j_repo: Optional[Any] = None
    ltm_index: Optional[Any] = None
    sensory_cache: Optional[Any] = None
    working_memory: Optional[Any] = None
    history_store: Optional[Any] = None


def build_molt_dependencies(local_rag: Any, include_ltm_index: bool = True) -> MoltDependencies:
    tiers = build_memory_tiers(local_rag)
    database = _database(local_rag)
    return MoltDependencies(
        neo4j_repo=getattr(local_rag, "neo4j_repo", None),
        ltm_index=_ltm_index(local_rag) if include_ltm_index else None,
        sensory_cache=tiers.sensory_cache,
        working_memory=tiers.working_memory,
        history_store=_history_store(database),
    )


def build_memory_tiers(local_rag: Any) -> MoltDependencies:
    return MoltDependencies(
        sensory_cache=_sensory_cache(_database(local_rag)),
        working_memory=RingWorkingMemory(capacity=MOLT_WORKING_MEMORY_CAPACITY),
    )


def _database(local_rag: Any) -> Optional[Any]:
    return getattr(getattr(local_rag, "mongo_repo", None), "db", None)


def _sensory_cache(database: Any) -> Optional[MongoSensoryCache]:
    if database is None:
        return None
    return MongoSensoryCache(database, MOLT_SENSORY_COLLECTION, MOLT_SENSORY_TTL_SECONDS)


def _history_store(database: Any) -> Optional[Any]:
    if database is None:
        return None
    from repositories.strategy_history_repository import MongoStrategyHistoryStore

    return MongoStrategyHistoryStore(database, MOLT_STRATEGY_HISTORY_COLLECTION)


def _ltm_index(local_rag: Any) -> Optional[MilvusEsLtmIndex]:
    if local_rag is None:
        return None
    es_client = getattr(getattr(local_rag, "es_retriever", None), "es_client", None)
    if es_client is None:
        return None
    try:
        return _connect_ltm_index(es_client)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        logger.exception("molt ltm index unavailable", extra={"error": str(error)})
        return None


def _connect_ltm_index(es_client: Any) -> MilvusEsLtmIndex:
    from pymilvus import MilvusClient
    from infrastructure.config.settings import get_settings
    from infrastructure.embedding.sentence_transformer_provider import SentenceTransformerEmbeddingProvider

    return MilvusEsLtmIndex(
        embedding_provider=SentenceTransformerEmbeddingProvider(get_settings()),
        milvus_client=MilvusClient(uri=f"http://{DB_HOST}:{DB_PORT}"),
        es_client=es_client,
        collection_name=MOLT_LTM_COLLECTION,
        index_name=MOLT_LTM_INDEX,
        ttl_days=MOLT_LTM_TTL_DAYS,
    )
