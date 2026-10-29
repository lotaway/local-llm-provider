import logging
import re
from typing import List

logger = logging.getLogger(__name__)

MAX_QUERY_TERMS = 8
MIN_TERM_LENGTH = 2
QUERY_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "this",
        "that",
        "from",
        "into",
        "about",
        "请",
        "如何",
        "什么",
        "哪些",
        "怎么",
        "为什",
        "以及",
        "是否",
        "可以",
    }
)
TERM_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")


class GraphQueryCoordinator:
    def __init__(self, neo4j_repo, max_chunk_ids: int = 100):
        self._neo4j_repo = neo4j_repo
        self._max_chunk_ids = max_chunk_ids

    def find_chunk_ids(self, query: str) -> List[str]:
        terms = self._query_terms(query)
        if not terms:
            return []
        chunk_ids = self._neo4j_repo.find_chunk_ids_by_concepts(
            terms, limit=self._max_chunk_ids
        )
        if not chunk_ids:
            logger.debug(
                "graph coordination found no concept chunks", extra={"terms": terms}
            )
            return []
        logger.info(
            "graph coordination matched chunks",
            extra={"terms": terms, "chunk_count": len(chunk_ids)},
        )
        return chunk_ids

    def _query_terms(self, query: str) -> List[str]:
        if not query:
            return []
        terms = []
        for raw in TERM_PATTERN.findall(query):
            term = raw.strip().lower()
            if len(term) < MIN_TERM_LENGTH or term in QUERY_STOPWORDS:
                continue
            if term not in terms:
                terms.append(term)
        return terms[:MAX_QUERY_TERMS]
