import re
from collections import Counter
from typing import Any, Dict, List

from interfaces.abstraction_types import ITopicExtractor

MAX_TOPICS = 10
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}")
STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "have", "has", "had", "not", "but", "you", "your", "they", "their", "will",
        "can", "could", "should", "would", "about", "into", "than", "then", "when",
        "what", "which", "who", "how", "why", "where", "there", "here", "also",
        "using", "used", "use", "based", "such", "more", "most", "some", "any",
        "的", "了", "是", "在", "和", "与", "及", "中", "对", "为", "把", "被",
        "就", "也", "都", "而", "以及", "一个", "我们", "他们", "这个", "那个",
        "可以", "需要", "进行", "通过", "如果", "因为", "所以", "但是", "并且",
    }
)


class GraphKeywordTopicExtractor(ITopicExtractor):
    def __init__(self, neo4j_repo=None, max_topics: int = MAX_TOPICS):
        self._neo4j_repo = neo4j_repo
        self._max_topics = max_topics

    def extract(self, episodes: List[Dict[str, Any]]) -> List[str]:
        texts = [text for text in (self._text(episode) for episode in episodes) if text]
        if not texts:
            return []
        return self._merge(self._concepts(texts), self._keywords(texts))

    def _text(self, episode: Dict[str, Any]) -> str:
        return str(episode.get("text") or episode.get("content") or "").strip()

    def _concepts(self, texts: List[str]) -> List[str]:
        if self._neo4j_repo is None:
            return []
        corpus = " ".join(texts).lower()
        return [
            concept["name"]
            for concept in self._neo4j_repo.list_concepts()
            if concept.get("name") and concept["name"].lower() in corpus
        ]

    def _keywords(self, texts: List[str]) -> List[str]:
        counts = Counter()
        for text in texts:
            counts.update(self._tokens(text))
        return [token for token, _ in counts.most_common()]

    def _tokens(self, text: str) -> List[str]:
        return [
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
            if not self._is_stopword(token)
        ]

    def _is_stopword(self, token: str) -> bool:
        return token.lower() in STOPWORDS

    def _merge(self, concepts: List[str], keywords: List[str]) -> List[str]:
        merged = []
        seen = set()
        for topic in concepts + keywords:
            normalized = topic.strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
            if len(merged) == self._max_topics:
                break
        return merged
