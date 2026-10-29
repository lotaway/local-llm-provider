from services.topic_extractor import GraphKeywordTopicExtractor


class FakeConceptRepository:
    def list_concepts(self):
        return [{"name": "Vector Search", "stable_id": "concept_vector_search"}]


class TestGraphKeywordTopicExtractor:
    def test_merges_concepts_and_keywords_without_stopwords(self):
        episodes = [
            {"text": "Vector Search uses the index and the embedding model."},
            {"text": "Vector Search improves the embedding retrieval quality."},
        ]
        extractor = GraphKeywordTopicExtractor(FakeConceptRepository(), max_topics=5)

        topics = extractor.extract(episodes)

        assert topics[0] == "Vector Search"
        assert "the" not in [topic.lower() for topic in topics]
        assert len(topics) == len({topic.lower() for topic in topics})
        assert len(topics) <= 5

    def test_returns_empty_for_empty_episodes(self):
        extractor = GraphKeywordTopicExtractor()

        assert extractor.extract([]) == []

    def test_chinese_tokens_are_supported(self):
        episodes = [{"content": "向量检索 提升 检索质量 与 召回率"}]

        topics = GraphKeywordTopicExtractor(max_topics=3).extract(episodes)

        assert "向量检索" in topics
        assert "与" not in topics
