import json

from infrastructure.rag.ltm_index import MilvusEsLtmIndex


class FakeEmbedding:
    dimension = 3

    def embed(self, text):
        return [0.1, 0.2, 0.3]


class FakeMilvus:
    def __init__(self):
        self.collections = {}
        self.rows = []

    def has_collection(self, name):
        return name in self.collections

    def create_collection(self, collection_name, **kwargs):
        self.collections[collection_name] = kwargs

    def upsert(self, collection_name, data):
        self.rows.extend(data)


class FakeIndices:
    def __init__(self):
        self.created = {}

    def exists(self, index):
        return index in self.created

    def create(self, index, mappings):
        self.created[index] = mappings


class FakeElasticsearch:
    def __init__(self):
        self.indices = FakeIndices()
        self.documents = {}

    def index(self, index, id, document):
        self.documents[id] = document


def build_index():
    milvus = FakeMilvus()
    es = FakeElasticsearch()
    index = MilvusEsLtmIndex(
        embedding_provider=FakeEmbedding(),
        milvus_client=milvus,
        es_client=es,
        collection_name="ltm_vectors",
        index_name="ltm_keywords",
        ttl_days=30,
    )
    return index, milvus, es


def ltm_record():
    return {
        "topic": "Vector Search",
        "version": 2,
        "conclusion": "Vector Search improves retrieval quality with embeddings.",
        "conditions": ["embedding model available"],
        "confidence": 0.8,
        "sources": ["doc-1"],
        "source_chunk_ids": ["doc-1_chunk_0001"],
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-02T00:00:00+00:00",
    }


class TestMilvusEsLtmIndex:
    def test_vector_contains_explainability_metadata(self):
        index, milvus, _ = build_index()

        document_id = index.save_ltm_vector(ltm_record())

        assert document_id == "ltm::Vector Search::v2"
        assert milvus.collections["ltm_vectors"]["dimension"] == 3
        metadata = json.loads(milvus.rows[0]["metadata"])
        assert metadata["source_chunk_ids"] == ["doc-1_chunk_0001"]
        assert metadata["last_reinforced_at"]
        assert metadata["expires_at"]
        assert metadata["version_relations"] == {}

    def test_keywords_are_indexed_with_tags(self):
        index, _, es = build_index()

        document_id = index.index_ltm_keywords(ltm_record())

        document = es.documents[document_id]
        assert "vector" in document["keywords"]
        assert document["source_chunk_ids"] == ["doc-1_chunk_0001"]
        assert document["version"] == 2
