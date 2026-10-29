import pytest
from langchain_core.documents import Document

from retrievers.chunk_filter import (
    GraphFilteredVectorRetriever,
    build_chunk_expr,
    similarity_search_with_chunk_filter,
)
from retrievers.es_retriever import ESBM25Retriever
from retrievers.graph_coordinator import GraphQueryCoordinator
from services.ingestion_checkpoint import (
    checksum_matches,
    is_complete,
    needs_rebuild,
    normalize_status,
    pending_steps,
)


class FakeConceptRepository:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def find_chunk_ids_by_concepts(self, terms, limit=100):
        self.calls.append((list(terms), limit))
        found = []
        for term in terms:
            for concept, chunk_ids in self.mapping.items():
                if concept in term:
                    found.extend(chunk_ids)
        return list(dict.fromkeys(found))[:limit]


class FakeVectorStore:
    def __init__(self, filtered_docs=None):
        self.filtered_docs = filtered_docs if filtered_docs is not None else []
        self.calls = []

    def similarity_search_with_score(self, query, k=4, expr=None):
        self.calls.append(("score", query, k, expr))
        if expr is not None:
            return self.filtered_docs
        return [(Document(page_content="fallback", metadata={"chunk_id": "c-fallback"}), 0.1)]

    def similarity_search(self, query, k=4, expr=None):
        self.calls.append(("plain", query, k, expr))
        if expr is not None:
            return self.filtered_docs
        return [Document(page_content="fallback", metadata={"chunk_id": "c-fallback"})]


class FakeBM25:
    def get_documents_with_scores(self, query):
        return [(Document(page_content="keyword", metadata={"chunk_id": "c-kw"}), 1.0)]


def test_coordinator_resolves_concepts_to_chunk_ids():
    repo = FakeConceptRepository({"向量": ["chunk-a", "chunk-b"]})
    coordinator = GraphQueryCoordinator(repo)

    chunk_ids = coordinator.find_chunk_ids("如何做向量检索")

    assert chunk_ids == ["chunk-a", "chunk-b"]
    assert repo.calls[0][0]


def test_chunk_expression_is_escaped_and_deduplicated():
    expr = build_chunk_expr(['a"b', "a\"b", "", "c"])

    assert expr == 'chunk_id in ["a\\"b", "c"]'


def test_filtered_vector_search_falls_back_when_store_rejects_expr():
    class StrictStore(FakeVectorStore):
        def similarity_search_with_score(self, query, k=4, expr=None):
            if expr is not None:
                raise TypeError("expr unsupported")
            return super().similarity_search_with_score(query, k, expr)

    store = StrictStore()
    results = similarity_search_with_chunk_filter(store, "q", 2, ["chunk-a"])

    assert len(results) == 1
    assert results[0][0].page_content == "fallback"


def test_filtered_vector_retriever_uses_graph_chunk_ids():
    vectorstore = FakeVectorStore(
        filtered_docs=[Document(page_content="graph hit", metadata={"chunk_id": "chunk-a"})]
    )
    retriever = GraphFilteredVectorRetriever(
        vectorstore=vectorstore, k=3, chunk_id_provider=lambda query: ["chunk-a"]
    )

    docs = retriever.invoke("vector search")

    assert docs[0].page_content == "graph hit"
    assert vectorstore.calls[0][3] == 'chunk_id in ["chunk-a"]'


def test_hybrid_retriever_applies_graph_filter_without_losing_bm25():
    from retrievers.hybrid_retriever import HybridRetriever

    vectorstore = FakeVectorStore(
        filtered_docs=[(Document(page_content="graph hit", metadata={"chunk_id": "chunk-a"}), 0.9)]
    )
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_retriever=FakeBM25(),
        k=3,
        chunk_id_provider=lambda query: ["chunk-a"],
    )

    docs = retriever.invoke("vector search")

    assert any(doc.metadata.get("chunk_id") == "chunk-a" for doc in docs)
    assert any("keyword" in doc.page_content for doc in docs)


def test_es_index_action_contains_flat_fields():
    retriever = ESBM25Retriever.__new__(ESBM25Retriever)
    object.__setattr__(retriever, "index_name", "rag_docs")
    document = Document(
        page_content="hello",
        metadata={"chunk_id": "chunk-a", "doc_id": "doc-a", "filename": "a.md"},
    )

    action = retriever._index_action(document)

    assert action["_id"] == "chunk-a"
    assert action["_source"]["chunk_id"] == "chunk-a"
    assert action["_source"]["doc_id"] == "doc-a"
    assert action["_source"]["filename"] == "a.md"
    assert action["_source"]["text"] == "hello"


def test_es_hit_exposes_highlight_and_scores():
    retriever = ESBM25Retriever.__new__(ESBM25Retriever)
    hit = {
        "_score": 2.5,
        "_source": {
            "chunk_id": "chunk-a",
            "doc_id": "doc-a",
            "filename": "a.md",
            "text": "hello world",
            "metadata": {"author": "me"},
        },
        "highlight": {"text": ["hello <em>world</em>"]},
    }

    document, score = retriever._parse_hit(hit)

    assert score == 2.5
    assert document.metadata["_highlight"]["text"] == ["hello <em>world</em>"]
    assert document.metadata["chunk_id"] == "chunk-a"


def test_ingestion_checkpoint_detects_resume_and_rebuild():
    status = normalize_status({"chunked": True, "embedded": True, "graphed": False})
    document = {"checksum": "sha256:abc", "status": status}

    assert pending_steps(status) == ["graphed"]
    assert not is_complete(status)
    assert checksum_matches(document, "abc")
    assert not needs_rebuild(document, "abc")
    assert needs_rebuild(document, "def")
    assert needs_rebuild(None, "abc") is False


def test_milvus_store_builds_filters():
    pymilvus = pytest.importorskip("pymilvus")
    assert pymilvus is not None
    from infrastructure.rag.milvus_store import MilvusVectorStore

    store = MilvusVectorStore.__new__(MilvusVectorStore)

    assert store._build_filter({"chunk_id": ["a", "b"]}) == 'chunk_id in ["a", "b"]'
    assert store._build_filter({"doc_id": {"$ne": "x"}}) == 'doc_id != "x"'
    assert store._build_filter(None) == ""
