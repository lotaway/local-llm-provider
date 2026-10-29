from retrievers.graph_evidence import GraphEvidenceExtractor


def subgraph():
    return [
        {
            "nodes": [
                {"stable_id": "concept_a", "properties": {"chunk_id": "chunk-1"}},
                {
                    "stable_id": "concept_b",
                    "evidence": [
                        {"source_chunk_ids": ["chunk-2", "chunk-1"]},
                        {"metadata": {"evidence_chunk_id": "chunk-3"}},
                    ],
                },
            ],
            "edges": [
                {
                    "from": "concept_a",
                    "to": "concept_b",
                    "type": "uses",
                    "properties": {"source_chunk_id": "chunk-4"},
                }
            ],
        }
    ]


class TestGraphEvidenceExtractor:
    def test_extracts_chunk_ids_from_nodes_edges_and_evidence(self):
        chunk_ids = GraphEvidenceExtractor().extract(subgraph())

        assert chunk_ids == ["chunk-1", "chunk-2", "chunk-3", "chunk-4"]

    def test_returns_empty_without_chunk_evidence(self):
        subgraphs = [{"nodes": [{"stable_id": "entity"}], "edges": []}]

        assert GraphEvidenceExtractor().extract(subgraphs) == []

    def test_accepts_list_and_scalar_identifiers(self):
        subgraphs = [{"nodes": [{"chunk_ids": ["a", 2]}, {"chunk_id": 3}], "edges": []}]

        assert GraphEvidenceExtractor().extract(subgraphs) == ["a", "2", "3"]
