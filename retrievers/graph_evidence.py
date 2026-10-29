from typing import Any, Dict, Iterable, List

CHUNK_ID_FIELDS = frozenset(
    {
        "chunk_id",
        "chunk_ids",
        "source_chunk_id",
        "source_chunk_ids",
        "evidence_chunk_id",
        "evidence_chunk_ids",
        "evidence_chunks",
    }
)
NESTED_FIELDS = frozenset({"evidence", "properties", "metadata", "payload", "attributes"})


class GraphEvidenceExtractor:
    def extract(self, subgraphs: Iterable[Dict[str, Any]]) -> List[str]:
        chunk_ids = []
        for subgraph in subgraphs:
            for node in subgraph.get("nodes", []):
                self._collect(node, chunk_ids)
            for edge in subgraph.get("edges", []):
                self._collect(edge, chunk_ids)
        return list(dict.fromkeys(chunk_ids))

    def _collect(self, value: Any, chunk_ids: List[str]) -> None:
        if isinstance(value, list):
            for item in value:
                self._collect(item, chunk_ids)
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key in CHUNK_ID_FIELDS:
                chunk_ids.extend(self._identifiers(item))
            elif key in NESTED_FIELDS or isinstance(item, (dict, list)):
                self._collect(item, chunk_ids)

    def _identifiers(self, value: Any) -> List[str]:
        values = value if isinstance(value, list) else [value]
        return [str(item) for item in values if isinstance(item, (str, int)) and str(item)]
