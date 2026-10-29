import logging
from typing import List, Dict, Any, Optional
from repositories.neo4j_repository import Neo4jRepository
from model_providers import LocalLLModel
from retrievers.graph_evidence import GraphEvidenceExtractor

logger = logging.getLogger(__name__)

class GraphRetriever:
    ENTITY_EXTRACTION_PROMPT = """Extract core entity names from the user's question.
Return only a comma-separated list of entity names. If no entities are found, return "None".

Question: {query}
"""

    def __init__(self, neo4j_repo: Neo4jRepository, llm: LocalLLModel, feedback_service=None):
        self.neo4j_repo = neo4j_repo
        self.llm = llm
        self.feedback_service = feedback_service
        self._evidence_extractor = GraphEvidenceExtractor()

    async def retrieve_context(self, query: str, hops: int = 2, limit_nodes: int = 15) -> str:
        try:
            entity_names = await self._extract_entities(query)
            if not entity_names: return ""
            
            start_nodes = self._find_start_nodes(entity_names)
            if not start_nodes: return ""

            all_subgraphs = self._expand_subgraphs(start_nodes, hops)
            if not all_subgraphs: return ""

            result = self._format_aggregated_subgraphs(all_subgraphs, limit_nodes)
            self._report_recall(query, all_subgraphs)
            return result
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}", exc_info=True)
            return ""

    def _report_recall(self, query: str, subgraphs: List[Dict[str, Any]]) -> List[str]:
        if self.feedback_service is None:
            return []
        chunk_ids = self._evidence_extractor.extract(subgraphs)
        if not chunk_ids:
            logger.warning(
                "graph feedback skipped: no chunk_id evidence",
                extra={
                    "query": query,
                    "subgraph_count": len(subgraphs),
                    "node_count": sum(len(item.get("nodes", [])) for item in subgraphs),
                    "edge_count": sum(len(item.get("edges", [])) for item in subgraphs),
                },
            )
            return []
        self.feedback_service.on_recall_batch(chunk_ids)
        return chunk_ids

    async def _extract_entities(self, query: str) -> List[str]:
        response = await self.llm.chat_at_once(self.ENTITY_EXTRACTION_PROMPT.format(query=query), temperature=0.0)
        response = self.llm.extract_after_think(response)
        if "None" in response or not response.strip(): return []
        return [e.strip() for e in response.split(",") if e.strip()]

    def _find_start_nodes(self, entity_names: List[str]) -> list:
        nodes = []
        for name in entity_names:
            nodes.extend(self.neo4j_repo.find_entities_by_name(name, limit=3))
        return nodes

    def _expand_subgraphs(self, start_nodes: list, hops: int) -> list:
        seen, subgraphs = set(), []
        for node in start_nodes[:5]:
            stable_id = node.get("stable_id")
            if not stable_id or stable_id in seen: continue
            
            subgraph = self.neo4j_repo.query_subgraph(stable_id, hops=hops)
            if subgraph["nodes"]:
                subgraphs.append(subgraph)
                seen.update(n.get("stable_id") for n in subgraph["nodes"])
        return subgraphs

    def _format_aggregated_subgraphs(self, subgraphs: List[Dict[str, Any]], limit_nodes: int) -> str:
        nodes, edges = {}, set()
        for sg in subgraphs:
            for n in sg["nodes"]: nodes[n["stable_id"]] = n
            for e in sg["edges"]: edges.add((e["from"], e["to"], e["type"]))
        
        if not nodes: return ""

        parts = ["\n### Structured Graph Context:"]
        parts.append("Entities:")
        for n in list(nodes.values())[:limit_nodes]:
            name = n.get("canonical_name", n.get("stable_id", "Unknown"))
            props = {k: v for k, v in n.items() if k not in ["stable_id", "canonical_name", "type", "created_at"]}
            parts.append(f"- {name} ({n.get('type', 'Entity')}) {' '+str(props) if props else ''}")
        
        parts.append("\nRelationships:")
        for frm, to, typ in list(edges)[:20]:
            frm_name = nodes.get(frm, {}).get("canonical_name", frm)
            to_name = nodes.get(to, {}).get("canonical_name", to)
            parts.append(f"- {frm_name} --[{typ}]--> {to_name}")
            
        return "\n".join(parts)
