import logging
from typing import List, Dict, Any, Optional
from repositories.neo4j_repository import Neo4jRepository
from model_providers import LocalLLModel

logger = logging.getLogger(__name__)

class GraphRetriever:
    """
    Retriever that leverages Neo4j to find structural knowledge and relationships.
    """
    ENTITY_EXTRACTION_PROMPT = """从用户的问题中提取核心实体（Entity）名称。
只返回实体名称列表，用逗号分隔。如果没有找到明显的实体，返回 "None"。

问题：{query}
"""

    def __init__(self, neo4j_repo: Neo4jRepository, llm: LocalLLModel):
        self.neo4j_repo = neo4j_repo
        self.llm = llm

    async def retrieve_context(self, query: str, hops: int = 2, limit_nodes: int = 15) -> str:
        """
        Identify entities in query, expand their subgraphs, and format as context.
        """
        try:
            # Step 1: Extract entities from query using LLM
            # Note: Using small temperature for deterministic results
            entities_response = await self.llm.chat_at_once(
                self.ENTITY_EXTRACTION_PROMPT.format(query=query), 
                temperature=0.0
            )
            
            entities_response = self.llm.extract_after_think(entities_response)
            
            if "None" in entities_response or not entities_response.strip():
                logger.info("No entities extracted from query for graph retrieval.")
                return ""

            entity_names = [e.strip() for e in entities_response.split(",") if e.strip()]
            logger.info(f"Extracted entities for graph search: {entity_names}")
            
            # Step 2: Find actual entity nodes in Neo4j
            start_nodes = []
            for name in entity_names:
                matches = self.neo4j_repo.find_entities_by_name(name, limit=3)
                start_nodes.extend(matches)
            
            if not start_nodes:
                logger.info("No matching nodes found in Neo4j.")
                return ""

            # Step 3: Expand subgraphs and aggregate
            seen_nodes = set()
            all_subgraphs = []
            
            # Limit starting points to avoid context explosion
            for node in start_nodes[:5]:
                stable_id = node.get("stable_id")
                if not stable_id or stable_id in seen_nodes:
                    continue
                    
                subgraph = self.neo4j_repo.query_subgraph(stable_id, hops=hops)
                if subgraph["nodes"]:
                    all_subgraphs.append(subgraph)
                    for n in subgraph["nodes"]:
                        seen_nodes.add(n.get("stable_id"))

            if not all_subgraphs:
                return ""

            return self._format_aggregated_subgraphs(all_subgraphs, limit_nodes)

        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}", exc_info=True)
            return ""

    def _format_aggregated_subgraphs(self, subgraphs: List[Dict[str, Any]], limit_nodes: int) -> str:
        unique_nodes = {}
        unique_edges = set()
        
        for sg in subgraphs:
            for n in sg["nodes"]:
                unique_nodes[n["stable_id"]] = n
            for e in sg["edges"]:
                # (from, to, type)
                edge_key = (e["from"], e["to"], e["type"])
                unique_edges.add(edge_key)
        
        if not unique_nodes:
            return ""

        # Build context string
        context_parts = ["\n### 知识图谱信息（结构化上下文）："]
        
        # Nodes
        context_parts.append("相关实体：")
        node_list = list(unique_nodes.values())[:limit_nodes]
        for n in node_list:
            name = n.get("canonical_name", n.get("stable_id", "Unknown"))
            etype = n.get("type", "Entity")
            # Filter out internal/boilerplate properties
            props = {k: v for k, v in n.items() if k not in ["stable_id", "canonical_name", "type", "created_at"]}
            prop_str = f" {props}" if props else ""
            context_parts.append(f"- {name} ({etype}){prop_str}")
        
        # Edges
        context_parts.append("\n实体关系：")
        # Only show edges where both nodes are in the limited nodes list if there's balance, 
        # but here we show top relations.
        edge_list = list(unique_edges)[:20] 
        for u_edge in edge_list:
            from_name = unique_nodes.get(u_edge[0], {}).get("canonical_name", u_edge[0])
            to_name = unique_nodes.get(u_edge[1], {}).get("canonical_name", u_edge[1])
            context_parts.append(f"- {from_name} --[{u_edge[2]}]--> {to_name}")
            
        return "\n".join(context_parts)
