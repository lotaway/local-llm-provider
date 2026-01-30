import os
import logging
from typing import List, Optional, Dict, Any
from neo4j import GraphDatabase
from schemas.graph import Entity, Relation

logger = logging.getLogger(__name__)

class Neo4jRepository:
    def __init__(self):
        uri = os.getenv("NEO4J_BOLT_URL", "bolt://localhost:7687")
        auth_env = os.getenv("NEO4J_AUTH", "neo4j/123123123")
        try:
            parts = auth_env.split("/")
            user = parts[0]
            password = parts[1]
        except (IndexError, AttributeError):
            user = "neo4j"
            password = "password"
            logger.warning(f"Invalid NEO4J_AUTH format: {auth_env}. Using defaults.")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def is_empty(self) -> bool:
        """Check if the database has any Entity nodes"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n:Entity) RETURN count(n) AS count")
                record = result.single()
                return record["count"] == 0 if record else True
        except Exception as e:
            logger.error(f"Error checking if Neo4j is empty: {e}")
            return True

    def merge_entity(self, entity: Entity):
        """Idempotent merge of an entity node"""
        with self.driver.session() as session:
            session.execute_write(self._merge_entity_tx, entity)

    @staticmethod
    def _merge_entity_tx(tx, entity: Entity):
        # We use apoc.util.merge if available, but here we use standard MERGE for core fields
        # and APOC for dynamic properties to adhere to the spec's flexibility.
        query = (
            "MERGE (e:Entity {stable_id: $stable_id}) "
            "SET e.type = $type, "
            "    e.canonical_name = $canonical_name, "
            "    e.created_at = $created_at "
            "WITH e "
            "CALL apoc.create.setProperties(e, keys($properties), [k in keys($properties) | $properties[k]]) YIELD node "
            "RETURN node"
        )
        tx.run(query, 
               stable_id=entity.stable_id, 
               type=entity.type, 
               canonical_name=entity.canonical_name, 
               created_at=entity.created_at.isoformat(), 
               properties=entity.properties)

    def merge_relation(self, relation: Relation):
        """Idempotent merge of a relationship between entities"""
        with self.driver.session() as session:
            session.execute_write(self._merge_relation_tx, relation)

    @staticmethod
    def _merge_relation_tx(tx, relation: Relation):
        # Ensure nodes exist first (though they should have been merged already)
        tx.run("MERGE (a:Entity {stable_id: $from_id})", from_id=relation.from_entity_id)
        tx.run("MERGE (b:Entity {stable_id: $to_id})", to_id=relation.to_entity_id)
        
        # Merge relationship with specific type
        # Note: Cypher doesn't support dynamic relationship types in MERGE easily without APOC
        query = (
            "MATCH (a:Entity {stable_id: $from_id}), (b:Entity {stable_id: $to_id}) "
            "CALL apoc.merge.relationship(a, $relation_type, {}, {confidence: $confidence, source_doc_id: $source_doc_id, created_at: $created_at}, b) "
            "YIELD rel "
            "CALL apoc.create.setRelProperties(rel, keys($properties), [k in keys($properties) | $properties[k]]) YIELD rel as r "
            "RETURN r"
        )
        tx.run(query, 
               from_id=relation.from_entity_id, 
               to_id=relation.to_entity_id,
               relation_type=relation.relation_type, 
               confidence=relation.confidence,
               source_doc_id=relation.source_doc_id, 
               created_at=relation.created_at.isoformat(),
               properties=relation.properties)

    def query_subgraph(self, entity_id: str, hops: int = 2) -> Dict[str, Any]:
        """Retrieve N-hop subgraph around an entity"""
        with self.driver.session() as session:
            return session.execute_read(self._query_subgraph_tx, entity_id, hops)

    @staticmethod
    def _query_subgraph_tx(tx, entity_id: str, hops: int) -> Dict[str, Any]:
        query = (
            "MATCH (start:Entity {stable_id: $entity_id}) "
            "CALL apoc.path.subgraphAll(start, {maxLevel: $hops}) "
            "YIELD nodes, relationships "
            "RETURN nodes, relationships"
        )
        result = tx.run(query, entity_id=entity_id, hops=hops)
        record = result.single()
        if not record:
            return {"nodes": [], "edges": []}
        
        nodes = []
        for node in record["nodes"]:
            nodes.append(dict(node))
            
        edges = []
        for rel in record["relationships"]:
            # rel.start_node and rel.end_node are accessible in the driver
            edges.append({
                "from": rel.start_node["stable_id"],
                "to": rel.end_node["stable_id"],
                "type": rel.type,
                "properties": dict(rel)
            })
            
        return {"nodes": nodes, "edges": edges}

    def find_entities_by_name(self, name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search entities by canonical name (fuzzy match)"""
        with self.driver.session() as session:
            query = (
                "MATCH (e:Entity) "
                "WHERE e.canonical_name CONTAINS $name OR e.stable_id CONTAINS $name "
                "RETURN e LIMIT $limit"
            )
            result = session.run(query, name=name, limit=limit)
            return [dict(record["e"]) for record in result]
