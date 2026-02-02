import logging
from typing import List, Optional, Dict, Any
from constants import NEO4J_BOLT_URL, NEO4J_AUTH
from schemas.graph import Entity, Relation, LTMNode
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class Neo4jRepository:
    def __init__(self):
        uri = NEO4J_BOLT_URL
        auth_env = NEO4J_AUTH
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
        tx.run(
            query,
            stable_id=entity.stable_id,
            type=entity.type,
            canonical_name=entity.canonical_name,
            created_at=entity.created_at.isoformat(),
            properties=entity.properties,
        )

    def merge_relation(self, relation: Relation):
        """Idempotent merge of a relationship between entities"""
        with self.driver.session() as session:
            session.execute_write(self._merge_relation_tx, relation)

    @staticmethod
    def _merge_relation_tx(tx, relation: Relation):
        # Ensure nodes exist first (though they should have been merged already)
        tx.run(
            "MERGE (a:Entity {stable_id: $from_id})", from_id=relation.from_entity_id
        )
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
        tx.run(
            query,
            from_id=relation.from_entity_id,
            to_id=relation.to_entity_id,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            source_doc_id=relation.source_doc_id,
            created_at=relation.created_at.isoformat(),
            properties=relation.properties,
        )

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
            edges.append(
                {
                    "from": rel.start_node["stable_id"],
                    "to": rel.end_node["stable_id"],
                    "type": rel.type,
                    "properties": dict(rel),
                }
            )

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

    def save_ltm(self, ltm: LTMNode) -> str:
        """Save or update an LTM node, returns the topic_version key"""
        topic_version = f"{ltm.topic}_v{ltm.version}"
        with self.driver.session() as session:
            session.execute_write(self._save_ltm_tx, ltm, topic_version)
        return topic_version

    @staticmethod
    def _save_ltm_tx(tx, ltm: LTMNode, topic_version: str):
        query = (
            "MERGE (l:LTMNode {topic_version: $topic_version}) "
            "SET l.topic = $topic, "
            "    l.version = $version, "
            "    l.conclusion = $conclusion, "
            "    l.conditions = $conditions, "
            "    l.confidence = $confidence, "
            "    l.sources = $sources, "
            "    l.source_chunk_ids = $source_chunk_ids, "
            "    l.properties = $properties, "
            "    l.created_at = $created_at, "
            "    l.updated_at = $updated_at "
            "RETURN l.topic_version"
        )
        tx.run(
            query,
            topic_version=topic_version,
            topic=ltm.topic,
            version=ltm.version,
            conclusion=ltm.conclusion,
            conditions=ltm.conditions,
            confidence=ltm.confidence,
            sources=ltm.sources,
            source_chunk_ids=ltm.source_chunk_ids,
            properties=ltm.properties,
            created_at=ltm.created_at.isoformat(),
            updated_at=ltm.updated_at.isoformat(),
        )

    def get_ltm_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Get all versions of LTM nodes for a topic"""
        with self.driver.session() as session:
            query = (
                "MATCH (l:LTMNode) "
                "WHERE l.topic = $topic "
                "RETURN l ORDER BY l.version DESC"
            )
            result = session.run(query, topic=topic)
            return [dict(record["l"]) for record in result]

    def get_ltm_by_topic_version(
        self, topic: str, version: int
    ) -> Optional[Dict[str, Any]]:
        """Get specific version of LTM node"""
        with self.driver.session() as session:
            query = (
                "MATCH (l:LTMNode) "
                "WHERE l.topic = $topic AND l.version = $version "
                "RETURN l"
            )
            result = session.run(query, topic=topic, version=version)
            record = result.single()
            return dict(record["l"]) if record else None

    def get_latest_ltm(self, topic: str) -> Optional[Dict[str, Any]]:
        """Get the latest version of LTM for a topic"""
        with self.driver.session() as session:
            query = (
                "MATCH (l:LTMNode) "
                "WHERE l.topic = $topic "
                "RETURN l ORDER BY l.version DESC LIMIT 1"
            )
            result = session.run(query, topic=topic)
            record = result.single()
            return dict(record["l"]) if record else None

    def link_episodic_to_ltm(self, chunk_id: str, ltm_topic_version: str):
        """Create relationship from episodic chunk to LTM node"""
        with self.driver.session() as session:
            query = (
                "MATCH (c:Chunk {chunk_id: $chunk_id}) "
                "MATCH (l:LTMNode {topic_version: $ltm_topic_version}) "
                "MERGE (c)-[:EVOLVED_TO]->(l)"
            )
            session.run(query, chunk_id=chunk_id, ltm_topic_version=ltm_topic_version)

    def get_episodic_for_ltm(self, ltm_topic_version: str) -> List[Dict[str, Any]]:
        """Get all episodic chunks that evolved into this LTM"""
        with self.driver.session() as session:
            query = (
                "MATCH (c:Chunk)-[:EVOLVED_TO]->(l:LTMNode {topic_version: $topic_version}) "
                "RETURN c"
            )
            result = session.run(query, topic_version=ltm_topic_version)
            return [dict(record["c"]) for record in result]

    def create_ltm_version_relation(
        self, old_topic_version: str, new_topic_version: str
    ):
        """Create supersedes relationship between versions"""
        with self.driver.session() as session:
            query = (
                "MATCH (old:LTMNode {topic_version: $old_version}) "
                "MATCH (new:LTMNode {topic_version: $new_version}) "
                "MERGE (old)-[:SUPERSEDED_BY]->(new)"
            )
            session.run(
                query, old_version=old_topic_version, new_version=new_topic_version
            )

    def get_ltm_versions_chain(self, topic: str) -> List[Dict[str, Any]]:
        """Get version chain for a topic with supersedes relationships"""
        with self.driver.session() as session:
            query = (
                "MATCH (l:LTMNode {topic: $topic}) "
                "OPTIONAL MATCH (l)-[:SUPERSEDED_BY]->(next:LTMNode) "
                "RETURN l, next "
                "ORDER BY l.version DESC"
            )
            result = session.run(query, topic=topic)
            versions = []
            for record in result:
                node = dict(record["l"])
                node["superseded_by"] = (
                    dict(record["next"])["topic_version"] if record["next"] else None
                )
                versions.append(node)
            return versions

    def search_ltm_by_conclusion(
        self, query_text: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search LTM nodes by conclusion content (simple substring match)"""
        with self.driver.session() as session:
            query = (
                "MATCH (l:LTMNode) "
                "WHERE l.conclusion CONTAINS $query "
                "RETURN l ORDER BY l.confidence DESC LIMIT $limit"
            )
            result = session.run(query, query=query_text, limit=limit)
            return [dict(record["l"]) for record in result]

    def delete_ltm(self, topic: str, version: int) -> bool:
        """Delete a specific LTM version"""
        with self.driver.session() as session:
            topic_version = f"{topic}_v{version}"
            result = session.run(
                "MATCH (l:LTMNode {topic_version: $topic_version}) "
                "DETACH DELETE l RETURN count(*) as deleted",
                topic_version=topic_version,
            )
            record = result.single()
            return record["deleted"] > 0 if record else False
