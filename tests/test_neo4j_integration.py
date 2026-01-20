import os
import asyncio
from repositories.neo4j_repository import Neo4jRepository
from schemas.graph import Entity, Relation
from datetime import datetime

async def test_neo4j():
    repo = Neo4jRepository()
    
    # Test merging entities
    e1 = Entity(stable_id="Test_A", type="Test", canonical_name="Node A", properties={"val": 1})
    e2 = Entity(stable_id="Test_B", type="Test", canonical_name="Node B", properties={"val": 2})
    
    print("Merging entities...")
    repo.merge_entity(e1)
    repo.merge_entity(e2)
    
    # Test merging relation
    r = Relation(
        from_entity_id="Test_A",
        to_entity_id="Test_B",
        relation_type="TESTS",
        confidence=0.8,
        properties={"how": "async"}
    )
    
    print("Merging relation...")
    repo.merge_relation(r)
    
    # Test subgraph query
    print("Querying subgraph...")
    subgraph = repo.query_subgraph("Test_A", hops=1)
    print(f"Nodes found: {len(subgraph['nodes'])}")
    print(f"Edges found: {len(subgraph['edges'])}")
    
    # Test search
    print("Testing search...")
    matches = repo.find_entities_by_name("Node A")
    print(f"Matches for 'Node A': {len(matches)}")
    
    repo.close()
    print("Neo4j test finished.")

if __name__ == "__main__":
    asyncio.run(test_neo4j())
