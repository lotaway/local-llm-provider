#!/usr/bin/env python3
"""
RAG Data Cleanup Script
Cleanup all RAG data in Milvus, Elasticsearch, and Neo4j
"""

import asyncio
import os
import sys
from datetime import datetime

# Add project root to path first
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from constants import (
    DB_HOST,
    DB_PORT,
    DB_COLLECTION,
    ES_HOST,
    ES_PORT1,
    ES_INDEX_NAME,
    NEO4J_BOLT_URL,
    NEO4J_AUTH,
    MONGO_URI,
    MONGO_DB_NAME,
)

from pymilvus import connections, utility, Collection
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase
from pymongo import MongoClient


def get_milvus_stats():
    try:
        host = DB_HOST
        port = DB_PORT
        collection_name = DB_COLLECTION

        connections.connect(host=host, port=port)

        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            stats = {
                "entities": collection.num_entities,
                "collection_name": collection_name,
                "connected": True,
            }
            collection.release()
            return stats
        else:
            return {
                "entities": 0,
                "collection_name": collection_name,
                "connected": True,
                "message": "Collection does not exist",
            }
    except Exception as e:
        return {
            "entities": -1,
            "collection_name": DB_COLLECTION,
            "connected": False,
            "error": str(e),
        }


def get_es_stats():
    try:
        host = ES_HOST
        port = ES_PORT1
        index_name = ES_INDEX_NAME

        es_client = Elasticsearch(f"http://{host}:{port}")

        if es_client.indices.exists(index=index_name):
            stats = es_client.count(index=index_name)
            return {
                "documents": stats["count"],
                "index_name": index_name,
                "connected": True,
            }
        else:
            return {
                "documents": 0,
                "index_name": index_name,
                "connected": True,
                "message": "Index does not exist",
            }
    except Exception as e:
        return {
            "documents": -1,
            "index_name": ES_INDEX_NAME,
            "connected": False,
            "error": str(e),
        }


def get_neo4j_stats():
    try:
        uri = NEO4J_BOLT_URL
        auth_env = NEO4J_AUTH

        parts = auth_env.split("/")
        user = parts[0]
        password = parts[1]

        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            entity_result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_record = entity_result.single()
            entity_count = entity_record["count"] if entity_record else 0

            relation_result = session.run(
                "MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count"
            )
            relation_record = relation_result.single()
            relation_count = relation_record["count"] if relation_record else 0

            return {
                "entities": entity_count,
                "relations": relation_count,
                "connected": True,
            }
    except Exception as e:
        return {"entities": -1, "relations": -1, "connected": False, "error": str(e)}


def get_mongodb_stats():
    """Get MongoDB statistics for documents and chunks"""
    try:
        mongo_uri = MONGO_URI
        db_name = MONGO_DB_NAME

        client = MongoClient(mongo_uri)
        db = client[db_name]

        documents_count = db["documents"].count_documents({})
        chunks_count = db["chunks"].count_documents({})

        client.close()

        return {
            "documents": documents_count,
            "chunks": chunks_count,
            "db_name": db_name,
            "connected": True,
        }
    except Exception as e:
        return {
            "documents": -1,
            "chunks": -1,
            "db_name": MONGO_DB_NAME,
            "connected": False,
            "error": str(e),
        }


def cleanup_mongodb():
    """Cleanup all documents and chunks in MongoDB"""
    try:
        mongo_uri = MONGO_URI
        db_name = MONGO_DB_NAME

        client = MongoClient(mongo_uri)
        db = client[db_name]

        documents_count = db["documents"].count_documents({})
        chunks_count = db["chunks"].count_documents({})

        db["documents"].delete_many({})
        db["chunks"].delete_many({})

        client.close()

        return {
            "success": True,
            "documents_deleted": documents_count,
            "chunks_deleted": chunks_count,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def cleanup_milvus():
    try:
        host = DB_HOST
        port = DB_PORT
        collection_name = DB_COLLECTION

        connections.connect(host=host, port=port)

        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            entities_before = collection.num_entities
            collection.release()

            asyncio.run(utility.drop_collection(collection_name))

            return {"success": True, "entities_deleted": entities_before}
        else:
            return {
                "success": True,
                "entities_deleted": 0,
                "message": "Collection does not exist",
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def cleanup_es():
    try:
        host = ES_HOST
        port = ES_PORT1
        index_name = ES_INDEX_NAME

        es_client = Elasticsearch(f"http://{host}:{port}")

        if es_client.indices.exists(index=index_name):
            stats = es_client.count(index=index_name)
            docs_before = stats["count"]

            es_client.indices.delete(index=index_name)

            settings = {
                "analysis": {"analyzer": {"default": {"type": "standard"}}},
                "similarity": {"default": {"type": "BM25"}},
            }
            mappings = {
                "properties": {
                    "content": {"type": "text", "similarity": "default"},
                    "metadata": {
                        "type": "object",
                        "enabled": True,
                    },
                }
            }
            es_client.indices.create(
                index=index_name, settings=settings, mappings=mappings
            )

            return {"success": True, "documents_deleted": docs_before}
        else:
            return {
                "success": True,
                "documents_deleted": 0,
                "message": "Index does not exist",
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def cleanup_neo4j():
    try:
        uri = NEO4J_BOLT_URL
        auth_env = NEO4J_AUTH

        parts = auth_env.split("/")
        user = parts[0]
        password = parts[1]

        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            entity_result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_record = entity_result.single()
            entities_before = entity_record["count"] if entity_record else 0

            relation_result = session.run(
                "MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count"
            )
            relation_record = relation_result.single()
            relations_before = relation_record["count"] if relation_record else 0

            session.run("MATCH ()-[r:RELATED_TO]->() DELETE r")
            session.run("MATCH (e:Entity) DELETE e")

            return {
                "success": True,
                "entities_deleted": entities_before,
                "relations_deleted": relations_before,
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def print_stats(milvus_stats, es_stats, neo4j_stats, mongo_stats):
    print("\n" + "=" * 60)
    print("RAG Data Statistics")
    print("=" * 60)

    print("\nMilvus (Vector Database)")
    if milvus_stats.get("connected"):
        if milvus_stats.get("entities", -1) >= 0:
            print(f"   Collection: {milvus_stats['collection_name']}")
            print(f"   Documents: {milvus_stats['entities']:,}")
        else:
            print(
                f"   Warning: {milvus_stats.get('message', milvus_stats.get('error', 'Unknown error'))}"
            )
    else:
        print(f"   Connection failed: {milvus_stats.get('error', 'Unknown error')}")

    print("\nElasticsearch (BM25 Retrieval)")
    if es_stats.get("connected"):
        if es_stats.get("documents", -1) >= 0:
            print(f"   Index: {es_stats['index_name']}")
            print(f"   Documents: {es_stats['documents']:,}")
        else:
            print(
                f"   Warning: {es_stats.get('message', es_stats.get('error', 'Unknown error'))}"
            )
    else:
        print(f"   Connection failed: {es_stats.get('error', 'Unknown error')}")

    print("\nNeo4j (Graph Database)")
    if neo4j_stats.get("connected"):
        if neo4j_stats.get("entities", -1) >= 0:
            print(f"   Entities: {neo4j_stats['entities']:,}")
            print(f"   Relations: {neo4j_stats['relations']:,}")
        else:
            print(
                f"   Warning: {neo4j_stats.get('message', neo4j_stats.get('error', 'Unknown error'))}"
            )
    else:
        print(f"   Connection failed: {neo4j_stats.get('error', 'Unknown error')}")

    print("\nMongoDB (Source of Truth)")
    if mongo_stats.get("connected"):
        if mongo_stats.get("documents", -1) >= 0:
            print(f"   Database: {mongo_stats['db_name']}")
            print(f"   Documents: {mongo_stats['documents']:,}")
            print(f"   Chunks: {mongo_stats['chunks']:,}")
        else:
            print(
                f"   Warning: {mongo_stats.get('message', mongo_stats.get('error', 'Unknown error'))}"
            )
    else:
        print(f"   Connection failed: {mongo_stats.get('error', 'Unknown error')}")

    total_docs = 0
    if milvus_stats.get("entities", -1) > 0:
        total_docs += milvus_stats["entities"]
    if es_stats.get("documents", -1) > 0:
        total_docs += es_stats["documents"]
    total_entities = (
        neo4j_stats.get("entities", 0) if neo4j_stats.get("entities", -1) > 0 else 0
    )
    total_relations = (
        neo4j_stats.get("relations", 0) if neo4j_stats.get("relations", -1) > 0 else 0
    )

    print("\n" + "-" * 60)
    print(
        f"Total: Vector Docs {total_docs:,} | Entities {total_entities:,} | Relations {total_relations:,}"
    )
    print("=" * 60)


def confirm_cleanup():
    print("\nWARNING: This operation will permanently delete all RAG data!")
    print("   - All vectors in Milvus")
    print("   - All indexed documents in Elasticsearch")
    print("   - All entities and relations in Neo4j")
    print("   - All documents and chunks in MongoDB")
    print("\nThis operation is irreversible!")

    print("\n" + "-" * 60)
    response = (
        input("Confirm to clear all RAG data? (yes/no, default no): ").strip().lower()
    )
    print("-" * 60)

    if response != "yes":
        print("\nOperation cancelled. No data was deleted.")
        return False

    return True


def main():
    print("=" * 60)
    print("RAG Data Cleanup Tool")
    print("=" * 60)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nCollecting statistics from all databases...")

    milvus_stats = get_milvus_stats()
    es_stats = get_es_stats()
    neo4j_stats = get_neo4j_stats()
    mongo_stats = get_mongodb_stats()

    print_stats(milvus_stats, es_stats, neo4j_stats, mongo_stats)

    has_data = False
    if milvus_stats.get("entities", 0) > 0:
        has_data = True
    if es_stats.get("documents", 0) > 0:
        has_data = True
    if neo4j_stats.get("entities", 0) > 0 or neo4j_stats.get("relations", 0) > 0:
        has_data = True
    if mongo_stats.get("documents", 0) > 0:
        has_data = True

    if not has_data:
        print("\nAll databases are empty. No cleanup needed.")
        return

    if not confirm_cleanup():
        return

    print("\nStarting data cleanup...")

    print("\nCleaning MongoDB...")
    mongo_result = cleanup_mongodb()
    if mongo_result["success"]:
        print(
            f"   MongoDB cleanup complete. Deleted {mongo_result.get('documents_deleted', 0)} documents and {mongo_result.get('chunks_deleted', 0)} chunks"
        )
    else:
        print(
            f"   MongoDB cleanup failed: {mongo_result.get('error', 'Unknown error')}"
        )

    print("\nCleaning Milvus...")
    milvus_result = cleanup_milvus()
    if milvus_result["success"]:
        print(
            f"   Milvus cleanup complete. Deleted {milvus_result.get('entities_deleted', 0)} vectors"
        )
    else:
        print(
            f"   Milvus cleanup failed: {milvus_result.get('error', 'Unknown error')}"
        )

    print("\nCleaning Elasticsearch...")
    es_result = cleanup_es()
    if es_result["success"]:
        print(
            f"   Elasticsearch cleanup complete. Deleted {es_result.get('documents_deleted', 0)} documents"
        )
    else:
        print(
            f"   Elasticsearch cleanup failed: {es_result.get('error', 'Unknown error')}"
        )

    print("\nCleaning Neo4j...")
    neo4j_result = cleanup_neo4j()
    if neo4j_result["success"]:
        print(
            f"   Neo4j cleanup complete. Deleted {neo4j_result.get('entities_deleted', 0)} entities and {neo4j_result.get('relations_deleted', 0)} relations"
        )
    else:
        print(f"   Neo4j cleanup failed: {neo4j_result.get('error', 'Unknown error')}")

    print("\nVerifying cleanup results...")

    milvus_final = get_milvus_stats()
    es_final = get_es_stats()
    neo4j_final = get_neo4j_stats()
    mongo_final = get_mongodb_stats()

    print("\nPost-cleanup statistics:")
    print(f"   Milvus: {milvus_final.get('entities', 'N/A')} vectors")
    print(f"   Elasticsearch: {es_final.get('documents', 'N/A')} documents")
    print(
        f"   Neo4j: {neo4j_final.get('entities', 'N/A')} entities, {neo4j_final.get('relations', 'N/A')} relations"
    )
    print(
        f"   MongoDB: {mongo_final.get('documents', 'N/A')} documents, {mongo_final.get('chunks', 'N/A')} chunks"
    )

    print("\n" + "=" * 60)
    print("Data cleanup completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
