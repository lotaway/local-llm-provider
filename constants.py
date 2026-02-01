import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", "models")
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "offload")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 16))
DEFAULT_CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", 8192))
QUANTIZATION = os.getenv("QUANTIZATION", "4bit")
TORCH_DTYPE = os.getenv("TORCH_DTYPE")

# RAG Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "19530")
DB_COLLECTION = os.getenv("DB_COLLECTION", "rag_docs")

# Elasticsearch Configuration
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT1 = int(os.getenv("ES_PORT1", 9200))
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "rag_docs")

# Neo4j Configuration
NEO4J_BOLT_URL = os.getenv("NEO4J_BOLT_URL", "bolt://localhost:7687")
NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/123123123")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_USER = os.getenv("MONGO_USER", "")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "rag_system")

# Data Path Configuration
DATA_PATH = os.getenv("DATA_PATH", os.getenv("DOCS_PATH", "./docs"))
