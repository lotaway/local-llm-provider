import os
from dotenv import load_dotenv

# Load environment variables from .env file first
load_dotenv()

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
DATA_PATH = os.getenv("DOCS_PATH", "./docs")

# MOLT Learning Configuration
LEARNING = os.getenv("LEARNING", "false").lower() == "true"

LLP_SKILLS_DIRS = os.getenv("LLP_SKILLS_DIRS", "")
LLP_ENABLE_CLAUDE_GLOBAL = os.getenv("LLP_ENABLE_CLAUDE_GLOBAL", "0")
LLP_CLAUDE_SKILLS_DIR = os.getenv("LLP_CLAUDE_SKILLS_DIR", "")
LLP_ENABLE_OPENCLAW = os.getenv("LLP_ENABLE_OPENCLAW", "0")
LLP_OPENCLAW_ROOT = os.getenv("LLP_OPENCLAW_ROOT", "")

POE_API_KEY = os.getenv("POE_API_KEY", "")
POE_DEFAULT_MODEL = os.getenv("POE_DEFAULT_MODEL", "Claude-Sonnet-4.5")
CUSTOM_LLM_API_KEY = os.getenv("CUSTOM_LLM_API_KEY", "")
CUSTOM_LLM_BASE_URL = os.getenv("CUSTOM_LLM_BASE_URL", "")
CUSTOM_LLM_MODEL = os.getenv("CUSTOM_LLM_MODEL", "")
CUSTOM_LLM_PROTOCOL = os.getenv("CUSTOM_LLM_PROTOCOL", "openai")

# Protocol to Base URL mapping for common providers
PROTOCOL_BASE_URL_MAP = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "copilot": "https://api.githubcopilot.com",
}

# Auto-fallback for common protocols if BASE_URL is missing
if not CUSTOM_LLM_BASE_URL:
    CUSTOM_LLM_BASE_URL = PROTOCOL_BASE_URL_MAP.get(CUSTOM_LLM_PROTOCOL.lower(), "")

VIBEVOICE_DIR = os.getenv("VIBEVOICE_DIR", "")
VIBEVOICE_MODEL = os.getenv("VIBEVOICE_MODEL", "microsoft/VibeVoice-ASR")
VIBEVOICE_SCRIPT = os.getenv(
    "VIBEVOICE_SCRIPT",
    os.path.join(VIBEVOICE_DIR, "demo", "vibevoice_asr_inference_from_file.py")
    if VIBEVOICE_DIR
    else "",
)

EMBEDDING_MODEL="Alibaba-NLP/gte-Qwen2-1.5B-instruct"

# TTS Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTS_PREFIX = "tts:" # Strict Scheme B prefix

# MongoDB TTS Collections (Isolated from RAG/Brain)
TTS_MONGO_COLLECTION_SPEAKERS = "tts_speakers"
TTS_MONGO_COLLECTION_PROFILES = "tts_user_profiles"