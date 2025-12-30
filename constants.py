import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", "models")
OFFLOAD_DIR = os.getenv("OFFLOAD_DIR", "offload")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 16))
