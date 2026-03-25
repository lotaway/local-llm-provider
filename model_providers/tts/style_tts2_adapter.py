import os
import sys
import logging
import torch
import numpy as np
import redis
from pathlib import Path
from typing import Dict, Any, Optional

from utils.device import get_optimal_device, setup_hardware_env
from constants import (
    PROJECT_ROOT, 
    REDIS_HOST, REDIS_PORT, REDIS_TTS_PREFIX, 
    TTS_MONGO_COLLECTION_SPEAKERS, TTS_MONGO_COLLECTION_PROFILES
)

logger = logging.getLogger(__name__)

# Add StyleTTS2 to sys.path for internal imports
STYLETTS2_PATH = os.path.join(PROJECT_ROOT, "third_party", "StyleTTS2")
if STYLETTS2_PATH not in sys.path:
    sys.path.append(STYLETTS2_PATH)

class StyleTTS2Adapter:
    """
    Adapter for StyleTTS2 to bridge the vendor code with the project.
    Strict isolation for hardware (NVIDIA/AMD/MPS) and Storage (Redis Prefix).
    """

    def __init__(self, model_checkpoint: str = None):
        setup_hardware_env()
        self.device = get_optimal_device()
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        self.prefix = REDIS_TTS_PREFIX
        self.model = None # To be initialized with third_party weights
        logger.info(f"StyleTTS2 initialized on {self.device} with storage prefix '{self.prefix}'")

    def get_speaker_embedding(self, speaker_id: str) -> Optional[np.ndarray]:
        """Strict Scheme B: Redis KeyPrefix check."""
        redis_key = f"{self.prefix}speaker:{speaker_id}"
        cached_data = self.redis_client.get(redis_key)
        
        if cached_data:
            return np.frombuffer(cached_data, dtype=np.float32)
        
        # Fallback to MongoDB (to be implemented with DB client)
        return None

    def save_speaker_embedding(self, speaker_id: str, embedding: np.ndarray):
        """Strict Scheme B: Redis KeyPrefix save."""
        redis_key = f"{self.prefix}speaker:{speaker_id}"
        self.redis_client.set(redis_key, embedding.tobytes())
        logger.info(f"Speaker embedding saved to Redis with prefix {self.prefix}")

    async def generate_speech(self, text: str, emotion_params: Dict[str, float], speaker_id: str):
        """
        Implementation of the VAD-based emotion generation.
        Integration point with StyleTTS2 code.
        """
        # Placeholder for actual generation using third_party.StyleTTS2
        logger.info(f"Generating speech: {text[:20]}... with emotion: {emotion_params}")
        pass

    async def clone_voice(self, audio_data: bytes, user_id: str, alias: str):
        """
        Implementation of voice cloning and storage isolation.
        """
        # Placeholder for cloning logic
        logger.info(f"Cloning voice for user {user_id} as {alias}")
        pass
