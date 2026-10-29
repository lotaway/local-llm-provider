import os
import sys
import logging
import time
import numpy as np
import redis
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from utils.device import get_optimal_device, setup_hardware_env
from services.emotion_parser import EmotionParser
from constants import (
    PROJECT_ROOT,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_TTS_PREFIX,
)

logger = logging.getLogger(__name__)

STYLETTS2_PATH = os.path.join(PROJECT_ROOT, "third_party", "StyleTTS2")
if STYLETTS2_PATH not in sys.path:
    sys.path.append(STYLETTS2_PATH)


class StyleTTS2Adapter:
    """Adapter for StyleTTS2 to bridge the vendor code with the project."""

    def __init__(
        self,
        model_checkpoint: str = None,
        redis_client: redis.Redis = None,
        emotion_parser: EmotionParser = None,
        vendor_streamer: Callable[[str, Dict[str, float], str], AsyncGenerator[str, None]] = None,
        preference_store=None,
    ):
        setup_hardware_env()
        self.device = get_optimal_device()
        self.redis_client = redis_client or redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        self.prefix = REDIS_TTS_PREFIX
        self.model = None
        self.emotion_parser = emotion_parser or EmotionParser()
        self.vendor_streamer = vendor_streamer
        self.preference_store = preference_store
        logger.info(f"StyleTTS2 initialized on {self.device} with storage prefix '{self.prefix}'")

    def get_speaker_embedding(self, speaker_id: str) -> Optional[np.ndarray]:
        redis_key = f"{self.prefix}speaker:{speaker_id}"
        cached_data = self.redis_client.get(redis_key)

        if cached_data:
            return np.frombuffer(cached_data, dtype=np.float32)

        return None

    def save_speaker_embedding(self, speaker_id: str, embedding: np.ndarray):
        redis_key = f"{self.prefix}speaker:{speaker_id}"
        self.redis_client.set(redis_key, embedding.tobytes())
        logger.info(f"Speaker embedding saved to Redis with prefix {self.prefix}")

    async def generate_speech_stream(
        self,
        text: str,
        emotion_params: Optional[Dict[str, float]] = None,
        speaker_id: str = "",
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        user_id: str = "",
    ) -> AsyncGenerator[str, None]:
        started_at = time.monotonic()
        resolved_emotion = self._resolve_emotion(text, emotion_params, user_id)
        stream = self.vendor_streamer or self._default_streamer
        first_chunk = True
        async for chunk in stream(text, resolved_emotion, speaker_id):
            if not chunk:
                continue
            if first_chunk and callback is not None:
                callback({"type": "first_chunk", "elapsed_ms": round((time.monotonic() - started_at) * 1000, 3)})
                first_chunk = False
            yield chunk

    async def _default_streamer(
        self, text: str, emotion_params: Dict[str, float], speaker_id: str
    ) -> AsyncGenerator[str, None]:
        if self.model is None:
            return
        yield ""

    def _resolve_emotion(self, text: str, emotion_params: Optional[Dict[str, float]], user_id: str = "") -> Dict[str, float]:
        resolved = dict(self.emotion_parser.parse_params(text))
        resolved.update(emotion_params or {})
        if self.preference_store is not None:
            stored = self.preference_store.get(user_id)
            preferences = (stored or {}).get("preferences") or {}
            resolved.update({key: value for key, value in preferences.items() if key in resolved})
        return resolved

    async def generate_speech(
        self,
        text: str,
        emotion_params: Optional[Dict[str, float]] = None,
        speaker_id: str = "",
        callback: Optional[Callable[[str], Any]] = None,
        user_id: str = "",
    ) -> Dict[str, Any]:
        resolved_emotion = self._resolve_emotion(text, emotion_params, user_id)
        generator = self.generate_speech_stream(text, resolved_emotion, speaker_id, callback, user_id)
        full_text = ""
        async for chunk in generator:
            full_text += chunk
        return {"audio_text": full_text, "emotion": resolved_emotion}
