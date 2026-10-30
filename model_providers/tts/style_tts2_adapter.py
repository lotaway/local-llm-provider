import asyncio
import base64
import logging
import os
import sys
import time
from typing import Any, AsyncGenerator, Callable, Dict, Optional

import numpy as np
import redis

from constants import REDIS_HOST, REDIS_PORT, REDIS_TTS_PREFIX
from interfaces.tts_types import ISpeakerRepository
from infrastructure.tts.speaker_repository import MongoSpeakerRepository
from model_providers.tts.audio_processing import process_samples
from model_providers.tts.style_tts2_engine import StyleTTS2Engine, StyleTTS2Unavailable
from services.emotion_parser import EmotionParser
from services.tts_post_processing import TTSPostProcessor
from utils.device import get_optimal_device, setup_hardware_env
from utils.tts_runtime import configure_phonemizer_espeak, get_styletts2_path

logger = logging.getLogger(__name__)

STYLETTS2_PATH = str(get_styletts2_path())
if STYLETTS2_PATH not in sys.path:
    sys.path.append(STYLETTS2_PATH)

_STREAM_END = object()


class StyleTTS2Adapter:
    """Bridge between the project API and the vendored StyleTTS2 engine."""

    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        redis_client: Optional[redis.Redis] = None,
        emotion_parser: Optional[EmotionParser] = None,
        vendor_streamer: Optional[Callable[[str, Dict[str, float], str], AsyncGenerator[str, None]]] = None,
        vendor_cloner: Optional[Callable[[bytes, str], Any]] = None,
        preference_store=None,
        engine: Optional[StyleTTS2Engine] = None,
        speaker_repository: Optional[ISpeakerRepository] = None,
        post_processor: Optional[TTSPostProcessor] = None,
    ):
        setup_hardware_env()
        configure_phonemizer_espeak()
        self.device = get_optimal_device()
        self.redis_client = redis_client or redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=False
        )
        self.prefix = REDIS_TTS_PREFIX
        self.model_checkpoint = model_checkpoint or os.getenv("TTS_MODEL_CHECKPOINT")
        self.emotion_parser = emotion_parser or EmotionParser()
        self.vendor_streamer = vendor_streamer
        self.vendor_cloner = vendor_cloner
        self.preference_store = preference_store
        self.speaker_repository = speaker_repository
        self.post_processor = post_processor or TTSPostProcessor()
        self._engine = engine
        logger.info(
            "StyleTTS2 adapter initialized",
            extra={"device": str(self.device), "prefix": self.prefix},
        )

    @property
    def engine(self) -> StyleTTS2Engine:
        if self._engine is None:
            self._engine = StyleTTS2Engine(
                checkpoint_path=self.model_checkpoint,
                device=str(self.device),
                postprocess=process_samples,
            )
        return self._engine

    def ensure_ready(self) -> None:
        checkpoint = self.engine.checkpoint_path
        if not checkpoint or not os.path.isfile(checkpoint):
            raise StyleTTS2Unavailable(
                "StyleTTS2 checkpoint not found. Set TTS_MODEL_CHECKPOINT or place "
                "an epoch_2nd_*.pth file under models/tts/."
            )

    def get_speaker_embedding(self, speaker_id: str) -> Optional[np.ndarray]:
        redis_key = f"{self.prefix}speaker:{speaker_id}"
        cached_data = self.redis_client.get(redis_key)
        if cached_data:
            return np.frombuffer(cached_data, dtype=np.float32)
        return None

    def save_speaker_embedding(self, speaker_id: str, embedding: np.ndarray) -> None:
        redis_key = f"{self.prefix}speaker:{speaker_id}"
        self.redis_client.set(redis_key, embedding.astype(np.float32).tobytes())

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
                callback(
                    {
                        "type": "first_chunk",
                        "elapsed_ms": round((time.monotonic() - started_at) * 1000, 3),
                    }
                )
                first_chunk = False
            yield chunk

    async def _default_streamer(
        self, text: str, emotion_params: Dict[str, float], speaker_id: str
    ) -> AsyncGenerator[str, None]:
        self.ensure_ready()
        reference = self._resolve_reference(speaker_id)
        iterator = iter(self.engine.synthesize_stream(text, emotion_params, reference))
        while True:
            chunk = await asyncio.to_thread(next, iterator, _STREAM_END)
            if chunk is _STREAM_END:
                break
            yield base64.b64encode(chunk).decode("ascii")

    async def clone_voice(
        self,
        audio_data: bytes,
        user_id: str,
        alias: str,
        vendor_cloner: Optional[Callable[[bytes, str], Any]] = None,
    ) -> Dict[str, Any]:
        if not alias:
            raise ValueError("alias must not be empty")
        if not audio_data:
            raise ValueError("audio data must not be empty")
        cloner = vendor_cloner or self.vendor_cloner
        if cloner is not None:
            result = await cloner(audio_data, alias)
        else:
            result = await self._default_cloner(audio_data, alias, user_id)
        self._persist_alias(user_id, alias)
        return {"status": "ok", "user_id": user_id, "alias": alias, "result": result}

    async def _default_cloner(
        self, audio_data: bytes, alias: str, user_id: str
    ) -> Dict[str, Any]:
        embedding = await asyncio.to_thread(self.engine.encode_reference, audio_data)
        await asyncio.to_thread(self.save_speaker_embedding, alias, embedding)
        record = {
            "alias": alias,
            "user_id": user_id,
            "dimensions": int(embedding.size),
            "sample_rate": self.engine.sample_rate,
        }
        await asyncio.to_thread(self._speaker_repository().save, alias, record)
        logger.info(
            "voice cloned",
            extra={"alias": alias, "user_id": user_id, "dimensions": int(embedding.size)},
        )
        return {"cloned": True, "alias": alias, "dimensions": int(embedding.size)}

    def _speaker_repository(self) -> ISpeakerRepository:
        if self.speaker_repository is None:
            self.speaker_repository = MongoSpeakerRepository()
        return self.speaker_repository

    def _persist_alias(self, user_id: str, alias: str) -> None:
        if self.preference_store is None or not user_id:
            return
        save = getattr(self.preference_store, "save", None)
        if not callable(save):
            logger.debug("Preference store has no save() method; alias was not persisted")
            return
        stored = getattr(self.preference_store, "get", lambda _: None)(user_id) or {}
        preferences = dict(stored.get("preferences") or {})
        preferences["voice_alias"] = alias
        save(user_id, preferences)

    def _resolve_reference(self, speaker_id: str) -> Optional[np.ndarray]:
        if not speaker_id:
            return None
        embedding = self.get_speaker_embedding(speaker_id)
        if embedding is None:
            logger.info("speaker embedding not found; using default style", extra={"speaker_id": speaker_id})
            return None
        return embedding.reshape(1, -1).astype(np.float32)

    def _resolve_emotion(
        self,
        text: str,
        emotion_params: Optional[Dict[str, float]],
        user_id: str = "",
    ) -> Dict[str, float]:
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
        generator = self.generate_speech_stream(
            text, resolved_emotion, speaker_id, callback, user_id
        )
        chunks = [chunk async for chunk in generator]
        return {
            "audio_text": "".join(chunks),
            "audio_chunks": len(chunks),
            "emotion": resolved_emotion,
        }
