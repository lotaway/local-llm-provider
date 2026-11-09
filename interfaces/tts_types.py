from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol


class ITTSCollection(Protocol):
    """Minimal persistent-store port backed by a Mongo-style collection."""

    def update_one(self, filter: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> Any: ...

    def find_one(self, filter: Dict[str, Any], projection: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]: ...

    def find(self, filter: Dict[str, Any], projection: Optional[Dict[str, Any]] = None) -> Any: ...

    def delete_one(self, filter: Dict[str, Any]) -> Any: ...


class ITTSQualityEvaluator(ABC):
    """Port for the TTS quality evaluation pipeline (WER/DNSMOS/similarity)."""

    def evaluate(
        self,
        ref_text: str,
        ref_audio: bytes,
        gen_audio: bytes,
        thresholds: Dict[str, float],
    ) -> Any: ...


class ITTSNotifier(ABC):
    """Port for TTS quality alert notifications."""

    def notify(self, result: Dict[str, Any]) -> None: ...


class ITTSQualityRecordStore(Protocol):
    """Port for persisting sanitized TTS quality records."""

    def save(self, record: Dict[str, Any]) -> None: ...


class ITTSQualityObserver(Protocol):
    """Port for sampled, asynchronous quality observation after a TTS request."""

    def observe(
        self,
        result: Dict[str, Any],
        audio: bytes,
        reference_audio: Optional[bytes],
    ) -> None: ...


class ISpeakerRepository(ABC):
    @abstractmethod
    def save(self, speaker_id: str, record: Dict[str, Any]) -> None: ...

    @abstractmethod
    def get(self, speaker_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def list(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def delete(self, speaker_id: str) -> bool: ...


class ITTSAdapter(ABC):
    @abstractmethod
    async def generate_speech_stream(self, text: str, emotion_params=None, speaker_id: str = "", callback=None, user_id: str = ""): ...

    @abstractmethod
    async def clone_voice(self, audio_data: bytes, user_id: str, alias: str, vendor_cloner=None) -> Dict[str, Any]: ...
