import base64
import binascii
import logging
import re
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from globals import limiter
from interfaces.tts_types import ITTSAdapter
from model_providers.tts.style_tts2_adapter import StyleTTS2Adapter
from services.tts_preference_store import TTSPreferenceStore
from services.tts_service import TTSQualityRejected, TTSService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])

TEXT_MAX_LENGTH = 2000
AUDIO_MAX_BYTES = 10 * 1024 * 1024
ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_service: Optional[TTSService] = None


class GenerateSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=TEXT_MAX_LENGTH)
    speaker_id: str = Field(default="", max_length=128)
    emotion_params: Optional[Dict[str, float]] = None
    user_id: str = Field(default="", max_length=128)

    @field_validator("text")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("emotion_params")
    @classmethod
    def _validate_emotion_params(cls, value):
        if value is None:
            return value
        for key, item in value.items():
            if not 0.0 <= float(item) <= 1.0:
                raise ValueError(f"emotion parameter {key} must be within [0, 1]")
        return value


class CloneVoiceRequest(BaseModel):
    audio_data: str = Field(min_length=1)
    user_id: str = Field(default="", max_length=128)
    alias: str = Field(min_length=1, max_length=64)

    @field_validator("alias")
    @classmethod
    def _validate_alias(cls, value: str) -> str:
        if not ALIAS_PATTERN.match(value):
            raise ValueError("alias must contain only letters, digits, '-' or '_'")
        return value


class FeedbackRequest(BaseModel):
    user_id: str = Field(default="", max_length=128)
    rating: int = Field(ge=1, le=5)
    complaint: str = Field(default="", max_length=1000)
    regenerate_count: int = Field(default=0, ge=0)


def set_tts_service(service: Optional[TTSService]) -> None:
    global _service
    _service = service


def get_tts_service() -> TTSService:
    global _service
    if _service is None:
        _service = TTSService(
            adapter=StyleTTS2Adapter(preference_store=TTSPreferenceStore()),
            preference_store=TTSPreferenceStore(),
        )
    return _service


def set_tts_adapter(adapter: ITTSAdapter) -> None:
    set_tts_service(TTSService(adapter=adapter))


def _decode_clone_audio(audio_data: str) -> bytes:
    try:
        decoded = base64.b64decode(audio_data, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=422, detail="audio_data must be valid base64") from error
    if len(decoded) > AUDIO_MAX_BYTES:
        raise HTTPException(status_code=413, detail="audio_data exceeds the size limit")
    return decoded


@router.post("")
@limiter.limit("10/minute")
async def generate_speech(req: GenerateSpeechRequest, request: Request):
    service = get_tts_service()
    try:
        service.ensure_ready()
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    stream = service.stream(
        req.text, req.emotion_params, req.speaker_id, req.user_id
    )
    return StreamingResponse(
        stream,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.post("/clone")
@limiter.limit("5/minute")
async def clone_voice(req: CloneVoiceRequest, request: Request):
    audio_data = _decode_clone_audio(req.audio_data)
    service = get_tts_service()
    try:
        result = await service.adapter.clone_voice(audio_data, req.user_id, req.alias)
    except Exception as error:
        logger.error("voice clone failed", extra={"alias": req.alias, "error": str(error)})
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": result.get("status"), "alias": result.get("alias")}


@router.post("/feedback")
@limiter.limit("30/minute")
async def record_feedback(req: FeedbackRequest, request: Request):
    record = get_tts_service().record_feedback(req.model_dump())
    return {"status": "recorded", "feedback": record}


@router.get("/metrics")
async def tts_metrics():
    return get_tts_service().metrics()


@router.post("/regression")
@limiter.limit("2/minute")
async def tts_regression(request: Request, speaker_id: str = ""):
    return await get_tts_service().run_regression(speaker_id)
