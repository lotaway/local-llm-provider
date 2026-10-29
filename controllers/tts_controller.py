from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional, Dict

from globals import limiter
from model_providers.tts.style_tts2_adapter import StyleTTS2Adapter

router = APIRouter(prefix="/tts", tags=["tts"])


class GenerateSpeechRequest(BaseModel):
    text: str
    speaker_id: str = ""
    emotion_params: Optional[Dict[str, float]] = None
    user_id: str = ""


class CloneVoiceRequest(BaseModel):
    audio_data: str
    user_id: str = ""
    alias: str = ""


def get_tts_adapter() -> StyleTTS2Adapter:
    return StyleTTS2Adapter()


@router.post("")
@limiter.limit("10/minute")
async def generate_speech(req: GenerateSpeechRequest, request: Request):
    adapter = get_tts_adapter()
    audio_text = ""
    async for chunk in adapter.generate_speech_stream(
        req.text, req.emotion_params, req.speaker_id, user_id=req.user_id
    ):
        audio_text += chunk
    return {"text": req.text, "audio_text": audio_text}


@router.post("/clone")
@limiter.limit("5/minute")
async def clone_voice(req: CloneVoiceRequest, request: Request):
    import base64

    adapter = get_tts_adapter()
    audio_data = base64.b64decode(req.audio_data)
    result = await adapter.clone_voice(audio_data, req.user_id, req.alias)
    return {"status": result.get("status"), "alias": result.get("alias")}
