from typing import Optional

import whisper
from ..domain.interfaces import ASRProcessor


class WhisperASRProcessor(ASRProcessor):
    def __init__(self, model_name: str = "large", download_root: Optional[str] = None):
        self._model = whisper.load_model(model_name, download_root=download_root)

    def transcribe(self, audio_path: str) -> str:
        result = self._model.transcribe(audio_path)
        return result["text"].strip()
