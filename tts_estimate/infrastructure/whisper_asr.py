import whisper
from ..domain.interfaces import ASRProcessor

class WhisperASRProcessor(ASRProcessor):
    def __init__(self, model_name: str = "large"):
        self._model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str) -> str:
        result = self._model.transcribe(audio_path)
        return result["text"].strip()
