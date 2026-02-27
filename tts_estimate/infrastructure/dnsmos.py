import numpy as np
import onnxruntime as ort
import librosa
from typing import Dict
from ..domain.interfaces import MOSCalculator

class MicrosoftDNSMOSCalculator(MOSCalculator):
    def __init__(self, model_path: str):
        self._session = ort.InferenceSession(model_path)

    def calculate_score(self, audio_path: str) -> Dict[str, float]:
        audio = self._load_audio(audio_path)
        return self._run_inference(audio)

    def _load_audio(self, path: str) -> np.ndarray:
        audio, _ = librosa.load(path, sr=16000)
        return audio.astype(np.float32)

    def _run_inference(self, audio: np.ndarray) -> Dict[str, float]:
        # Placeholder for actual DNSMOS inference logic
        # In real scenario, it involves framing and session.run
        input_data = {self._session.get_inputs()[0].name: [audio]}
        outputs = self._session.run(None, input_data)
        return {
            "overall": float(outputs[0]),
            "signal": float(outputs[1]),
            "background": float(outputs[2])
        }
