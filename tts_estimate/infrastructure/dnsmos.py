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
        frames = self._frames(audio)
        frame_results = [self._infer_frame(frame) for frame in frames]
        if not frame_results:
            return {"overall": 0.0, "signal": 0.0, "background": 0.0}
        return {
            "overall": float(sum(item["overall"] for item in frame_results) / len(frame_results)),
            "signal": float(sum(item["signal"] for item in frame_results) / len(frame_results)),
            "background": float(sum(item["background"] for item in frame_results) / len(frame_results)),
        }

    def _frames(self, audio: np.ndarray, frame_seconds: float = 10.0) -> list:
        frame_samples = max(1, int(len(audio) / 16000 * frame_seconds))
        frames = []
        for start in range(0, len(audio), frame_samples):
            chunk = audio[start : start + frame_samples]
            if len(chunk) == 0:
                break
            frames.append(chunk.astype(np.float32))
        return frames

    def _infer_frame(self, frame: np.ndarray) -> Dict[str, float]:
        input_data = {self._session.get_inputs()[0].name: [frame]}
        outputs = self._session.run(None, input_data)
        return {
            "overall": float(outputs[0]),
            "signal": float(outputs[1]),
            "background": float(outputs[2]),
        }
