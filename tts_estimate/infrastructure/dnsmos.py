from typing import Dict, List

import numpy as np

from ..domain.interfaces import MOSCalculator

SAMPLING_RATE = 16000
INPUT_LENGTH_SECONDS = 9.01
FRAME_SAMPLES = int(INPUT_LENGTH_SECONDS * SAMPLING_RATE)
HOP_SAMPLES = SAMPLING_RATE
SIG_POLYFIT = (-0.08397278, 1.22083953, 0.0052439)
BAK_POLYFIT = (-0.13166888, 1.60915514, -0.39604546)
OVR_POLYFIT = (-0.06766283, 1.11546468, 0.04602535)
SCORE_KEYS = ("overall", "signal", "background")


class MicrosoftDNSMOSCalculator(MOSCalculator):
    """Port adapter for the official DNSMOS P.835 sig_bak_ovr.onnx model."""

    def __init__(self, model_path: str = "", session=None, apply_calibration: bool = True):
        self._session = session if session is not None else self._create_session(model_path)
        self._apply_calibration = apply_calibration

    @staticmethod
    def _create_session(model_path: str):
        import onnxruntime as ort

        if not model_path:
            raise ValueError("model_path is required when no ONNX session is injected")
        return ort.InferenceSession(model_path)

    def calculate_score(self, audio_path: str) -> Dict[str, float]:
        """Return mean calibrated OVRL/SIG/BAK scores over 9.01-second windows."""
        return self._run_inference(self._load_audio(audio_path))

    def _load_audio(self, path: str) -> np.ndarray:
        import librosa

        audio, _ = librosa.load(path, sr=SAMPLING_RATE)
        return np.asarray(audio, dtype=np.float32)

    def _run_inference(self, audio: np.ndarray) -> Dict[str, float]:
        results = [self._infer_frame(frame) for frame in self._frames(audio)]
        if not results:
            return {key: 0.0 for key in SCORE_KEYS}
        return {key: float(np.mean([item[key] for item in results])) for key in SCORE_KEYS}

    def _frames(self, audio: np.ndarray) -> List[np.ndarray]:
        padded = self._pad(audio)
        hops = max(1, int(np.floor(len(padded) / SAMPLING_RATE) - INPUT_LENGTH_SECONDS) + 1)
        starts = (index * HOP_SAMPLES for index in range(hops))
        return [
            np.asarray(padded[start : start + FRAME_SAMPLES], dtype=np.float32)
            for start in starts
            if start + FRAME_SAMPLES <= len(padded)
        ]

    def _pad(self, audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return np.zeros(FRAME_SAMPLES, dtype=np.float32)
        padded = np.asarray(audio, dtype=np.float32)
        while padded.size < FRAME_SAMPLES:
            padded = np.concatenate((padded, padded))
        return padded

    def _infer_frame(self, frame: np.ndarray) -> Dict[str, float]:
        name = self._session.get_inputs()[0].name
        inputs = {name: np.asarray(frame, dtype=np.float32)[np.newaxis, :]}
        outputs = self._session.run(None, inputs)
        scores = np.asarray(outputs[0]).reshape(-1)
        if scores.size < 3:
            raise ValueError("DNSMOS first output must contain SIG/BAK/OVR scores")
        sig, bak, ovr = (float(scores[0]), float(scores[1]), float(scores[2]))
        if self._apply_calibration:
            sig, bak, ovr = (
                float(np.polyval(SIG_POLYFIT, sig)),
                float(np.polyval(BAK_POLYFIT, bak)),
                float(np.polyval(OVR_POLYFIT, ovr)),
            )
        return {"overall": ovr, "signal": sig, "background": bak}
