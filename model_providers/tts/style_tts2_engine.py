"""Real StyleTTS2 inference and speaker embedding extraction.

The vendored StyleTTS2 checkout contains the model code and the auxiliary
ASR/JDC/PLBERT weights, but not the final StyleTTS2 checkpoint.  This module
loads a checkpoint lazily so importing the adapter stays cheap and the service
can start without the optional weights.

Vendor modules use generic names (``models``, ``text_utils``) that can collide
with project modules, therefore they are loaded through private module names
inside a temporary vendor working directory.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from utils.tts_runtime import configure_phonemizer_espeak, get_styletts2_path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SENTENCE_PATTERN = re.compile(r"(?<=[.!?。！？])\s*")
MAX_SENTENCE_LENGTH = 400


class StyleTTS2Unavailable(RuntimeError):
    """Raised when the checkpoint or a required vendor module is missing."""


class StyleTTS2Engine:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        vendor_path: Optional[os.PathLike] = None,
        max_sentence_length: int = MAX_SENTENCE_LENGTH,
        postprocess: Optional[Callable[[np.ndarray, Dict[str, float]], np.ndarray]] = None,
    ):
        self._checkpoint_path = checkpoint_path or os.getenv("TTS_MODEL_CHECKPOINT")
        self._config_path = config_path or os.getenv("TTS_MODEL_CONFIG")
        self._device_name = device
        self._vendor_path = Path(vendor_path) if vendor_path else get_styletts2_path()
        self._max_sentence_length = max_sentence_length
        self._postprocess = postprocess
        self._torch = None
        self._model = None
        self._sampler = None
        self._text_cleaner = None
        self._phonemizer = None
        self._device = None
        self._multispeaker = False
        self._style_dim = 128
        self._lock = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def checkpoint_path(self) -> Optional[str]:
        return self._checkpoint_path or self._discover_checkpoint()

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def load(self) -> None:
        if self._loaded:
            return
        import threading

        if self._lock is None:
            self._lock = threading.Lock()
        with self._lock:
            if self._loaded:
                return
            self._load_vendor()

    def synthesize(self, text: str, emotion_params: Dict[str, float]) -> np.ndarray:
        self.load()
        wave = self._synthesize_wave(text, emotion_params, None)
        return self._postprocess_wave(wave, emotion_params)

    def synthesize_stream(
        self,
        text: str,
        emotion_params: Dict[str, float],
        reference: Optional[np.ndarray] = None,
    ) -> Iterator[bytes]:
        self.load()
        for sentence in self._split_sentences(text):
            wave = self._synthesize_wave(sentence, emotion_params, reference)
            yield self._to_wav(self._postprocess_wave(wave, emotion_params))

    def _postprocess_wave(
        self, wave: np.ndarray, emotion_params: Dict[str, float]
    ) -> np.ndarray:
        if self._postprocess is None:
            return wave
        return self._postprocess(wave, emotion_params)

    def encode_reference(self, audio_bytes: bytes) -> np.ndarray:
        self.load()
        wave = self._decode_audio(audio_bytes)
        mel = self._preprocess(wave)
        with self._torch.no_grad():
            ref_s = self._model.style_encoder(mel.unsqueeze(1))
            if self._multispeaker:
                ref_p = self._model.predictor_encoder(mel.unsqueeze(1))
                embedding = self._torch.cat([ref_s, ref_p], dim=1)
            else:
                embedding = ref_s
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def _load_vendor(self) -> None:
        checkpoint = self.checkpoint_path
        if not checkpoint or not Path(checkpoint).is_file():
            raise StyleTTS2Unavailable(
                "StyleTTS2 checkpoint not found. Set TTS_MODEL_CHECKPOINT or place "
                "an epoch_2nd_*.pth file under models/tts/."
            )
        configure_phonemizer_espeak()
        with self._vendor_context():
            import torch
            from munch import Munch

            models = self._load_private_module("models")
            text_utils = self._load_private_module("text_utils")
            from Modules.diffusion.sampler import (
                ADPM2Sampler,
                DiffusionSampler,
                KarrasSchedule,
            )
            from Utils.PLBERT.util import load_plbert

            config = self._read_config(models)
            self._device = self._resolve_device(torch)
            self._multispeaker = bool(config["model_params"].get("multispeaker", False))
            self._style_dim = int(config["model_params"].get("style_dim", 128))

            text_aligner = models.load_ASR_models(
                self._vendor_file(config["ASR_path"]), self._vendor_file(config["ASR_config"])
            )
            pitch_extractor = models.load_F0_models(self._vendor_file(config["F0_path"]))
            plbert = load_plbert(self._vendor_file(config["PLBERT_dir"]))
            model = models.build_model(
                self._recursive_munch(Munch, config["model_params"]),
                text_aligner,
                pitch_extractor,
                plbert,
            )
            self._load_state_dict(torch, model, checkpoint)
            for key in model:
                model[key].eval()
                model[key].to(self._device)
            self._model = model
            self._torch = torch
            self._text_cleaner = text_utils.TextCleaner()
            self._sampler = DiffusionSampler(
                model.diffusion.diffusion,
                sampler=ADPM2Sampler(),
                sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                clamp=False,
            )
            self._phonemizer = self._build_phonemizer()
            self._loaded = True
            logger.info(
                "StyleTTS2 checkpoint loaded",
                extra={
                    "checkpoint": str(checkpoint),
                    "device": str(self._device),
                    "multispeaker": self._multispeaker,
                },
            )

    def _synthesize_wave(
        self,
        text: str,
        emotion_params: Dict[str, float],
        reference: Optional[np.ndarray],
    ) -> np.ndarray:
        text = (text or "").strip().replace('"', "")
        if not text:
            raise ValueError("text must not be empty")
        torch = self._torch
        tokens = self._phoneme_tokens(text)
        token_tensor = torch.LongTensor(tokens).to(self._device).unsqueeze(0)
        input_lengths = torch.LongTensor([token_tensor.shape[-1]]).to(self._device)
        text_mask = self._length_to_mask(torch, input_lengths).to(self._device)
        embedding_scale = 1.0 + float(emotion_params.get("arousal", 0.5)) * 1.5
        diffusion_steps = int(os.getenv("TTS_DIFFUSION_STEPS", "10"))
        noise = torch.randn(1, 1, self._style_dim * 2).to(self._device)

        with torch.no_grad():
            t_en = self._model.text_encoder(token_tensor, input_lengths, text_mask)
            bert_dur = self._model.bert(
                token_tensor, attention_mask=(~text_mask).int()
            )
            d_en = self._model.bert_encoder(bert_dur).transpose(-1, -2)
            if reference is None:
                s_pred = self._sampler(
                    noise,
                    embedding=bert_dur[0].unsqueeze(0),
                    num_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                ).squeeze(0)
            else:
                s_pred = self._sampler(
                    noise=noise,
                    embedding=bert_dur,
                    embedding_scale=embedding_scale,
                    features=torch.from_numpy(reference).to(self._device),
                    num_steps=diffusion_steps,
                ).squeeze(1)
            style, ref = s_pred[:, self._style_dim :], s_pred[:, : self._style_dim]
            if reference is not None:
                ref_tensor = torch.from_numpy(reference).to(self._device)
                ref = 0.3 * ref + 0.7 * ref_tensor[:, : self._style_dim]
                style = 0.7 * style + 0.3 * ref_tensor[:, self._style_dim :]

            durations, d = self._predict_durations(
                d_en, style, input_lengths, text_mask, emotion_params
            )
            alignment = self._alignment(torch, input_lengths, durations)
            prosody = d.transpose(-1, -2) @ alignment.unsqueeze(0).to(self._device)
            f0_pred, n_pred = self._model.predictor.F0Ntrain(prosody, style)
            f0_pred = f0_pred * (0.75 + float(emotion_params.get("pitch", 0.5)) * 0.5)
            audio = self._model.decoder(
                t_en @ alignment.unsqueeze(0).to(self._device),
                f0_pred,
                n_pred,
                ref.squeeze().unsqueeze(0),
            )
        wave = audio.squeeze().cpu().numpy().astype(np.float32)
        return self._apply_energy(wave, emotion_params)

    def _predict_durations(self, d_en, style, input_lengths, text_mask, emotion_params):
        torch = self._torch
        d = self._model.predictor.text_encoder(d_en, style, input_lengths, text_mask)
        x, _ = self._model.predictor.lstm(d)
        duration = self._model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        speed = max(0.5, float(emotion_params.get("speed", 0.5))) * 1.0
        speed_factor = max(0.5, min(2.0, 0.75 + speed * 0.5))
        durations = torch.round(duration.squeeze() / speed_factor).clamp(min=1)
        durations[-1] += 5
        return durations, d

    def _alignment(self, torch, input_lengths, durations):
        alignment = torch.zeros(
            input_lengths, int(durations.sum().data), device=self._device
        )
        frame = 0
        for index in range(alignment.size(0)):
            length = int(durations[index].data)
            alignment[index, frame : frame + length] = 1
            frame += length
        return alignment

    def _apply_energy(self, wave: np.ndarray, emotion_params: Dict[str, float]) -> np.ndarray:
        energy = np.clip(0.75 + float(emotion_params.get("energy", 0.5)) * 0.5, 0.5, 1.5)
        return np.clip(wave * energy, -1.0, 1.0)

    def _phoneme_tokens(self, text: str) -> List[int]:
        from nltk.tokenize import word_tokenize

        phonemes = self._phonemizer.phonemize([text])[0]
        try:
            words = word_tokenize(phonemes)
        except LookupError:
            words = phonemes.split()
        tokens = self._text_cleaner(" ".join(words))
        tokens.insert(0, 0)
        return tokens

    def _preprocess(self, wave: np.ndarray):
        torch = self._torch
        import torchaudio

        transform = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        ).to(self._device)
        tensor = torch.from_numpy(wave).float().to(self._device)
        mel = transform(tensor)
        return (torch.log(1e-5 + mel.unsqueeze(0)) + 4.0) / 4.0

    def _decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        import soundfile as sf

        try:
            wave, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            import librosa

            wave, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        if getattr(wave, "ndim", 1) > 1:
            wave = wave.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            import librosa

            wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        import librosa

        trimmed, _ = librosa.effects.trim(wave, top_db=30)
        return trimmed.astype(np.float32)

    def _to_wav(self, wave: np.ndarray) -> bytes:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, wave, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def _split_sentences(self, text: str) -> List[str]:
        sentences = [item.strip() for item in SENTENCE_PATTERN.split(text or "") if item.strip()]
        if not sentences:
            raise ValueError("text must not be empty")
        return [
            sentence[index : index + self._max_sentence_length]
            for sentence in sentences
            for index in range(0, len(sentence), self._max_sentence_length)
        ]

    def _read_config(self, models_module) -> Dict[str, Any]:
        import yaml

        path = self._config_path
        if not path:
            checkpoint = Path(self.checkpoint_path or "")
            candidate = checkpoint.parent / "config.yml"
            path = str(candidate) if candidate.is_file() else str(self._vendor_path / "Configs" / "config.yml")
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _vendor_file(self, value: Any) -> str:
        path = Path(str(value))
        if path.is_absolute():
            return str(path)
        return str(self._vendor_path / path)

    def _discover_checkpoint(self) -> Optional[str]:
        roots = [Path(os.getenv("TTS_MODEL_DIR", Path.cwd() / "models" / "tts")), self._vendor_path]
        for root in roots:
            if not root.is_dir():
                continue
            for pattern in ("**/epoch_2nd_*.pth", "**/epochs_2nd_*.pth", "**/styletts2*.pth"):
                matches = sorted(root.glob(pattern))
                if matches:
                    return str(matches[0])
        return None

    def _resolve_device(self, torch):
        if self._device_name:
            return torch.device(self._device_name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_state_dict(self, torch, model, checkpoint: str) -> None:
        state = torch.load(checkpoint, map_location="cpu")
        params = state.get("net", state)
        for key in model:
            if key not in params:
                continue
            weights = params[key]
            try:
                model[key].load_state_dict(weights, strict=False)
            except RuntimeError:
                weights = {name[7:] if name.startswith("module.") else name: value for name, value in weights.items()}
                model[key].load_state_dict(weights, strict=False)

    def _build_phonemizer(self):
        from phonemizer.backend import EspeakBackend

        return EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)

    @staticmethod
    def _recursive_munch(munch_type, value):
        if isinstance(value, dict):
            return munch_type({key: StyleTTS2Engine._recursive_munch(munch_type, item) for key, item in value.items()})
        return value

    @staticmethod
    def _length_to_mask(torch, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        return torch.gt(mask + 1, lengths.unsqueeze(1))

    @contextmanager
    def _vendor_context(self) -> Iterator[None]:
        previous_path = list(sys.path)
        previous_cwd = os.getcwd()
        sys.path.insert(0, str(self._vendor_path))
        os.chdir(self._vendor_path)
        try:
            yield
        finally:
            os.chdir(previous_cwd)
            sys.path[:] = previous_path

    def _load_private_module(self, name: str):
        module_name = f"_styletts2_vendor_{name}"
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached
        path = self._vendor_path / f"{name}.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise StyleTTS2Unavailable(f"cannot load StyleTTS2 vendor module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def encode_wav_base64(wave: bytes) -> str:
    return base64.b64encode(wave).decode("ascii")
