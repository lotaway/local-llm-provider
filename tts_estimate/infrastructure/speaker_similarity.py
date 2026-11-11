from typing import Optional

import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from ..domain.interfaces import SpeakerSimilarityCalculator


class ECAPASpeakerSimilarityCalculator(SpeakerSimilarityCalculator):
    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: Optional[str] = None,
    ):
        self._classifier = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=savedir,
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,
            run_opts={"device": _accelerator()},
        )

    def calculate_similarity(self, reference_audio: str, generated_audio: str) -> float:
        ref_emb = self._compute_embedding(reference_audio).flatten()
        gen_emb = self._compute_embedding(generated_audio).flatten()
        similarity = torch.nn.functional.cosine_similarity(ref_emb, gen_emb, dim=0)
        return float(similarity.item())

    def _compute_embedding(self, audio_path: str) -> torch.Tensor:
        signal = self._classifier.load_audio(audio_path)
        return self._classifier.encode_batch(signal)


def _accelerator() -> str:
    return f"cuda:{torch.cuda.current_device()}"
