import torch
from speechbrain.inference.speaker import EncoderClassifier
from ..domain.interfaces import SpeakerSimilarityCalculator

class ECAPASpeakerSimilarityCalculator(SpeakerSimilarityCalculator):
    def __init__(self, model_source: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self._classifier = EncoderClassifier.from_hparams(source=model_source)

    def calculate_similarity(self, reference_audio: str, generated_audio: str) -> float:
        ref_emb = self._compute_embedding(reference_audio)
        gen_emb = self._compute_embedding(generated_audio)
        similarity = torch.nn.functional.cosine_similarity(ref_emb, gen_emb)
        return float(similarity.item())

    def _compute_embedding(self, audio_path: str) -> torch.Tensor:
        signal, fs = self._classifier.load_audio(audio_path)
        return self._classifier.encode_batch(signal)
