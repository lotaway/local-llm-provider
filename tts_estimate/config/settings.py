from pydantic_settings import BaseSettings
from pydantic import Field

class EvaluationThresholds(BaseSettings):
    wer_max: float = Field(default=0.08, description="Maximum Word Error Rate")
    dnsmos_min: float = Field(default=3.8, description="Minimum DNSMOS overall score")
    similarity_min: float = Field(default=0.78, description="Minimum Speaker Similarity")

class ModelConfig(BaseSettings):
    whisper_model: str = "large"
    dnsmos_model_path: str = "models/dnsmos.onnx"
    ecapa_model_source: str = "speechbrain/spkrec-ecapa-voxceleb"

class AppConfig(BaseSettings):
    thresholds: EvaluationThresholds = EvaluationThresholds()
    models: ModelConfig = ModelConfig()

    class Config:
        env_prefix = "TTS_EVAL_"
