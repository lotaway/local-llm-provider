from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationThresholds(BaseModel):
    wer_max: float = Field(default=0.08, description="Maximum Word Error Rate")
    dnsmos_min: float = Field(default=3.8, description="Minimum DNSMOS overall score")
    similarity_min: float = Field(default=0.78, description="Minimum Speaker Similarity")


class ModelConfig(BaseModel):
    whisper_model: str = "large"
    dnsmos_model_path: str = "models/dnsmos.onnx"
    ecapa_model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    models_root: str = "models"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_EVAL_", env_nested_delimiter="__")

    thresholds: EvaluationThresholds = EvaluationThresholds()
    models: ModelConfig = ModelConfig()
