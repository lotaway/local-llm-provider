"""Single composition point for the real TTS quality evaluator stack."""

import logging
from pathlib import Path
from typing import Optional

from ..config.settings import AppConfig

logger = logging.getLogger(__name__)


def build_quality_evaluator(config: Optional[AppConfig] = None):
    from ..application.evaluator_service import TTSQualityEvaluator
    from ..infrastructure.model_assets import DNSMOS_ASSET, ensure_local_model
    from .dnsmos import MicrosoftDNSMOSCalculator
    from .speaker_similarity import ECAPASpeakerSimilarityCalculator
    from .wer_calculator import JiwerWERCalculator
    from .whisper_asr import WhisperASRProcessor

    config = config or AppConfig()
    root = Path(config.models.models_root)
    dnsmos_path = ensure_local_model(Path(config.models.dnsmos_model_path), DNSMOS_ASSET)
    evaluator = TTSQualityEvaluator(
        WhisperASRProcessor(config.models.whisper_model, download_root=str(root / "whisper")),
        MicrosoftDNSMOSCalculator(model_path=str(dnsmos_path)),
        ECAPASpeakerSimilarityCalculator(config.models.ecapa_model_source, savedir=str(root / "ecapa")),
        JiwerWERCalculator(),
    )
    _log_resolved_models(config, root, dnsmos_path)
    return evaluator


def _log_resolved_models(config: AppConfig, root: Path, dnsmos_path: Path) -> None:
    logger.info(
        "tts_evaluation_models_resolved",
        extra={
            "models_root": str(root),
            "whisper": config.models.whisper_model,
            "dnsmos": str(dnsmos_path),
            "ecapa": config.models.ecapa_model_source,
        },
    )
