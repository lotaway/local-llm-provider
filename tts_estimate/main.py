import logging
import json
from .config.settings import AppConfig
from .infrastructure.whisper_asr import WhisperASRProcessor
from .infrastructure.dnsmos import MicrosoftDNSMOSCalculator
from .infrastructure.speaker_similarity import ECAPASpeakerSimilarityCalculator
from .infrastructure.wer_calculator import JiwerWERCalculator
from .application.evaluator_service import TTSQualityEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def bootstrap() -> TTSQualityEvaluator:
    config = AppConfig()
    
    asr = WhisperASRProcessor(model_name=config.models.whisper_model)
    mos = MicrosoftDNSMOSCalculator(model_path=config.models.dnsmos_model_path)
    speaker = ECAPASpeakerSimilarityCalculator(model_source=config.models.ecapa_model_source)
    wer_calc = JiwerWERCalculator()
    
    return TTSQualityEvaluator(asr, mos, speaker, wer_calc)

def run_evaluation(
    evaluator: TTSQualityEvaluator,
    ref_text: str,
    ref_audio: str,
    gen_audio: str
) -> str:
    config = AppConfig()
    result = evaluator.evaluate(
        ref_text, ref_audio, gen_audio, config.thresholds.model_dump()
    )
    
    output = {
        "wer": result.metrics.wer,
        "dnsmos": {
            "overall": result.metrics.dnsmos_overall,
            "signal": result.metrics.dnsmos_signal,
            "background": result.metrics.dnsmos_background
        },
        "speaker_similarity": result.metrics.speaker_similarity,
        "status": "pass" if result.is_passed else "reject"
    }
    return json.dumps(output, indent=2)

def main():
    # Example usage (would typically come from CLI args)
    evaluator = bootstrap()
    # Mock data for demonstration
    report = run_evaluation(
        evaluator, 
        "Hello world", 
        "path/to/ref.wav", 
        "path/to/gen.wav"
    )
    logger.info(f"Evaluation Report:\n{report}")

if __name__ == "__main__":
    main()
