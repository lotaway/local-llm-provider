from constants import TTS_QUALITY_MAX_PENDING, TTS_QUALITY_SAMPLE_RATE
from infrastructure.tts.logging_notifier import StructuredLogNotifier
from infrastructure.tts.preference_collection import build_preference_collection
from infrastructure.tts.quality_evaluator_adapter import ProductionTTSQualityEvaluator
from infrastructure.tts.speaker_repository import MongoSpeakerRepository
from model_providers.tts.style_tts2_adapter import StyleTTS2Adapter
from services.tts_alerts import TTSAlertService
from services.tts_preference_store import TTSPreferenceStore
from services.tts_quality_gateway import TTSQualityGateway
from services.tts_quality_observer import TTSQualityObserver
from services.tts_quality_statistics import TTSQualityStatistics
from services.tts_request_sampler import TTSRequestSampler
from services.tts_service import TTSService


def _build_quality_observer(alerts, trends, sampler) -> TTSQualityObserver:
    gateway = TTSQualityGateway(evaluator=ProductionTTSQualityEvaluator())
    return TTSQualityObserver(
        gateway=gateway,
        trends=trends,
        sampler=sampler,
        alerts=alerts,
        max_pending=TTS_QUALITY_MAX_PENDING,
    )


def build_tts_service() -> TTSService:
    preferences = TTSPreferenceStore(collection=build_preference_collection())
    alerts = TTSAlertService(notifier=StructuredLogNotifier())
    trends = TTSQualityStatistics()
    sampler = TTSRequestSampler(sample_rate=TTS_QUALITY_SAMPLE_RATE)
    observer = _build_quality_observer(alerts, trends, sampler)
    return TTSService(
        adapter=StyleTTS2Adapter(
            preference_store=preferences,
            speaker_repository=MongoSpeakerRepository(),
        ),
        preference_store=preferences,
        alerts=alerts,
        trends=trends,
        sampling_monitor=sampler,
        quality_observer=observer,
    )
