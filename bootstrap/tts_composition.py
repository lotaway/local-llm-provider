from infrastructure.tts.logging_notifier import StructuredLogNotifier
from infrastructure.tts.preference_collection import build_preference_collection
from infrastructure.tts.speaker_repository import MongoSpeakerRepository
from model_providers.tts.style_tts2_adapter import StyleTTS2Adapter
from services.tts_alerts import TTSAlertService
from services.tts_preference_store import TTSPreferenceStore
from services.tts_service import TTSService


def build_tts_service() -> TTSService:
    preferences = TTSPreferenceStore(collection=build_preference_collection())
    return TTSService(
        adapter=StyleTTS2Adapter(
            preference_store=preferences,
            speaker_repository=MongoSpeakerRepository(),
        ),
        preference_store=preferences,
        alerts=TTSAlertService(notifier=StructuredLogNotifier()),
    )
