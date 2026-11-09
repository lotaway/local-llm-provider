import base64
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from interfaces.tts_types import ITTSAdapter, ITTSQualityObserver
from services.tts_alerts import TTSAlertService
from services.tts_audio_quality import TTSAudioQualityService
from services.tts_feedback_monitor import TTSFeedbackMonitor
from services.tts_post_processing import TTSPostProcessor
from services.tts_request_sampler import TTSRequestSampler
from services.tts_regression_suite import TTSRegressionSuite
from services.tts_quality_statistics import TTSQualityStatistics

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 24000


class TTSService:
    def __init__(
        self,
        adapter: ITTSAdapter,
        preference_store=None,
        post_processor: Optional[TTSPostProcessor] = None,
        quality_observer: Optional[ITTSQualityObserver] = None,
        sampling_monitor: Optional[TTSRequestSampler] = None,
        audio_quality: Optional[TTSAudioQualityService] = None,
        alerts: Optional[TTSAlertService] = None,
        trends: Optional[TTSQualityStatistics] = None,
        feedback: Optional[TTSFeedbackMonitor] = None,
        regression: Optional[TTSRegressionSuite] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        self.adapter = adapter
        self.preference_store = preference_store
        self.post_processor = post_processor or TTSPostProcessor()
        self.quality_observer = quality_observer
        self.sampling_monitor = sampling_monitor or TTSRequestSampler()
        self.audio_quality = audio_quality or TTSAudioQualityService(sample_rate=sample_rate)
        self.alerts = alerts or TTSAlertService()
        self.trends = trends or TTSQualityStatistics()
        self.feedback = feedback or TTSFeedbackMonitor()
        self.regression = regression or TTSRegressionSuite()
        self.sample_rate = sample_rate

    def ensure_ready(self) -> None:
        ensure_ready = getattr(self.adapter, "ensure_ready", None)
        if callable(ensure_ready):
            ensure_ready()

    async def stream(
        self,
        text: str,
        emotion_params: Optional[Dict[str, float]] = None,
        speaker_id: str = "",
        user_id: str = "",
        reference_audio: Optional[bytes] = None,
    ) -> AsyncGenerator[bytes, None]:
        base64_chunks: List[str] = []
        audio_chunks: List[bytes] = []
        text_value = text or ""
        async for chunk in self.adapter.generate_speech_stream(
            text, emotion_params, speaker_id, user_id=user_id
        ):
            base64_chunks.append(chunk)
            try:
                payload = base64.b64decode(chunk, validate=True)
            except (ValueError, TypeError):
                payload = chunk.encode("utf-8")
            audio_chunks.append(payload)
            yield payload
        audio = b"".join(audio_chunks)
        self._record_request(text_value, "".join(base64_chunks), audio, reference_audio)

    def record_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.feedback.record(payload)

    def metrics(self) -> Dict[str, Any]:
        return {
            "sampled_requests": self.sampling_monitor.sampled_count,
            "wer": self.trends.wer_statistics(),
            "dnsmos": self.trends.dnsmos_statistics(),
            "speaker_similarity": self.trends.similarity_statistics(),
            "failure_rate": self.trends.failure_rate(),
            "daily": self.trends.daily_breakdown(),
            "alerts": self.alerts.alerts(limit=20),
            "feedback": self.feedback.distribution(),
        }

    async def run_regression(self, speaker_id: str = "") -> Dict[str, Any]:
        results = []
        for sentence in self.regression.sentences:
            audio = b""
            async for chunk in self.stream(sentence, speaker_id=speaker_id):
                audio += chunk
            results.append(
                {
                    "sentence": sentence,
                    "duration_seconds": self.audio_quality.duration_seconds(audio),
                    "bytes": len(audio),
                }
            )
        return {"results": results}

    def _record_request(
        self, text: str, text_chunks: str, audio: bytes, reference_audio: Optional[bytes]
    ) -> None:
        anomaly = self.audio_quality.is_duration_anomaly(audio) if audio else True
        duplicate = self.audio_quality.has_duplicate(audio) if audio else False
        status = "reject" if anomaly or duplicate else "pass"
        result = {
            "status": status,
            "duration_seconds": self.audio_quality.duration_seconds(audio),
            "duration_anomaly": anomaly,
            "duplicate": duplicate,
            "text": text,
        }
        self.alerts.check([result])
        if self.quality_observer is None:
            self.trends.add(result)
            return
        self.quality_observer.observe(result, audio, reference_audio)
