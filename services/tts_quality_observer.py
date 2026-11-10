import logging
import threading
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, Optional

from constants import TTS_QUALITY_DAILY_LIMIT, TTS_QUALITY_MAX_PENDING, TTS_QUALITY_MAX_WORKERS
from interfaces.tts_types import ITTSQualityRecordStore
from services.tts_alerts import TTSAlertService
from services.tts_quality_gateway import TTSQualityGateway
from services.tts_quality_quota import DailyEvaluationQuota, EvaluationQuotaExceeded
from services.tts_quality_statistics import TTSQualityStatistics
from services.tts_request_sampler import TTSRequestSampler

logger = logging.getLogger(__name__)


class TTSQualityObserver:
    def __init__(
        self,
        gateway: TTSQualityGateway,
        trends: TTSQualityStatistics,
        sampler: TTSRequestSampler,
        alerts: TTSAlertService,
        executor: Optional[Executor] = None, max_pending: int = TTS_QUALITY_MAX_PENDING,
        record_store: Optional[ITTSQualityRecordStore] = None,
        quota: Optional[DailyEvaluationQuota] = None,
    ) -> None:
        if max_pending < 1:
            raise ValueError(f"max_pending must be at least 1, got {max_pending}")
        self._gateway = gateway
        self._trends = trends
        self._sampler = sampler
        self._alerts = alerts
        self._executor = executor or self._build_executor()
        self._capacity = threading.BoundedSemaphore(max_pending)
        self._record_store = record_store
        self._quota = quota or DailyEvaluationQuota(TTS_QUALITY_DAILY_LIMIT)

    @staticmethod
    def _build_executor() -> Executor:
        return ThreadPoolExecutor(
            max_workers=TTS_QUALITY_MAX_WORKERS, thread_name_prefix="tts-quality"
        )

    def observe(
        self, result: Dict[str, Any], audio: bytes, reference_audio: Optional[bytes]
    ) -> None:
        if not self._sampler.select(result):
            self._record_lightweight(result)
            return
        if not self._capacity.acquire(blocking=False):
            self._skip(result, {"reason": "queue_full"})
            return
        try:
            self._quota.consume()
        except EvaluationQuotaExceeded as error:
            self._capacity.release()
            self._skip(result, {"reason": "quota_exceeded", "limit": error.limit})
            return
        self._submit(result, audio, reference_audio)

    def _skip(self, result: Dict[str, Any], extra: Dict[str, Any]) -> None:
        logger.warning("tts_quality_evaluation_skipped", extra=extra)
        self._record_lightweight(result)

    def _record_lightweight(self, result: Dict[str, Any]) -> None:
        self._trends.add(result)
        self._alerts.check([result])

    def _submit(
        self, result: Dict[str, Any], audio: bytes, reference_audio: Optional[bytes]
    ) -> None:
        try:
            future = self._executor.submit(self._evaluate_and_record, result, audio, reference_audio)
        except RuntimeError:
            self._capacity.release()
            logger.exception("tts_quality_evaluation_submission_failed")
            self._record_lightweight(result)
            return
        future.add_done_callback(partial(self._on_completed, result=result))

    def _evaluate_and_record(
        self, result: Dict[str, Any], audio: bytes, reference_audio: Optional[bytes]
    ) -> None:
        report = self._gateway.evaluate(result["text"], reference_audio or b"", audio)
        evaluated = {**result, **report, "evaluated": True}
        self._trends.add(evaluated)
        self._sampler.record(evaluated)
        self._alerts.check([evaluated])
        self._persist(evaluated)

    def _persist(self, record: Dict[str, Any]) -> None:
        if self._record_store is None:
            return
        try:
            self._record_store.save(record)
        except Exception as error:
            logger.error(
                "tts_quality_record_persist_failed", extra={"error_type": type(error).__name__}
            )

    def _on_completed(self, future: Future, result: Dict[str, Any]) -> None:
        self._capacity.release()
        error = future.exception()
        if error is None:
            return
        logger.error(
            "tts_quality_evaluation_failed",
            extra={"error_type": type(error).__name__, "error": str(error)},
        )
        self._trends.add(result)
        self._alerts.check([{"status": "reject", "reason": "evaluation_failed"}])
