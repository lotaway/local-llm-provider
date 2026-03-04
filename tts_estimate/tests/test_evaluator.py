import pytest
from unittest.mock import MagicMock
from ..application.evaluator_service import TTSQualityEvaluator
from ..domain.interfaces import AudioQualityMetrics

@pytest.fixture
def mock_evaluator():
    asr = MagicMock()
    mos = MagicMock()
    speaker = MagicMock()
    wer_calc = MagicMock()
    return TTSQualityEvaluator(asr, mos, speaker, wer_calc), asr, mos, speaker, wer_calc

def test_evaluate_pass(mock_evaluator):
    evaluator, asr, mos, speaker, wer_calc = mock_evaluator
    
    # Setup mocks
    asr.transcribe.return_value = "hello world"
    wer_calc.compute_wer.return_value = 0.02
    mos.calculate_score.return_value = {"overall": 4.2, "signal": 4.0, "background": 4.1}
    speaker.calculate_similarity.return_value = 0.85
    
    thresholds = {"wer_max": 0.08, "dnsmos_min": 3.8, "similarity_min": 0.78}
    
    result = evaluator.evaluate("hello world", "ref.wav", "gen.wav", thresholds)
    
    assert result.is_passed is True
    assert result.metrics.wer == 0.02

def test_evaluate_reject_wer(mock_evaluator):
    evaluator, asr, mos, speaker, wer_calc = mock_evaluator
    
    asr.transcribe.return_value = "wrong text"
    wer_calc.compute_wer.return_value = 0.5
    mos.calculate_score.return_value = {"overall": 4.2, "signal": 4.0, "background": 4.1}
    speaker.calculate_similarity.return_value = 0.85
    
    thresholds = {"wer_max": 0.08, "dnsmos_min": 3.8, "similarity_min": 0.78}
    
    result = evaluator.evaluate("hello world", "ref.wav", "gen.wav", thresholds)
    
    assert result.is_passed is False
