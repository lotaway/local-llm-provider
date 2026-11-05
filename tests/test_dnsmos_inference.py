import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tts_estimate.infrastructure.dnsmos import (  # noqa: E402
    BAK_POLYFIT,
    FRAME_SAMPLES,
    HOP_SAMPLES,
    OVR_POLYFIT,
    SIG_POLYFIT,
    MicrosoftDNSMOSCalculator,
)


class FakeSession:
    def __init__(self, scores=(3.0, 2.5, 3.4)):
        self.scores = scores
        self.feeds = []

    def get_inputs(self):
        return [SimpleNamespace(name="input_1")]

    def run(self, _output_names, feed):
        self.feeds.append(feed["input_1"])
        return [np.asarray([list(self.scores)], dtype=np.float32)]


def build(session, apply_calibration=True):
    return MicrosoftDNSMOSCalculator(session=session, apply_calibration=apply_calibration)


def test_feed_shape_matches_official_901_second_window():
    session = FakeSession()
    build(session, False)._run_inference(np.zeros(16000 * 11, dtype=np.float32))
    assert session.feeds[0].shape == (1, FRAME_SAMPLES)
    assert session.feeds[0].dtype == np.float32
    assert FRAME_SAMPLES == 144160


def test_first_output_maps_sig_bak_ovr_to_domain_scores():
    session = FakeSession((3.0, 2.5, 3.4))
    result = build(session, False)._run_inference(np.zeros(16000 * 11, dtype=np.float32))
    assert result == {"overall": 3.4, "signal": 3.0, "background": 2.5}


def test_short_audio_is_padded_to_one_window():
    session = FakeSession()
    build(session, False)._run_inference(np.ones(8000, dtype=np.float32))
    assert len(session.feeds) == 1
    assert session.feeds[0].shape == (1, FRAME_SAMPLES)


def test_long_audio_uses_one_second_hop():
    session = FakeSession()
    build(session, False)._run_inference(np.zeros(16000 * 20, dtype=np.float32))
    assert len(session.feeds) == 11
    assert HOP_SAMPLES == 16000
    assert all(feed.shape == (1, FRAME_SAMPLES) for feed in session.feeds)


def test_official_polyfit_calibration_is_applied():
    session = FakeSession((3.0, 2.5, 3.4))
    result = build(session)._run_inference(np.zeros(16000 * 11, dtype=np.float32))
    assert result["signal"] == pytest.approx(float(np.polyval(SIG_POLYFIT, 3.0)))
    assert result["background"] == pytest.approx(float(np.polyval(BAK_POLYFIT, 2.5)))
    assert result["overall"] == pytest.approx(float(np.polyval(OVR_POLYFIT, 3.4)))


def test_model_without_three_scores_is_rejected():
    session = FakeSession((1.0,))
    with pytest.raises(ValueError):
        build(session, False)._run_inference(np.zeros(16000, dtype=np.float32))


def test_empty_audio_returns_zero_scores():
    session = FakeSession()
    result = build(session, False)._run_inference(np.array([], dtype=np.float32))
    assert result == {"overall": 0.0, "signal": 0.0, "background": 0.0}
