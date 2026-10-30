from services.tts_quality_gateway import TTSQualityGateway


class FakeEvaluator:
    def __init__(self, is_passed: bool = True):
        self.is_passed = is_passed

    def evaluate(self, ref_text, ref_audio, gen_audio):
        from types import SimpleNamespace

        return SimpleNamespace(
            is_passed=self.is_passed,
            metrics=SimpleNamespace(
                wer=0.12,
                dnsmos_overall=3.2,
                dnsmos_signal=3.4,
                dnsmos_background=3.1,
            ),
            speaker_similarity=0.88,
        )


def test_quality_gateway_blocks_failed_generation():
    gateway = TTSQualityGateway(FakeEvaluator(is_passed=False))

    report = gateway.evaluate("Hello world", b"audio", b"audio")

    assert report["status"] == "reject"
    assert gateway.should_block(report) is True


def test_quality_gateway_passes_clean_generation():
    gateway = TTSQualityGateway(FakeEvaluator(is_passed=True))

    report = gateway.evaluate("Hello world", b"audio", b"audio")

    assert report["status"] == "pass"
    assert gateway.should_block(report) is False
