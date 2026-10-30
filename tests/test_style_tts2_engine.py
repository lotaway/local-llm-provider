import pytest

from model_providers.tts.style_tts2_engine import StyleTTS2Engine, StyleTTS2Unavailable


def test_synthesize_without_checkpoint_raises_unavailable(tmp_path):
    engine = StyleTTS2Engine(vendor_path=tmp_path)

    with pytest.raises(StyleTTS2Unavailable):
        engine.synthesize("Hello world.", {})


def test_sentence_splitting_respects_sentence_boundaries():
    engine = StyleTTS2Engine(vendor_path="/tmp")

    sentences = engine._split_sentences("Hello world. Good night!")

    assert sentences == ["Hello world.", "Good night!"]


def test_long_sentence_is_chunked_to_max_length(tmp_path):
    engine = StyleTTS2Engine(vendor_path=tmp_path, max_sentence_length=5)

    sentences = engine._split_sentences("Hello world and goodbye")

    assert "".join(sentences) == "Hello world and goodbye"
    assert all(len(sentence) <= 5 for sentence in sentences)
    assert len(sentences) > 1


def test_empty_text_is_rejected():
    engine = StyleTTS2Engine(vendor_path="/tmp")

    with pytest.raises(ValueError):
        engine._split_sentences("   ")
