from model_providers.tts.style_tts2_adapter import StyleTTS2Adapter
from services.emotion_parser import EmotionParser


class FakeRedisClient:
    def get(self, key):
        return None

    def set(self, key, value):
        return None


async def fake_streamer(text, emotion_params, speaker_id):
    for chunk in ("Hel", "lo"):
        yield chunk


def test_generate_speech_stream_yields_vendor_chunks():
    adapter = StyleTTS2Adapter(
        redis_client=FakeRedisClient(),
        vendor_streamer=fake_streamer,
        emotion_parser=EmotionParser(),
    )

    collected = []

    async def run():
        return [
            chunk
            async for chunk in adapter.generate_speech_stream("Hello", None, "spk", collected.append)
        ]

    import asyncio

    chunks = asyncio.run(run())

    assert chunks == ["Hel", "lo"]
    assert collected[0]["type"] == "first_chunk"
    assert collected[0]["elapsed_ms"] >= 0


def test_generate_speech_joins_streamed_chunks():
    adapter = StyleTTS2Adapter(
        redis_client=FakeRedisClient(),
        vendor_streamer=fake_streamer,
        emotion_parser=EmotionParser(),
    )

    import asyncio

    async def run():
        return await adapter.generate_speech("Hello", None, "spk")

    result = asyncio.run(run())

    assert result["audio_text"] == "Hello"
    assert result["emotion"]["valence"] >= 0.0
