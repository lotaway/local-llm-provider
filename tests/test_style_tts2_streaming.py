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


class FakePreferenceStore:
    def __init__(self, preferences):
        self._preferences = preferences

    def get(self, user_id):
        return {"preferences": self._preferences} if self._preferences else None


def test_generate_speech_applies_user_preferences():
    adapter = StyleTTS2Adapter(
        redis_client=FakeRedisClient(),
        vendor_streamer=fake_streamer,
        emotion_parser=EmotionParser(),
        preference_store=FakePreferenceStore({"valence": 0.9, "arousal": 0.8}),
    )

    import asyncio

    async def run():
        return await adapter.generate_speech("Hello", None, "spk", user_id="u1")

    result = asyncio.run(run())

    assert result["emotion"]["valence"] == 0.9
    assert result["emotion"]["arousal"] == 0.8


async def fake_cloner(audio_data, alias):
    return {"cloned": True, "alias": alias, "bytes": len(audio_data)}


def test_clone_voice_uses_injected_cloner_and_persists_preference():
    adapter = StyleTTS2Adapter(
        redis_client=FakeRedisClient(),
        emotion_parser=EmotionParser(),
        preference_store=FakePreferenceStore({}),
        vendor_cloner=fake_cloner,
    )

    import asyncio

    async def run():
        return await adapter.clone_voice(b"audio", user_id="u2", alias="voice-a")

    result = asyncio.run(run())

    assert result["status"] == "ok"
    assert result["result"]["cloned"] is True
