import pytest

from services.emotion_parser import EmotionParser


def test_emotion_parser_returns_stable_normalized_params():
    parser = EmotionParser()

    params = parser.parse("This is a great good day")

    assert params.valence > 0.5
    assert set(params.to_dict()) == {
        "valence",
        "arousal",
        "dominance",
        "speed",
        "pitch",
        "energy",
    }


def test_emotion_parser_marks_emotional_text():
    parser = EmotionParser()

    assert parser.parse_params("What a terrible bad bad day!")["arousal"] > parser.parse_params("What a bad day.")["arousal"]


def test_emotion_parser_ignores_questions_for_dominance():
    parser = EmotionParser()

    questions_only = parser.parse("What? Why? How?")
    statements = parser.parse("This is fine.")

    assert questions_only.dominance < statements.dominance
    assert questions_only.arousal > statements.arousal


def test_emotion_parser_accepts_custom_defaults():
    parser = EmotionParser(
        default_params={
            "valence": 0.2,
            "arousal": 0.9,
            "dominance": 0.1,
            "speed": 0.2,
            "pitch": 0.2,
            "energy": 0.8,
        }
    )

    result = parser.parse("")

    assert result.valence == pytest.approx(0.2)
    assert result.arousal == pytest.approx(0.9)
    assert result.dominance == pytest.approx(0.1)
    assert result.speed == pytest.approx(0.2)
    assert result.energy == pytest.approx(0.9)


def test_emotion_parser_uses_default_dominance_for_questions():
    parser = EmotionParser(default_params={"dominance": 0.25})

    assert parser.parse("What? Why?").dominance == pytest.approx(0.25)
