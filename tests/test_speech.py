import numpy as np
import pytest

pytest.importorskip("piper_plus")

from yomibench import speech  # noqa: E402


def test_kana_is_spoken_as_written():
    assert speech.kana_is_spoken_as_written("ブタイノカミテカラトウジョウスル")
    assert not speech.kana_is_spoken_as_written("")


def test_synthesize_returns_int16_audio():
    sr, audio = speech.synthesize(speech.load_voice(), "キョーワイチバニイク")
    assert sr == 22050
    assert audio.dtype == np.int16 and audio.ndim == 1 and len(audio) > sr // 2


def test_speak_rejects_empty_and_too_long():
    assert speech.speak("") is None
    assert speech.speak("ア" * (speech.MAX_CHARS + 1)) is None
