"""Speak readings with piper-plus and the Tsukuyomi-chan voice.

Audio is generated on demand. This code never saves it; the Space deletes Gradio's temp copy after 10 minutes.

Tsukuyomi-chan's terms forbid distributing synthesized speech as material, so the demo makes audio
only when a button is pressed and does not keep files.
"""

import io
import wave
from functools import lru_cache

import numpy as np
from huggingface_hub import hf_hub_download
from piper_plus import PiperVoice

from yomibench.normalize import normalize
from yomibench.systems import openjtalk

VOICE_REPO = "ayousanz/piper-plus-tsukuyomi-chan"
MODEL_FILE = "tsukuyomi-chan-6lang-fp16.onnx"
CONFIG_FILE = "config.json"
MAX_CHARS = 200

# From the "credit required" template linked from https://tyc.rei-yumesaki.net/material/corpus/
# (tsukuyomi-chan_credit_template1.html, 2026-10-22). The author confirms the wording before publishing.
CREDIT = (
    "**Voice.** 本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が"
    "無料公開している音声データを使用しています。"
    "■[つくよみちゃんコーパス（CV.夢前黎）](https://tyc.rei-yumesaki.net/material/corpus/) "
    "■[つくよみちゃん公式サイト](https://tyc.rei-yumesaki.net) "
    "【禁止事項】出力した音声を次の目的で使用することを禁止します。■人を批判・攻撃すること。"
    "（「批判・攻撃」の定義は、[つくよみちゃんキャラクターライセンス](https://tyc.rei-yumesaki.net/about/terms/#condition3)"
    "に準じます）■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。"
    "■刺激の強い表現をゾーニングなしで公開すること。■他者に対して二次利用（素材としての利用）を許可する形で公開すること。"
    "■Beam Japanese Reading Fixer 本体の利用規約に違反すること。"
    "【クレジット表記】■Beam Japanese Reading Fixer:つくよみちゃん "
    "Voice model: [ayousanz/piper-plus-tsukuyomi-chan](https://huggingface.co/ayousanz/piper-plus-tsukuyomi-chan) "
    "with [piper-plus](https://github.com/ayutaz/piper-plus) (MIT)."
)


def load_voice() -> PiperVoice:
    model = hf_hub_download(VOICE_REPO, MODEL_FILE)
    config = hf_hub_download(VOICE_REPO, CONFIG_FILE)
    return PiperVoice.load(model, config)


@lru_cache(maxsize=1)
def get_voice() -> PiperVoice:
    return load_voice()


def synthesize(voice: PiperVoice, text: str) -> tuple[int, np.ndarray]:
    """(sample_rate, int16 mono samples), the form gr.Audio(type="numpy") takes."""
    lang = (voice.config.language_id_map or {}).get("ja")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        voice.synthesize(text, wav, language_id=lang)
    buf.seek(0)
    with wave.open(buf, "rb") as wav:
        frames = wav.readframes(wav.getnframes())
        rate = wav.getframerate()
    return rate, np.frombuffer(frames, dtype=np.int16)


def speak(text: str) -> tuple[int, np.ndarray] | None:
    """Speech for text or a katakana reading; None for empty or over-long input."""
    text = (text or "").strip()
    if not text or len(text) > MAX_CHARS:
        return None
    return synthesize(get_voice(), text)


def kana_is_spoken_as_written(kana: str) -> bool:
    """Would the voice (which reads through OpenJTalk) say this katakana reading exactly as written?"""
    return bool(normalize(kana)) and normalize(openjtalk.read(kana)) == normalize(kana)
