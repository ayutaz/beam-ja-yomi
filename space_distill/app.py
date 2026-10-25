"""Gradio demo for Yomi-Distill: a small model, trained on Beam's corrections, tries to fix OpenJTalk's reading.

Beam is the teacher only. This Space never calls Beam and holds no API key: the student runs on ZeroGPU.
"""

import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):  # the Space keeps yomibench/ next to app.py; the repo keeps it one level up
    if (p / "yomibench").is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gradio as gr  # noqa: E402

from yomibench.display import diff_html, kana_to_romaji  # noqa: E402
from yomibench.normalize import normalize  # noqa: E402
from yomibench.student import format_prompt, gate, generate  # noqa: E402
from yomibench.systems import openjtalk  # noqa: E402

try:
    from yomibench import speech
except ImportError:  # the demo works without the voice
    speech = None

try:
    import spaces
    gpu = spaces.GPU(duration=20)
except ImportError:  # not on Hugging Face
    def gpu(fn):
        return fn

MODEL_REPO = "ayousanz/yomi-distill-ja"
MAX_CHARS = 200
MAX_NEW_TOKENS = 160
EXAMPLE = "宇部市岬町に向かいます"  # OpenJTalk: ミサキマチ; the town is ミサキチョウ, which the student gets
_JAPANESE = re.compile(r"[぀-ヿ㐀-鿿々〆ヵヶｦ-ﾟ]")  # ｦ-ﾟ: half-width katakana

INTRO = (
    "Japanese text-to-speech first turns text into a katakana reading with a dictionary-based converter "
    "(OpenJTalk), which misreads numbers, words whose reading depends on context, and proper nouns. "
    "**Beam** (Reflection AI) corrects many of these errors, but a 501B model is too large to run inside a "
    "speech pipeline. This demo runs a 1.8B model trained on Beam's corrections of Wikipedia sentences. "
    "Beam is the teacher only: nothing here calls Beam. In the pre-registered test the student did **not** beat "
    "OpenJTalk overall: it mostly keeps OpenJTalk's reading, reads 町 in place names as チョウ, and sometimes "
    "breaks a name (see Results)."
)
CREDITS = (
    "**Credits.** Teacher: [Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B) by "
    "Reflection AI. Reading converter: [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) "
    "(OpenJTalk), as used by [piper-plus](https://github.com/ayutaz/piper-plus) (MIT). Training sentences: "
    "Japanese Wikipedia (CC BY-SA 3.0)."
)

_student = None  # (model, tokenizer), loaded at startup by load_student()


def gate_ratios() -> tuple[float, float]:
    """The gate frozen before the final scoring (frozen.json next to app.py), else the defaults."""
    path = HERE / "frozen.json"
    if path.exists():
        g = json.loads(path.read_text(encoding="utf-8"))
        return g["min_ratio"], g["max_ratio"]
    return 0.8, 1.25


def load_student(repo: str = MODEL_REPO) -> bool:
    """At module level on ZeroGPU: CUDA placement outside @spaces.GPU is emulated and done once.
    False if the model cannot be loaded (e.g. a private repo without a token): the page then shows OpenJTalk only."""
    global _student
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
    except Exception as err:
        print(f"student not loaded from {repo}: {type(err).__name__}: {err}", file=sys.stderr)
        _student = None
        return False
    _student = (model, tok)
    return True


@gpu
def run_student(prompts: list[str]) -> list[tuple[str, bool]]:
    model, tok = _student
    return generate(model, tok, prompts, MAX_NEW_TOKENS)


def _reading_block(label: str, kana: str, marked: str) -> str:
    return (f"<div style='margin:.4em 0'><b>{html.escape(label)}</b><br>"
            f"<span style='font-size:1.25em'>{marked}</span><br>"
            f"<code>{html.escape(kana_to_romaji(kana))}</code></div>")


def read(text: str, student) -> tuple[str, str, tuple[str, str]]:
    """(html, message, (OpenJTalk reading, student reading)). Never raises on bad input or GPU trouble.

    student: callable taking a list of prompts and returning (output, truncated) per prompt, or None.
    """
    text = (text or "").strip()
    if not text:
        return "", "Please enter a Japanese sentence.", ("", "")
    if len(text) > MAX_CHARS:
        return "", f"Please keep it to {MAX_CHARS} characters or fewer.", ("", "")
    if not _JAPANESE.search(text):
        return "", "Please enter Japanese text (kana or kanji).", ("", "")
    draft = openjtalk.read(text)
    parts = [_reading_block("OpenJTalk reading", draft, html.escape(draft))]
    if student is None:
        return "".join(parts), "The student model is not loaded here. Showing OpenJTalk only.", (draft, "")
    try:
        output, truncated = student([format_prompt(text, draft)])[0]
    except Exception as err:  # GPU queue or quota trouble: keep the page usable
        return "".join(parts), f"The student could not run ({type(err).__name__}). Showing OpenJTalk only.", \
            (draft, "")
    reading, fellback = gate(draft, output, truncated, *gate_ratios())
    parts.append(_reading_block("Student's reading (changes highlighted)", reading, diff_html(draft, reading)))
    if fellback:
        msg = "The student's answer did not look like a reading, so OpenJTalk's reading is kept (fallback)."
    elif normalize(reading) == normalize(draft):
        msg = "The student kept the OpenJTalk reading unchanged."
    else:
        msg = "Highlighted = where the student changed the OpenJTalk reading."
    return "".join(parts), msg, (draft, reading)


def _report() -> str:
    path = HERE / "report.md"
    return path.read_text(encoding="utf-8") if path.exists() else "_Results are added after the final scoring run._"


def build_demo(student) -> gr.Blocks:
    # delete_cache: generated speech is removed from Gradio's temp files after 10 minutes
    with gr.Blocks(title="Yomi-Distill", delete_cache=(600, 600)) as demo:
        gr.Markdown("# Yomi-Distill: can a small model learn Beam's Japanese reading fixes?")
        gr.Markdown(INTRO)
        with gr.Tabs():
            with gr.Tab("Try it"):
                box = gr.Textbox(label=f"Japanese text (up to {MAX_CHARS} characters)", value=EXAMPLE, max_lines=4)
                go = gr.Button("Read it", variant="primary")
                out = gr.HTML()
                msg = gr.Markdown()
                readings = gr.State(("", ""))
                go.click(lambda t: read(t, student), box, [out, msg, readings], concurrency_limit=1)
                if speech is not None:
                    with gr.Row():
                        b1 = gr.Button("Speak OpenJTalk reading")
                        b2 = gr.Button("Speak student's reading")
                    audio = gr.Audio(label="Speech (generated on demand, not for download)", type="numpy",
                                     buttons=[])
                    b1.click(lambda r: speech.speak(r[0]), readings, audio, concurrency_limit=1)
                    b2.click(lambda r: speech.speak(r[1]), readings, audio, concurrency_limit=1)
            with gr.Tab("Results"):
                gr.Markdown(_report())
        gr.Markdown(CREDITS)
        if speech is not None:
            gr.Markdown(speech.credit("Yomi-Distill"))
    return demo


if __name__ == "__main__":
    build_demo(run_student if load_student() else None).launch()
