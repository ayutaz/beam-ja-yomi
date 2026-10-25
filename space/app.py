"""Gradio demo: can Beam fix OpenJTalk's misreadings of Japanese text?

Tabs: Results (tables from the benchmark), Examples (readings side by side), Try it (live correction).
Live correction needs REFLECTION_API_KEY (a Space secret, or .env when run locally).
"""

import html
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):  # the Space keeps yomibench/ next to app.py; the repo keeps it one level up
    if (p / "yomibench").is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gradio as gr  # noqa: E402
import openai  # noqa: E402

from yomibench.display import diff_html, gold_diff_html, kana_to_romaji  # noqa: E402
from yomibench.summary import render_markdown, summarize  # noqa: E402
from yomibench.systems import openjtalk  # noqa: E402
from yomibench.systems.beam import BeamClient, DailyLimitReached, build_messages, parse_response  # noqa: E402

try:
    from yomibench import speech
except ImportError:  # the demo works without the voice
    speech = None

try:
    import spaces
except ImportError:  # not on Hugging Face
    spaces = None

if spaces is not None:
    @spaces.GPU(duration=1)
    def _zero_gpu_placeholder():
        """ZeroGPU Spaces must declare a GPU function at startup. Nothing here uses the GPU; the app runs on CPU."""
        return None

MAX_CHARS = 200
FIX_CONDITION = "beam_fix-low-en-r0"
TRY_IT_EXAMPLE = "このカレーは辛い"  # OpenJTalk: ツライ (painful); every Beam-fix run: カライ (spicy)
CATEGORIES = ("all", "number", "homograph", "place", "person", "medical", "government", "disaster")
KINDS = ("fixed", "broken", "all")
_JAPANESE = re.compile(r"[぀-ヿ㐀-鿿々〆ヵヶｦ-ﾟ]")  # ｦ-ﾟ: half-width katakana

INTRO = (
    "Japanese text-to-speech usually turns text into a katakana reading with a dictionary-based converter "
    "(OpenJTalk) before a voice model speaks it. The converter misreads numbers, units, words whose reading "
    "depends on context, and proper nouns. This demo asks **Beam** (Reflection AI) to correct the converter's "
    "reading, and measures how often it fixes a misreading and how often it breaks a correct one, on a "
    "set of 300 Japanese sentences whose readings a native speaker approved "
    "([dataset](https://huggingface.co/datasets/ayousanz/beam-ja-yomi-bench))."
)
CREDITS = (
    "**Credits.** Model: [Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B) by "
    "Reflection AI. Reading converter: [pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) "
    "(OpenJTalk), as used by [piper-plus](https://github.com/ayutaz/piper-plus) (MIT). Place names: Japan Post "
    "postal code data. Benchmark data: CC BY 4.0."
)


def default_results_dir() -> Path:
    for p in (Path(os.environ.get("YOMI_RESULTS_DIR", HERE / "results")), HERE.parent / "results",
              HERE.parent / "cache" / "smoke"):
        if (p / "ojt.jsonl").exists():
            return p
    return HERE / "results"


def figure_path() -> Path | None:
    """The results figure: figures/ next to app.py on the Space, hf/figures/ in the repository."""
    for p in (HERE / "figures" / "results.png", HERE.parent / "hf" / "figures" / "results.png"):
        if p.exists():
            return p
    return None


def default_beam() -> BeamClient | None:
    try:
        # one stuck request must not hold the single Try-it slot for long
        # memory cache only: visitors' text is not written to the Space's disk
        return BeamClient(cache_dir=None, timeout=60, max_retries=1)
    except RuntimeError:  # no REFLECTION_API_KEY
        return None


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _reading_block(label: str, kana: str, marked: str) -> str:
    return (f"<div style='margin:.4em 0'><b>{html.escape(label)}</b><br>"
            f"<span style='font-size:1.25em'>{marked}</span><br>"
            f"<code>{html.escape(kana_to_romaji(kana))}</code></div>")


def try_it(text: str, beam: BeamClient | None) -> tuple[str, str]:
    """(html, message). Never raises on bad input or API trouble."""
    text = (text or "").strip()
    if not text:
        return "", "Please enter a Japanese sentence."
    if len(text) > MAX_CHARS:
        return "", f"Please keep it to {MAX_CHARS} characters or fewer."
    if not _JAPANESE.search(text):
        return "", "Please enter Japanese text (kana or kanji)."
    draft = openjtalk.read(text)
    parts = [_reading_block("OpenJTalk reading", draft, html.escape(draft))]
    if beam is None:
        return "".join(parts), "Live Beam correction is disabled here (no API key). Showing OpenJTalk only."
    try:
        parsed = parse_response(beam.complete(build_messages("fix", "en", text, draft), "low", 0))
    except DailyLimitReached:
        return "".join(parts), "Beam's daily allowance for this demo is used up. Please try again tomorrow."
    except openai.RateLimitError:
        return "".join(parts), "Beam is busy right now. Showing OpenJTalk only; please try again in a minute."
    except openai.APITimeoutError:
        return "".join(parts), "Beam took too long to answer. Showing OpenJTalk only; please try again."
    except Exception as err:  # network or API error: keep the page usable
        return "".join(parts), f"Beam could not be reached ({type(err).__name__}). Showing OpenJTalk only."
    fixed = parsed["output"]
    if parsed["truncated"] or not fixed:
        return "".join(parts), "Beam did not return a reading this time. Please try again."
    parts.append(_reading_block("Beam-corrected reading (changes highlighted)", fixed, diff_html(draft, fixed)))
    if fixed == draft:
        return "".join(parts), "Beam kept the OpenJTalk reading unchanged."
    return "".join(parts), "Highlighted = where Beam changed the OpenJTalk reading."


def example_rows(results_dir: Path, category: str, kind: str) -> list[dict]:
    """Join OpenJTalk and Beam-fix rows. kind: fixed (OpenJTalk wrong, Beam right), broken, or all."""
    ojt = {r["id"]: r for r in _load(Path(results_dir) / "ojt.jsonl")}
    out = []
    for f in _load(Path(results_dir) / f"{FIX_CONDITION}.jsonl"):
        o = ojt.get(f["id"])
        if o is None or (category != "all" and f["category"] != category):
            continue
        if kind == "fixed" and not (not o["match"] and f["match"]):
            continue
        if kind == "broken" and not (o["match"] and not f["match"]):
            continue
        out.append({"id": f["id"], "category": f["category"], "text": f["text"], "reading": f["reading"],
                    "reading_romaji": kana_to_romaji(f["reading"]),
                    "ojt": o["output"], "ojt_match": o["match"],
                    "ojt_html": gold_diff_html(f["reading"], f.get("reading_alts", []), o["output"], o["match"]),
                    "fix": f["output"], "fix_match": f["match"],
                    "fix_html": gold_diff_html(f["reading"], f.get("reading_alts", []), f["output"], f["match"])})
    return out


def _examples_html(rows: list[dict]) -> str:
    if not rows:
        return "<p>No examples for this choice.</p>"
    mark = lambda ok: "✅" if ok else "❌"  # noqa: E731
    body = "".join(
        f"<tr><td>{html.escape(r['id'])}</td><td>{html.escape(r['text'])}</td>"
        f"<td>{html.escape(r['reading'])}<br><code>{html.escape(r['reading_romaji'])}</code></td>"
        f"<td>{mark(r['ojt_match'])} {r['ojt_html']}</td><td>{mark(r['fix_match'])} {r['fix_html']}</td></tr>"
        for r in rows)
    return ("<table><thead><tr><th>id</th><th>text</th><th>correct reading</th><th>OpenJTalk</th>"
            "<th>Beam fix (low)</th></tr></thead><tbody>" + body + "</tbody></table>"
            "<p>Wrong readings are shown in the normalized form used for scoring (long vowels spelled out, punctuation removed); highlighted = differs from the nearest correct reading.</p>")


def _headline(summary: dict) -> str:
    conds, fx = summary["conditions"], summary["fix_vs_ojt"].get(FIX_CONDITION)
    if "ojt" not in conds or FIX_CONDITION not in conds or fx is None:
        return "_Results are not available yet._"
    o, f = conds["ojt"], conds[FIX_CONDITION]
    return (f"**OpenJTalk alone:** {o['exact']}/{o['n']} exactly right · "
            f"**OpenJTalk + Beam fix (low):** {f['exact']}/{f['n']} · "
            f"**Fixed:** {fx['fixed']} · **Broken:** {fx['broken']}")


def build_demo(results_dir: Path, beam: BeamClient | None) -> gr.Blocks:
    results_dir = Path(results_dir)
    summary = summarize(results_dir)
    # delete_cache: generated speech is removed from Gradio's temp files after 10 minutes
    with gr.Blocks(title="Beam Japanese reading fixer", delete_cache=(600, 600)) as demo:
        gr.Markdown("# Can Beam fix Japanese TTS misreadings?")
        gr.Markdown(INTRO)
        gr.Markdown(_headline(summary))
        with gr.Tabs():
            with gr.Tab("Results"):
                if figure_path() is not None:
                    gr.Image(str(figure_path()), show_label=False, interactive=False, buttons=[])
                gr.Markdown(render_markdown(summary) if summary["conditions"] else "_No results yet._")
            with gr.Tab("Examples"):
                with gr.Row():
                    cat = gr.Dropdown(list(CATEGORIES), value="all", label="Category")
                    kind = gr.Dropdown(list(KINDS), value="fixed", label="Show",
                                       info="fixed = OpenJTalk wrong, Beam right; broken = the reverse")
                table = gr.HTML(_examples_html(example_rows(results_dir, "all", "fixed")))
                for dd in (cat, kind):
                    dd.change(lambda c, k: _examples_html(example_rows(results_dir, c, k)), [cat, kind], table)
                if speech is not None:
                    _example_audio(results_dir)
            with gr.Tab("Try it"):
                box = gr.Textbox(label=f"Japanese text (up to {MAX_CHARS} characters)",
                                 value=TRY_IT_EXAMPLE, max_lines=4)
                go = gr.Button("Read it", variant="primary")
                out = gr.HTML()
                msg = gr.Markdown()
                go.click(lambda t: try_it(t, beam), box, [out, msg], concurrency_limit=1)
                if speech is not None:
                    _try_audio(box, beam)
        gr.Markdown(CREDITS)
        if speech is not None:
            gr.Markdown(speech.CREDIT)
    return demo


def _example_audio(results_dir: Path) -> None:
    rows = {r["id"]: r for r in example_rows(results_dir, "all", "all")}
    pick = gr.Dropdown(sorted(rows), label="Listen to an example (id)")
    with gr.Row():
        b1 = gr.Button("OpenJTalk reading")
        b2 = gr.Button("Beam-fixed reading")
    audio = gr.Audio(label="Speech (generated on demand, not for download)", type="numpy", buttons=[])
    b1.click(lambda i: speech.speak(rows[i]["text"]) if i in rows else None, pick, audio, concurrency_limit=1)
    b2.click(lambda i: speech.speak(rows[i]["fix"]) if i in rows else None, pick, audio, concurrency_limit=1)


def _try_audio(box, beam) -> None:
    with gr.Row():
        b1 = gr.Button("Speak OpenJTalk reading")
        b2 = gr.Button("Speak Beam-fixed reading")
    audio = gr.Audio(label="Speech (generated on demand, not for download)", type="numpy", buttons=[])

    def fixed_kana(text):
        if beam is None or not (text or "").strip() or len(text) > MAX_CHARS:
            return None
        draft = openjtalk.read(text)
        try:
            out = parse_response(beam.complete(build_messages("fix", "en", text, draft), "low", 0))["output"]
        except Exception:
            return None
        return speech.speak(out) if out else None

    b1.click(speech.speak, box, audio, concurrency_limit=1)
    b2.click(fixed_kana, box, audio, concurrency_limit=1)


if __name__ == "__main__":
    build_demo(default_results_dir(), default_beam()).launch()
