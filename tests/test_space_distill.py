import pytest

gr = pytest.importorskip("gradio")

from space_distill.app import MAX_CHARS, build_demo, read  # noqa: E402
from yomibench.systems import openjtalk  # noqa: E402


def fake_student(output, truncated=False):
    calls = []

    def run(prompts):
        calls.append(prompts)
        return [(output, truncated) for _ in prompts]

    run.calls = calls
    return run


def test_read_shows_openjtalk_and_the_student_reading():
    student = fake_student("コノカレーワカライ")
    html, msg, (draft, reading) = read("このカレーは辛い", student)
    assert draft == openjtalk.read("このカレーは辛い") and draft in html
    assert reading == "コノカレーワカライ" and "konokareewakarai" in html
    assert student.calls == [["文: このカレーは辛い\n下書き: " + draft + "\n読み: "]]
    assert "fallback" not in msg.lower()


def test_read_refuses_long_empty_and_non_japanese_text_without_calling_the_student():
    student = fake_student("ア")
    assert str(MAX_CHARS) in read("あ" * (MAX_CHARS + 1), student)[1]
    assert "Please enter" in read("  ", student)[1]
    assert "Japanese" in read("hello", student)[1]
    assert student.calls == []


def test_read_falls_back_to_openjtalk_when_the_student_writes_kanji():
    html, msg, (draft, reading) = read("このカレーは辛い", fake_student("辛い"))
    assert reading == draft and "fallback" in msg.lower()


def test_read_without_a_student_shows_openjtalk_only():
    html, msg, (draft, reading) = read("このカレーは辛い", None)
    assert draft in html and reading == "" and "not loaded" in msg


def test_read_keeps_the_page_usable_when_the_student_fails():
    def broken(prompts):
        raise RuntimeError("GPU task aborted")

    html, msg, (draft, reading) = read("このカレーは辛い", broken)
    assert draft in html and reading == "" and "RuntimeError" in msg


def test_build_demo_never_mentions_an_api_key():
    demo = build_demo(None)
    assert isinstance(demo, gr.Blocks)
    assert "REFLECTION_API_KEY" not in str(demo.config)


def test_load_student_returns_false_when_the_model_cannot_be_read(tmp_path):
    from space_distill import app

    assert app.load_student(str(tmp_path)) is False  # an empty folder holds no model
    assert app._student is None


def test_read_says_unchanged_when_only_punctuation_differs():
    draft = openjtalk.read("雨が降る、風が吹く")
    assert "、" in draft
    html, msg, (_, reading) = read("雨が降る、風が吹く", fake_student(draft.replace("、", "")))
    assert msg == "The student kept the OpenJTalk reading unchanged."
