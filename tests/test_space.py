import json

import pytest

gr = pytest.importorskip("gradio")

from space.app import MAX_CHARS, build_demo, example_rows, try_it  # noqa: E402
from yomibench.systems.beam import BeamClient  # noqa: E402


def _row(id_, cat, reading, output, match, **extra):
    return {"id": id_, "category": cat, "text": f"文{id_}", "reading": reading, "reading_alts": [],
            "output": output, "match": match, "cer": 0.0 if match else 0.5, "format_issue": False, **extra}


@pytest.fixture
def results_dir(tmp_path):
    ojt = [_row("number-0001", "number", "イチ", "ニ", False), _row("number-0002", "number", "サン", "サン", True),
           _row("place-0001", "place", "ハナテン", "ハナテン", True)]
    fix = [_row("number-0001", "number", "イチ", "イチ", True), _row("number-0002", "number", "サン", "シ", False),
           _row("place-0001", "place", "ハナテン", "ハナテン", True)]
    for name, rows in (("ojt", ojt), ("beam_fix-low-en-r0", fix)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                encoding="utf-8")
    return tmp_path


def test_try_it_rejects_bad_input_without_raising():
    assert "Please enter" in try_it("", None)[1]
    assert str(MAX_CHARS) in try_it("あ" * (MAX_CHARS + 1), None)[1]
    assert "Japanese" in try_it("hello world", None)[1]


def test_try_it_without_api_key_still_shows_openjtalk():
    html, msg = try_it("今日は市場に行く", None)
    assert "シジョー" in html and "Live Beam correction is disabled" in msg


def test_try_it_with_beam_shows_both_readings(fake_openai, tmp_path):
    beam = BeamClient(client=fake_openai, cache_dir=tmp_path)
    html, msg = try_it("放出", beam)
    assert "ハナテン" in html and "Beam" in html
    assert "Draft reading:" in fake_openai.last_request["messages"][1]["content"]
    assert fake_openai.last_request["reasoning_effort"] == "low"


def test_example_rows_filters_by_kind_and_category(results_dir):
    assert [r["id"] for r in example_rows(results_dir, "all", "fixed")] == ["number-0001"]
    assert [r["id"] for r in example_rows(results_dir, "all", "broken")] == ["number-0002"]
    assert [r["id"] for r in example_rows(results_dir, "place", "all")] == ["place-0001"]
    row = example_rows(results_dir, "number", "fixed")[0]
    assert row["reading_romaji"] == "ichi" and "<mark>" in row["ojt_html"] and "<mark>" not in row["fix_html"]


def test_build_demo_returns_blocks(results_dir):
    assert isinstance(build_demo(results_dir, None), gr.Blocks)


def test_build_demo_without_results(tmp_path):
    assert isinstance(build_demo(tmp_path, None), gr.Blocks)


def test_speech_buttons_and_no_download(results_dir):
    pytest.importorskip("piper_plus")
    demo = build_demo(results_dir, None)
    audios = [b for b in demo.blocks.values() if isinstance(b, gr.Audio)]
    labels = {b.value for b in demo.blocks.values() if isinstance(b, gr.Button)}
    assert len(audios) == 2 and all(not a.buttons for a in audios)
    assert {"OpenJTalk reading", "Beam-fixed reading"} <= labels
    assert any(isinstance(b, gr.Markdown) and "つくよみちゃん" in (b.value or "") for b in demo.blocks.values())


def test_space_client_gives_up_quickly(monkeypatch, tmp_path):
    from space import app
    monkeypatch.setenv("REFLECTION_API_KEY", "test-key")
    monkeypatch.setattr(app, "HERE", tmp_path)
    beam = app.default_beam()
    assert beam.client.timeout == 60 and beam.client.max_retries == 1


def test_try_it_accepts_halfwidth_katakana():
    html, msg = try_it("ｱｲｳ", None)
    assert "Japanese" not in msg and html


def test_space_keeps_no_visitor_text_or_audio_on_disk(monkeypatch, tmp_path, results_dir):
    from space import app
    monkeypatch.setenv("REFLECTION_API_KEY", "test-key")
    assert app.default_beam().cache_dir is None
    assert build_demo(results_dir, None).delete_cache == (600, 600)


def test_try_it_starts_with_a_sentence_beam_fixed_in_every_run(results_dir):
    from space.app import TRY_IT_EXAMPLE
    assert TRY_IT_EXAMPLE == "このカレーは辛い"


def test_try_it_says_when_the_daily_allowance_is_spent(fake_openai_daily_limit, tmp_path):
    html, msg = try_it("放出", BeamClient(client=fake_openai_daily_limit, cache_dir=tmp_path))
    assert "OpenJTalk" in html and "tomorrow" in msg


def test_try_it_says_beam_is_busy_on_a_plain_rate_limit(tmp_path):
    from tests.conftest import FakeOpenAI

    busy = FakeOpenAI(limit_after=0, error_body={"error": {"code": "rate_limit_exceeded"}})
    html, msg = try_it("放出", BeamClient(client=busy, cache_dir=tmp_path))
    assert "OpenJTalk" in html and "busy" in msg and "minute" in msg


def test_try_it_says_beam_was_too_slow_on_a_timeout(tmp_path):
    import httpx
    import openai

    class Slow:
        def __init__(self):
            self.chat = self.completions = self

        def create(self, **req):
            raise openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))

    html, msg = try_it("放出", BeamClient(client=Slow(), cache_dir=tmp_path))
    assert "OpenJTalk" in html and "too long" in msg


def test_results_tab_shows_the_figure_when_it_exists(results_dir):
    from space.app import figure_path

    assert figure_path() is not None  # hf/figures/results.png in the repository
    demo = build_demo(results_dir, None)
    assert any(isinstance(b, gr.Image) for b in demo.blocks.values())
