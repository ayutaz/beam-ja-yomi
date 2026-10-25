from yomibench.corpus import LeakFilter, build_pool, keep_sentence, split_sentences


def test_split_sentences():
    assert split_sentences("東京は晴れ。大阪は雨！") == ["東京は晴れ。", "大阪は雨！"]


def test_split_sentences_on_newlines_and_strips():
    assert split_sentences("第一段落です。\n\n 第二段落です？") == ["第一段落です。", "第二段落です？"]


def test_keep_sentence():
    assert keep_sentence("昨日は3.5kgの荷物を運んで駅まで歩いた。")
    assert not keep_sentence("短い文。")
    assert not keep_sentence("Pythonはプログラミング言語である。")
    assert not keep_sentence("東京（とうきょう）は日本の首都である。")
    assert not keep_sentence("ひらがなだけでかかれたながいぶんしょうです。")
    assert not keep_sentence("詳細は http://example.jp を参照のこと。")
    assert not keep_sentence("長" * 61 + "。")


def test_leak_exact_after_nfkc():
    leak = LeakFilter(["体重が2.5kg増えた"])
    assert leak.leaks("体重が２．５ｋｇ増えた。")


def test_leak_content_ngram():
    leak = LeakFilter(["下手から役者が退場する"])
    assert leak.leaks("舞台の下手から役者が出た。")


def test_kana_only_ngram_is_not_a_leak():
    leak = LeakFilter(["彼は毎日走っています"])
    assert not leak.leaks("私は図書館で本を読んでいます。")


def test_leak_place():
    leak = LeakFilter(["斜里郡小清水町共和に向かいます"])
    assert leak.leaks("小清水町共和の農場。")
    assert not leak.leaks("共和国の首都である。")


def _articles():
    return [
        {"id": "1", "title": "甲", "url": "https://ja.wikipedia.org/wiki/甲",
         "text": "甲は日本の地名である。甲の人口は千人である。甲には駅が一つある。下手から役者が退場する場面がある。"},
        {"id": "2", "title": "乙", "url": "https://ja.wikipedia.org/wiki/乙",
         "text": "乙は東京都にある寺院である。乙の本堂は国の重要文化財である。"},
    ]


def test_build_pool_deterministic_and_capped():
    leak = LeakFilter(["下手から役者が退場する"])
    a = build_pool(_articles(), leak, n=10, seed=1, per_article=2)
    b = build_pool(_articles(), leak, n=10, seed=1, per_article=2)
    assert a == b
    assert sum(r["title"] == "甲" for r in a) <= 2
    assert all("下手から役者" not in r["text"] for r in a)
    assert all(set(r) == {"id", "text", "title", "url"} for r in a)
    assert [r["id"] for r in a] == [f"wiki-{i:06d}" for i in range(1, len(a) + 1)]


def test_build_pool_stats():
    leak = LeakFilter(["下手から役者が退場する"])
    stats = {}
    build_pool(_articles(), leak, n=10, seed=1, per_article=5, stats=stats)
    assert stats["articles"] == 2
    assert stats["leak"] == 1


def test_keep_sentence_rejects_list_fragments():
    assert not keep_sentence("アメリカ合衆国の男子バスケットボール選手")
    assert not keep_sentence("玉樹チベット族自治州 - 中華人民共和国青海省の自治州。")
