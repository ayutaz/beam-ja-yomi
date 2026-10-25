"""Sentences from Japanese Wikipedia for the teacher, with sentences close to the evaluation sets removed.

Rules (plan 2026-10-24, Task 2): 15-60 characters after NFKC, at least one kanji, at most three ASCII
letters, no brackets, no URLs or wiki markup, and it must end with 。！？ and hold no " - " (list
items in articles look like sentences otherwise). A sentence leaks when, after NFKC and dropping spaces and
punctuation, it equals an evaluation sentence or shares a character 5-gram that holds two or more kanji
or digits with one (kana-only 5-grams such as しています are too common to count).
"""

import random
import re
import unicodedata
from collections.abc import Iterable

MIN_CHARS, MAX_CHARS = 15, 60
NGRAM = 5
_SENTENCE_END = re.compile(r"(?<=[。！？!?])|\n")
_KANJI = re.compile(r"[一-鿿々〆]")
_CONTENT = re.compile(r"[一-鿿々〆0-9]")
_ASCII_LETTER = re.compile(r"[A-Za-z]")
_REJECT = re.compile(r"[（）()\[\]「」『』|=]|http| - ")
_DROP = re.compile(r"[\s\W_]")  # spaces, punctuation and symbols; \w keeps kana, kanji, letters, digits


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def keep_sentence(s: str) -> bool:
    s = unicodedata.normalize("NFKC", s)
    return (MIN_CHARS <= len(s) <= MAX_CHARS and _KANJI.search(s) is not None
            and len(_ASCII_LETTER.findall(s)) <= 3 and _REJECT.search(s) is None and s[-1] in "。！？!?")


def _canon(s: str) -> str:
    return _DROP.sub("", unicodedata.normalize("NFKC", s))


def _content_ngrams(s: str) -> set[str]:
    grams = (s[i:i + NGRAM] for i in range(len(s) - NGRAM + 1))
    return {g for g in grams if len(_CONTENT.findall(g)) >= 2}


class LeakFilter:
    def __init__(self, eval_texts: list[str]):
        canon = [_canon(t) for t in eval_texts]
        self.exact = set(canon)
        self.grams = set().union(*(_content_ngrams(c) for c in canon)) if canon else set()

    def leaks(self, s: str) -> bool:
        c = _canon(s)
        return c in self.exact or not self.grams.isdisjoint(_content_ngrams(c))


def build_pool(articles: Iterable[dict], leak: LeakFilter, n: int, seed: int = 20261024,
               per_article: int = 2, stats: dict | None = None) -> list[dict]:
    """Up to `per_article` kept, non-leaking sentences per article, then `n` of them in a seeded order."""
    rng = random.Random(seed)
    counts = {"articles": 0, "sentences": 0, "kept": 0, "filtered": 0, "leak": 0}
    picked = []
    for art in articles:
        counts["articles"] += 1
        good = []
        for s in split_sentences(art["text"]):
            counts["sentences"] += 1
            if not keep_sentence(s):
                counts["filtered"] += 1
            elif leak.leaks(s):
                counts["leak"] += 1
            else:
                good.append(s)
        counts["kept"] += len(good)
        rng.shuffle(good)
        picked += [{"text": s, "title": art["title"], "url": art["url"]} for s in good[:per_article]]
    rng.shuffle(picked)
    if stats is not None:
        stats.update(counts, candidates=len(picked))
    return [{"id": f"wiki-{i:06d}", **row} for i, row in enumerate(picked[:n], 1)]
