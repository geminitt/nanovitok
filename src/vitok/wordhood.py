"""H4: do SuperBPE's superwords line up with Vietnamese word boundaries?

    python -m vitok.wordhood --tokenizers tokenizers-16k --docs test.jsonl --out results/wordhood.json

A superword is a token that covers more than one syllable. For every superword occurrence we ask
whether the text it covers is exactly a multi-syllable word according to underthesea's segmenter.

Two baselines, both over runs of adjacent syllables of the same length in the same documents:
- `baseline_rate`: every run. Too easy a bar: superwords are the most frequent runs, and a frequent
  run is more often a word, so frequency alone beats this baseline.
- `frequency_matched_baseline` (the pre-registered one): the N most frequent run types, N being
  the number of distinct superword types of that length. It asks whether SuperBPE picks words more
  often than simply taking the most frequent syllable runs would.
"""

import argparse
import collections
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

from vitok.tokenizer_spec import CONDITIONS

SYLLABLE_RE = re.compile(r"[^\W\d_]+")


def word_spans(text: str) -> dict[int, set[tuple[int, int]]] | None:
    """Character spans of the words underthesea finds, keyed by syllable count.

    The segmenter normalizes whitespace, so a word is matched back with `\\s+` between its syllables.
    A word that still cannot be found is skipped; if more than a fifth of them are, the document is
    unalignable and None is returned so the caller can drop it.
    """
    from underthesea import word_tokenize

    spans, pos, missed, words = collections.defaultdict(set), 0, 0, word_tokenize(text)
    for word in words:
        pattern = r"\s+".join(re.escape(p) for p in word.split())
        m = re.compile(pattern).search(text, pos) if pattern else None
        if m is None:
            missed += 1
            continue
        spans[len(SYLLABLE_RE.findall(word))].add(m.span())
        pos = m.end()
    return None if missed > len(words) / 5 else spans


def syllable_runs(text: str, k: int) -> list[tuple[int, int]]:
    """Spans of k adjacent syllables separated by single spaces — the candidates a superword competes with."""
    assert k >= 2, "a superword covers at least two syllables"
    syl = [m.span() for m in SYLLABLE_RE.finditer(text)]
    runs = []
    for i in range(len(syl) - k + 1):
        window = syl[i:i + k]
        if all(text[a[1]:b[0]] == " " for a, b in zip(window, window[1:])):
            runs.append((window[0][0], window[-1][1]))
    return runs


def superword_spans(tok: Tokenizer, text: str) -> list[tuple[int, int]]:
    """Spans of the tokens that cover more than one syllable, trimmed to the text they actually cover."""
    out = []
    for start, end in tok.encode(text, add_special_tokens=False).offsets:
        piece = text[start:end]
        trimmed = piece.strip()
        if not trimmed or " " not in trimmed:
            continue
        start += len(piece) - len(piece.lstrip())
        end -= len(piece) - len(piece.rstrip())
        out.append((start, end))
    return out


def frequency_matched_rate(run_count: collections.Counter, run_hit: collections.Counter, n_types: int) -> float | None:
    """Share of the occurrences of the `n_types` most frequent runs that are exactly one word."""
    top = [key for key, _ in run_count.most_common(n_types)]
    occ = sum(run_count[key] for key in top)
    return sum(run_hit[key] for key in top) / occ if occ else None


SYLLABLES = (2, 3, 4)  # the fork caps a token at 4 words, and a Vietnamese word is written as syllables


def measure(tok: Tokenizer, docs: list[str]) -> dict:
    hit, seen = collections.Counter(), collections.Counter()
    base_hit, base_seen = collections.Counter(), collections.Counter()
    superword_types = collections.defaultdict(set)  # k -> distinct superword texts (lowercased)
    run_count = collections.defaultdict(collections.Counter)  # k -> run text -> occurrences
    run_hit = collections.defaultdict(collections.Counter)  # k -> run text -> occurrences that are one word
    skipped = other = 0
    for text in docs:
        words = word_spans(text)
        if words is None:
            skipped += 1
            continue
        for span in superword_spans(tok, text):
            k = len(SYLLABLE_RE.findall(text[span[0]:span[1]]))
            if k not in SYLLABLES:  # e.g. a token that merges a word with punctuation
                other += 1
                continue
            seen[k] += 1
            hit[k] += span in words.get(k, ())
            superword_types[k].add(text[span[0]:span[1]].lower())
        for k in SYLLABLES:  # the baselines depend on the text only, not on the tokenizer
            for run in syllable_runs(text, k):
                is_word = run in words.get(k, ())
                base_seen[k] += 1
                base_hit[k] += is_word
                key = text[run[0]:run[1]].lower()
                run_count[k][key] += 1
                run_hit[k][key] += is_word

    matched = lambda k: frequency_matched_rate(run_count[k], run_hit[k], len(superword_types[k]))

    total = sum(seen.values())
    weight = lambda k: seen[k] / total

    def weighted(rate):
        return sum(weight(k) * rate(k) for k in seen if rate(k) is not None) if total else 0.0

    all_runs = lambda k: base_hit[k] / base_seen[k] if base_seen[k] else None
    return {
        "superword_occurrences": total,
        "superwords_outside_2_4_syllables": other,
        "share_outside_2_4": other / (total + other) if total + other else 0.0,
        "match_rate": sum(hit.values()) / total if total else 0.0,
        "baseline_rate": weighted(all_runs),
        "frequency_matched_baseline": weighted(matched),
        "by_syllables": {k: {"occurrences": seen[k], "types": len(superword_types[k]), "match_rate": hit[k] / seen[k],
                             "baseline_rate": all_runs(k), "frequency_matched_baseline": matched(k)}
                         for k in sorted(seen)},
        "docs_skipped": skipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", type=Path, required=True)
    ap.add_argument("--docs", type=Path, required=True, help="test.jsonl")
    ap.add_argument("--n-docs", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    docs = [json.loads(l)["text"] for l in args.docs.read_text(encoding="utf-8").splitlines()][:args.n_docs]
    result = {"n_docs": len(docs)}
    print(f"{'condition':12s} {'superwords':>11s} {'match word':>11s} {'freq-matched':>13s} {'all runs':>9s}")
    for cond in CONDITIONS:
        tok = Tokenizer.from_file(str(args.tokenizers / cond / "tokenizer.json"))
        r = measure(tok, docs)
        result[cond] = r
        print(f"{cond:12s} {r['superword_occurrences']:11,d} {r['match_rate']:11.1%} "
              f"{r['frequency_matched_baseline']:13.1%} {r['baseline_rate']:9.1%}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
