"""Build tone minimal pairs: a test sentence vs. the same sentence with one syllable's tone swapped
to another real syllable ("má" -> "mà"). A model is right when it prefers the original.

    python -m vitok.minimal_pairs --test test.jsonl --syllables syllables.json --out minimal_pairs.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path

from vitok.text import TONE_MARKS, tone_of, with_tone

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[^\W\d_]+")


def build_pairs(docs: list[str], syllables: dict[str, int], n: int, min_count: int = 50,
                min_chars: int = 30, max_chars: int = 200, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    valid = {s for s, c in syllables.items() if c >= min_count}
    tones = [*TONE_MARKS, None]
    sentences = [s.strip() for d in docs for s in SENTENCE_RE.split(d) if min_chars <= len(s.strip()) <= max_chars]
    rng.shuffle(sentences)
    pairs = []
    for sent in sentences:
        spans = [m for m in WORD_RE.finditer(sent) if m.group().lower() in valid and tone_of(m.group())]
        rng.shuffle(spans)
        for m in spans:
            word = m.group()
            options = [w for t in tones if t != tone_of(word)
                       if (w := with_tone(word, t)) and w.lower() in valid and w.lower() != word.lower()]
            if options:
                swapped = rng.choice(options)
                pairs.append({"id": len(pairs), "good": sent,
                              "bad": sent[:m.start()] + swapped + sent[m.end():],
                              "from": word, "to": swapped})
                break
        if len(pairs) >= n:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--syllables", type=Path, required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    docs = [json.loads(line)["text"] for line in args.test.read_text(encoding="utf-8").splitlines()]
    syllables = json.loads(args.syllables.read_text(encoding="utf-8"))
    pairs = build_pairs(docs, syllables, args.n)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs to {args.out}")
    for p in pairs[:5]:
        print(f"  {p['from']} -> {p['to']}: {p['bad'][:80]}")


if __name__ == "__main__":
    main()
