"""Compression statistics of the trained tokenizers (no model needed).

    python -m vitok.compression --tokenizers tokenizers --docs shards/shard_99999.parquet --out compression.json
"""

import argparse
import collections
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

from vitok.tokenizer_spec import CONDITIONS

SYLLABLE_RE = re.compile(r"[^\W\d_]+")
# GPT-2 byte-level alphabet encodes a space as "Ġ"
SPACE = "Ġ"


def stats_for(tok: Tokenizer, docs: list[str], top_k: int = 30) -> dict:
    enc = tok.encode_batch(docs, add_special_tokens=False)
    n_tokens = sum(len(e.ids) for e in enc)
    n_chars = sum(len(d) for d in docs)
    n_syll = sum(len(SYLLABLE_RE.findall(d)) for d in docs)

    vocab = tok.get_vocab(with_added_tokens=False)
    superword_ids = {i for t, i in vocab.items() if SPACE in t.strip(SPACE)}
    used = collections.Counter(i for e in enc for i in e.ids)
    super_uses = sum(c for i, c in used.items() if i in superword_ids)
    top = [(tok.decode([i]), c) for i, c in used.most_common() if i in superword_ids][:top_k]
    return {
        "tokens": n_tokens,
        "chars_nfc": n_chars,
        "chars_per_token": n_chars / n_tokens,
        "tokens_per_syllable": n_tokens / max(n_syll, 1),
        "superword_vocab": len(superword_ids),
        "superword_token_share": super_uses / n_tokens,
        "top_superwords": top,
        "vocab_size": len(vocab),
    }


def load_docs(path: Path, n: int) -> list[str]:
    return pq.read_table(path, columns=["text"]).column("text").to_pylist()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", type=Path, required=True)
    ap.add_argument("--docs", type=Path, required=True, help="parquet with a text column (use the val shard)")
    ap.add_argument("--n-docs", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    docs = load_docs(args.docs, args.n_docs)
    result = {}
    for cond in CONDITIONS:
        tok = Tokenizer.from_file(str(args.tokenizers / cond / "tokenizer.json"))
        result[cond] = stats_for(tok, docs)
    for norm in ("nfc", "nfd"):
        b, s = result[f"bpe-{norm}"], result[f"super-{norm}"]
        result[f"token_reduction_{norm}"] = 1 - s["tokens"] / b["tokens"]
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for cond in CONDITIONS:
        r = result[cond]
        print(f"{cond:10s} chars/token={r['chars_per_token']:.3f} tokens/syllable={r['tokens_per_syllable']:.3f} "
              f"superword share={r['superword_token_share']:.1%}")
    for norm in ("nfc", "nfd"):
        red = result[f"token_reduction_{norm}"]
        print(f"SuperBPE token reduction ({norm}): {red:.1%}  -> Gate 1 {'PASS' if red >= 0.15 else 'FAIL'} (>=15%)")


if __name__ == "__main__":
    main()
