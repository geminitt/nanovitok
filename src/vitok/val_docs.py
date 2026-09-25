"""The val shard as an evaluation set, prepared exactly like test.jsonl.

    python -m vitok.val_docs --shard vitok-data/shards/shard_99999.parquet --out val_docs.jsonl

nanochat keeps this shard out of training (it is the last parquet file). Scoring it per document,
like the test set, is the pre-registered sensitivity check: does the H1 difference keep its sign on
documents from FineWeb-2's train split instead of its test split?
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from vitok.data import eval_text


def val_docs(shard: Path, min_chars: int = 300, max_chars: int = 2500) -> list[dict]:
    texts = pq.read_table(shard, columns=["text"]).column("text").to_pylist()
    docs = (eval_text(t, min_chars, max_chars) for t in texts)
    return [{"id": i, "text": d} for i, d in enumerate(d for d in docs if d is not None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    docs = val_docs(args.shard)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} documents to {args.out}")


if __name__ == "__main__":
    main()
