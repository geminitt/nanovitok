"""Evaluation inputs that every condition is scored on, downloaded at pinned revisions.

    python -m vitok.heldout --out data/heldout

- `belebele_vie_Latn.jsonl`: Belebele's Vietnamese test items (`facebook/belebele`, `data/vie_Latn.jsonl`), the
  fields `vitok.score` reads, NFC.
- `en_docs.jsonl`: English documents from FineWeb `sample-10BT`, read in file order from the first row groups of
  its first parquet file (only those byte ranges are fetched), cut like the Vietnamese test set (`eval_text`: at
  least 300 characters, cut to 2,500 at a space), exact duplicates dropped.
- `heldout_manifest.json`: repositories, revisions, files, counts and the sha256 of each output.

The same revision and arguments give byte-identical files.
"""

import argparse
import hashlib
import json
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

from vitok.data import eval_text

BELEBELE_REPO, BELEBELE_FILE = "facebook/belebele", "data/vie_Latn.jsonl"
BELEBELE_REVISION = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
BELEBELE_ITEMS = 900
BELEBELE_FIELDS = ("flores_passage", "question", "mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4",
                   "correct_answer_num")
FINEWEB_REPO, FINEWEB_FILE = "HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet"
FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"


def belebele_items(records: Iterable[dict]) -> list[dict]:
    """The fields vitok.score reads, NFC; refuses items with a missing field or an answer number outside 1-4."""
    out = []
    for rec in records:
        missing = [f for f in BELEBELE_FIELDS if not isinstance(rec.get(f), str)]
        if missing:
            raise ValueError(f"Belebele item lacks {missing}: {str(rec)[:200]}")
        if rec["correct_answer_num"] not in ("1", "2", "3", "4"):
            raise ValueError(f"Belebele answer number {rec['correct_answer_num']!r} is not 1-4")
        out.append({f: unicodedata.normalize("NFC", rec[f]) for f in BELEBELE_FIELDS})
    return out


def english_docs(texts: Iterable[str], n: int, min_chars: int = 300, max_chars: int = 2500) -> list[str]:
    """The first n texts (NFC) that are long enough, cut like the test set, without exact duplicates."""
    out, seen = [], set()
    for text in texts:
        doc = eval_text(unicodedata.normalize("NFC", text), min_chars, max_chars)
        if doc is None or doc in seen:
            continue
        seen.add(doc)
        out.append(doc)
        if len(out) == n:
            return out
    raise ValueError(f"only {len(out)} usable documents, {n} needed")


def _fineweb_texts(revision: str) -> Iterator[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    with fs.open(f"datasets/{FINEWEB_REPO}@{revision}/{FINEWEB_FILE}", "rb") as f:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            yield from pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()


def _belebele_records(revision: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(BELEBELE_REPO, BELEBELE_FILE, repo_type="dataset", revision=revision)
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--en-docs", type=int, default=2000)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    items = belebele_items(_belebele_records(BELEBELE_REVISION))
    if len(items) != BELEBELE_ITEMS:
        raise SystemExit(f"Belebele vie_Latn has {len(items)} items, expected {BELEBELE_ITEMS}")
    en = english_docs(_fineweb_texts(FINEWEB_REVISION), args.en_docs)
    manifest = {
        "belebele": {"repo": BELEBELE_REPO, "file": BELEBELE_FILE, "revision": BELEBELE_REVISION, "items": len(items),
                     "sha256": write_jsonl(args.out / "belebele_vie_Latn.jsonl", items)},
        "en_docs": {"repo": FINEWEB_REPO, "file": FINEWEB_FILE, "revision": FINEWEB_REVISION, "docs": len(en),
                    "min_chars": 300, "max_chars": 2500,
                    "sha256": write_jsonl(args.out / "en_docs.jsonl", [{"text": t} for t in en])},
    }
    (args.out / "heldout_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
