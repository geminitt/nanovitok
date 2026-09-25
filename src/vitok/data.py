"""Download FineWeb-2 Vietnamese and write every split used by the project.

    python -m vitok.data --out /kaggle/working/vitok-data

Output layout:
    tok_train.txt           NFC text for tokenizer training (one document per line block)
    shards/shard_NNNNN.parquet   nanochat pretraining shards ("text" column, NFC)
    shards/shard_99999.parquet   nanochat validation shard (nanochat uses the last file as val)
    test.jsonl              held-out documents from FineWeb-2's test split, char-truncated
    syllables.json          syllable counts from a sample of the pretraining text
    stats.json
"""

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "HuggingFaceFW/fineweb-2"
SUBSET = "data/vie_Latn"
SYLLABLE_RE = re.compile(r"[^\W\d_]+")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _digest(s: str) -> bytes:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=12).digest()


def iter_docs(split: str, cache_dir: Path, max_files: int | None = None):
    """Yield NFC documents from FineWeb-2 vie_Latn, downloading one parquet file at a time."""
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(f"{SUBSET}/{split}/") and f.endswith(".parquet"))
    for f in files[:max_files]:
        # local_dir (not cache_dir) stores a real file, so unlink() below actually frees disk
        path = Path(hf_hub_download(REPO, f, repo_type="dataset", local_dir=cache_dir))
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                yield _nfc(text)
        path.unlink()  # free disk: raw files are ~5GB each


def truncate_chars(text: str, max_chars: int) -> str:
    """Cut at the last whitespace before max_chars so no syllable is split."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    return text[: cut if cut > 0 else max_chars]


def eval_text(text: str, min_chars: int = 300, max_chars: int = 2500) -> str | None:
    """A document as the held-out sets use it: at least `min_chars` long, cut to `max_chars` at a space.

    Short enough to fit one context for every tokenizer, so no score depends on where a window falls.
    """
    return None if len(text) < min_chars else truncate_chars(text, max_chars)


class ShardWriter:
    def __init__(self, out_dir: Path, shard_bytes: int, row_group_docs: int = 1024):
        self.out_dir, self.shard_bytes, self.row_group_docs = out_dir, shard_bytes, row_group_docs
        self.idx, self.buf, self.buf_bytes, self.writer, self.cur_bytes = 0, [], 0, None, 0
        out_dir.mkdir(parents=True, exist_ok=True)

    def _flush_rows(self):
        if self.buf:
            if self.writer is None:
                path = self.out_dir / f"shard_{self.idx:05d}.parquet"
                self.writer = pq.ParquetWriter(path, pa.schema([("text", pa.string())]), compression="zstd")
            self.writer.write_table(pa.table({"text": self.buf}))
            self.buf = []

    def add(self, text: str):
        self.buf.append(text)
        n = len(text.encode("utf-8"))
        self.cur_bytes += n
        if len(self.buf) >= self.row_group_docs:
            self._flush_rows()
        if self.cur_bytes >= self.shard_bytes:
            self.close_shard()

    def close_shard(self):
        self._flush_rows()
        if self.writer is not None:
            self.writer.close()
            self.writer, self.cur_bytes = None, 0
            self.idx += 1

    def close(self):
        self.close_shard()


def write_single_shard(path: Path, docs: list[str]):
    pq.write_table(pa.table({"text": docs}), path, row_group_size=1024, compression="zstd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/hf_cache"))
    ap.add_argument("--tok-train-bytes", type=float, default=5e8)
    ap.add_argument("--val-docs", type=int, default=5000)
    ap.add_argument("--pretrain-bytes", type=float, default=1e10, help="UTF-8 bytes of pretraining text")
    ap.add_argument("--shard-bytes", type=float, default=5e8)
    ap.add_argument("--test-docs", type=int, default=2000)
    ap.add_argument("--test-max-chars", type=int, default=2500)
    ap.add_argument("--test-min-chars", type=int, default=300)
    ap.add_argument("--syllable-sample-bytes", type=float, default=2e8)
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seen = set()
    docs = iter_docs("train", args.cache_dir, args.max_files)
    stats = collections.Counter()

    # 1) tokenizer training corpus
    with open(args.out / "tok_train.txt", "w", encoding="utf-8") as f:
        for text in docs:
            seen.add(_digest(text))
            f.write(text.replace("\n\n\n", "\n\n") + "\n\n")
            stats["tok_train_bytes"] += len(text.encode("utf-8"))
            stats["tok_train_docs"] += 1
            if stats["tok_train_bytes"] >= args.tok_train_bytes:
                break

    # 2) nanochat validation shard (written last so it sorts after the train shards)
    val = []
    for text in docs:
        seen.add(_digest(text))
        val.append(text)
        if len(val) >= args.val_docs:
            break

    # 3) pretraining shards + syllable counts from the first part
    syllables = collections.Counter()
    writer = ShardWriter(args.out / "shards", int(args.shard_bytes))
    for text in docs:
        seen.add(_digest(text))
        writer.add(text)
        n = len(text.encode("utf-8"))
        stats["pretrain_bytes"] += n
        stats["pretrain_chars"] += len(text)
        stats["pretrain_docs"] += 1
        if stats["pretrain_bytes"] <= args.syllable_sample_bytes:
            syllables.update(m.lower() for m in SYLLABLE_RE.findall(text))
        if stats["pretrain_bytes"] >= args.pretrain_bytes:
            break
    writer.close()
    write_single_shard(args.out / "shards" / "shard_99999.parquet", val)
    stats["train_shards"] = writer.idx

    # 4) test set from FineWeb-2's own test split, deduplicated against everything above
    test = []
    for text in iter_docs("test", args.cache_dir):
        doc = eval_text(text, args.test_min_chars, args.test_max_chars)
        if doc is None:
            continue
        if _digest(text) in seen:
            stats["test_dups_dropped"] += 1
            continue
        test.append({"id": len(test), "text": doc})
        if len(test) >= args.test_docs:
            break
    with open(args.out / "test.jsonl", "w", encoding="utf-8") as f:
        for row in test:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["test_docs"] = len(test)

    (args.out / "syllables.json").write_text(
        json.dumps(dict(syllables.most_common()), ensure_ascii=False), encoding="utf-8")
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
