"""Training data for adaptation: FineWeb-2 Vietnamese, deduplicated against the evaluation and merge-learning text,
tokenized per condition and packed into fixed-length sequences.

Every condition reads the same documents in the same order, so they differ only in how the text is cut. A step is
`batch` sequences of `seq_len` tokens for every condition (equal compute); a condition whose tokens cover more
characters therefore reads more text per step, and each batch records how many characters its tokens cover, so
results can be plotted against compute and against text read.
"""

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vitok.data import eval_text


def digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).digest()


def held_out_digests(texts: Iterable[str]) -> set[bytes]:
    """Digests of the held-out texts (test documents, merge-learning documents), as stored: NFC and cut."""
    return {digest(unicodedata.normalize("NFC", t)) for t in texts}


def training_docs(docs: Iterable[str], held_out: set[bytes]) -> Iterator[str]:
    """NFC documents, skipping any whose full text or held-out form (cut like the test set) is held out."""
    for doc in docs:
        doc = unicodedata.normalize("NFC", doc)
        cut = eval_text(doc)
        if digest(doc) in held_out or (cut is not None and digest(cut) in held_out):
            continue
        yield doc


@dataclass
class Batch:
    input_ids: np.ndarray  # (batch, seq_len) int64
    chars: int             # characters of text the batch's tokens cover
    state: dict            # everything needed to resume right after this batch (documents read, leftover tokens)


def token_chars(offsets: list[tuple[int, int]]) -> list[int]:
    """Characters each token adds, counting every character once (byte pieces of one character share its span)."""
    out, seen = [], 0
    for start, end in offsets:
        out.append(max(0, end - max(start, seen)))
        seen = max(seen, end)
    return out


def packed_batches(tokenizer, docs: Iterable[str], eos_id: int, seq_len: int, batch: int,
                   encode_batch: int = 256, state: dict | None = None) -> Iterator[Batch]:
    """Concatenate documents (each followed by eos) and cut the stream into batches of seq_len-token rows.

    `docs` must start after the documents `state` says were read; the leftover tokens in `state` come first, so a
    resumed stream yields exactly the batches the uninterrupted one would have.
    """
    if seq_len < 2 or batch < 1:
        raise ValueError(f"seq_len {seq_len} and batch {batch} must be at least 2 and 1")
    need = seq_len * batch
    state = state or {"docs_read": 0, "ids": [], "chars": []}
    ids_buf, chars_buf = list(state["ids"]), list(state["chars"])
    read = state["docs_read"]
    pending: list[str] = []

    def flush():
        nonlocal read
        for enc in tokenizer.encode_batch(pending, add_special_tokens=False):
            ids_buf.extend(enc.ids)
            chars_buf.extend(token_chars(enc.offsets))
            ids_buf.append(eos_id)
            chars_buf.append(0)
        read += len(pending)
        pending.clear()

    def emit():
        rows = np.asarray(ids_buf[:need], dtype=np.int64).reshape(batch, seq_len)
        chars = int(sum(chars_buf[:need]))
        del ids_buf[:need], chars_buf[:need]
        return Batch(rows, chars, {"docs_read": read, "ids": list(ids_buf), "chars": list(chars_buf)})

    while len(ids_buf) >= need:
        yield emit()
    for doc in docs:
        pending.append(doc)
        if len(pending) == encode_batch:
            flush()
            while len(ids_buf) >= need:
                yield emit()
    if pending:
        flush()
    while len(ids_buf) >= need:
        yield emit()


def parquet_texts(paths: Iterable[Path]) -> Iterator[str]:
    """The "text" column of parquet shards, file by file in the given order, row group by row group."""
    import pyarrow.parquet as pq

    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            yield from pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()


def jsonl_texts(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]
