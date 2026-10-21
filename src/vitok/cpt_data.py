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
    input_ids: np.ndarray    # (batch, seq_len) int64
    chars: int               # characters of text the batch's tokens cover
    state: dict              # everything needed to resume right after this batch (documents read, leftover tokens)
    token_chars: np.ndarray  # (batch, seq_len) characters each token adds (eos: 0); sums to `chars`


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
        per_token = np.asarray(chars_buf[:need], dtype=np.int64).reshape(batch, seq_len)
        del ids_buf[:need], chars_buf[:need]
        return Batch(rows, int(per_token.sum()), {"docs_read": read, "ids": list(ids_buf), "chars": list(chars_buf)},
                     per_token)

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


def jsonl_rows(path: Path) -> list[dict]:
    """Records of a JSONL file, split at newlines only: str.splitlines() also breaks at U+0085, U+2028 and other
    separators that JSON keeps raw inside strings (one Global-MMLU question holds a U+0085)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_texts(path: Path) -> list[str]:
    return [r["text"] for r in jsonl_rows(path)]


# ---- document-aligned steps (Token Distillation study): every condition reads the same text at every step ----

@dataclass
class Step:
    pieces: list[tuple[str, bool]]  # (text, whether it ends its document), in stream order
    chars: int
    domain_chars: dict[str, int]
    digest: str                     # chain digest of every step's text up to and including this one
    state: dict                     # resume position after this step: next record, characters of it read, digest


CUT_WINDOW = 4096  # characters before the limit searched for a line break or space (a step stays >= 98% full)


def cut_point(text: str, limit: int) -> int:
    """Characters of `text` that fit in `limit`: all of it, else up to the last line break in the last
    min(CUT_WINDOW, limit // 2) characters of the room, else the last space there, else exactly `limit` (one rule
    for every tokenizer, so steps match)."""
    if limit < 1:
        raise ValueError(f"no room for a piece ({limit} characters)")
    if len(text) <= limit:
        return len(text)
    start = limit - min(CUT_WINDOW, limit // 2)
    for sep in ("\n", " "):
        i = text.rfind(sep, start, limit)
        if i >= 0:
            return i + 1
    return limit


def step_digest(previous: str, pieces: list[tuple[str, bool]]) -> str:
    """The chain digest after a step: equal digests at step k mean the same text was read at every step up to k."""
    h = hashlib.blake2b(bytes.fromhex(previous), digest_size=16)
    for text, ends in pieces:
        h.update(text.encode("utf-8") + (b"\x00" if ends else b"\x01"))
    return h.hexdigest()


def aligned_steps(records: Iterable[dict], chars_per_step: int, state: dict | None = None) -> Iterator[Step]:
    """Cut a stream of {"text"[, "domain"]} records into steps of at most `chars_per_step` characters, without
    tokenizing: a document that does not fit is cut (`cut_point`) and continues in the next step. `records` must
    start at `state["record"]`; a final step that is not full is not yielded."""
    if chars_per_step < 1:
        raise ValueError(f"chars_per_step must be positive, got {chars_per_step}")
    state = state or {"record": 0, "offset": 0, "digest": ""}
    index, offset, chain = state["record"], state["offset"], state["digest"]
    pieces: list[tuple[str, bool]] = []
    chars = 0
    domains: dict[str, int] = {}
    for rec in records:
        text, domain = rec["text"][offset:], rec.get("domain") or ""
        if not text:
            raise ValueError(f"record {index} is empty at offset {offset}")
        while text:
            n = cut_point(text, chars_per_step - chars)
            pieces.append((text[:n], n == len(text)))
            chars += n
            offset += n
            domains[domain] = domains.get(domain, 0) + n
            text = text[n:]
            if text or chars == chars_per_step:  # a cut means the step is full
                chain = step_digest(chain, pieces)
                nxt = {"record": index, "offset": offset} if text else {"record": index + 1, "offset": 0}
                yield Step(pieces, chars, domains, chain, nxt | {"digest": chain})
                pieces, chars, domains = [], 0, {}
        index, offset = index + 1, 0


def step_rows(tokenizer, pieces: list[tuple[str, bool]], eos_id: int, seq_len: int
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(input_ids, labels, token_chars) of one step, each (rows, seq_len): the pieces' tokens, eos after a piece that
    ends its document, cut into rows; the last row is padded with eos and label -100 (not predicted)."""
    if seq_len < 2:
        raise ValueError(f"seq_len {seq_len} must be at least 2")
    ids: list[int] = []
    chars: list[int] = []
    for (_, ends), enc in zip(pieces, tokenizer.encode_batch([p for p, _ in pieces], add_special_tokens=False),
                              strict=True):
        ids += enc.ids
        chars += token_chars(enc.offsets)
        if ends:
            ids.append(eos_id)
            chars.append(0)
    rows = -(-len(ids) // seq_len)
    pad = rows * seq_len - len(ids)
    x = np.asarray(ids + [eos_id] * pad, dtype=np.int64).reshape(rows, seq_len)
    labels = x.copy()
    labels.reshape(-1)[len(ids):] = -100
    per_token = np.asarray(chars + [0] * pad, dtype=np.int64).reshape(rows, seq_len)
    assert int(per_token.sum()) == sum(len(p) for p, _ in pieces), "token characters do not add up to the step's"
    return x, labels, per_token
