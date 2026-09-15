"""SuperBPE stage 2 without the forked `tokenizers` trainer.

The fork (alisawuffles/tokenizers-superbpe, `do_train_extend`) cannot finish stage 2 on 500MB with
4 CPUs / 30GB: stage-2 pretokens are whole phrases that almost never repeat, so its per-word
data structures hold the entire corpus, and it replays every inherited merge through the trainer.

This module computes the same result differently:

1. `encode_corpus`: apply the inherited merges by *encoding* each stage-2 pretoken with a BPE model
   holding only those merges. Applying merges lowest-rank-first equals replaying them in order,
   because a merge always ranks after the merges that built its parts.
2. `train_stage2`: continue BPE on the resulting id array with a doubly linked list, so a merge
   costs one scan for its left token plus work proportional to its occurrences. Selection follows
   the fork: highest count, ties to the smallest (left, right) id pair, skip tokens with more than
   4 words or containing ":Ġ", stop when the vocabulary is full.

Known deviation: the fork drops an inherited merge whose pair never occurs in the stage-2 pieces
("not found in queue"); here every inherited merge is kept. Notebook 01 compares both on a small
corpus before the full run.
"""

import time
from itertools import chain

import numpy as np
from tokenizers import Tokenizer, models

SEP = np.uint32(0xFFFFFFFF)  # between pretokens: pairs never cross it
HOLE = np.uint32(0xFFFFFFFE)  # right half of a merged pair
BANNED = np.int64(-(1 << 62))  # count of a pair whose token breaks the superword rules


def _key(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)


def encode_corpus(files: list[str], vocab: dict[str, int], merges: list[tuple[str, str]], pre_tokenizer,
                  batch_lines: int = 20000) -> np.ndarray:
    """Ids of every stage-2 pretoken after the inherited merges, separated by SEP.

    Lines are split on b"\\n" and keep it, like the HF trainer's `lines_with_ending`.
    """
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizer
    parts = []

    def flush(lines):
        encs = tok.encode_batch(lines, add_special_tokens=False)
        ids = np.fromiter(chain.from_iterable(e.ids for e in encs), dtype=np.uint32)
        words = np.fromiter(chain.from_iterable(e.word_ids for e in encs), dtype=np.int64)
        lens = np.fromiter((len(e.ids) for e in encs), dtype=np.int64, count=len(encs))
        line = np.repeat(np.arange(len(encs)), lens)
        starts = np.flatnonzero((np.diff(words) != 0) | (np.diff(line) != 0)) + 1
        parts.append(np.insert(ids, starts, SEP))
        parts.append(np.array([SEP], dtype=np.uint32))

    for path in files:
        with open(path, "rb") as f:
            batch = []
            for raw in f:
                batch.append(raw.decode("utf-8"))
                if len(batch) == batch_lines:
                    flush(batch)
                    batch = []
            if batch:
                flush(batch)
    return np.concatenate(parts) if parts else np.array([SEP], dtype=np.uint32)


class _PairCounts:
    """Pair counts: a sorted table for the pairs present at the start, a growable one for new pairs."""

    def __init__(self, tok: np.ndarray):
        left, right = tok[:-1], tok[1:]
        ok = (left < HOLE) & (right < HOLE)
        self.keys, counts = np.unique(_key(left[ok], right[ok]), return_counts=True)
        self.counts = counts.astype(np.int64)
        self.extra_keys = np.zeros(1024, dtype=np.uint64)
        self.extra_counts = np.zeros(1024, dtype=np.int64)
        self.n_extra = 0
        self.slot = {}

    def best(self) -> tuple[int, int]:
        """(key, count) of the most frequent pair; ties go to the smallest key."""
        best_key, best_count = 0, 0
        for keys, counts in ((self.keys, self.counts), (self.extra_keys[:self.n_extra], self.extra_counts[:self.n_extra])):
            if len(counts) == 0:
                continue
            m = counts.max()
            if m < 1:
                continue
            k = int(keys[counts == m].min())
            if m > best_count or (m == best_count and k < best_key):
                best_key, best_count = k, int(m)
        return best_key, best_count

    def _locate(self, key: int) -> tuple[np.ndarray, int]:
        i = int(np.searchsorted(self.keys, key))
        if i < len(self.keys) and self.keys[i] == key:
            return self.counts, i
        return self.extra_counts, self.slot[key]

    def ban(self, key: int):
        counts, i = self._locate(key)
        counts[i] = BANNED

    def update(self, keys: np.ndarray, deltas: np.ndarray):
        pos = np.minimum(np.searchsorted(self.keys, keys), max(len(self.keys) - 1, 0))
        found = self.keys[pos] == keys if len(self.keys) else np.zeros(len(keys), dtype=bool)
        self.counts[pos[found]] += deltas[found]
        for k, d in zip(keys[~found].tolist(), deltas[~found].tolist()):
            s = self.slot.get(k)
            if s is None:
                assert d > 0, "a pair can only appear by being created"
                if self.n_extra == len(self.extra_keys):
                    self.extra_keys = np.resize(self.extra_keys, 2 * self.n_extra)
                    self.extra_counts = np.resize(self.extra_counts, 2 * self.n_extra)
                s = self.slot[k] = self.n_extra
                self.extra_keys[s], self.extra_counts[s] = k, 0
                self.n_extra += 1
            self.extra_counts[s] += d


def _interleave(*columns: np.ndarray) -> np.ndarray:
    """Row-wise interleave of position columns that are sorted along the linked list, minus repeats.

    For occurrences k < k+1: prv[i_k] < i_k < j_k <= prv[i_k+1], so the interleaved sequence is
    already sorted and a repeat can only sit next to its twin (adjacent occurrences).
    """
    x = np.stack(columns, axis=1).ravel()
    return x[np.concatenate(([True], x[1:] != x[:-1]))]


def _pair_keys(tok, nxt, starts):
    starts = starts[starts >= 0]
    ends = nxt[starts]
    starts, ends = starts[ends >= 0], ends[ends >= 0]
    left, right = tok[starts], tok[ends]
    ok = (left < HOLE) & (right < HOLE)
    return _key(left[ok], right[ok])


def _merge(tok, nxt, prv, a: int, b: int, new: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge every (a, b) left to right in place; return the pair-count deltas."""
    i = np.flatnonzero(tok == a)
    j = nxt[i]
    ok = j >= 0
    i, j = i[ok], j[ok]
    ok = tok[j] == b
    i, j = i[ok], j[ok]
    if a == b and len(i) > 1:
        # "a a a": occurrences chain when one's right half is the next one's left half; keep every other
        linked = np.concatenate(([False], i[1:] == j[:-1]))
        chain_start = np.flatnonzero(~linked)
        offset = np.arange(len(i)) - chain_start[np.cumsum(~linked) - 1]
        i, j = i[offset % 2 == 0], j[offset % 2 == 0]
    if len(i) == 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)

    old = _pair_keys(tok, nxt, _interleave(prv[i], i, j))
    after = nxt[j]
    tok[i], tok[j] = new, HOLE
    nxt[i] = after
    prv[after[after >= 0]] = i[after >= 0]
    fresh = _pair_keys(tok, nxt, _interleave(prv[i], i))

    keys, inverse = np.unique(np.concatenate((old, fresh)), return_inverse=True)
    sign = np.concatenate((np.full(len(old), -1), np.ones(len(fresh)))).astype(np.int64)
    deltas = np.bincount(inverse, weights=sign, minlength=len(keys)).astype(np.int64)
    return keys[deltas != 0], deltas[deltas != 0]


def _breaks_rules(token: str, max_words: int) -> bool:
    """The fork skips tokens with more than `max_words` words or containing ":Ġ"."""
    return ":Ġ" in token or sum(1 for w in token.split("Ġ") if w) > max_words


def train_stage2(ids: np.ndarray, vocab: dict[str, int], vocab_size: int, max_words: int = 4,
                 log_every: int = 100) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Continue BPE on `ids` (from `encode_corpus`) until the vocabulary has `vocab_size` tokens.

    Returns the full vocabulary and the new merges only.
    """
    vocab = dict(vocab)
    id_to_token = {i: t for t, i in vocab.items()}
    tok = ids.copy()
    n = len(tok)
    nxt = np.arange(1, n + 1, dtype=np.int64)
    nxt[-1] = -1
    prv = np.arange(-1, n - 1, dtype=np.int64)
    counts = _PairCounts(tok)
    merges = []
    start = time.time()
    while len(vocab) < vocab_size:
        key, count = counts.best()
        if count < 1:
            break
        a, b = key >> 32, key & 0xFFFFFFFF
        token = id_to_token[a] + id_to_token[b]
        if _breaks_rules(token, max_words):
            counts.ban(key)
            continue
        new = vocab.setdefault(token, len(vocab))
        id_to_token[new] = token
        merges.append((id_to_token[a], id_to_token[b]))
        counts.update(*_merge(tok, nxt, prv, a, b, new))
        if log_every and len(merges) % log_every == 0:
            print(f"stage 2: {len(merges)} merges, vocab {len(vocab)}/{vocab_size}, last {token!r} x{count}, "
                  f"{time.time() - start:.0f}s", flush=True)
    return vocab, merges
