import json
import unicodedata

import numpy as np
import pytest

from tokenizers import pre_tokenizers

from conftest import SENTENCES
from vitok import superbpe
from vitok.tokenizer_spec import STAGE1_REGEX, STAGE2_REGEX
from vitok.train_tokenizers import _pre_tokenizer, _train

# repeated tokens exercise the "a a a" overlap rule; ": " exercises the ":Ġ" rule
# "zzz" holds exactly two overlapping (z, z) pairs
EXTRA = ["ha ha ha ha ha ha ha", "aaaaaaa bbbb aaaaaaa", "Ghi chú: xem thêm. Ghi chú: xem thêm.", "zzz qq"]


def reference_stage2(lines, vocab, inherited, vocab_size, max_words=4):
    """Slow, literal version of the fork's do_train_extend: replay merges word by word, then
    recount every pair each step (highest count, ties to the smallest id pair)."""
    pre = _pre_tokenizer(STAGE2_REGEX)
    vocab = dict(vocab)
    id_to_token = {i: t for t, i in vocab.items()}
    words = [[vocab[c] for c in piece] for line in lines for piece, _ in pre.pre_tokenize_str(line)]

    def merge_word(w, a, b, new):  # Word::merge: left to right, merged symbol can't match again
        out, i = [], 0
        while i < len(w):
            if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                out.append(new)
                i += 2
            else:
                out.append(w[i])
                i += 1
        return out

    for a, b in inherited:
        words = [merge_word(w, vocab[a], vocab[b], vocab[a + b]) for w in words]

    merges, banned = [], set()
    while len(vocab) < vocab_size:
        counts = {}
        for w in words:
            for p in zip(w, w[1:]):
                counts[p] = counts.get(p, 0) + 1
        counts = {p: c for p, c in counts.items() if p not in banned}
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        token = id_to_token[pair[0]] + id_to_token[pair[1]]
        if ":Ġ" in token or len([x for x in token.split("Ġ") if x]) > max_words:
            banned.add(pair)
            continue
        new = vocab.setdefault(token, len(vocab))
        id_to_token[new] = token
        merges.append((id_to_token[pair[0]], id_to_token[pair[1]]))
        words = [merge_word(w, *pair, new) for w in words]
    return vocab, merges


@pytest.mark.parametrize("norm", ["NFC", "NFD"])
@pytest.mark.parametrize("target", [700, 100_000], ids=["vocab-700", "until-no-pair-is-left"])
@pytest.mark.parametrize("capacity", [1024, 2], ids=["room", "table-grows"])
def test_fast_stage2_matches_reference(tmp_path, monkeypatch, norm, target, capacity):
    # the reference recounts every pair after every merge; the fast version must give the same merges, in the same
    # order, whether it stops at the vocabulary size or runs out of pairs (counts down to 1), and whether its table of
    # new pairs fits or has to grow
    monkeypatch.setattr(superbpe, "EXTRA_CAPACITY", capacity)
    lines = [unicodedata.normalize(norm, s) + "\n" for s in (SENTENCES + EXTRA) * 3]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(lines), encoding="utf-8")
    _train(tmp_path, [str(corpus)], 400, STAGE1_REGEX)
    merges = [tuple(m.split(" ")) for m in (tmp_path / "merges.txt").read_text(encoding="utf-8").splitlines()[1:]]
    vocab = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))
    n_inherit = 100
    inherited = merges[:n_inherit]
    inherited_vocab = {t: i for t, i in vocab.items() if i < 256 + n_inherit}

    ids = superbpe.encode_corpus([str(corpus)], inherited_vocab, inherited, _pre_tokenizer(STAGE2_REGEX))
    got_vocab, got_merges = superbpe.train_stage2(ids, inherited_vocab, target, log_every=0)
    want_vocab, want_merges = reference_stage2(lines, inherited_vocab, inherited, target)

    assert got_merges == want_merges
    assert got_vocab == want_vocab
    assert any("Ġ" in t.strip("Ġ") for t in got_vocab)  # superwords were learned
    if target > len(want_vocab):
        assert len(got_vocab) < target  # it stopped because no pair was left


def test_encode_corpus_does_not_depend_on_batching(tmp_path):
    vocab = {c: i for i, c in enumerate(sorted(set("".join(pre_tokenizers.ByteLevel.alphabet()))))}
    corpus = tmp_path / "c.txt"
    corpus.write_text("".join(s + "\n" for s in SENTENCES), encoding="utf-8")
    pre = _pre_tokenizer(STAGE2_REGEX)
    whole = superbpe.encode_corpus([str(corpus)], vocab, [], pre)
    assert np.array_equal(whole, superbpe.encode_corpus([str(corpus)], vocab, [], pre, batch_lines=3))
    assert np.array_equal(whole, superbpe.encode_corpus([str(corpus)] * 2, vocab, [], pre)[: len(whole)])
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert superbpe.encode_corpus([str(empty)], vocab, [], pre).tolist() == [int(superbpe.SEP)]


def test_train_stage2_reports_progress(capsys):
    vocab = {"a": 0, "b": 1}
    ids = np.array([0, 1, 0, 1, superbpe.SEP, 0, 1, 1], dtype=np.uint32)
    superbpe.train_stage2(ids, vocab, 4, log_every=1)
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("stage 2: 1 merges, vocab 3/4, last 'ab' x3") and out[0].endswith("s")
    assert len(out) == 2


def _literal_merge(seq, a, b, new):
    """Left-to-right merge of (a, b) in a flat id list where SEP separates pretokens."""
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _pairs(seq):
    import collections
    sep = int(superbpe.SEP)
    return collections.Counter((x, y) for x, y in zip(seq, seq[1:]) if sep not in (x, y))


S = int(superbpe.SEP)


@pytest.mark.parametrize("seq,a,b", [
    ([0, 0, 0, S, 0, 0, 0, 0, 1], 0, 0),   # runs of 3 and 4
    ([0, 0, 0, S], 0, 0),                  # exactly two overlapping occurrences
    ([0, 0, 1, 0, 0, 0, S], 0, 0),         # a run after a lone pair: every other one counts from each run's start
    ([1, 0, 0, 0, 0, 0, 1, S], 0, 0),      # a run of 5
    ([0, 1, 0, 1, S, 0, 1], 0, 1),         # distinct tokens
    ([1, 0, 1, 1], 0, 1),                  # a single occurrence
])
def test_merge_matches_a_literal_left_to_right_merge(seq, a, b):
    new = 9
    tok = np.array(seq, dtype=np.uint32)
    nxt = np.append(np.arange(1, len(tok), dtype=np.int64), -1)
    prv = np.arange(-1, len(tok) - 1, dtype=np.int64)
    keys, deltas = superbpe._merge(tok, nxt, prv, a, b, new)
    alive = [int(t) for t in tok if t != superbpe.HOLE]
    want = _literal_merge(seq, a, b, new)
    assert alive == want
    # the linked list still walks exactly the live tokens, both ways
    walk, pos = [], 0
    while pos >= 0:
        walk.append(int(tok[pos]))
        pos = int(nxt[pos])
    assert walk == want
    change = {(int(k) >> 32, int(k) & 0xFFFFFFFF): int(d) for k, d in zip(keys, deltas)}
    expected = _pairs(want)
    expected.subtract(_pairs(seq))
    assert change == {k: v for k, v in expected.items() if v}


def test_overlapping_pairs_merge_left_to_right():
    vocab = {"a": 0, "b": 1}
    ids = np.array([0, 0, 0, superbpe.SEP, 0, 0, 0, 0, 1], dtype=np.uint32)
    got_vocab, merges = superbpe.train_stage2(ids, vocab, 3, log_every=0)
    assert merges == [("a", "a")]


def test_super_tokenizer_starts_with_bpe_merges(tokenizers_dir):
    for norm in ("nfc", "nfd"):
        n = json.loads((tokenizers_dir / "train_meta.json").read_text())[norm]["n_inherited_merges"]
        bpe = json.loads((tokenizers_dir / f"bpe-{norm}" / "tokenizer.json").read_text())["model"]
        sup = json.loads((tokenizers_dir / f"super-{norm}" / "tokenizer.json").read_text())["model"]
        n = min(n, len(bpe["merges"]))  # the tiny test corpus runs out of stage-1 merges
        assert sup["merges"][:n] == bpe["merges"][:n]
