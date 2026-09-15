import json
import unicodedata

import numpy as np
import pytest

from conftest import SENTENCES
from vitok import superbpe
from vitok.tokenizer_spec import STAGE1_REGEX, STAGE2_REGEX
from vitok.train_tokenizers import _pre_tokenizer, _train

# repeated tokens exercise the "a a a" overlap rule; ": " exercises the ":Ġ" rule
EXTRA = ["ha ha ha ha ha ha ha", "aaaaaaa bbbb aaaaaaa", "Ghi chú: xem thêm. Ghi chú: xem thêm."]


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
def test_fast_stage2_matches_reference(tmp_path, norm):
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
    got_vocab, got_merges = superbpe.train_stage2(ids, inherited_vocab, 700, log_every=0)
    want_vocab, want_merges = reference_stage2(lines, inherited_vocab, inherited, 700)

    assert got_merges == want_merges
    assert got_vocab == want_vocab
    assert any("Ġ" in t.strip("Ġ") for t in got_vocab)  # superwords were learned


def test_overlapping_pairs_merge_left_to_right():
    vocab = {"a": 0, "b": 1}
    ids = np.array([0, 0, 0, superbpe.SEP, 0, 0, 0, 0, 1], dtype=np.uint32)
    got_vocab, merges = superbpe.train_stage2(ids, vocab, 3, log_every=0)
    assert merges == [("a", "a")]
    tok = ids.copy()
    nxt = np.append(np.arange(1, len(tok), dtype=np.int64), -1)
    prv = np.arange(-1, len(tok) - 1, dtype=np.int64)
    keys, deltas = superbpe._merge(tok, nxt, prv, 0, 0, 2)
    alive = [int(t) for t in tok if t != superbpe.HOLE]
    assert alive == [2, 0, superbpe.SEP, 2, 2, 1]
    change = {(int(k) >> 32, int(k) & 0xFFFFFFFF): int(d) for k, d in zip(keys, deltas)}
    assert change == {(0, 0): -5, (2, 0): 1, (2, 2): 1, (0, 1): -1, (2, 1): 1}


def test_super_tokenizer_starts_with_bpe_merges(tokenizers_dir):
    for norm in ("nfc", "nfd"):
        n = json.loads((tokenizers_dir / "train_meta.json").read_text())[norm]["n_inherited_merges"]
        bpe = json.loads((tokenizers_dir / f"bpe-{norm}" / "tokenizer.json").read_text())["model"]
        sup = json.loads((tokenizers_dir / f"super-{norm}" / "tokenizer.json").read_text())["model"]
        n = min(n, len(bpe["merges"]))  # the tiny test corpus runs out of stage-1 merges
        assert sup["merges"][:n] == bpe["merges"][:n]
