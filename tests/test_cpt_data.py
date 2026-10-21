"""Training-data contracts: held-out text never enters, documents keep their order, batches resume exactly."""

import json
import unicodedata

import numpy as np
import pytest
from conftest import SENTENCES, qwen_like_base
from tokenizers import Tokenizer

from vitok import cpt_data

DOCS = [s * (1 + i % 3) for i, s in enumerate(SENTENCES * 6)]


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.from_str(json.dumps(qwen_like_base()))


def test_held_out_documents_are_skipped_by_full_text_or_test_cut():
    long = "chữ " * 1000  # cut to 2,500 characters for the test set
    held = cpt_data.held_out_digests([SENTENCES[0], long[:2500].rstrip()])
    docs = [SENTENCES[0], unicodedata.normalize("NFD", SENTENCES[0]), long, SENTENCES[1]]
    kept = list(cpt_data.training_docs(docs, held))
    assert kept == [SENTENCES[1]]  # same text, NFD form, and a document whose cut is a test document are all out


def test_token_chars_count_each_character_once(tok):
    for text in SENTENCES + ["€ 🎉 Việt", "  hai  dấu  cách  "]:
        enc = tok.encode(text, add_special_tokens=False)
        assert sum(cpt_data.token_chars(enc.offsets)) == len(text)


def test_batches_are_the_documents_in_order_with_eos_between(tok):
    eos = tok.token_to_id("<|endoftext|>")
    batches = list(cpt_data.packed_batches(tok, DOCS, eos, seq_len=16, batch=3, encode_batch=5))
    stream = np.concatenate([b.input_ids.ravel() for b in batches]).tolist()
    want = [i for d in DOCS for i in tok.encode(d, add_special_tokens=False).ids + [eos]]
    assert len(stream) == len(want) - len(want) % 48 and stream == want[: len(stream)]
    assert all(b.input_ids.shape == (3, 16) for b in batches)
    covered = sum(b.chars for b in batches)
    assert sum(map(len, DOCS[: batches[-1].state["docs_read"]])) >= covered > 0
    # each token carries the characters it adds (eos adds none), so any slice of a batch knows the text it covers
    per_token = [c for d in DOCS for c in cpt_data.token_chars(tok.encode(d, add_special_tokens=False).offsets) + [0]]
    flat = np.concatenate([b.token_chars.ravel() for b in batches]).tolist()
    assert flat == per_token[: len(flat)]
    assert all(b.token_chars.shape == b.input_ids.shape and int(b.token_chars.sum()) == b.chars for b in batches)


@pytest.mark.parametrize("stop", [0, 1, 4, 9])
def test_a_resumed_stream_continues_exactly(tok, stop):
    eos = tok.token_to_id("<|endoftext|>")
    run = lambda docs, state=None: list(cpt_data.packed_batches(tok, docs, eos, seq_len=16, batch=3, encode_batch=5,
                                                                state=state))
    full = run(DOCS)
    head = full[: stop + 1]
    state = head[-1].state
    rest = run(DOCS[state["docs_read"]:], json.loads(json.dumps(state)))  # the state survives JSON
    assert [b.input_ids.tolist() for b in head + rest] == [b.input_ids.tolist() for b in full]
    assert [b.chars for b in head + rest] == [b.chars for b in full]
    assert [b.token_chars.tolist() for b in head + rest] == [b.token_chars.tolist() for b in full]


def test_packing_refuses_degenerate_shapes(tok):
    with pytest.raises(ValueError, match="must be at least"):
        next(cpt_data.packed_batches(tok, DOCS, 0, seq_len=1, batch=3))


def test_a_stream_that_fills_its_last_batch_exactly_yields_it(tok):
    eos = tok.token_to_id("<|endoftext|>")
    n = len(tok.encode(SENTENCES[0], add_special_tokens=False).ids) + 1  # the document and its eos
    out = list(cpt_data.packed_batches(tok, [SENTENCES[0]], eos, seq_len=n, batch=1))
    assert len(out) == 1 and out[0].input_ids[0, -1] == eos and out[0].chars == len(SENTENCES[0])
    two = list(cpt_data.packed_batches(tok, [SENTENCES[0]] * 2, eos, seq_len=n, batch=2))
    assert len(two) == 1 and two[0].state == {"docs_read": 2, "ids": [], "chars": []}
    assert len(list(cpt_data.packed_batches(tok, DOCS, eos, seq_len=2, batch=1))) > 0  # the smallest shape


def test_texts_are_read_from_parquet_and_jsonl_in_order(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"text": SENTENCES[:3], "id": [1, 2, 3]}), tmp_path / "a.parquet", row_group_size=2)
    pq.write_table(pa.table({"text": SENTENCES[3:5], "id": [4, 5]}), tmp_path / "b.parquet")
    assert list(cpt_data.parquet_texts([tmp_path / "a.parquet", tmp_path / "b.parquet"])) == SENTENCES[:5]
    (tmp_path / "t.jsonl").write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in SENTENCES)
                                      + "\n", encoding="utf-8")
    assert cpt_data.jsonl_texts(tmp_path / "t.jsonl") == SENTENCES
    assert len(cpt_data.digest("x")) == 12 and cpt_data.digest("x") != cpt_data.digest("y")


def test_exactly_full_buffers_are_emitted_wherever_they_fill(tok):
    eos = tok.token_to_id("<|endoftext|>")
    n = len(tok.encode(SENTENCES[0], add_special_tokens=False).ids) + 1
    # filled right after a full encode batch (one document per encode batch)
    mid = list(cpt_data.packed_batches(tok, [SENTENCES[0], SENTENCES[1]], eos, seq_len=n, batch=1, encode_batch=1))
    assert len(mid) >= 1 and mid[0].state["docs_read"] == 1
    # filled by the leftover tokens of a resumed state, before any document is read
    first = mid[0]
    state = {"docs_read": 1, "ids": first.input_ids.ravel().tolist(), "chars": [1] * n}
    again = list(cpt_data.packed_batches(tok, [SENTENCES[1]], eos, seq_len=n, batch=1, state=state))
    assert again[0].input_ids.tolist() == first.input_ids.tolist() and again[0].chars == n
    assert again[0].state["docs_read"] == 1  # emitted before the next document was read


# ---- document-aligned steps ----

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

TEXTS = st.lists(st.text(alphabet="ab \nđ", min_size=1, max_size=60), min_size=1, max_size=25)


def records(texts):
    return [{"text": t, "domain": "abc"[i % 3]} for i, t in enumerate(texts)]


@given(TEXTS, st.integers(1, 40))
def test_steps_cover_the_stream_in_order_and_fill_their_room(texts, size):
    steps = list(cpt_data.aligned_steps(records(texts), size))
    pieces = [p for s in steps for p in s.pieces]
    read = "".join(p + ("\x00" if ends else "") for p, ends in pieces)
    whole = "".join(t + "\x00" for t in texts)
    assert whole.startswith(read)  # nothing lost, repeated or reordered; only a final partial step is left out
    assert len(whole) - len(read) < size + len(texts) + 1
    for s in steps:
        assert size // 2 <= s.chars <= size and s.chars == sum(len(p) for p, _ in s.pieces) == sum(
            s.domain_chars.values())


@given(TEXTS, st.integers(1, 40), st.data())
def test_a_step_state_resumes_to_the_same_steps(texts, size, data):
    recs = records(texts)
    steps = list(cpt_data.aligned_steps(recs, size))
    if not steps:
        return
    k = data.draw(st.integers(0, len(steps) - 1))
    st_ = steps[k].state
    rest = list(cpt_data.aligned_steps(recs[st_["record"]:], size, st_))
    assert [(s.pieces, s.digest, s.state) for s in rest] == [(s.pieces, s.digest, s.state) for s in steps[k + 1:]]


def test_digests_match_exactly_while_the_text_read_matches():
    a = [{"text": "một hai ba bốn năm"}, {"text": "sáu bảy tám chín mười"}]
    b = [{"text": "một hai ba bốn năm"}, {"text": "sáu bảy tâm chín mười"}]  # differs inside the third step
    da = [s.digest for s in cpt_data.aligned_steps(a, 8)]
    db = [s.digest for s in cpt_data.aligned_steps(b, 8)]
    assert da[0] == db[0] and da[1] == db[1] and da[2:] != db[2:]
    # the same characters with a document boundary moved
    moved = [{"text": "một hai ba bốn"}, {"text": " năm"}, {"text": "sáu bảy tám chín mười"}]
    assert [s.digest for s in cpt_data.aligned_steps(moved, 8)][1] != da[1]


def test_documents_are_cut_at_line_breaks_then_spaces_in_the_second_half():
    assert cpt_data.cut_point("ab\ncd ef gh", 9) == 9  # line break at 2 is before the second half; space at 8
    assert cpt_data.cut_point("abcdef\ngh ij", 10) == 7  # line break at 6 lies in the second half (>= 5)
    assert cpt_data.cut_point("ab\ncdefgh ij", 10) == 10  # line break at 2 is in the first half; space at 9
    assert cpt_data.cut_point("abcdefghijkl", 10) == 10
    assert cpt_data.cut_point("short", 10) == 5
    big = "x" * 5000 + "\n" + "y" * 4500 + " " + "z" * 6000  # the window starts at 5,904: line break outside
    assert cpt_data.cut_point(big, 10_000) == 5000 + 1 + 4500 + 1  # the space inside it
    assert cpt_data.cut_point("x" * 9000 + "\n" + "y" * 100 + " " + "z" * 6000, 10_000) == 9001
    assert cpt_data.cut_point("x" * 5000 + "\n" + "y" * 9000, 10_000) == 10_000
    with pytest.raises(ValueError, match="no room"):
        cpt_data.cut_point("x", 0)


@given(st.lists(st.tuples(st.sampled_from(SENTENCES + ["", "Việt Nam "]), st.booleans()), min_size=1, max_size=8),
       st.integers(2, 20))
def test_step_rows_hold_the_pieces_tokens_with_eos_and_pad_only_the_end(pieces, seq_len):
    tok = Tokenizer.from_str(json.dumps(qwen_like_base()))
    eos = tok.token_to_id("<|endoftext|>")
    x, labels, chars = cpt_data.step_rows(tok, pieces, eos, seq_len)
    want = [i for p, ends in pieces for i in tok.encode(p, add_special_tokens=False).ids + ([eos] if ends else [])]
    flat = x.reshape(-1).tolist()
    assert x.shape[1] == seq_len and flat[:len(want)] == want and len(flat) - len(want) < seq_len
    assert (labels.reshape(-1)[:len(want)] == np.asarray(want)).all() and (labels.reshape(-1)[len(want):] == -100).all()
    assert int(chars.sum()) == sum(len(p) for p, _ in pieces)
