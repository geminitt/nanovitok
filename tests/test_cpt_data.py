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
