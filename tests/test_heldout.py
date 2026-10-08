"""Evaluation-input contracts: the fields scoring reads, NFC, test-set cuts, exact counts, byte-identical reruns."""

import hashlib
import json
import unicodedata

import pytest
from conftest import SENTENCES

from vitok import heldout
from vitok.data import eval_text


def item(i, **over):
    rec = {f: unicodedata.normalize("NFD", f"{f} {SENTENCES[i % 8]}") for f in heldout.BELEBELE_FIELDS}
    rec |= {"correct_answer_num": str(1 + i % 4), "link": "x", "dialect": "vie_Latn"}
    return rec | over


def test_belebele_items_keep_the_scored_fields_in_nfc():
    out = heldout.belebele_items([item(i) for i in range(5)])
    assert [set(r) for r in out] == [set(heldout.BELEBELE_FIELDS)] * 5
    assert all(v == unicodedata.normalize("NFC", v) for r in out for v in r.values())
    assert [r["correct_answer_num"] for r in out] == ["1", "2", "3", "4", "1"]


@pytest.mark.parametrize("bad,message", [({"question": None}, "lacks"), ({"correct_answer_num": "5"}, "not 1-4"),
                                         ({"correct_answer_num": 2}, "lacks")])
def test_belebele_items_refuse_malformed_records(bad, message):
    with pytest.raises(ValueError, match=message):
        heldout.belebele_items([item(0), item(1, **bad)])


def test_english_docs_are_cut_like_the_test_set_without_duplicates():
    long = "word " * 900
    texts = ["too short", long, long, "x" * 400, unicodedata.normalize("NFD", "Café " * 100), "y" * 500]
    out = heldout.english_docs(texts, 3)
    assert out == [eval_text(long), "x" * 400, eval_text("Café " * 100)]
    assert all(300 <= len(d) <= 2500 for d in out)
    with pytest.raises(ValueError, match="only 4 usable documents, 5 needed"):
        heldout.english_docs(texts, 5)


def test_outputs_regenerate_byte_identically_and_match_the_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(heldout, "_belebele_records", lambda rev: [item(i) for i in range(heldout.BELEBELE_ITEMS)])
    monkeypatch.setattr(heldout, "_fineweb_texts", lambda rev: iter(f"{i} " + "text " * 80 for i in range(50)))
    for out in (tmp_path / "a", tmp_path / "b"):
        heldout.main(["--out", str(out), "--en-docs", "20"])
    for name in ("belebele_vie_Latn.jsonl", "en_docs.jsonl", "heldout_manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    man = json.loads((tmp_path / "a" / "heldout_manifest.json").read_text())
    for key, name in (("belebele", "belebele_vie_Latn.jsonl"), ("en_docs", "en_docs.jsonl")):
        assert man[key]["sha256"] == hashlib.sha256((tmp_path / "a" / name).read_bytes()).hexdigest()
    assert man["belebele"]["items"] == 900 and man["en_docs"]["docs"] == 20
    assert len((tmp_path / "a" / "en_docs.jsonl").read_text().splitlines()) == 20


def test_a_belebele_file_of_the_wrong_size_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(heldout, "_belebele_records", lambda rev: [item(i) for i in range(899)])
    with pytest.raises(SystemExit, match="899 items, expected 900"):
        heldout.main(["--out", str(tmp_path)])


def test_the_manifest_and_files_record_exactly_what_was_fetched(tmp_path, monkeypatch):
    revs = []
    monkeypatch.setattr(heldout, "_belebele_records",
                        lambda rev: revs.append(rev) or [item(i) for i in range(heldout.BELEBELE_ITEMS)])
    monkeypatch.setattr(heldout, "_fineweb_texts",
                        lambda rev: revs.append(rev) or iter(f"Đoạn {i} " + "chữ " * 90 for i in range(3000)))
    heldout.main(["--out", str(tmp_path)])
    assert revs == [heldout.BELEBELE_REVISION, heldout.FINEWEB_REVISION]
    man = json.loads((tmp_path / "heldout_manifest.json").read_text())
    sha = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert man == {
        "belebele": {"repo": "facebook/belebele", "file": "data/vie_Latn.jsonl", "revision": heldout.BELEBELE_REVISION,
                     "items": 900, "sha256": sha("belebele_vie_Latn.jsonl")},
        "en_docs": {"repo": "HuggingFaceFW/fineweb", "file": "sample/10BT/000_00000.parquet",
                    "revision": heldout.FINEWEB_REVISION, "docs": 2000, "min_chars": 300, "max_chars": 2500,
                    "sha256": sha("en_docs.jsonl")}}
    lines = (tmp_path / "en_docs.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines[0] == json.dumps({"text": eval_text("Đoạn 0 " + "chữ " * 90)}, ensure_ascii=False)  # UTF-8, not \\u
    first = json.loads((tmp_path / "belebele_vie_Latn.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first == heldout.belebele_items([item(0)])[0]
