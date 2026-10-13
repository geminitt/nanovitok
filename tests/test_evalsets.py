"""Held-out set contracts: seeded deduplicated samples in source order, the windows each domain uses, NFC, the
multiple-choice prompts and gold answers, the reserved-file split, byte-identical reruns and the manifest."""

import hashlib
import json
import unicodedata

import pytest
from conftest import SENTENCES
from hypothesis import given
from hypothesis import strategies as st

from vitok import evalsets as es
from vitok.data import eval_text


@given(st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=40), st.integers(0, 3), st.data())
def test_a_sample_is_a_seeded_deduplicated_subset_in_source_order(texts, seed, data):
    unique = list(dict.fromkeys(texts))
    n = data.draw(st.integers(1, len(unique)))
    out = es.sample(((f"k{i}", t) for i, t in enumerate(texts)), n, seed)
    assert len(out) == n and len({r["text"] for r in out}) == n
    first = {t: f"k{texts.index(t)}" for t in unique}
    assert all(r["id"] == first[r["text"]] for r in out)  # a duplicate keeps its first occurrence
    order = [unique.index(r["text"]) for r in out]
    assert order == sorted(order)
    assert out == es.sample(((f"k{i}", t) for i, t in enumerate(texts)), n, seed)
    with pytest.raises(ValueError, match=f"only {len(unique)} usable documents"):
        es.sample(((f"k{i}", t) for i, t in enumerate(texts)), len(unique) + 1, seed)


def test_wikipedia_keeps_long_articles_cut_from_the_start_in_nfc():
    long = unicodedata.normalize("NFD", "Hà Nội là thủ đô. " * 200)
    assert es.wiki_window(long) == eval_text(unicodedata.normalize("NFC", long))
    assert es.wiki_window("x" * (es.WIKI_MIN_SOURCE - 1)) is None
    assert es.wiki_window("x " * es.WIKI_MIN_SOURCE) is not None


def test_legal_windows_skip_the_letterhead_to_a_line_start():
    head = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n" * 28  # 980 characters of letterhead, shorter than the skip
    body = "Điều 1. Phạm vi điều chỉnh của văn bản này.\n" * 100
    w = es.legal_window(head + body)
    assert w is not None and w.startswith("Điều 1.") and len(w) <= 2500
    assert "CỘNG HÒA" not in w
    assert es.legal_window("a" * (es.LEGAL_MIN_SOURCE - 1)) is None
    assert es.legal_window("a" * 5000) is None  # no line break after the letterhead


def test_html_text_keeps_visible_text_one_line_per_block():
    html = ("<html><head><title>t</title><style>p{x}</style></head><body><div>Số: 1/NQ</div>"
            "<p>Điều  1.\n Nội&nbsp;dung</p><script>var a;</script><table><tr><td>a</td><td>b</td></tr></table>"
            "<br>cuối</body></html>")
    assert es.html_text(html) == "Số: 1/NQ\nĐiều 1. Nội dung\nab\ncuối"


def belebele_rec(i, **over):
    rec = {f: unicodedata.normalize("NFD", f"{f} {SENTENCES[i % 8]}") for f in es.heldout.BELEBELE_FIELDS}
    return rec | {"correct_answer_num": str(1 + i % 4)} | over


def test_belebele_uses_the_original_prompt_and_zero_based_gold():
    out = es.belebele_mc([belebele_rec(i) for i in range(4)])
    assert [r["gold"] for r in out] == [0, 1, 2, 3]
    r = out[1]
    assert r["context"].endswith("\nTrả lời:") and "\nCâu hỏi: " in r["context"]
    assert all(c.startswith(" ") for c in r["choices"]) and len(r["choices"]) == 4
    assert r["context"] == unicodedata.normalize("NFC", r["context"])


def test_letter_items_map_the_answer_letter_and_skip_incomplete_items():
    recs = [{"sample_id": "s0", "question": "Q0", "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d",
             "answer": "C"},
            {"sample_id": "s1", "question": "Q1", "option_a": "a", "option_b": "", "option_c": "c", "option_d": "d",
             "answer": "A"},
            {"sample_id": "s2", "question": "Q2", "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d",
             "answer": "E"}]
    out = es.letter_mc(recs, "m", "question", "{q}\nTrả lời:")
    assert out == [{"id": "m-s0", "context": "Q0\nTrả lời:", "choices": [" a", " b", " c", " d"], "gold": 2}]


def test_xnli_labels_follow_entailment_neutral_contradiction():
    recs = [{"premise": "Trời mưa.", "hypothesis": "Đường ướt.", "label": str(k)} for k in range(3)]
    out = es.xnli_mc(recs)
    assert [r["gold"] for r in out] == [0, 1, 2]
    assert out[0]["context"] == "Trời mưa, đúng không?"
    assert out[0]["choices"] == [" Đúng, Đường ướt.", " Ngoài ra, Đường ướt.", " Không, Đường ướt."]


def test_hellaswag_is_cleaned_like_the_harness_and_reads_string_endings():
    rec = {"ind": "7", "activity_label": "Cooking", "ctx_a": "A man [header] cuts", "ctx_b": "the bread",
           "endings": "['slowly. [title] Done', 'fast', 'with a knife', 'and eats']", "label": "2"}
    out = es.hellaswag_mc([rec, rec | {"endings": ["one", "two"]}, rec | {"label": ""}], "h")
    assert out == [{"id": "h-0-7", "context": "Cooking: A man cuts The bread",
                    "choices": [" slowly.. Done", " fast", " with a knife", " and eats"], "gold": 2}]
    # an ending that starts with a section marker keeps one leading space (the harness would leave two)
    marked = es.hellaswag_mc([rec | {"endings": ["[title] Tòa án", "b", "c", "d"]}], "h")
    assert marked[0]["choices"][0] == " Tòa án"


def test_every_choice_starts_with_exactly_one_space_and_ids_stay_unique():
    assert es._mc("x", "ctx", ["  a", "b ", " c"], 0)["choices"] == [" a", " b", " c"]
    rec = {"ind": "7", "activity_label": "A", "ctx_a": "b", "ctx_b": "", "endings": ["e", "f", "g", "h"], "label": "1"}
    out = es.hellaswag_mc([rec, rec], "h")
    assert len({r["id"] for r in out}) == 2


@pytest.mark.parametrize("choices,gold", [([" a"], 0), ([" a", " b"], 2), ([" a", "  "], 0)])
def test_malformed_items_are_refused(choices, gold):
    with pytest.raises(ValueError, match="malformed"):
        es._mc("x", "ctx", choices, gold)


def test_the_reserved_file_gives_merge_text_then_dev_documents(monkeypatch):
    monkeypatch.setattr(es, "MERGE_CHARS", 1000)
    monkeypatch.setattr(es, "DEV_DOCS", 3)
    texts = ["m" * 600, "n" * 600, "short", "d" * 400, "d" * 400, "e" * 3000, "f" * 500, "g" * 500]
    monkeypatch.setattr(es, "_streamed_texts", lambda repo, name: iter(texts))
    (merge, dev), meta = es.build_merge_and_dev()
    assert [r["text"] for r in merge] == texts[:2] and meta["merge_chars"] == 1200
    assert [r["text"] for r in dev] == ["d" * 400, eval_text("e" * 3000), "f" * 500]
    assert meta["train_files"] == es.TRAIN_FILES and es.RESERVED_FILE not in es.TRAIN_FILES
    monkeypatch.setattr(es, "_streamed_texts", lambda repo, name: iter(texts[:5]))
    with pytest.raises(ValueError, match="only 1 dev documents"):
        es.build_merge_and_dev()


def fake_sources(monkeypatch):
    def rows(repo_rev, files, columns):
        if repo_rev == es.LEGAL:
            head = "<p>" + "Tiêu đề\n" * 200 + "</p>"
            return iter({"id": i, "content_html": head + f"<p>{i} " + "điều " * 600 + "</p>"} for i in range(30))
        if repo_rev == es.FINEWEB2:
            return iter({"id": f"f{i}", "text": f"{i} " + "chữ " * 100} for i in range(30))
        return iter({"id": f"w{i}", "text": f"{i} " + "bài " * 1000} for i in range(30))
    monkeypatch.setattr(es, "_parquet_rows", rows)
    monkeypatch.setattr(es, "DOCS", 10)
    monkeypatch.setattr(es.heldout, "_fineweb_texts", lambda rev: iter(f"{i} " + "word " * 80 for i in range(30)))


def test_language_sets_regenerate_byte_identically_and_match_the_manifest(tmp_path, monkeypatch):
    fake_sources(monkeypatch)
    sets = ["fineweb2", "wikipedia", "legal", "english"]
    for out in (tmp_path / "a", tmp_path / "b"):
        es.main(["--out", str(out), "--sets", *sets])
    man = json.loads((tmp_path / "a" / "evalsets_manifest.json").read_text())
    for name in sets:
        data = (tmp_path / "a" / f"{name}.jsonl").read_bytes()
        assert data == (tmp_path / "b" / f"{name}.jsonl").read_bytes()
        assert man[name]["sha256"] == hashlib.sha256(data).hexdigest() and man[name]["count"] == 10
        rows = [json.loads(line) for line in data.decode().splitlines()]
        assert all(300 <= len(r["text"]) <= 2500 for r in rows)
    assert man["fineweb2"]["revision"] == es.FINEWEB2[1] and man["legal"]["revision"] == es.LEGAL[1]
    # a later run of one set keeps the other entries of the manifest
    es.main(["--out", str(tmp_path / "a"), "--sets", "fineweb2"])
    assert json.loads((tmp_path / "a" / "evalsets_manifest.json").read_text()) == man


def test_a_gated_flores_names_the_owner_action(tmp_path, monkeypatch):
    def denied(repo_rev, name):
        raise PermissionError("403")
    monkeypatch.setattr(es, "_jsonl_rows", denied)
    with pytest.raises(SystemExit, match="accept its terms"):
        es.main(["--out", str(tmp_path), "--sets", "flores"])
