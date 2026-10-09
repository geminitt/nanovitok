"""K1 contracts: the retrofit keeps the base tokenizer intact and adds reachable Vietnamese tokens after its ids."""

import json
from pathlib import Path

import pytest
from conftest import QWEN_RE, SENTENCES, SPECIALS, qwen_like_base
from hypothesis import given, settings
from hypothesis import strategies as st
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE

from vitok import retrofit

TEXT = st.text(alphabet="aăâbcdđeêghiklmnoôơpqrstuưvxyáàảãạếềểễệọộớờởỡợúùủũụứừửữựABCĐHNTV 0123456789.,:;!?'\"-()\n\t",
               max_size=80)


@pytest.fixture(scope="module")
def base() -> dict:
    return qwen_like_base()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    path = tmp_path_factory.mktemp("k1") / "vi.txt"
    path.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    return path


def as_tok(j: dict) -> Tokenizer:
    return Tokenizer.from_str(json.dumps(j))


def ids(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


@pytest.mark.parametrize("kind", retrofit.KINDS)
@settings(max_examples=200, deadline=None)
@given(text=TEXT)
def test_without_new_merges_the_retrofit_encodes_exactly_like_the_base(base, kind, text):
    # the extra word-run alternative only groups words; with the base's merges alone no token can span a space
    assert ids(as_tok(retrofit.retrofit(base, [], kind)), text) == ids(as_tok(base), text)


@pytest.fixture(params=retrofit.KINDS)
def learned(request, base, corpus):
    # function-scoped on purpose: a cached fixture would hide learn_merges/retrofit from all but the first test
    kind = request.param
    merges = retrofit.learn_merges(base, [corpus], kind, 120)
    return kind, merges, retrofit.retrofit(base, merges, kind)


def test_retrofitted_tokenizer_round_trips(learned):
    tok = as_tok(learned[2])

    @settings(max_examples=200, deadline=None)
    @given(text=TEXT)
    def round_trip(text):
        assert tok.decode(ids(tok, text)) == normalizers.NFC().normalize_str(text)

    round_trip()


def test_new_tokens_follow_the_old_ids_and_specials_keep_theirs(base, learned):
    kind, merges, out = learned
    old, new = as_tok(base), as_tok(out)
    old_vocab, new_vocab = old.get_vocab(True), new.get_vocab(True)
    assert all(new_vocab[t] == i for t, i in old_vocab.items())          # nothing moved
    added = sorted(i for t, i in new_vocab.items() if t not in old_vocab)
    assert added == list(range(max(old_vocab.values()) + 1, max(old_vocab.values()) + 1 + len(added)))
    assert len(added) == len({a + b for a, b in merges} - set(old_vocab)) > 0
    for special in SPECIALS:
        assert new.token_to_id(special) == old.token_to_id(special)
    assert out["model"]["merges"][: len(base["model"]["merges"])] == base["model"]["merges"]


def test_kinds_differ_only_in_whether_tokens_may_join_words(learned):
    kind, merges, out = learned
    spans = [a + b for a, b in merges if "Ġ" in (a + b).strip("Ġ")]
    if kind == "single":
        assert not spans and out["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == QWEN_RE
    else:
        regex = out["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        assert spans and regex == retrofit.WORD_RUN + "|" + QWEN_RE


def test_report_counts_savings_reachability_and_unchanged_text(base, learned):
    kind, merges, out = learned
    new, old = as_tok(out), as_tok(base)
    report = retrofit.measure(new, old, SENTENCES, {"english": ["The quick brown fox, don't stop! 2026"]})
    n_old = sum(len(ids(old, d)) for d in SENTENCES)
    n_new = sum(len(ids(new, d)) for d in SENTENCES)
    chars = sum(map(len, SENTENCES))
    assert report["docs"] == len(SENTENCES) and report["chars"] == chars
    assert report["chars_per_token_base"] == pytest.approx(chars / n_old)
    assert report["chars_per_token_new"] == pytest.approx(chars / n_new)
    assert report["token_change"] == pytest.approx(n_new / n_old - 1)
    assert report["new_tokens"] == len([t for t in new.get_vocab(True) if t not in old.get_vocab(True)])
    # every new token that is whole characters encodes to itself; byte pieces of a character are only counted
    assert report["new_tokens_reachable"] == report["new_tokens_whole_chars"] > 0
    assert report["new_tokens_whole_chars"] + report["new_tokens_partial_chars"] == report["new_tokens"]
    assert 0 < report["new_tokens_used_on_docs"] <= report["new_tokens"]
    assert report["round_trip"] == len(SENTENCES) and report["token_change"] < 0
    assert report["chars_per_token_new"] > report["chars_per_token_base"]
    assert report["identical_ids"] == {"english": 1} and report["identical_ids_of"] == {"english": 1}


def test_cli_writes_tokenizer_manifest_and_report(base, corpus, tmp_path, monkeypatch, capsys):
    import hashlib
    import subprocess
    import sys
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    docs = tmp_path / "docs.jsonl"
    docs.write_text("".join(json.dumps({"id": i, "text": s}) + "\n" for i, s in enumerate(SENTENCES)), encoding="utf-8")
    other = tmp_path / "english.txt"
    other.write_text("The quick brown fox, don't stop!", encoding="utf-8")

    def run(out, n_new, kind="multi"):
        monkeypatch.setattr(sys, "argv", ["retrofit", "--base", str(base_path), "--corpus", str(corpus), "--kind", kind,
                                          "--n-new", str(n_new), "--docs", str(docs), "--others", str(other),
                                          "--out", str(out)])
        retrofit.main()
        return json.loads(capsys.readouterr().out)

    out = tmp_path / "deep" / "c2"  # parent folders are created
    printed = run(out, 50)
    manifest = json.loads((out / "manifest.json").read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=Path(retrofit.__file__).parent).stdout.strip()
    assert manifest == {**manifest, "base": str(base_path), "base_sha256": sha(base_path), "kind": "multi",
                        "n_new_requested": 50, "merges_learned": 50, "git_commit": head or "unknown",
                        "corpus": [{"path": str(corpus), "sha256": sha(corpus), "bytes": corpus.stat().st_size}]}
    import re
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4}", manifest["created"])
    assert set(manifest) == {"base", "base_sha256", "corpus", "kind", "n_new_requested", "merges_learned", "git_commit",
                             "packages", "created"} and set(manifest["packages"]) == {"tokenizers", "numpy"}
    report = json.loads((out / "report.json").read_text())
    assert printed == report and report["new_tokens"] > 0 and report["identical_ids"] == {"english.txt": 1}
    grown = Tokenizer.from_file(str(out / "tokenizer.json")).get_vocab_size()
    assert grown == as_tok(base).get_vocab_size() + report["new_tokens"]
    # zero new tokens is allowed and gives the base back, into an existing folder
    assert run(out, 0, "single")["new_tokens"] == 0
    assert json.loads((out / "manifest.json").read_text())["merges_learned"] == 0
    zero = Tokenizer.from_file(str(out / "tokenizer.json"))
    assert all(ids(zero, s) == ids(as_tok(base), s) for s in SENTENCES)
    monkeypatch.setattr(sys, "argv", ["retrofit", "--base", str(base_path), "--corpus", str(corpus), "--kind", "multi",
                                      "--n-new", "-1", "--out", str(out)])
    with pytest.raises(SystemExit, match="--n-new must be >= 0"):
        retrofit.main()


def test_merges_written_as_strings_are_kept_as_strings(base, corpus):
    flat = json.loads(json.dumps(base))
    flat["model"]["merges"] = [" ".join(m) for m in base["model"]["merges"]]
    merges = retrofit.learn_merges(flat, [corpus], "multi", 40)
    out = retrofit.retrofit(flat, merges, "multi")
    assert merges == retrofit.learn_merges(base, [corpus], "multi", 40)
    assert all(isinstance(m, str) for m in out["model"]["merges"])
    assert out["model"]["merges"][-len(merges):] == [f"{a} {b}" for a, b in merges]
    assert ids(as_tok(out), SENTENCES[1]) == ids(as_tok(retrofit.retrofit(base, merges, "multi")), SENTENCES[1])


def test_corpus_is_normalized_like_the_base_and_bases_without_specials_work(base, corpus, tmp_path):
    import unicodedata
    nfd = tmp_path / "nfd.txt"
    nfd.write_text(unicodedata.normalize("NFD", corpus.read_text(encoding="utf-8")), encoding="utf-8")
    assert retrofit.learn_merges(base, [nfd], "multi", 40) == retrofit.learn_merges(base, [corpus], "multi", 40)
    bare = json.loads(json.dumps(base))
    bare.pop("added_tokens")
    bare["normalizer"] = None  # a base without a normalizer learns from the text as it is
    out = retrofit.retrofit(bare, retrofit.learn_merges(bare, [corpus], "single", 30), "single")
    assert len(out["model"]["vocab"]) > len(bare["model"]["vocab"])


def test_jsonl_and_text_corpora_give_the_same_merges(base, corpus, tmp_path):
    jsonl = tmp_path / "vi.jsonl"
    jsonl.write_text("".join(json.dumps({"id": i, "text": line}) + "\n" for i, line in enumerate(SENTENCES * 40)),
                     encoding="utf-8")
    txt = tmp_path / "vi.txt"
    txt.write_text("".join(line + "\n\n" for line in SENTENCES * 40), encoding="utf-8")
    assert retrofit.learn_merges(base, [jsonl], "multi", 60) == retrofit.learn_merges(base, [txt], "multi", 60)


def test_byte_decoder_inverts_the_byte_level_alphabet():
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    assert len(alphabet) == 256 and set(retrofit.BYTE_DECODER) == set(alphabet)
    assert sorted(retrofit.BYTE_DECODER.values()) == list(range(256))
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    for text in ["Việt Nam", " học\n", "€ 🎉"]:
        pieces = [p for p, _ in tok.pre_tokenizer.pre_tokenize_str(text)]
        assert b"".join(retrofit.token_bytes(p) for p in pieces) == text.encode("utf-8")


@pytest.mark.parametrize("kind", retrofit.KINDS)
def test_merges_are_learned_on_the_pieces_the_final_tokenizer_cuts(base, kind):
    # learning and use must pre-tokenize alike, or the learned merges would never fire
    _, _, base_regex = retrofit.base_parts(base)
    learning = retrofit._pre_tokenizer(retrofit.regex_for(kind, base_regex))
    final = as_tok(retrofit.retrofit(base, [], kind)).pre_tokenizer
    for text in SENTENCES + ["Năm 2026, giá 1.250.000 đồng!\n\tdon't stop", "  hai  dấu cách  "]:
        assert learning.pre_tokenize_str(text) == final.pre_tokenize_str(text)


@pytest.mark.parametrize("argv", [
    [],                                                                  # nothing
    ["--corpus", "x.txt", "--kind", "multi", "--n-new", "5", "--out", "o"],   # no --base
    ["--base", "b.json", "--kind", "multi", "--n-new", "5", "--out", "o"],    # no --corpus
    ["--base", "b.json", "--corpus", "x.txt", "--n-new", "5", "--out", "o"],  # no --kind
    ["--base", "b.json", "--corpus", "x.txt", "--kind", "multi", "--out", "o"],  # no --n-new
    ["--base", "b.json", "--corpus", "x.txt", "--kind", "multi", "--n-new", "5"],  # no --out
    ["--base", "b.json", "--corpus", "x.txt", "--kind", "triple", "--n-new", "5", "--out", "o"],  # unknown kind
])
def test_cli_refuses_incomplete_arguments(argv, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", ["retrofit", *argv])
    with pytest.raises(SystemExit) as stop:
        retrofit.main()
    assert stop.value.code == 2  # argparse's usage error, before any file is read
    capsys.readouterr()
