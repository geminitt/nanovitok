import hashlib
import json
import sys

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from vitok import survey


def _tokenizer():
    tokenizer = Tokenizer(models.WordLevel({"hello": 0, "world": 1, "[UNK]": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["[BOS]"])
    tokenizer.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 3)])
    return tokenizer


def test_survey_counts_the_same_text_without_special_tokens():
    assert survey.measure(_tokenizer(), ["hello world", "hello"]) == {
        "chars": [11, 5], "tokens": [2, 1], "chars_per_token": 16 / 3,
    }
    for docs in ([], [""], ["   "]):
        with pytest.raises(ValueError, match="nonempty|empty encoding"):
            survey.measure(_tokenizer(), docs)


def _inputs(tmp_path):
    docs_root = tmp_path / "inputs"
    path = docs_root / "tokenizers-16k/bpe-nfc/tokenizer.json"
    path.parent.mkdir(parents=True)
    _tokenizer().save(str(path))
    inputs = {}
    measured = {"chars": [11, 5], "tokens": [2, 1], "chars_per_token": 16 / 3}
    for part, name in (("test", "test.jsonl"), ("val", "val_docs.jsonl")):
        document_path = docs_root / name
        document_path.write_text(' {"text": "hello world"}\n {"text": "hello"}\n')
        inputs[part] = {"sha256": hashlib.sha256(document_path.read_bytes()).hexdigest(), "documents": 2}
    row = {"name": "nanovitok/bpe-nfc-16k", "revision": None,
           "tokenizer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tokenizer_bytes": path.stat().st_size,
           "inputs": inputs, "results": {"test": measured, "val": measured}, "vocab": 4, "roundtrips": 2}
    return docs_root, path, [row]


@pytest.mark.parametrize("remote", [False, True])
def test_survey_reproduction_pins_inputs_and_refuses_changed_measurements(tmp_path, monkeypatch, remote):
    import huggingface_hub

    docs_root, tokenizer_path, rows = _inputs(tmp_path)
    if remote:
        rows[0].update(name="example/tokenizer", revision="fixed-revision")

        def download(name, filename, revision, local_files_only):
            assert (name, filename, revision, local_files_only) == (
                "example/tokenizer", "tokenizer.json", "fixed-revision", True)
            return str(tokenizer_path)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    recorded, out = tmp_path / "recorded.json", tmp_path / "regenerated.json"
    recorded.write_text(json.dumps(rows, indent=2) + "\n")
    monkeypatch.setattr(sys, "argv", ["survey", "--recorded", str(recorded), "--docs-root", str(docs_root),
                                     "--out", str(out), "--offline"])
    survey.main()
    assert out.read_bytes() == recorded.read_bytes()
    rows[0]["results"]["test"]["tokens"][0] = 999
    recorded.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="remeasurement differs"):
        survey.main()
    rows[0]["tokenizer_sha256"] = "wrong tokenizer"
    with pytest.raises(ValueError, match="tokenizer checksum differs"):
        survey.regenerate(rows, docs_root, offline=True)
    (docs_root / "test.jsonl").write_text('{"text": "different documents"}\n')
    with pytest.raises(ValueError, match="documents differ"):
        survey.regenerate(rows, docs_root, offline=True)
    for invalid in ([], rows + rows):
        with pytest.raises(ValueError, match="nonempty and unique"):
            survey.regenerate(invalid, docs_root, offline=True)


@pytest.mark.parametrize("offline,failures", [(False, 1), (False, 3), (True, 1)])
def test_survey_download_retries_are_bounded_and_offline_never_retries(tmp_path, monkeypatch, offline, failures):
    import huggingface_hub

    docs_root, path, rows = _inputs(tmp_path)
    rows[0].update(name="example/tokenizer", revision="fixed-revision")
    calls, waits = [], []

    def download(name, filename, revision, local_files_only):
        assert (name, filename, revision, local_files_only) == (
            "example/tokenizer", "tokenizer.json", "fixed-revision", offline)
        calls.append(name)
        if len(calls) <= failures:
            raise OSError("download failed")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(survey.time, "sleep", waits.append)
    if offline or failures == 3:
        with pytest.raises(OSError, match="download failed"):
            survey.regenerate(rows, docs_root, offline=offline)
    else:
        assert survey.regenerate(rows, docs_root, offline=offline) == rows
    assert len(calls) == (1 if offline else min(failures + 1, 3))
    assert waits == ([] if offline else [1] if failures == 1 else [1, 2])


def test_survey_roundtrip_sample_is_limited_to_the_original_200_documents(tmp_path):
    docs_root, _, rows = _inputs(tmp_path)
    path = docs_root / "test.jsonl"
    path.write_text('{"text": "hello"}\n' * 201)
    rows[0]["inputs"]["test"] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "documents": 201}
    rows[0]["results"]["test"] = {"chars": [5] * 201, "tokens": [1] * 201, "chars_per_token": 5.0}
    rows[0]["roundtrips"] = 200
    assert survey.regenerate(rows, docs_root, offline=True) == rows
