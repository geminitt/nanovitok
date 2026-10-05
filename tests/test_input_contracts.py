"""Contract shared by every module: input from outside the code (CLI flags, files, downloads) that cannot be used is
refused with a clear error, never with a bare assert (asserts are for this code's own invariants)."""

import sys
import types

import pytest


def _parse_bad_condition(tmp_path, tokenizers_dir):
    from vitok.tokenizer_spec import parse
    parse("bpe-nfkc")


def _unknown_special_token(tmp_path, tokenizers_dir):
    from vitok.hf_tokenizer import HFTokenizer
    HFTokenizer.from_directory(tokenizers_dir / "bpe-nfc").encode_special("<|nope|>")


def _device_batch_not_dividing(tmp_path, tokenizers_dir):
    from vitok.conditions import train_args
    train_args({"bpe-nfc": 4.0, "super-nfc": 5.0}, "super-nfc", depth=8, device_batch=48)


def _missing_chars_per_token(tmp_path, tokenizers_dir):
    from vitok.conditions import train_args
    train_args({"bpe-nfc": 4.0}, "super-nfc", depth=8)


def _no_merges_to_inherit(tmp_path, tokenizers_dir):
    from vitok.train_tokenizers import train_all
    corpus = tmp_path / "c.txt"
    corpus.write_text("học sinh\n" * 50, encoding="utf-8")
    train_all(corpus, tmp_path / "out", vocab_size=260, transition=0.9, workroot=tmp_path)


def _compression_on_no_text(tmp_path, tokenizers_dir):
    from tokenizers import Tokenizer
    from vitok.compression import stats_for
    stats_for(Tokenizer.from_file(str(tokenizers_dir / "bpe-nfc" / "tokenizer.json")), ["", ""])


def _two_copies_of_a_run(tmp_path, tokenizers_dir):
    from vitok.eval_checkpoints import find_runs
    for copy in ("a", "b"):
        ckpt = tmp_path / copy / "runs" / "bpe-nfc_d6_s0" / "base_checkpoints" / "d6" / "model_000001.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"")
    find_runs(tmp_path)


def _download_returns_no_file(tmp_path, tokenizers_dir):
    from vitok.fetch_checkpoints import kaggle_download
    fake = types.SimpleNamespace(notebook_output_download=lambda *a, **k: str(tmp_path / "missing.pt"))
    saved = sys.modules.get("kagglehub")
    sys.modules["kagglehub"] = fake
    try:
        kaggle_download(4, "runs/x/model.pt", tmp_path, attempts=1)
    finally:
        if saved is None:
            del sys.modules["kagglehub"]
        else:
            sys.modules["kagglehub"] = saved


CASES = [
    (_parse_bad_condition, ValueError, "unknown condition"),
    (_unknown_special_token, ValueError, "no special token"),
    (_device_batch_not_dividing, ValueError, "must divide"),
    (_missing_chars_per_token, ValueError, "chars-per-token"),
    (_no_merges_to_inherit, ValueError, "leaves no merges"),
    (_compression_on_no_text, ValueError, "no text"),
    (_two_copies_of_a_run, ValueError, "two copies"),
    (_download_returns_no_file, FileNotFoundError, "not a file"),
]


@pytest.mark.parametrize("case,error,message", CASES, ids=[c.__name__.strip("_") for c, _, _ in CASES])
def test_outside_input_is_refused_with_a_clear_error(tmp_path, tokenizers_dir, case, error, message):
    with pytest.raises(error, match=message):
        case(tmp_path, tokenizers_dir)


@pytest.mark.parametrize("condition,parsed", [("bpe-nfc", ("bpe", "nfc")), ("bpe-nfd", ("bpe", "nfd")),
                                              ("super-nfc", ("super", "nfc")), ("super-nfd", ("super", "nfd")),
                                              ("bpe-nfkc", None), ("super", None), ("gpt-nfc", None), ("BPE-NFC", None)])
def test_condition_names_parse_exactly(condition, parsed):
    from vitok.tokenizer_spec import parse
    if parsed is None:
        with pytest.raises(ValueError, match="unknown condition"):
            parse(condition)
    else:
        assert parse(condition) == parsed
