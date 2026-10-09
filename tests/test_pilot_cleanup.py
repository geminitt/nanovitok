"""Contracts of the retained NFC pilot launchers after removing obsolete consumers."""

import json
import shutil
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from vitok import compression, eval_checkpoints, fetch_checkpoints, kaggle_run, train_tokenizers


def test_pilot_queue_keeps_train_eval_commands_and_resumes(tmp_path, tokenizers_dir, monkeypatch):
    data, work, nanochat = tmp_path / "data", tmp_path / "work", tmp_path / "nanochat"
    shutil.copytree(tokenizers_dir, data / "tokenizers")
    (data / "shards").mkdir()
    nanochat.mkdir()
    (data / "compression.json").write_text(json.dumps({
        "bpe-nfc": {"chars_per_token": 4}, "super-nfc": {"chars_per_token": 5}}))
    (data / "test.jsonl").write_text('{"id": 0, "text": "hello"}\n')
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert cmd[:2] == [sys.executable, "-m"]
        base = Path(kwargs["env"]["NANOCHAT_BASE_DIR"])
        assert kwargs["check"] and kwargs["env"]["NANOCHAT_DTYPE"] == "float16"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0" and kwargs["env"]["NANOCHAT_SEED"] == "1"
        assert kwargs["env"]["PYTHONPATH"].split(":")[0] == str(nanochat)
        assert kwargs["stdout"].name.endswith("train.log" if cmd[2] == "scripts.base_train" else "eval.log")
        if cmd[2] == "scripts.base_train":
            assert kwargs["cwd"] == nanochat and "--num-iterations=1" in cmd
            checkpoint = base / "base_checkpoints/d6"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model_000001.pt").write_bytes(b"checkpoint")
            (checkpoint / "optim_000001.pt").write_bytes(b"optimizer")
        else:
            assert cmd[2] == "vitok.eval" and "--pairs" not in cmd and "--variants" not in cmd
            assert cmd[cmd.index("--test") + 1] == str(data / "test.jsonl")
            assert "--model-tag=d6" in cmd
            Path(cmd[cmd.index("--out") + 1]).write_text('{}')

    monkeypatch.setattr(kaggle_run.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["queue", "--gpu", "0", "--conditions", "bpe-nfc", "super-nfc",
                                     "--depth", "6", "--seed", "1", "--data", str(data), "--work", str(work),
                                     "--nanochat", str(nanochat), "--num-iterations", "1"])
    kaggle_run.main()
    assert [cmd[2] for cmd, _ in calls] == ["scripts.base_train", "vitok.eval"] * 2
    assert not list(work.rglob("optim_*.pt"))
    assert len(list((work / "results").glob("*.json"))) == 2
    for condition in ("bpe-nfc", "super-nfc"):
        base = work / "runs" / f"{condition}_d6_s1"
        assert (base / "base_data_climbmix").resolve() == data / "shards"
        assert (base / "tokenizer/token_bytes.pt").is_file()
    calls.clear()
    kaggle_run.main()
    assert not calls


def test_pilot_checkpoint_launcher_filters_nfd_and_resumes(tmp_path, monkeypatch):
    root, out, nanochat = tmp_path / "runs", tmp_path / "out", tmp_path / "nanochat"
    for tag in ("bpe-nfc_d6_s0", "super-nfc_d6_s0", "bpe-nfd_d6_s0"):
        checkpoint = root / tag / "base_checkpoints/d6/model_000001.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        tokenizer = root / tag / "tokenizer/tokenizer.json"
        tokenizer.parent.mkdir()
        tokenizer.write_text('{}')
    assert set(eval_checkpoints.find_runs(root)) == {"bpe-nfc_d6_s0", "super-nfc_d6_s0"}
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        assert cmd[:3] == [sys.executable, "-m", "vitok.eval"]
        assert cmd[cmd.index("--test") + 1] == str((tmp_path / "docs.jsonl").resolve())
        assert cmd[cmd.index("--batch-size") + 1] == "8"
        assert "--variants" not in cmd and "--pairs" not in cmd and "--model-tag=d6" in cmd
        assert kwargs["env"]["NANOCHAT_DTYPE"] == "float32" and kwargs["check"]
        assert kwargs["env"]["NANOCHAT_BASE_DIR"] == str((root / "super-nfc_d6_s0").resolve())
        assert kwargs["env"]["PYTHONPATH"].split(":")[0] == str(nanochat.resolve())
        Path(cmd[cmd.index("--out") + 1]).write_text('{}')
        kwargs["stdout"].write("Scored NFC documents\n")

    monkeypatch.setattr(eval_checkpoints.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["checkpoints", "--runs", str(root), "--docs", str(tmp_path / "docs.jsonl"),
                                     "--nanochat", str(nanochat), "--out", str(out), "--dtype", "float32",
                                     "--only", "super-nfc_d6_s0"])
    eval_checkpoints.main()
    assert len(calls) == 1 and (out / "super-nfc_d6_s0.json").is_file()
    calls.clear()
    eval_checkpoints.main()
    assert not calls
    (out / "super-nfc_d6_s0.json").unlink()
    (root / "super-nfc_d6_s0/tokenizer/tokenizer.json").unlink()
    with pytest.raises(SystemExit, match="tokenizer is missing"):
        eval_checkpoints.main()


@pytest.mark.parametrize("vocab_size", [400, 600])
def test_pilot_tokenizer_and_compression_clis_keep_the_nfc_pair(tmp_path, corpus_file, monkeypatch, vocab_size):
    import pyarrow as pa
    import pyarrow.parquet as pq

    tokenizers, work = tmp_path / "tokenizers", tmp_path / "work"
    monkeypatch.setattr(sys, "argv", ["train", "--corpus", str(corpus_file), "--out", str(tokenizers),
                                     "--workdir", str(work), "--vocab-size", str(vocab_size), "--transition", "0.9"])
    train_tokenizers.main()
    assert {path.name for path in tokenizers.iterdir()} == {"bpe-nfc", "super-nfc", "train_meta.json"}
    assert not (work / "corpus_nfc.txt").exists()
    meta = json.loads((tokenizers / "train_meta.json").read_text())
    assert meta["vocab_size"] == vocab_size and meta["transition"] == 0.9 and meta["nfc"]["n_alphabet"] == 256
    assert meta["nfc"]["n_inherited_merges"] == round(vocab_size * 0.9) - 256
    assert meta["corpus_bytes"] == corpus_file.stat().st_size
    docs = ["Hà Nội là thủ đô.", "Học sinh đi học."]
    parquet, out = tmp_path / "docs.parquet", tmp_path / "compression.json"
    pq.write_table(pa.table({"text": docs + ["This document is excluded."]}), parquet)
    monkeypatch.setattr(sys, "argv", ["compression", "--tokenizers", str(tokenizers), "--docs", str(parquet),
                                     "--n-docs", "2", "--out", str(out)])
    compression.main()
    measured = json.loads(out.read_text())
    assert set(measured) == {"bpe-nfc", "super-nfc", "token_reduction_nfc"}
    for condition in ("bpe-nfc", "super-nfc"):
        tokenizer = Tokenizer.from_file(str(tokenizers / condition / "tokenizer.json"))
        assert measured[condition]["chars_nfc"] == sum(map(len, docs))
        assert measured[condition]["tokens"] == sum(len(tokenizer.encode(text).ids) for text in docs)
        assert all(tokenizer.decode(tokenizer.encode(text).ids) == text for text in docs)
        assert tokenizer.normalizer.normalize_str("Ha\u0300 No\u0323\u0302i") == "Hà Nội"
    assert measured["token_reduction_nfc"] == 1 - measured["super-nfc"]["tokens"] / measured["bpe-nfc"]["tokens"]


def test_pilot_checkpoint_fetch_cli_restores_only_the_ten_retained_runs(tmp_path, tokenizers_dir, monkeypatch):
    import kagglehub

    root = tmp_path / "outputs"
    shutil.copytree(tokenizers_dir, root / "vitok-data/tokenizers-16k")
    calls = []

    def download(handle, path, output_dir):
        assert handle in {f"spritker/tokenizer-vietnamese-train-eval/versions/{version}"
                          for version in (4, 5, 6, 8, 9)}
        assert "nfd" not in path
        calls.append((handle, path))
        target = Path(output_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path)
        return str(target)

    monkeypatch.setattr(kagglehub, "notebook_output_download", download)
    monkeypatch.setattr(fetch_checkpoints.time, "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["fetch", "--out", str(root)])
    fetch_checkpoints.main()
    assert len(calls) == 20
    assert len(list(root.glob("results-v*/runs/*/base_checkpoints/d*/model_*.pt"))) == 10
    assert all(path.parent.joinpath("token_bytes.pt").is_file()
               for path in root.glob("results-v*/runs/*/tokenizer/tokenizer.json"))
    calls.clear()
    fetch_checkpoints.main()
    assert not calls and not (root / ".fetch-staging").exists()
    subset = tmp_path / "selected"
    monkeypatch.setattr(sys, "argv", ["fetch", "--out", str(subset), "--tokenizers",
                                     str(root / "vitok-data/tokenizers-16k"), "--only", "super-nfc_d6_s0"])
    fetch_checkpoints.main()
    assert len(calls) == 2 and all("super-nfc_d6_s0" in path for _, path in calls)
