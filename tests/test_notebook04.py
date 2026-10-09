"""Run every speed-session cell with real tiny models and subprocesses on CPU."""

import hashlib
import json
import runpy
import subprocess
import sys
import time

import pytest
import torch
from conftest import REPO
from test_notebook03 import kaggle_input as kaggle_input
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from vitok import adapt, extend, kaggle_session

NOTEBOOK = REPO / "kaggle" / "notebooks" / "04_generation_speed.ipynb"


def test_every_speed_model_runs_in_one_session_and_old_timings_are_refused(kaggle_input, tmp_path, monkeypatch):
    root = kaggle_input
    r = root / "input" / "vitok-code" / "retrofit"
    base = Tokenizer.from_file(str(r / "base" / "tokenizer.json"))
    previous = root / "input" / "finished-training" / "runs"
    for name in ("base", "syllable-1000", "multisyllable-1000", "multisyllable-1000-s1"):
        folder = previous / name
        (folder / "train").mkdir(parents=True, exist_ok=True)
        kind = name.removesuffix("-s1")
        new = Tokenizer.from_file(str(r / kind / "tokenizer.json"))
        model = AutoModelForCausalLM.from_pretrained(root / "qwen")
        if name != "base":
            extend.extend(model, extend.new_token_pieces(base, new))
            model.save_pretrained(folder / "extended")
            (folder / "extended" / "extend_manifest.json").write_text(json.dumps(
                {"model": str(root / "qwen"), "revision": "test"}))
        ids = adapt.new_token_ids(base, new)
        trainable = adapt.set_trainable(model, ids, [0, 1, 2, 3])
        torch.save({"params": adapt.trainable_state(trainable, ids), "new_ids": [ids.start, ids.stop],
                    "step": 1200, "tokens": 19660800, "chars": 100, "kind": "final"}, folder / "train" / "final.pt")
        (folder / "score-final").mkdir(exist_ok=True)
        sha = hashlib.sha256((folder / "train" / "final.pt").read_bytes()).hexdigest()
        (folder / "score-final" / "summary.json").write_text(json.dumps({"signature": {"snapshot_sha256": sha}}))
        (folder / "train" / "manifest.json").write_text(json.dumps(
            {"git_commit": "test", "signature": {"steps": 1200, "batch": 32, "seq_len": 512}}))
    monkeypatch.setenv("VITOK_KAGGLE_INPUT", str(root / "input"))
    monkeypatch.setenv("VITOK_KAGGLE_WORK", str(tmp_path / "speed"))
    monkeypatch.setattr(kaggle_session, "preflight", lambda need_gpus: [{"name": "cpu"}] * need_gpus)
    monkeypatch.setattr(kaggle_session, "prefetch", lambda model, revision: str(model))
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]
    ns: dict = {}
    exec(cells[0], ns)
    assert (ns["SPEED_PROMPTS"], ns["NEW_TOKENS"], ns["PROMPT_CHARS"], ns["SESSION_MINUTES"]) == (20, 256, 500, 60)
    ns.update(MODEL=str(root / "qwen"), REVISION="test", DTYPE="fp32", SPEED_PROMPTS=2, NEW_TOKENS=4)
    for cell in cells[1:]:
        exec(cell, ns)
    result = json.loads((tmp_path / "speed" / "session-results.json").read_text())
    assert result["status"] == "done", result
    names = {"original", "syllable-1000-before", "multisyllable-1000-before", "base",
             "syllable-1000", "multisyllable-1000", "multisyllable-1000-s1"}
    assert set(result["models"]) == names
    signatures = set()
    for name, rec in result["models"].items():
        assert rec["status"] == "done" and rec["session_id"] == result["session_id"]
        summary = json.loads((tmp_path / "speed" / name / "summary.json").read_text())
        assert summary["generation"]["prompts"] == 2 and summary["generation"]["new_tokens"] == 4
        signatures.add(summary["signature"]["vi_docs_sha256"])
    assert len(signatures) == 1
    assert sorted(map(len, ns["QUEUES"].values())) == [3, 4]
    # A new notebook invocation may not splice timing records from a previous session.
    with pytest.raises(RuntimeError, match="new empty work directory"):
        exec(cells[1], ns)
    # A killed worker may leave only a prefix of its queue; that is never a complete session.
    queue_path = tmp_path / "speed" / "queue-gpu0.json"
    original_queue = queue_path.read_text()
    partial = json.loads(original_queue)
    partial.pop("original")
    queue_path.write_text(json.dumps(partial))
    with pytest.raises(RuntimeError, match="speed session incomplete"):
        exec(cells[-1], ns)
    queue_path.write_text(original_queue)
    # The report must reject stale score signatures, even inside the same output folder.
    summary_path = tmp_path / "speed" / "original" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["signature"]["new_tokens"] += 1
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="signature does not match"):
        exec(cells[-1], ns)
    # Guard the input contracts before paying for any GPU scoring.
    final_summary = previous / "base" / "score-final" / "summary.json"
    original = final_summary.read_text()
    final_summary.write_text(json.dumps({"signature": {"snapshot_sha256": "wrong"}}))
    ns["WORK"] = tmp_path / "bad-checkpoint"
    with pytest.raises(RuntimeError, match="checkpoint differs"):
        exec(cells[1], ns)
    final_summary.write_text(original)
    training = previous / "base" / "train" / "manifest.json"
    original = training.read_text()
    changed = json.loads(original)
    changed["signature"]["steps"] = 1199
    training.write_text(json.dumps(changed))
    ns["WORK"] = tmp_path / "bad-budget"
    with pytest.raises(RuntimeError, match="unexpected training budget"):
        exec(cells[1], ns)
    training.write_text(original)
    ext = previous / "syllable-1000" / "extended" / "extend_manifest.json"
    original = ext.read_text()
    ext.write_text(json.dumps({"model": "wrong", "revision": "wrong"}))
    ns["WORK"] = tmp_path / "bad-revision"
    with pytest.raises(RuntimeError, match="wrong starting-model revision"):
        exec(cells[1], ns)
    ext.write_text(original)
    # Execute the exact generated worker for deadline, timeout and isolated failure paths.
    worker = tmp_path / "speed" / "speed_worker.py"
    cfg = json.loads((tmp_path / "speed" / "session-manifest.json").read_text())
    cfg["queues"] = {"0": ["original", "base", "syllable-1000"]}
    for mode in ("expired", "timeout", "failed"):
        work = tmp_path / mode
        work.mkdir()
        cfg.update(work=str(work), deadline=time.monotonic() + (-1 if mode == "expired" else 60))
        config_path = work / "config.json"
        config_path.write_text(json.dumps(cfg))
        calls = []

        def run(argv, mode=mode, calls=calls, **kwargs):
            calls.append(argv)
            if mode == "timeout":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            return subprocess.CompletedProcess(argv, 2 if len(calls) == 1 else 0)

        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(sys, "argv", [str(worker), str(config_path), "0"])
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(worker), run_name="__main__")
        outcomes = json.loads((work / "queue-gpu0.json").read_text())
        assert stopped.value.code == 1 and len(outcomes) == 3
        if mode == "failed":
            assert [r["status"] for r in outcomes.values()] == ["failed", "done", "done"]
        else:
            assert all(r["status"] == "incomplete" for r in outcomes.values())
        assert len(calls) == (0 if mode == "expired" else 3)
