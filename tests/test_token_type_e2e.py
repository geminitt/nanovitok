"""The token-type experiment end to end on CPU with a tiny Qwen3 and real subprocesses: seeded stream, extension,
scoring of named sets before, during and after training, full-parameter training with the dev curve and bf16
snapshots, a queue killed mid-training and resumed by the same command to the same weights as an uninterrupted run."""

import json
import os
import random
import signal
import subprocess
import sys
import time

import torch
from conftest import ROOT, SENTENCES, qwen_like_base
from test_extend import tiny_qwen3
from tokenizers import Tokenizer

from vitok import remote, retrofit, stream

STEPS = 8


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def setup(d):
    base = qwen_like_base()
    corpus = d / "merge.txt"
    corpus.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    multi = retrofit.retrofit(base, retrofit.learn_merges(base, [corpus], "multi", 120), "multi")
    (d / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (d / "multi.json").write_text(json.dumps(multi), encoding="utf-8")
    torch.manual_seed(0)
    tiny_qwen3(Tokenizer.from_str(json.dumps(base)).get_vocab_size(with_added_tokens=True), layers=3).save_pretrained(
        d / "qwen")
    rng = random.Random(0)
    docs = [" ".join(rng.choices(SENTENCES, k=rng.randint(1, 4))) for _ in range(400)]
    stream.build(d / "stream-0", 0, sum(map(len, docs)), 3000, set(), [("u", i) for i in range(8)],
                 lambda name, rg: docs[rg * 50:(rg + 1) * 50], {"test": 1})
    jsonl(d / "dev.jsonl", [{"text": t} for t in SENTENCES[:4]])
    jsonl(d / "web.jsonl", [{"id": str(i), "text": " ".join(SENTENCES[i:] + SENTENCES[:i])} for i in range(4)])
    jsonl(d / "law.jsonl", [{"id": "l", "text": "Điều 1. " + SENTENCES[2]}])
    jsonl(d / "quiz.jsonl", [{"id": f"q{i}", "context": SENTENCES[i] + "\nTrả lời:", "gold": i % 2,
                              "choices": [" có", " không"]} for i in range(3)])
    jsonl(d / "check.jsonl", [{"text": "The quick brown fox, don't stop!"}])
    cfg = {"model": str(d / "qwen"), "revision": "x", "base_tokenizer": str(d / "base.json"),
           "check_texts": str(d / "check.jsonl"),
           "conditions": {"base": {"tokenizer": str(d / "base.json")},
                          "multisyllable": {"tokenizer": str(d / "multi.json"), "lr": 2e-3}},
           "train": {"steps": STEPS, "batch": 4, "seq_len": 32, "lr": 3e-3, "warmup": 1, "layers": "all",
                     "weight_decay": 0.1, "save_every": 2, "log_every": 1, "dev_every": 3, "snapshot_steps": [4],
                     "snapshot_dtype": "bf16", "seed": 0},
           "score": {"lm": {"web": str(d / "web.jsonl"), "law": str(d / "law.jsonl")},
                     "mc": {"quiz": str(d / "quiz.jsonl")}, "speed_docs": str(d / "web.jsonl"), "speed_prompts": 1,
                     "new_tokens": 3, "batch_size": 2, "snapshots": True},
           "streams": {"0": str(d / "stream-0")}, "dev_docs": str(d / "dev.jsonl")}
    (d / "run.json").write_text(json.dumps(cfg), encoding="utf-8")


def test_the_queue_survives_a_kill_and_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # bit-identical CPU results across processes
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    setup(tmp_path)
    conds = ["base", "multisyllable"]
    whole = remote.run_queue(tmp_path / "run.json", conds, tmp_path / "whole", 30)
    assert whole["exit_code"] == 0, (tmp_path / "whole" / "queue.log").read_text()[-3000:]

    # the same queue in a child process group, killed once base has trained a few steps, then rerun as is
    work = tmp_path / "killed"
    code = ("import sys; from pathlib import Path; from vitok import remote; "
            "remote.run_queue(Path(sys.argv[1]), ['base', 'multisyllable'], Path(sys.argv[2]), 30)")
    proc = subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "run.json"), str(work)],
                            start_new_session=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    log = work / "base" / "train" / "train_log.jsonl"
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline and proc.poll() is None:
        if log.exists() and len(log.read_text().splitlines()) >= 5:
            break
        time.sleep(0.05)
    assert proc.poll() is None, proc.stderr.read().decode()
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(30)
    assert not (work / "base" / "train" / "final.pt").exists()
    again = remote.run_queue(tmp_path / "run.json", conds, work, 30)
    assert again["exit_code"] == 0, (work / "queue.log").read_text()[-3000:]

    for name in conds:
        a = torch.load(tmp_path / "whole" / name / "train" / "final.pt", map_location="cpu", weights_only=False)
        b = torch.load(work / name / "train" / "final.pt", map_location="cpu", weights_only=False)
        assert a["params"].keys() == b["params"].keys() and all(t.dtype == torch.bfloat16 for t in a["params"].values())
        assert all(torch.equal(a["params"][k], b["params"][k]) for k in a["params"]), name
        assert any(k.startswith("model.norm.") for k in a["params"])  # full-parameter mode trains the final norm
        dev = [json.loads(line) for line in (work / name / "train" / "dev_log.jsonl").read_text().splitlines()]
        ref = [json.loads(line) for line in (tmp_path / "whole" / name / "train" / "dev_log.jsonl").read_text()
               .splitlines()]
        assert [r["step"] for r in dev] == [0, 3, 6, 8] and [r["bpc"] for r in dev] == [r["bpc"] for r in ref]
        train = [json.loads(line) for line in (work / name / "train" / "train_log.jsonl").read_text().splitlines()]
        assert [r["step"] for r in train] == list(range(STEPS)) and all(r["train_bpc"] > 0 for r in train)
        for part in ("score-before", "score-step_000004", "score-final"):
            s = json.loads((work / name / part / "summary.json").read_text())
            assert set(s["lm"]) == {"web", "law"} and s["mc"]["quiz"]["items"] == 3
            assert ("generation" in s) == (part != "score-step_000004")
    m, b = (json.loads((work / n / "train" / "manifest.json").read_text()) for n in ("multisyllable", "base"))
    assert m["tokens"] == b["tokens"] and m["chars"] > b["chars"]
    assert m["signature"]["lr"] == 2e-3 and b["signature"]["lr"] == 3e-3
    assert m["signature"]["stream"] == b["signature"]["stream"] is not None
