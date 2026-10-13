"""Evaluation-queue contracts: each job's scoring command (model, weights, sets, speed only where asked), finished jobs
skipped, the deadline handed down and a stopped job ending the queue, failures isolated and reported; and one real
job on a tiny model."""

import json
import sys

import pytest

from vitok import evaluate
from vitok.adapt import EXIT_INCOMPLETE


def plan(d) -> dict:
    return {"score": {"batch_size": 8, "dtype": "fp16"},
            "sets": {"full": {"lm": {"web": "web.jsonl", "law": "law.jsonl"}, "mc": {"quiz": "quiz.jsonl"}},
                     "lm": {"lm": {"web": "web.jsonl", "law": "law.jsonl"}}},
            "speed": {"docs": "web.jsonl", "prompts": 20, "new_tokens": 256},
            "jobs": {"original": {"model": "/opt/model", "revision": "abc", "tokenizer": "base.json", "sets": "full",
                                  "speed": True, "out": str(d / "original")},
                     "multi-final": {"model": "/w/multi/extended", "tokenizer": "multi.json",
                                     "snapshot": "/w/multi/train/final.pt", "sets": "full", "speed": True,
                                     "out": str(d / "multi-final")},
                     "multi-step_000153": {"model": "/w/multi/extended", "tokenizer": "multi.json",
                                           "snapshot": "/w/multi/train/snapshots/step_000153.pt", "sets": "lm",
                                           "out": str(d / "multi-step_000153")}}}


def opt(a, k):
    return a[a.index(k) + 1]


def test_each_job_scores_its_weights_on_its_sets_and_measures_speed_only_where_asked(tmp_path):
    p = plan(tmp_path)
    full = evaluate.command(p, "multi-final")
    assert full[:3] == [sys.executable, "-m", "vitok.score"]
    assert (opt(full, "--model"), opt(full, "--tokenizer"), opt(full, "--snapshot")) == \
        ("/w/multi/extended", "multi.json", "/w/multi/train/final.pt")
    i = full.index("--lm")
    assert full[i + 1:i + 3] == ["web=web.jsonl", "law=law.jsonl"] and opt(full, "--mc") == "quiz=quiz.jsonl"
    assert (opt(full, "--speed-docs"), opt(full, "--speed-prompts"), opt(full, "--new-tokens")) == \
        ("web.jsonl", "20", "256")
    assert (opt(full, "--dtype"), opt(full, "--batch-size"), opt(full, "--out")) == \
        ("fp16", "8", str(tmp_path / "multi-final"))
    assert "--revision" not in full
    snap = evaluate.command(p, "multi-step_000153")
    assert "--mc" not in snap and "--speed-prompts" not in snap and opt(snap, "--snapshot").endswith("153.pt")
    orig = evaluate.command(p, "original")
    assert opt(orig, "--revision") == "abc" and "--snapshot" not in orig


class Fake:
    def __init__(self, codes=None):
        self.codes, self.calls = codes or {}, []

    def __call__(self, argv, log):
        out = argv[argv.index("--out") + 1]
        name = out.rsplit("/", 1)[-1]
        self.calls.append((name, float(argv[argv.index("--deadline-minutes") + 1])))
        code = self.codes.get(name, 0)
        if code == 0:
            import pathlib
            pathlib.Path(out).mkdir(parents=True, exist_ok=True)
            (pathlib.Path(out) / "summary.json").write_text("{}")
        return code


def write(tmp_path, p):
    (tmp_path / "plan.json").write_text(json.dumps(p))
    return tmp_path / "plan.json"


def test_finished_jobs_are_skipped_and_the_deadline_is_handed_down(tmp_path):
    path = write(tmp_path, plan(tmp_path))
    fake = Fake()
    args = ["--plan", str(path), "--jobs", "original", "multi-final", "--deadline-minutes", "30"]
    assert evaluate.main(args, run=fake) == 0
    assert [n for n, _ in fake.calls] == ["original", "multi-final"] and all(0 < m <= 30 for _, m in fake.calls)
    fake.calls.clear()
    assert evaluate.main(args, run=fake) == 0 and fake.calls == []
    out = json.loads((tmp_path / "evaluate-original-multi-final.json").read_text())
    assert out == {"original": {"status": "done"}, "multi-final": {"status": "done"}}


def test_a_failure_is_isolated_and_a_deadline_stop_ends_the_queue(tmp_path):
    path = write(tmp_path, plan(tmp_path))
    jobs = ["original", "multi-final", "multi-step_000153"]
    fake = Fake({"original": 1})
    assert evaluate.main(["--plan", str(path), "--jobs", *jobs, "--deadline-minutes", "30"], run=fake) == 1
    out = json.loads((tmp_path / f"evaluate-{'-'.join(jobs)}.json").read_text())
    assert out["original"]["status"] == "failed" and out["multi-final"]["status"] == "done"
    fake = Fake({"original": EXIT_INCOMPLETE})
    assert evaluate.main(["--plan", str(path), "--jobs", *jobs, "--deadline-minutes", "30"],
                         run=fake) == EXIT_INCOMPLETE
    out = json.loads((tmp_path / f"evaluate-{'-'.join(jobs)}.json").read_text())
    assert out["original"]["status"] == "incomplete" and out["multi-step_000153"]["status"] == "not started"


@pytest.mark.parametrize("change,message", [
    (lambda p: p["jobs"]["original"].update(sets="other"), "unknown set group"),
    (lambda p: p["jobs"]["original"].pop("out"), "lacks"),
])
def test_malformed_plans_and_unknown_jobs_are_refused(tmp_path, change, message):
    p = plan(tmp_path)
    change(p)
    with pytest.raises(ValueError, match=message):
        evaluate.load_plan(write(tmp_path, p))
    with pytest.raises(SystemExit, match="not in"):
        evaluate.main(["--plan", str(write(tmp_path, plan(tmp_path))), "--jobs", "nope", "--deadline-minutes", "1"])


def test_a_real_job_scores_a_tiny_model(tmp_path, monkeypatch):
    from conftest import ROOT, SENTENCES, qwen_like_base
    from test_extend import tiny_qwen3
    from tokenizers import Tokenizer
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    base = qwen_like_base()
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    tiny_qwen3(Tokenizer.from_str(json.dumps(base)).get_vocab_size(with_added_tokens=True)).save_pretrained(
        tmp_path / "qwen")
    (tmp_path / "web.jsonl").write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n"
                                                for t in SENTENCES[:3]), encoding="utf-8")
    p = {"score": {"batch_size": 2, "dtype": "fp32"}, "sets": {"lm": {"lm": {"web": str(tmp_path / "web.jsonl")}}},
         "speed": {"docs": str(tmp_path / "web.jsonl"), "prompts": 1, "new_tokens": 3},
         "jobs": {"tiny": {"model": str(tmp_path / "qwen"), "tokenizer": str(tmp_path / "base.json"), "sets": "lm",
                           "speed": True, "out": str(tmp_path / "tiny")}}}
    code = evaluate.main(["--plan", str(write(tmp_path, p)), "--jobs", "tiny", "--deadline-minutes", "10"])
    assert code == 0, (tmp_path / "tiny.log").read_text()[-2000:]
    s = json.loads((tmp_path / "tiny" / "summary.json").read_text())
    assert s["lm"]["web"]["docs"] == 3 and s["generation"]["prompts"] == 1
