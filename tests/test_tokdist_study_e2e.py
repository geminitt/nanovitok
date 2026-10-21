"""End to end on a tiny Qwen3 (CPU), the Token Distillation study's sequence (PLAN.md, Stage 2): a mixed-domain stream,
the mean extension, Token Distillation, then `cpt`, `mean-cpt` and `tokdist-cpt` on document-aligned steps with a dev
curve per domain; every condition reads the same text at every step, and a run killed midway resumes to the weights
of the uninterrupted run."""

import json
import os
import random
import signal
import subprocess
import sys
import time

import pytest
import torch
from conftest import ROOT, SENTENCES, qwen_like_base
from test_extend import tiny_qwen3
from tokenizers import Tokenizer

from vitok import adapt, extend, mixed_stream, retrofit, tokdist

STEPS, CHARS, SEQ = 6, 500, 32


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    torch.set_num_threads(1)
    d = tmp_path_factory.mktemp("study")
    base = qwen_like_base()
    corpus = d / "vi.txt"
    corpus.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    multi = retrofit.retrofit(base, retrofit.learn_merges(base, [corpus], "multi", 60), "multi")
    (d / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (d / "multi.json").write_text(json.dumps(multi), encoding="utf-8")
    base_tok, multi_tok = Tokenizer.from_str(json.dumps(base)), Tokenizer.from_str(json.dumps(multi))
    model = tiny_qwen3(base_tok.get_vocab_size(with_added_tokens=True), layers=2)
    model.save_pretrained(d / "cpt-model")
    extend.extend(model, extend.new_token_pieces(base_tok, multi_tok)).save_pretrained(d / "mean-model")
    rng = random.Random(3)
    docs = [(dom, " ".join(rng.choices(SENTENCES, k=rng.randint(1, 10)))) for dom in mixed_stream.DOMAINS
            for _ in range(70)]
    rng.shuffle(docs)
    mixed_stream.write_mixed(d / "stream", iter(docs), {"test": 1}, part_chars=5000)
    for i, dom in enumerate(mixed_stream.DOMAINS):
        (d / f"dev-{dom}.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in SENTENCES[i::3]),
                                            encoding="utf-8")
    assert tokdist.run(tokdist.parse_args([
        "--model", str(d / "mean-model"), "--base-tokenizer", str(d / "base.json"), "--tokenizer",
        str(d / "multi.json"), "--stream", str(d / "stream"), "--out", str(d / "tokdist"), "--snippets", "4",
        "--snippet-len", "16", "--batch", "4", "--lr", "1e-2", "--device", "cpu"])) == 0
    return d


CONDITIONS = {"cpt": ("cpt-model", "base.json"), "mean-cpt": ("mean-model", "multi.json"),
              "tokdist-cpt": ("tokdist/model", "multi.json")}


def argv(d, cond, out):
    model, tok = CONDITIONS[cond]
    dev = [x for dom in mixed_stream.DOMAINS for x in ("--dev-set", f"{dom}={d / f'dev-{dom}.jsonl'}")]
    return ["--model", str(d / model), "--base-tokenizer", str(d / "base.json"), "--tokenizer", str(d / tok),
            "--stream", str(d / "stream"), "--out", str(out), "--steps", str(STEPS), "--chars-per-step", str(CHARS),
            "--micro-batch", "4", "--seq-len", str(SEQ), "--lr", "3e-3", "--warmup", "1", "--layers", "all",
            "--log-every", "1", "--save-every", "2", "--dev-every", "3", "--snapshot-steps", "3", *dev]


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_the_three_conditions_read_the_same_text_and_record_per_domain_curves(study, tmp_path):
    logs = {}
    for cond in CONDITIONS:
        assert adapt.train(adapt.parse_args(argv(study, cond, tmp_path / cond))) == 0
        logs[cond] = lines(tmp_path / cond / "train_log.jsonl")
        dev = lines(tmp_path / cond / "dev_log.jsonl")
        assert [(r["step"], r["set"]) for r in dev] == [(s, n) for s in (0, 3, 6) for n in mixed_stream.DOMAINS]
        assert all(r["bpb"] > 0 for r in dev)
        assert (tmp_path / cond / "snapshots" / "step_000003.pt").exists()
    digests = {c: [r["digest"] for r in rec] for c, rec in logs.items()}
    assert digests["cpt"] == digests["mean-cpt"] == digests["tokdist-cpt"]
    assert [r["domain_chars"] for r in logs["cpt"]] == [r["domain_chars"] for r in logs["tokdist-cpt"]]
    assert logs["mean-cpt"][-1]["tokens"] == logs["tokdist-cpt"][-1]["tokens"] < logs["cpt"][-1]["tokens"]
    # the two extended conditions start from different new rows and the same everything else
    a = torch.load(tmp_path / "mean-cpt" / "final.pt", weights_only=False)
    b = torch.load(tmp_path / "tokdist-cpt" / "final.pt", weights_only=False)
    assert a["new_ids"] == b["new_ids"] and not torch.equal(a["params"][adapt.EMBED], b["params"][adapt.EMBED])


def test_a_killed_aligned_run_resumes_to_the_weights_of_the_whole_run(study, tmp_path):
    steps = 30
    whole, killed = tmp_path / "whole", tmp_path / "killed"
    a = argv(study, "tokdist-cpt", whole)
    a[a.index("--steps") + 1] = str(steps)
    assert adapt.train(adapt.parse_args(a)) == 0
    b = argv(study, "tokdist-cpt", killed)
    b[b.index("--steps") + 1] = str(steps)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src")}
    code = ("import sys, torch; torch.set_num_threads(1); from vitok import adapt; "
            "sys.exit(adapt.train(adapt.parse_args(sys.argv[1:])))")
    proc = subprocess.Popen([sys.executable, "-c", code, *b], env=env, cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    deadline = time.monotonic() + 120
    log = killed / "train_log.jsonl"
    while time.monotonic() < deadline and proc.poll() is None:
        if log.exists() and len(log.read_text().splitlines()) >= 7:
            break
        time.sleep(0.05)
    assert proc.poll() is None, proc.stderr.read().decode()
    proc.send_signal(signal.SIGKILL)
    assert proc.wait(60) == -signal.SIGKILL
    assert adapt.train(adapt.parse_args(b)) == 0
    x = torch.load(whole / "final.pt", weights_only=False)
    y = torch.load(killed / "final.pt", weights_only=False)
    assert all(torch.equal(x["params"][k], y["params"][k]) for k in x["params"])
    assert [r["digest"] for r in lines(log)] == [r["digest"] for r in lines(whole / "train_log.jsonl")]
    dev = lambda p: [{k: v for k, v in r.items() if k != "seconds"} for r in lines(p / "dev_log.jsonl")]  # noqa: E731
    assert dev(killed) == dev(whole)
