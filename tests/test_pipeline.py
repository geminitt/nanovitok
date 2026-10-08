"""Queue driver contracts: equal settings across conditions, resume by skipping finished steps, failures isolated,
the deadline handed down; and one end-to-end run of the whole queue on a tiny Qwen3 (CPU, real subprocesses)."""

import json
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from conftest import SENTENCES, qwen_like_base
from test_extend import tiny_qwen3
from tokenizers import Tokenizer

from vitok import adapt, pipeline, retrofit


def config(d, **over) -> dict:
    cfg = {"model": "Qwen/Qwen3-0.6B", "revision": "abc", "base_tokenizer": str(d / "base.json"),
           "check_texts": str(d / "check.jsonl"),
           "conditions": {"base": {"tokenizer": str(d / "base.json")},
                          "multisyllable": {"tokenizer": str(d / "multi.json")},
                          "multisyllable-s1": {"tokenizer": str(d / "multi.json"), "seed": 1}},
           "train": {"steps": 6, "batch": 4, "seq_len": 32, "lr": 3e-3, "warmup": 1, "layers": "0,-1",
                     "save_every": 2, "log_every": 1, "snapshot_steps": [3], "seed": 0},
           "score": {"vi_docs": str(d / "vi.jsonl"), "batch_size": 2, "snapshots": True, "speed_prompts": 1,
                     "new_tokens": 4},
           "shards": [str(d / "shard_00000.parquet")], "held_out": [str(d / "held.jsonl")]}
    cfg.update(over)
    return cfg


def adapt_argv(cfg, name, work):
    return next(a for s, _, a in pipeline.commands(cfg, name, work) if s == "adapt")


def test_every_condition_trains_with_the_same_settings(tmp_path):
    cfg = config(tmp_path)
    per = {}
    for name in cfg["conditions"]:
        a = adapt_argv(cfg, name, tmp_path)
        drop = {"--model", "--revision", "--tokenizer", "--out", "--seed"}
        per[name] = [x for i, x in enumerate(a) if x not in drop and a[i - 1] not in drop]
    assert per["base"] == per["multisyllable"] == per["multisyllable-s1"]
    seed = lambda n: (lambda a: a[a.index("--seed") + 1])(adapt_argv(cfg, n, tmp_path))
    assert (seed("base"), seed("multisyllable"), seed("multisyllable-s1")) == ("0", "0", "1")


def test_extended_conditions_start_from_their_extension_and_base_from_the_hub(tmp_path):
    cfg = config(tmp_path)
    steps = {n: [s for s, _, _ in pipeline.commands(cfg, n, tmp_path)] for n in ("base", "multisyllable")}
    assert steps["base"] == ["score-before", "adapt", "score-step_000003", "score-final"]
    assert steps["multisyllable"] == ["extend", *steps["base"]]
    a = adapt_argv(cfg, "base", tmp_path)
    assert a[a.index("--model") + 1] == "Qwen/Qwen3-0.6B" and a[a.index("--revision") + 1] == "abc"
    m = adapt_argv(cfg, "multisyllable", tmp_path)
    assert m[m.index("--model") + 1] == str(tmp_path / "multisyllable" / "extended") and "--revision" not in m
    speed = {s: "--speed-prompts" in a for s, _, a in pipeline.commands(cfg, "base", tmp_path)}
    assert speed == {"score-before": True, "adapt": False, "score-step_000003": False, "score-final": True}


class FakeRun:
    """Creates each step's output (or not) and returns a scripted exit code."""

    def __init__(self, cfg, work, codes=None):
        self.marks = {tuple(a): done for n in cfg["conditions"] for _, done, a in pipeline.commands(cfg, n, work)}
        self.codes, self.calls = codes or {}, []

    def __call__(self, argv, log):
        base = tuple(argv[:-2]) if "--deadline-minutes" in argv else tuple(argv)
        self.calls.append(argv)
        code = next((c for key, c in self.codes.items() if key in " ".join(argv)), 0)
        if code == 0:
            self.marks[base].parent.mkdir(parents=True, exist_ok=True)
            self.marks[base].write_text("{}")
        return code


def test_finished_steps_are_skipped_and_the_deadline_is_handed_down(tmp_path):
    cfg = config(tmp_path)
    run = FakeRun(cfg, tmp_path)
    assert pipeline.run_condition(cfg, "multisyllable", tmp_path, pipeline.time.monotonic() + 600, run) == \
        ["extend", "score-before", "adapt", "score-step_000003", "score-final"]
    assert "--deadline-minutes" not in run.calls[0]
    assert all(0 < float(a[a.index("--deadline-minutes") + 1]) <= 10 for a in run.calls[1:])
    assert pipeline.run_condition(cfg, "multisyllable", tmp_path, pipeline.time.monotonic() + 600, run) == []
    with pytest.raises(pipeline.Incomplete):
        pipeline.run_condition(cfg, "base", tmp_path, pipeline.time.monotonic() - 1, run)


@pytest.mark.parametrize("codes,error", [({"vitok.adapt": adapt.EXIT_INCOMPLETE}, pipeline.Incomplete),
                                         ({"vitok.adapt": 1}, RuntimeError)])
def test_a_step_that_stops_or_fails_ends_the_condition(tmp_path, codes, error):
    cfg = config(tmp_path)
    with pytest.raises(error):
        pipeline.run_condition(cfg, "base", tmp_path, pipeline.time.monotonic() + 600, FakeRun(cfg, tmp_path, codes))
    assert not (tmp_path / "base" / "score-final").exists()


def test_exit_zero_without_the_output_is_a_failure(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(RuntimeError, match="is missing"):
        pipeline.run_condition(cfg, "base", tmp_path, pipeline.time.monotonic() + 600, lambda a, log: 0)


def test_the_queue_continues_past_a_failing_condition_and_reports_each(tmp_path):
    cfg = config(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps(cfg))
    run = FakeRun(cfg, tmp_path, {"multi.json --check-texts": 1})  # the extension of multisyllable fails
    code = pipeline.main(["--config", str(tmp_path / "run.json"), "--conditions", "multisyllable", "base",
                          "--work", str(tmp_path), "--deadline-minutes", "10"], run=run)
    out = json.loads((tmp_path / "queue-multisyllable-base.json").read_text())
    assert code == 1 and out["multisyllable"]["status"] == "failed" and out["base"]["status"] == "done"


def test_a_deadline_stop_marks_the_rest_not_started(tmp_path):
    cfg = config(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps(cfg))
    run = FakeRun(cfg, tmp_path, {"vitok.adapt": adapt.EXIT_INCOMPLETE})
    code = pipeline.main(["--config", str(tmp_path / "run.json"), "--conditions", "base", "multisyllable",
                          "--work", str(tmp_path), "--deadline-minutes", "10"], run=run)
    out = json.loads((tmp_path / "queue-base-multisyllable.json").read_text())
    assert code == adapt.EXIT_INCOMPLETE
    assert out == {"base": {"status": "incomplete", "at": "adapt"}, "multisyllable": {"status": "not started"}}


@pytest.mark.parametrize("change,message", [(lambda c: c.pop("check_texts"), "lacks"),
                                            (lambda c: c["conditions"]["base"].pop("tokenizer"), "no tokenizer"),
                                            (lambda c: c["train"].update(stepz=3), "unknown training settings")])
def test_malformed_configs_are_refused(tmp_path, change, message):
    cfg = config(tmp_path)
    change(cfg)
    (tmp_path / "run.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=message):
        pipeline.load_config(tmp_path / "run.json")
    good = config(tmp_path)
    (tmp_path / "good.json").write_text(json.dumps(good))
    with pytest.raises(SystemExit, match="not in"):
        pipeline.main(["--config", str(tmp_path / "good.json"), "--conditions", "nope", "--work", str(tmp_path),
                       "--deadline-minutes", "1"])


def test_the_whole_queue_runs_end_to_end_on_a_tiny_model(tmp_path):
    """Every subprocess the Kaggle run takes, once, on CPU: extension, scoring before, training, snapshot and final
    scoring, for a base and an extended condition; a rerun finds everything done."""
    torch.manual_seed(0)
    base = qwen_like_base()
    corpus = tmp_path / "vi.txt"
    corpus.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    multi = retrofit.retrofit(base, retrofit.learn_merges(base, [corpus], "multi", 120), "multi")
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "multi.json").write_text(json.dumps(multi), encoding="utf-8")
    rows = Tokenizer.from_str(json.dumps(base)).get_vocab_size(with_added_tokens=True)
    tiny_qwen3(rows, layers=4).save_pretrained(tmp_path / "qwen")
    rng = random.Random(0)
    docs = [" ".join(rng.choices(SENTENCES, k=rng.randint(1, 4))) for _ in range(300)]
    pq.write_table(pa.table({"text": docs}), tmp_path / "shard_00000.parquet")
    write = lambda name, texts: (tmp_path / name).write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")
    write("held.jsonl", docs[:20:4])
    write("vi.jsonl", [" ".join(SENTENCES[i:] + SENTENCES[:i]) for i in range(4)])
    write("check.jsonl", ["The quick brown fox, don't stop!", "2026: 1.250.000"])
    cfg = config(tmp_path, model=str(tmp_path / "qwen"))
    del cfg["conditions"]["multisyllable-s1"]
    (tmp_path / "run.json").write_text(json.dumps(cfg))
    args = ["--config", str(tmp_path / "run.json"), "--conditions", "base", "multisyllable",
            "--work", str(tmp_path / "work"), "--deadline-minutes", "20"]
    code = pipeline.main(args)
    logs = {p.name: p.read_text()[-2000:] for p in (tmp_path / "work").rglob("*.log")}
    assert code == 0, logs
    for name in ("base", "multisyllable"):
        for part in ("score-before", "score-step_000003", "score-final"):
            assert (tmp_path / "work" / name / part / "summary.json").exists()
    m = json.loads((tmp_path / "work" / "multisyllable" / "train" / "manifest.json").read_text())
    b = json.loads((tmp_path / "work" / "base" / "train" / "manifest.json").read_text())
    assert m["tokens"] == b["tokens"] and m["chars"] > b["chars"] and m["new_tokens"] > 0 == b["new_tokens"]
    assert pipeline.main(args) == 0
    out = json.loads((tmp_path / "work" / "queue-base-multisyllable.json").read_text())
    assert out == {"base": {"status": "done", "ran": []}, "multisyllable": {"status": "done", "ran": []}}
    assert sys.executable  # subprocesses use the test's interpreter


def test_commands_follow_every_optional_setting(tmp_path):
    cfg = config(tmp_path)
    cfg["train"] |= {"grad_checkpointing": True}
    cfg["score"] |= {"en_docs": "en.jsonl", "belebele": "bb.jsonl"}
    steps = {s: a for s, _, a in pipeline.commands(cfg, "base", tmp_path)}
    assert "--grad-checkpointing" in steps["adapt"]
    snap = steps["score-step_000003"]
    assert snap[snap.index("--snapshot") + 1] == str(tmp_path / "base" / "train" / "snapshots" / "step_000003.pt")
    fin = steps["score-final"]
    assert fin[fin.index("--snapshot") + 1] == str(tmp_path / "base" / "train" / "final.pt")
    assert all(a[a.index("--en-docs") + 1] == "en.jsonl" and a[a.index("--belebele") + 1] == "bb.jsonl"
               for s, a in steps.items() if s.startswith("score"))
    bare = config(tmp_path)
    del bare["train"]["seed"], bare["train"]["snapshot_steps"]
    bare["score"] = {"vi_docs": "vi.jsonl", "before_training": False}
    steps = {s: a for s, _, a in pipeline.commands(bare, "base", tmp_path)}
    assert list(steps) == ["adapt", "score-final"]
    a = steps["adapt"]
    assert a[a.index("--seed") + 1] == "0" and "--snapshot-steps" not in a and "--grad-checkpointing" not in a
    assert "--en-docs" not in steps["score-final"] and "--batch-size" not in steps["score-final"]


def test_main_hands_down_its_deadline_and_keeps_finished_conditions(tmp_path):
    cfg = config(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps(cfg))
    run = FakeRun(cfg, tmp_path, {"multi.json --shards": adapt.EXIT_INCOMPLETE})  # multisyllable's training stops
    code = pipeline.main(["--config", str(tmp_path / "run.json"), "--conditions", "base", "multisyllable",
                          "multisyllable-s1", "--work", str(tmp_path), "--deadline-minutes", "10"], run=run)
    out = json.loads((tmp_path / "queue-base-multisyllable-multisyllable-s1.json").read_text())
    assert code == adapt.EXIT_INCOMPLETE and out["base"]["status"] == "done"
    assert out["multisyllable"] == {"status": "incomplete", "at": "adapt"}
    assert out["multisyllable-s1"] == {"status": "not started"}
    minutes = [float(a[a.index("--deadline-minutes") + 1]) for a in run.calls if "--deadline-minutes" in a]
    assert minutes and all(9.9 < m <= 10 for m in minutes)
    short = FakeRun(cfg, tmp_path / "s")
    (tmp_path / "s").mkdir()
    assert pipeline.main(["--config", str(tmp_path / "run.json"), "--conditions", "base", "--work",
                          str(tmp_path / "s"), "--deadline-minutes", "0.5"], run=short) == 0


def test_failures_name_their_error_type(tmp_path):
    cfg = config(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps(cfg))
    pipeline.main(["--config", str(tmp_path / "run.json"), "--conditions", "base", "--work", str(tmp_path),
                   "--deadline-minutes", "10"], run=lambda a, log: 2)
    out = json.loads((tmp_path / "queue-base.json").read_text())
    assert out["base"]["status"] == "failed" and out["base"]["error"].startswith("RuntimeError: score-before exited")
    for drop in ("--config", "--conditions", "--work", "--deadline-minutes"):
        full = ["--config", "c", "--conditions", "base", "--work", "w", "--deadline-minutes", "1"]
        i = full.index(drop)
        with pytest.raises(SystemExit):
            pipeline.main(full[:i] + full[i + 2:])


def test_a_step_log_holds_its_command_and_both_output_streams(tmp_path):
    log = tmp_path / "step.log"
    code = pipeline.subprocess_run([sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); "
                                    "sys.exit(3)"], log)
    text = log.read_text()
    assert code == 3 and text.startswith(f"$ {sys.executable} -c") and "out\n" in text and "err\n" in text
    pipeline.subprocess_run([sys.executable, "-c", "print('again')"], log)
    assert log.read_text().startswith(text) and log.read_text().endswith("again\n")  # appended, not replaced


def test_a_partly_finished_condition_runs_only_what_is_missing(tmp_path):
    cfg = config(tmp_path)
    run = FakeRun(cfg, tmp_path)
    pipeline.run_condition(cfg, "multisyllable", tmp_path, pipeline.time.monotonic() + 600, run)
    (tmp_path / "multisyllable" / "score-final" / "summary.json").unlink()
    (tmp_path / "multisyllable" / "train" / "final.pt").unlink()
    assert pipeline.run_condition(cfg, "multisyllable", tmp_path, pipeline.time.monotonic() + 600, run) == \
        ["adapt", "score-final"]


def test_a_condition_without_its_own_seed_takes_the_run_seed(tmp_path):
    cfg = config(tmp_path)
    cfg["train"]["seed"] = 5
    a = adapt_argv(cfg, "multisyllable", tmp_path)
    s1 = adapt_argv(cfg, "multisyllable-s1", tmp_path)
    assert a[a.index("--seed") + 1] == "5" and s1[s1.index("--seed") + 1] == "1"


def test_every_known_training_setting_is_accepted(tmp_path):
    cfg = config(tmp_path)
    cfg["train"] |= {k: 1 for k in pipeline.TRAIN_KEYS} | {"seed": 3, "snapshot_steps": [], "grad_checkpointing": True}
    (tmp_path / "run.json").write_text(json.dumps(cfg))
    assert pipeline.load_config(tmp_path / "run.json")["train"]["grad_checkpointing"] is True
