"""Kaggle session contracts (no GPU, no network): inputs found or refused, the machine checked, downloads retried,
one pipeline per GPU with its own device and the deadline, and a report built only from files the steps wrote."""

import json

import pytest

from vitok import kaggle_session as ks


def test_inputs_are_found_once_or_refused(tmp_path):
    (tmp_path / "vitok-code" / "src" / "vitok").mkdir(parents=True)
    (tmp_path / "vitok-code" / "pyproject.toml").write_text("")
    (tmp_path / "vitok-data" / "shards").mkdir(parents=True)
    (tmp_path / "vitok-data" / "test.jsonl").write_text("")
    assert ks.find_inputs(tmp_path) == {"code": tmp_path / "vitok-code", "data": tmp_path / "vitok-data"}
    (tmp_path / "copy" / "src" / "vitok").mkdir(parents=True)
    (tmp_path / "copy" / "pyproject.toml").write_text("")
    with pytest.raises(RuntimeError, match="expected one code bundle"):
        ks.find_inputs(tmp_path)


@pytest.mark.parametrize("smi,need,error", [
    ("Tesla T4, 15360, 3\nTesla T4, 15360, 3\n", 2, None),
    ("Tesla T4, 15360, 3\n", 2, "1 GPUs, 2 needed"),
    ("Tesla T4, 15360, 3\nTesla T4, 15360, 2100\n", 2, "already in use"),
])
def test_preflight_checks_gpu_count_and_memory_held_elsewhere(smi, need, error):
    if error:
        with pytest.raises(RuntimeError, match=error):
            ks.preflight(need, query=lambda: smi)
    else:
        assert ks.preflight(need, query=lambda: smi) == [{"name": "Tesla T4", "total_mib": 15360, "used_mib": 3}] * 2


def test_downloads_are_retried_with_backoff_then_given_up():
    waits, calls = [], []

    def flaky(model, revision):
        calls.append(revision)
        if len(calls) < 3:
            raise OSError("connection reset")
        return "/cache/qwen"

    assert ks.prefetch("Qwen/Qwen3-0.6B", "abc", wait=1, download=flaky, sleep=waits.append) == "/cache/qwen"
    assert calls == ["abc"] * 3 and waits == [1, 2]
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        ks.prefetch("m", "r", attempts=2, wait=1, download=lambda m, revision: 1 / 0, sleep=waits.append)


class FakeProc:
    def __init__(self, code):
        self.code = code

    def wait(self):
        return self.code


def test_one_pipeline_per_gpu_with_its_own_device_offline(tmp_path):
    seen = []

    def popen(argv, env, stdout, stderr):
        seen.append((argv, env["CUDA_VISIBLE_DEVICES"], env["HF_HUB_OFFLINE"]))
        return FakeProc(0 if env["CUDA_VISIBLE_DEVICES"] == "0" else 3)

    codes = ks.launch({0: ["base"], 1: ["multisyllable", "syllable"], 2: []}, tmp_path / "run.json", tmp_path, 600,
                      popen=popen)
    assert codes == {0: 0, 1: 3}
    assert [(a[a.index("--conditions") + 1:a.index("--work")], dev, off) for a, dev, off in seen] == \
        [(["base"], "0", "1"), (["multisyllable", "syllable"], "1", "1")]
    assert all(a[a.index("--deadline-minutes") + 1] == "600" for a, _, _ in seen)
    assert (tmp_path / "queue-gpu0.log").exists() and (tmp_path / "queue-gpu1.log").exists()


def test_report_reads_throughput_memory_and_scores_from_the_files(tmp_path):
    t = tmp_path / "base" / "train"
    t.mkdir(parents=True)
    rows = [{"step": 0, "loss": 3.0, "tokens": 100, "chars": 300, "elapsed_seconds": 1.0, "max_memory_gib": 2.0,
             "finite": True},
            {"step": 1, "loss": 2.0, "tokens": 200, "chars": 600, "elapsed_seconds": 4.0, "max_memory_gib": 2.5,
             "finite": False}]
    (t / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for part, bpc in (("score-before", 1.2), ("score-final", 1.1)):
        (tmp_path / "base" / part).mkdir()
        (tmp_path / "base" / part / "summary.json").write_text(json.dumps(
            {"vi": {"bpc": bpc}, "belebele": {"acc_char": 0.5}, "generation": {"decode_chars_per_second": 99.0}}))
    r = ks.report(tmp_path, ["base", "missing"])
    assert r["missing"] == {}
    assert r["base"]["train"] == {"steps": 2, "tokens": 200, "chars": 600, "loss_first": 3.0, "loss_last": 2.0,
                                  "tokens_per_second": 50.0, "chars_per_second": 150.0, "max_memory_gib": 2.5,
                                  "non_finite_steps": 1}
    assert r["base"]["score-final"] == {"vi": 1.1, "belebele_acc_char": 0.5, "decode_chars_per_second": 99.0}


def test_library_versions_below_the_tested_ones_are_refused():
    ks.check_versions({"transformers": "5.18.0", "torch": "2.14.0+cu130"}, {"transformers": "4.56", "torch": "2.9"})
    with pytest.raises(RuntimeError, match="older than tested"):
        ks.check_versions({"transformers": "4.55.2"}, {"transformers": "4.56"})
    with pytest.raises(RuntimeError, match="older than tested"):
        ks.check_versions({}, {"tokenizers": "0.22"})


def test_tokenizers_must_reproduce_their_measured_compression(tmp_path):
    from conftest import SENTENCES, qwen_like_base
    from tokenizers import Tokenizer
    (tmp_path / "base.json").write_text(json.dumps(qwen_like_base()))
    tok = Tokenizer.from_file(str(tmp_path / "base.json"))
    cpt = sum(map(len, SENTENCES)) / sum(len(tok.encode(s, add_special_tokens=False).ids) for s in SENTENCES)
    assert ks.check_tokenizers({"base": tmp_path / "base.json"}, SENTENCES, {"base": cpt}) == {"base": cpt}
    with pytest.raises(RuntimeError, match="do not reproduce"):
        ks.check_tokenizers({"base": tmp_path / "base.json"}, SENTENCES, {"base": cpt + 0.01})


def test_the_bundle_is_head_plus_the_retrofit_and_evaluation_files(tmp_path):
    import subprocess

    repo = tmp_path / "repo"
    (repo / "src" / "vitok").mkdir(parents=True)
    (repo / "LICENSE").write_text("MIT License\n\nCopyright (c) 2026\n\nPermission is granted.\n")
    (repo / "src" / "vitok" / "stats.py").write_text('"""Stats."""\n\nimport math\n')
    (repo / "pyproject.toml").write_text("")
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    (repo / "untracked.txt").write_text("not shipped")
    retro, held = tmp_path / "retro", tmp_path / "held"
    for i, name in enumerate(ks.RETROFIT_CONDITIONS):
        (retro / name).mkdir(parents=True)
        (retro / name / "tokenizer.json").write_text(name)
        if name != "base":
            (retro / name / "report.json").write_text(json.dumps({"chars_per_token_base": 3.4,
                                                                   "chars_per_token_new": 3.5 + i}))
    held.mkdir()
    for name in ks.HELDOUT_FILES:
        (held / name).write_text(name)
    docs = tmp_path / "test.jsonl"
    docs.write_text("".join(json.dumps({"text": f"Tài liệu {i}."}, ensure_ascii=False) + "\n" for i in range(9)))
    out = tmp_path / "bundle"
    man = ks.build_bundle(repo, out, retro, held, docs, "someone/vitok-code")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert man["commit"] == head and (out / "src" / "vitok" / "stats.py").exists()
    assert not (out / "untracked.txt").exists() and not (out / ".git").exists()
    assert (out / "retrofit" / "multisyllable" / "tokenizer.json").read_text() == "multisyllable"
    assert not (out / "retrofit" / "base" / "report.json").exists()
    assert all((out / "retrofit" / n).read_text() == n for n in ks.HELDOUT_FILES)
    texts = [json.loads(line)["text"] for line in (out / "retrofit" / "check_texts.jsonl").read_text().splitlines()]
    assert texts == ["MIT License", "Copyright (c) 2026", "Permission is granted.\n", '"""Stats."""', "import math\n",
                     *[f"Tài liệu {i}." for i in range(5)]]
    meta = json.loads((out / "dataset-metadata.json").read_text())
    assert meta["id"] == "someone/vitok-code" and meta["title"] == "vitok-code"
    assert ks.expected_compression(out / "retrofit") == {"base": 3.4, "syllable": 4.5, "multisyllable": 5.5,
                                                          "syllable-1000": 6.5, "multisyllable-1000": 7.5}
    (repo / "LICENSE").write_text("changed")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        ks.build_bundle(repo, out, retro, held, docs, "someone/vitok-code")
    (retro / "syllable" / "report.json").write_text(json.dumps({"chars_per_token_base": 3.3, "chars_per_token_new": 4}))
    with pytest.raises(RuntimeError, match="disagree on the base"):
        ks.expected_compression(retro)


def test_boundaries_of_the_machine_checks():
    ks.check_versions({"transformers": "4.56.0"}, {"transformers": "4.56"})  # the tested version itself passes
    ks.check_versions({"transformers": "4.56"}, {"transformers": "4.56"})
    ks.check_versions({"torch": "2.4.0.post1"}, {"torch": "2.4"})
    assert ks.preflight(1, max_used_mib=500, query=lambda: "T4, 15360, 500\n")[0]["used_mib"] == 500


def test_downloads_give_up_after_exactly_the_attempts_and_pass_the_model():
    calls = []
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        ks.prefetch("Qwen/Qwen3-0.6B", "r", attempts=3, wait=0,
                    download=lambda m, revision: calls.append((m, revision)) or 1 / 0, sleep=lambda s: None)
    assert calls == [("Qwen/Qwen3-0.6B", "r")] * 3


def test_queue_logs_take_both_output_streams(tmp_path):
    seen = []

    def popen(argv, env, stdout, stderr):
        seen.append(stderr)
        return FakeProc(0)

    ks.launch({0: ["base"]}, tmp_path / "c.json", tmp_path / "new" / "work", 1, popen=popen)
    assert seen == [ks.subprocess.STDOUT] and (tmp_path / "new" / "work" / "queue-gpu0.log").exists()


def test_report_memory_is_zero_when_no_step_measured_it(tmp_path):
    t = tmp_path / "base" / "train"
    t.mkdir(parents=True)
    row = {"step": 0, "loss": 3.0, "tokens": 10, "chars": 30, "elapsed_seconds": 1.0, "max_memory_gib": None,
           "finite": True}
    (t / "train_log.jsonl").write_text(json.dumps(row) + "\n")
    assert ks.report(tmp_path, ["base"])["base"]["train"]["max_memory_gib"] == 0


def test_an_empty_queue_does_not_stop_the_others(tmp_path):
    started = []

    def popen(argv, env, stdout, stderr):
        started.append(env["CUDA_VISIBLE_DEVICES"])
        return FakeProc(0)

    assert ks.launch({0: [], 1: ["base"]}, tmp_path / "c.json", tmp_path, 1, popen=popen) == {1: 0}
    assert started == ["1"]
