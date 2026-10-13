"""Remote-job contracts without the provider: the cost bound and the approval gate, preflight refusals, periodic
volume commits, and the queue run with its own deadline."""

import json
import threading
import time

import pytest

from vitok import remote


def test_the_bound_is_the_timeout_at_the_listed_rates_and_needs_an_approval_that_covers_it():
    job = remote.Job(gpu="A100-40GB", cpu=2, memory_gib=16, timeout_seconds=900)
    per_second = 0.000583 + 2 * 0.0000131 + 16 * 0.00000222
    assert job.max_usd() == pytest.approx(900 * per_second) == pytest.approx(0.580248, rel=1e-6)
    assert remote.check_approval(job, 1.0) == job.max_usd()
    for approved in (None, 0.5):
        with pytest.raises(SystemExit, match=r"may cost up to \$0.58"):
            remote.check_approval(job, approved)
    assert job.queue_minutes() == (900 - remote.MARGIN_SECONDS) / 60


def test_preflight_refuses_other_software_no_bf16_gpu_or_no_disk():
    ok = dict(remote.PACKAGES)
    assert remote.preflight(ok | {"torch": "2.14.0+cu128"}, True, True, 50, 10)["free_gib"] == 50
    with pytest.raises(RuntimeError, match="differ"):
        remote.preflight(ok | {"transformers": "5.18.0"}, True, True, 50, 10)
    with pytest.raises(RuntimeError, match="differ"):
        remote.preflight({k: v for k, v in ok.items() if k != "numpy"}, True, True, 50, 10)
    with pytest.raises(RuntimeError, match="bf16"):
        remote.preflight(ok, True, False, 50, 10)
    with pytest.raises(RuntimeError, match="GiB needed"):
        remote.preflight(ok, True, True, 5, 10)


def test_commits_repeat_while_running_and_once_more_at_the_end_even_after_errors():
    calls, lock = [], threading.Lock()

    def commit():
        with lock:
            calls.append(time.monotonic())
            if len(calls) == 2:
                raise OSError("volume busy")

    with remote.PeriodicCommit(commit, every=0.05) as pc:
        time.sleep(0.32)
    assert len(calls) >= 5 and pc.errors == ["OSError: volume busy"]
    assert not pc.thread.is_alive()


def test_the_queue_runs_the_pipeline_with_its_deadline_and_records_the_outcome(tmp_path):
    seen = {}

    def fake(argv, log):
        seen["argv"], seen["log"] = argv, log
        (tmp_path / "queue-base-multisyllable.json").write_text(json.dumps({"base": {"status": "done"}}))
        return 0

    out = remote.run_queue(tmp_path / "c.json", ["base", "multisyllable"], tmp_path, 12.345, run=fake)
    a = seen["argv"]
    assert a[1:3] == ["-m", "vitok.pipeline"] and a[a.index("--deadline-minutes") + 1] == "12.35"
    assert a[a.index("--conditions") + 1:a.index("--work")] == ["base", "multisyllable"]
    assert out["exit_code"] == 0 and out["conditions"] == {"base": {"status": "done"}}
    # parallel containers share the work folder: each queue writes only files named after its conditions
    # (Modal volumes: concurrent writes to one file are last-write-wins)
    assert seen["log"] == tmp_path / "queue-base-multisyllable.log"
    assert json.loads((tmp_path / "result-base-multisyllable.json").read_text())["exit_code"] == 0
