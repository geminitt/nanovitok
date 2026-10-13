"""What a paid remote job may use and what it does, without the provider SDK (tested locally; `modal_app.py` is
the thin Modal layer that imports this).

A job is a named queue of conditions run by `vitok.pipeline` on one GPU, inside fixed resource limits. Its largest
possible charge is its function timeout times the listed rates of its resources; the local entrypoint refuses to
launch unless the owner's approved maximum (`--approved-max-usd`) covers that bound. This only prevents an
unapproved launch: platform startup, image builds and storage are billed outside this bound, so the owner also sets
a provider usage cap (PLAN.md, Forbidden; global rules, paid compute).
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# the gpu environment's versions (`pixi run -e gpu`), which every local test ran with
PACKAGES = {"torch": "2.14.0", "transformers": "5.19.0", "tokenizers": "0.23.1", "numpy": "2.5.3",
            "pyarrow": "25.0.0", "huggingface_hub": "1.32.0"}
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
# Modal list prices verified 2026-10-12 at https://modal.com/pricing; re-verify before every launch
RATES = {"A100-40GB": 0.000583, "cpu_core": 0.0000131, "memory_gib": 0.00000222}  # USD per second
VOLUME_ROOT = "/vol"
COMMIT_EVERY_SECONDS = 300
MARGIN_SECONDS = 180  # the queue's own deadline ends this long before the function timeout, to save and commit


@dataclass(frozen=True)
class Job:
    gpu: str
    cpu: int
    memory_gib: int
    timeout_seconds: int

    def max_usd(self, rates: dict = RATES) -> float:
        """The charge if the function runs to its timeout: GPU, CPU cores and memory at their listed rates."""
        per_second = rates[self.gpu] + self.cpu * rates["cpu_core"] + self.memory_gib * rates["memory_gib"]
        return self.timeout_seconds * per_second

    def queue_minutes(self) -> float:
        return (self.timeout_seconds - MARGIN_SECONDS) / 60


def check_approval(job: Job, approved_usd: float | None, rates: dict = RATES) -> float:
    """The job's bound in USD; refuses a launch the owner's approved maximum does not cover."""
    bound = job.max_usd(rates)
    if approved_usd is None or approved_usd < bound:
        raise SystemExit(f"this job may cost up to ${bound:.2f} (timeout {job.timeout_seconds} s at listed rates); "
                         f"approved: {approved_usd}. Launch only with --approved-max-usd >= {bound:.2f} after the "
                         "owner approves it.")
    return bound


def preflight(installed: dict[str, str], cuda: bool, bf16: bool, free_gib: float, need_gib: float) -> dict:
    """Refuse to start on the wrong software, without a bf16 GPU, or without disk for the outputs."""
    wrong = {p: v for p, v in installed.items() if v.split("+")[0] != PACKAGES.get(p)}
    if set(installed) != set(PACKAGES) or wrong:
        raise RuntimeError(f"package versions differ from the tested ones: {wrong or installed}")
    if not (cuda and bf16):
        raise RuntimeError("a CUDA GPU with native bf16 is required")
    if free_gib < need_gib:
        raise RuntimeError(f"{free_gib:.1f} GiB free on the volume, {need_gib:.1f} GiB needed")
    return {"packages": installed, "free_gib": free_gib}


class PeriodicCommit:
    """Calls `commit` every `every` seconds in a thread, and once more on exit, so checkpoints written to a network
    volume survive a container that is stopped."""

    def __init__(self, commit: Callable[[], None], every: float = COMMIT_EVERY_SECONDS):
        self.commit, self.every, self.stop = commit, every, threading.Event()
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.wait(self.every):
            self._safe()

    def _safe(self):
        try:
            self.commit()
        except Exception as e:  # a failed commit is retried at the next tick and reported at the end
            self.errors.append(f"{type(e).__name__}: {e}")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._safe()


def run_queue(config: Path, conditions: list[str], work: Path, minutes: float,
              run: Callable[[list[str], Path], int] | None = None) -> dict:
    """Run the conditions' pipeline queue in a child process group with its own deadline; returns the outcome."""
    work.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "vitok.pipeline", "--config", str(config), "--conditions", *conditions,
            "--work", str(work), "--deadline-minutes", f"{minutes:.2f}"]
    tag = "-".join(conditions)
    log_path = work / f"queue-{tag}.log"  # one writer per file: parallel containers share `work`
    started = time.monotonic()
    if run is not None:
        code = run(argv, log_path)
    else:
        with open(log_path, "a", encoding="utf-8") as log:
            # same process group as the caller: stopping the caller stops the queue (no orphan keeps writing)
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
            code = proc.wait()
    queue = work / f"queue-{tag}.json"
    outcome = json.loads(queue.read_text()) if queue.exists() else {}
    result = {"exit_code": code, "seconds": time.monotonic() - started, "conditions": outcome}
    (work / f"result-{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_evaluation(plan: Path, jobs: list[str], minutes: float,
                   run: Callable[[list[str], Path], int] | None = None) -> dict:
    """Run evaluation jobs (`vitok.evaluate`) with their own deadline; logs and outcome next to the plan."""
    tag = "-".join(jobs)
    log_path = plan.parent / f"evaluate-{tag}.log"
    argv = [sys.executable, "-m", "vitok.evaluate", "--plan", str(plan), "--jobs", *jobs,
            "--deadline-minutes", f"{minutes:.2f}"]
    started = time.monotonic()
    if run is not None:
        code = run(argv, log_path)
    else:
        with open(log_path, "a", encoding="utf-8") as log:
            code = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "TOKENIZERS_PARALLELISM": "false"}).wait()
    outcome = plan.parent / f"evaluate-{tag}.json"
    result = {"exit_code": code, "seconds": time.monotonic() - started,
              "jobs": json.loads(outcome.read_text()) if outcome.exists() else {}}
    (plan.parent / f"result-evaluate-{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 2**30


def source_sha256(folder: Path) -> str:
    """Digest of the package source a job ran (relative path and bytes of every .py file), to name the code exactly."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(folder.rglob("*.py")):
        if "__pycache__" not in f.parts:
            h.update(f.relative_to(folder).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def launch_commit(repo: Path) -> str:
    """HEAD of `repo`, marked dirty when src/vitok differs from it; "unknown" without git (as inside an image)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/vitok"], cwd=repo, capture_output=True,
                               text=True).stdout.strip()
    except OSError:
        return "unknown"
    return (head or "unknown") + ("-dirty" if dirty else "")
