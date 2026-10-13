"""Modal layer of the token-type experiment: one function per approved job kind. Importing this file launches nothing.

Inputs live on the volume `nanovitok-token-type` (uploaded with `modal volume put`): a run config, the tokenizers,
the held-out sets and the seed streams, at the paths the config names under /vol. Launch only after the owner has
approved the job and its spending maximum, e.g.

    modal run --detach remote/modal_app.py::smoke --config /vol/smoke/config.json \
        --conditions base multisyllable-4000 --approved-max-usd 1

The function runs the pipeline queue with a deadline before its timeout, commits the volume every few minutes and at
the end, and returns the outcome. Rerunning the same command resumes from the checkpoints on the volume.
"""

import json
import sys
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vitok import remote  # noqa: E402  (after the path: the SDK's Python does not have the package installed)

MODEL = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots" / remote.MODEL_REVISION)
# timeouts from the smoke run's measurements (history/stage5-smoke-2026-10-13.md); each bound = timeout × listed rates
JOBS = {"smoke": remote.Job(gpu="A100-40GB", cpu=4, memory_gib=32, timeout_seconds=1200),
        "probe_n": remote.Job(gpu="A100-40GB", cpu=4, memory_gib=32, timeout_seconds=1500),
        "probe_lr": remote.Job(gpu="A100-40GB", cpu=4, memory_gib=32, timeout_seconds=2700),
        # one condition per call: 1,526 steps × ≈4.7 s + checkpoints, dev curve, snapshots ≈ 8,014 s measured basis
        "full": remote.Job(gpu="A100-40GB", cpu=4, memory_gib=32, timeout_seconds=9300),
        # fp16 scoring measured 3.65x faster than fp32 on the dev set; six models × (11 sets ≈ 383 s, 20 generations
        # ≈ 152 s, loading ≈ 40 s) ≈ 3,450 s; 27 snapshots × (5 bpc sets ≈ 68 s, loading ≈ 40 s) ≈ 2,920 s
        "eval_full": remote.Job(gpu="A100-40GB", cpu=4, memory_gib=32, timeout_seconds=4200),
        "eval_curve": remote.Job(gpu="A100-40GB", cpu=4, memory_gib=32, timeout_seconds=3600)}

app = modal.App("nanovitok-token-type")
FUNCTION = dict(retries=0, max_containers=1, scaledown_window=2, single_use_containers=True)
volume = modal.Volume.from_name("nanovitok-token-type", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*(f"{p}=={v}" for p, v in remote.PACKAGES.items()))
         .env({"PYTHONPATH": "/opt/src", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "VITOK_MODEL_DIR": "/opt/model"})
         .add_local_dir(REPO / "src" / "vitok", "/opt/src/vitok"))
for name in ("config.json", "generation_config.json", "model.safetensors"):
    image = image.add_local_file(MODEL / name, "/opt/model/" + name)


def _run(config: str, conditions: list[str], job: remote.Job, queue_minutes: float = 0.0) -> dict:
    from importlib.metadata import version

    import torch

    installed = {p: version(p) for p in remote.PACKAGES}
    pre = remote.preflight(installed, torch.cuda.is_available(), torch.cuda.is_available()
                           and torch.cuda.is_bf16_supported(), remote.free_gib(Path(remote.VOLUME_ROOT)), 20)
    work = Path(config).parent / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / f"environment-{'-'.join(conditions)}.json").write_text(json.dumps(
        pre | {"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "python": sys.version}, indent=2))
    with remote.PeriodicCommit(volume.commit) as pc:
        minutes = min(queue_minutes, job.queue_minutes()) if queue_minutes > 0 else job.queue_minutes()
        result = remote.run_queue(Path(config), conditions, work, minutes)
    result["commit_errors"] = pc.errors
    return result


@app.function(image=image, gpu=JOBS["smoke"].gpu, cpu=(JOBS["smoke"].cpu, JOBS["smoke"].cpu),
              memory=(JOBS["smoke"].memory_gib * 1024, JOBS["smoke"].memory_gib * 1024),
              timeout=JOBS["smoke"].timeout_seconds, retries=0, max_containers=1, scaledown_window=2,
              single_use_containers=True, volumes={remote.VOLUME_ROOT: volume})
def smoke_queue(config: str, conditions: list[str], queue_minutes: float = 0.0) -> dict:
    return _run(config, conditions, JOBS["smoke"], queue_minutes)


@app.local_entrypoint()
def smoke(config: str, conditions: str, approved_max_usd: float = 0.0, queue_minutes: float = 0.0):
    """`--queue-minutes` shortens the queue's deadline (a planned stop mid-run, to test resuming in a new container)."""
    bound = remote.check_approval(JOBS["smoke"], approved_max_usd or None)
    print(f"bound ${bound:.2f} (timeout at listed rates; startup, image build and storage are billed outside it)")
    print(json.dumps(smoke_queue.remote(config, conditions.split(), queue_minutes), indent=2))


def _resources(name: str) -> dict:
    job = JOBS[name]
    return dict(gpu=job.gpu, cpu=(job.cpu, job.cpu), memory=(job.memory_gib * 1024, job.memory_gib * 1024),
                timeout=job.timeout_seconds)


@app.function(image=image, volumes={remote.VOLUME_ROOT: volume}, **FUNCTION, **_resources("probe_n"))
def probe_n_queue(config: str, conditions: list[str], queue_minutes: float = 0.0) -> dict:
    return _run(config, conditions, JOBS["probe_n"], queue_minutes)


@app.function(image=image, volumes={remote.VOLUME_ROOT: volume}, **FUNCTION, **_resources("probe_lr"))
def probe_lr_queue(config: str, conditions: list[str], queue_minutes: float = 0.0) -> dict:
    return _run(config, conditions, JOBS["probe_lr"], queue_minutes)


def _launch(name: str, fn, config: str, conditions: str, approved_max_usd: float, queue_minutes: float):
    bound = remote.check_approval(JOBS[name], approved_max_usd or None)
    print(f"{name}: bound ${bound:.2f} (timeout at listed rates; startup, image build and storage are billed outside)")
    print(json.dumps(fn.remote(config, conditions.split(), queue_minutes), indent=2))


@app.local_entrypoint()
def probe_n(config: str, conditions: str, approved_max_usd: float = 0.0, queue_minutes: float = 0.0):
    _launch("probe_n", probe_n_queue, config, conditions, approved_max_usd, queue_minutes)


@app.local_entrypoint()
def probe_lr(config: str, conditions: str, approved_max_usd: float = 0.0, queue_minutes: float = 0.0):
    _launch("probe_lr", probe_lr_queue, config, conditions, approved_max_usd, queue_minutes)


@app.function(image=image, volumes={remote.VOLUME_ROOT: volume}, **(FUNCTION | {"max_containers": 3}),
              **_resources("full"))
def full_queue(config: str, conditions: list[str], queue_minutes: float = 0.0) -> dict:
    return _run(config, conditions, JOBS["full"], queue_minutes)


@app.local_entrypoint()
def full(config: str, conditions: str, approved_max_usd: float = 0.0, queue_minutes: float = 0.0):
    """One condition per launch; three launches run in parallel containers (each its own bound)."""
    _launch("full", full_queue, config, conditions, approved_max_usd, queue_minutes)


def _evaluate(plan: str, jobs: list[str], job: remote.Job) -> dict:
    from importlib.metadata import version

    import torch

    installed = {p: version(p) for p in remote.PACKAGES}
    pre = remote.preflight(installed, torch.cuda.is_available(), torch.cuda.is_available()
                           and torch.cuda.is_bf16_supported(), remote.free_gib(Path(remote.VOLUME_ROOT)), 5)
    (Path(plan).parent / f"environment-evaluate-{'-'.join(jobs)}.json").write_text(json.dumps(
        pre | {"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "python": sys.version}, indent=2))
    with remote.PeriodicCommit(volume.commit) as pc:
        result = remote.run_evaluation(Path(plan), jobs, job.queue_minutes())
    result["commit_errors"] = pc.errors
    return result


@app.function(image=image, volumes={remote.VOLUME_ROOT: volume}, **FUNCTION, **_resources("eval_full"))
def eval_full_queue(plan: str, jobs: list[str]) -> dict:
    return _evaluate(plan, jobs, JOBS["eval_full"])


@app.function(image=image, volumes={remote.VOLUME_ROOT: volume}, **FUNCTION, **_resources("eval_curve"))
def eval_curve_queue(plan: str, jobs: list[str]) -> dict:
    return _evaluate(plan, jobs, JOBS["eval_curve"])


def _launch_eval(name: str, fn, plan: str, jobs: str, approved_max_usd: float):
    bound = remote.check_approval(JOBS[name], approved_max_usd or None)
    print(f"{name}: bound ${bound:.2f} (timeout at listed rates; startup, image build and storage are billed outside)")
    print(json.dumps(fn.remote(plan, jobs.split()), indent=2))


@app.local_entrypoint()
def eval_full(plan: str, jobs: str, approved_max_usd: float = 0.0):
    _launch_eval("eval_full", eval_full_queue, plan, jobs, approved_max_usd)


@app.local_entrypoint()
def eval_curve(plan: str, jobs: str, approved_max_usd: float = 0.0):
    _launch_eval("eval_curve", eval_curve_queue, plan, jobs, approved_max_usd)
