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
JOBS = {"smoke": remote.Job(gpu="A100-40GB", cpu=4, memory_gib=32, timeout_seconds=1200)}

app = modal.App("nanovitok-token-type")
volume = modal.Volume.from_name("nanovitok-token-type", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*(f"{p}=={v}" for p, v in remote.PACKAGES.items()))
         .env({"PYTHONPATH": "/opt/src", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "VITOK_MODEL_DIR": "/opt/model"})
         .add_local_dir(REPO / "src" / "vitok", "/opt/src/vitok"))
for name in ("config.json", "generation_config.json", "model.safetensors"):
    image = image.add_local_file(MODEL / name, "/opt/model/" + name)


def _run(config: str, conditions: list[str], job: remote.Job) -> dict:
    from importlib.metadata import version

    import torch

    installed = {p: version(p) for p in remote.PACKAGES}
    pre = remote.preflight(installed, torch.cuda.is_available(), torch.cuda.is_available()
                           and torch.cuda.is_bf16_supported(), remote.free_gib(Path(remote.VOLUME_ROOT)), 20)
    work = Path(config).parent / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "environment.json").write_text(json.dumps(
        pre | {"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "python": sys.version}, indent=2))
    with remote.PeriodicCommit(volume.commit) as pc:
        result = remote.run_queue(Path(config), conditions, work, job.queue_minutes())
    result["commit_errors"] = pc.errors
    return result


@app.function(image=image, gpu=JOBS["smoke"].gpu, cpu=(JOBS["smoke"].cpu, JOBS["smoke"].cpu),
              memory=(JOBS["smoke"].memory_gib * 1024, JOBS["smoke"].memory_gib * 1024),
              timeout=JOBS["smoke"].timeout_seconds, retries=0, max_containers=1, scaledown_window=2,
              single_use_containers=True, volumes={remote.VOLUME_ROOT: volume})
def smoke_queue(config: str, conditions: list[str]) -> dict:
    return _run(config, conditions, JOBS["smoke"])


@app.local_entrypoint()
def smoke(config: str, conditions: str, approved_max_usd: float = 0.0):
    bound = remote.check_approval(JOBS["smoke"], approved_max_usd or None)
    print(f"bound ${bound:.2f} (timeout at listed rates; startup, image build and storage are billed outside it)")
    print(json.dumps(smoke_queue.remote(config, conditions.split()), indent=2))
