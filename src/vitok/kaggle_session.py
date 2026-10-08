"""A Kaggle session for the retrofit study: find the inputs, check the machine, fetch the base model, run one queue of
conditions per GPU in parallel, and report.

The notebook (kaggle/notebooks/03_adaptation.ipynb) only calls these functions, so every line it runs is tested on
CPU first. Steps:

- `find_inputs`: the code bundle (`vitok-code`, with the retrofit tokenizers and the held-out files) and the data
  (`vitok-data`, with the training shards and test.jsonl) under /kaggle/input.
- `preflight`: GPU count, memory already held by other processes, library versions.
- `prefetch`: download the pinned base model with retries and backoff; afterwards the queues run offline.
- `launch`: one `vitok.pipeline` process per GPU (CUDA_VISIBLE_DEVICES), each with the session's deadline; waits for
  all and returns their exit codes.
- `build_bundle` (run locally before a push): the `vitok-code` dataset folder — the committed code at HEAD, the
  retrofit tokenizers with their reports, the evaluation files, and check texts for the embedding extension.
- `report`: per condition, training throughput (tokens/s, characters/s), peak memory, final loss, and the bpc of each
  scoring run, read from the files the steps wrote.
"""

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path


def find_inputs(root: Path) -> dict[str, Path]:
    """The code bundle and the data folder; refuses when either is missing or ambiguous."""
    code = [p.parent for p in root.rglob("pyproject.toml") if (p.parent / "src" / "vitok").is_dir()]
    data = [p.parent for p in root.rglob("test.jsonl") if (p.parent / "shards").is_dir()]
    if len(code) != 1 or len(data) != 1:
        raise RuntimeError(f"expected one code bundle and one data folder under {root}, found {code} and {data}")
    return {"code": code[0], "data": data[0]}


def preflight(need_gpus: int, max_used_mib: int = 500, query: Callable | None = None) -> list[dict]:
    """One record per GPU (name, total and used memory); refuses too few GPUs or memory held outside this process."""
    if query is None:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    else:
        out = query()
    gpus: list[dict] = []
    for line in out.strip().splitlines():
        name, total, used = (x.strip() for x in line.split(","))
        gpus.append({"name": name, "total_mib": int(total), "used_mib": int(used)})
    if len(gpus) < need_gpus:
        raise RuntimeError(f"{len(gpus)} GPUs, {need_gpus} needed")
    busy = [g for g in gpus if g["used_mib"] > max_used_mib]
    if busy:
        raise RuntimeError(f"GPU memory already in use before the run: {busy}")
    return gpus


def check_versions(installed: dict[str, str], minimum: dict[str, str]) -> None:
    """Refuses a library older than the minimum the code was tested with (versions compared numerically)."""
    num = lambda v: tuple(int(x) for x in v.split("+")[0].split(".")[:3] if x.isdigit())
    old = {k: (installed.get(k), v) for k, v in minimum.items()
           if installed.get(k) is None or num(installed[k]) < num(v)}
    if old:
        raise RuntimeError(f"libraries older than tested (installed, minimum): {old}")


def check_tokenizers(tokenizers: dict[str, Path], docs: list[str], expected: dict[str, float]) -> dict[str, float]:
    """Manipulation check before any training: each condition's tokenizer gives, on the test documents, exactly the
    characters per token its retrofit report measured; a different `tokenizers` build or file would show here."""
    from tokenizers import Tokenizer

    chars = sum(map(len, docs))
    got = {}
    for name, path in tokenizers.items():
        tok = Tokenizer.from_file(str(path))
        got[name] = chars / sum(len(e.ids) for e in tok.encode_batch(docs, add_special_tokens=False))
    off = {n: (got[n], expected[n]) for n in expected if abs(got[n] - expected[n]) > 1e-9}
    if off:
        raise RuntimeError(f"tokenizers do not reproduce their reports (got, expected): {off}")
    return got


def prefetch(model: str, revision: str, attempts: int = 5, wait: float = 10.0, download: Callable | None = None,
             sleep: Callable = time.sleep) -> str:
    """Download the pinned model into the Hugging Face cache, retrying with exponential backoff; returns its path."""
    if download is None:
        from huggingface_hub import snapshot_download
        download = snapshot_download
    for i in range(attempts):
        try:
            return download(model, revision=revision)
        except Exception as e:  # network errors of any kind are retried
            if i == attempts - 1:
                raise RuntimeError(f"could not download {model}@{revision} after {attempts} attempts") from e
            print(f"download failed ({type(e).__name__}: {e}); retry in {wait * 2 ** i:.0f} s", flush=True)
            sleep(wait * 2 ** i)
    raise AssertionError("unreachable")


def launch(queues: dict[int, list[str]], config: Path, work: Path, deadline_minutes: float,
           popen: Callable = subprocess.Popen) -> dict[int, int]:
    """One vitok.pipeline per GPU, in parallel, offline; returns each GPU's exit code (logs in work/queue-gpuN.log)."""
    work.mkdir(parents=True, exist_ok=True)
    procs = {}
    for gpu, conds in queues.items():
        if not conds:
            continue
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        log = open(work / f"queue-gpu{gpu}.log", "a", encoding="utf-8")
        argv = [sys.executable, "-m", "vitok.pipeline", "--config", str(config), "--conditions", *conds,
                "--work", str(work), "--deadline-minutes", str(deadline_minutes)]
        procs[gpu] = (popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT), log)
    codes = {}
    for gpu, (proc, log) in procs.items():
        codes[gpu] = proc.wait()
        log.close()
    return codes


def report(work: Path, conditions: list[str]) -> dict[str, dict]:
    """What each condition's files say: training throughput and memory, and the bpc of every scoring run."""
    out = {}
    for name in conditions:
        rec: dict = {}
        log = work / name / "train" / "train_log.jsonl"
        if log.exists():
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows:
                last = rows[-1]
                rec["train"] = {"steps": last["step"] + 1, "tokens": last["tokens"], "chars": last["chars"],
                                "loss_first": rows[0]["loss"], "loss_last": last["loss"],
                                "tokens_per_second": last["tokens"] / last["elapsed_seconds"],
                                "chars_per_second": last["chars"] / last["elapsed_seconds"],
                                "max_memory_gib": max((r["max_memory_gib"] or 0) for r in rows),
                                "non_finite_steps": sum(not r["finite"] for r in rows)}
        for summary in sorted((work / name).glob("score-*/summary.json")):
            s = json.loads(summary.read_text(encoding="utf-8"))
            rec[summary.parent.name] = {k: s[k]["bpc"] for k in ("vi", "en") if k in s} | (
                {"belebele_acc_char": s["belebele"]["acc_char"]} if "belebele" in s else {}) | (
                {"decode_chars_per_second": s["generation"]["decode_chars_per_second"]} if "generation" in s else {})
        out[name] = rec
    return out


RETROFIT_CONDITIONS = ("base", "syllable", "multisyllable", "syllable-1000", "multisyllable-1000")
HELDOUT_FILES = ("belebele_vie_Latn.jsonl", "en_docs.jsonl", "heldout_manifest.json")
VAL_DOCS = Path("kaggle/outputs/vitok-data/val_docs.jsonl")  # the merge-learning text, not in the Kaggle vitok-data


def check_texts(repo: Path, test_docs: list[str], n_docs: int = 5) -> list[str]:
    """Texts for the extension's logit check: English (the license) and code (stats.py), split at blank lines, which
    the retrofit must leave alone, plus a few Vietnamese test documents, which it changes."""
    blocks = []
    for path in (repo / "LICENSE", repo / "src" / "vitok" / "stats.py"):
        blocks += [b for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    return blocks + test_docs[:n_docs]


def build_bundle(repo: Path, out: Path, retrofit_dir: Path, heldout_dir: Path, test_docs: Path, dataset_id: str,
                 run: Callable = subprocess.run) -> dict:
    """The vitok-code dataset folder: refuses uncommitted changes to tracked files, so the bundle is exactly HEAD."""
    import shutil
    import tarfile

    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True,
                check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"uncommitted changes to tracked files; commit first:\n{dirty}")
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    archive = out / "head.tar"
    run(["git", "archive", "--format=tar", "-o", str(archive), "HEAD"], cwd=repo, check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
    archive.unlink()
    r = out / "retrofit"
    for name in RETROFIT_CONDITIONS:
        (r / name).mkdir(parents=True)
        shutil.copy(retrofit_dir / name / "tokenizer.json", r / name / "tokenizer.json")
        if name != "base":
            shutil.copy(retrofit_dir / name / "report.json", r / name / "report.json")
    for name in HELDOUT_FILES:
        shutil.copy(heldout_dir / name, r / name)
    shutil.copy(repo / VAL_DOCS, r / "val_docs.jsonl")
    docs = [json.loads(line)["text"] for line in test_docs.read_text(encoding="utf-8").splitlines() if line.strip()]
    (r / "check_texts.jsonl").write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in check_texts(repo, docs)), encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": dataset_id.split("/")[1], "id": dataset_id, "licenses": [{"name": "MIT"}]}, indent=2))
    manifest = {"commit": commit, "retrofit": list(RETROFIT_CONDITIONS), "heldout": list(HELDOUT_FILES)}
    (r / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def expected_compression(retrofit: Path) -> dict[str, float]:
    """Characters per token on the test documents for every condition, from the retrofit reports."""
    out = {}
    for name in RETROFIT_CONDITIONS[1:]:
        rep = json.loads((retrofit / name / "report.json").read_text(encoding="utf-8"))
        out[name] = rep["chars_per_token_new"]
        base = out.setdefault("base", rep["chars_per_token_base"])
        if base != rep["chars_per_token_base"]:
            raise RuntimeError(f"reports disagree on the base tokenizer: {base} vs {rep['chars_per_token_base']}")
    return out
