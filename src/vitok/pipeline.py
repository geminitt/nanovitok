"""One GPU's queue of conditions: embedding extension, scoring before training, adaptation training, scoring after.

    python -m vitok.pipeline --config run.json --conditions multisyllable base --work /kaggle/working/runs \
        --deadline-minutes 690

The config (JSON) names the base model and its revision, the base tokenizer, every condition's tokenizer (and
optionally its seed), the training and scoring settings shared by all conditions, the training shards and the
held-out files. Every condition runs the same steps with the same settings; only the tokenizer (and the seed, for a
second-seed condition) differs.

Each step is a subprocess (`vitok.extend`, `vitok.adapt`, `vitok.score`) and is skipped when its output already
exists, so rerunning the command after a stop resumes; `vitok.adapt` and `vitok.score` resume inside a step too. A
failing condition does not stop the queue: the failure is recorded and the next condition runs; the summary lists
every condition's outcome. The time left before `--deadline-minutes` is passed to each step, which checkpoints and
exits with code 3 when it runs out.
"""

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from vitok.adapt import EXIT_INCOMPLETE

TRAIN_KEYS = ("steps", "batch", "micro_batch", "seq_len", "lr", "embed_lr", "warmup", "min_lr_ratio", "weight_decay",
              "grad_clip", "layers", "dtype", "save_every", "log_every")
SCORE_KEYS = ("batch_size", "dtype")
SPEED_KEYS = ("speed_prompts", "speed_prompt_chars", "new_tokens")  # measured before training and on the final weights


class Incomplete(Exception):
    """A step stopped at the deadline with its progress saved."""


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("model", "revision", "base_tokenizer", "check_texts", "conditions", "train", "score",
                           "shards", "held_out") if k not in cfg]
    if missing:
        raise ValueError(f"{path} lacks {missing}")
    for name, cond in cfg["conditions"].items():
        if "tokenizer" not in cond:
            raise ValueError(f"condition {name} has no tokenizer")
    unknown = set(cfg["train"]) - set(TRAIN_KEYS) - {"seed", "snapshot_steps", "grad_checkpointing"}
    if unknown:
        raise ValueError(f"unknown training settings {sorted(unknown)}")
    return cfg


def _flags(settings: dict, keys) -> list[str]:
    out = []
    for k in keys:
        if k in settings and settings[k] is not None:
            out += [f"--{k.replace('_', '-')}", str(settings[k])]
    return out


def is_extended(cfg: dict, cond: dict) -> bool:
    return Path(cond["tokenizer"]).resolve() != Path(cfg["base_tokenizer"]).resolve()


def commands(cfg: dict, name: str, work: Path) -> list[tuple[str, Path, list[str]]]:
    """(step, output that marks it done, argv) for one condition, in order."""
    cond, out = cfg["conditions"][name], work / name
    py = [sys.executable, "-m"]
    model = str(out / "extended") if is_extended(cfg, cond) else cfg["model"]
    revision = [] if is_extended(cfg, cond) else ["--revision", cfg["revision"]]
    steps = []
    if is_extended(cfg, cond):
        steps.append(("extend", out / "extended" / "extend_manifest.json",
                      py + ["vitok.extend", "--model", cfg["model"], "--revision", cfg["revision"],
                            "--base-tokenizer", cfg["base_tokenizer"], "--tokenizer", cond["tokenizer"],
                            "--check-texts", cfg["check_texts"], "--out", str(out / "extended")]))
    score = py + ["vitok.score", "--model", model, *revision, "--tokenizer", cond["tokenizer"],
                  *_flags(cfg["score"], SCORE_KEYS)]
    for part in ("vi_docs", "en_docs", "belebele"):
        if cfg["score"].get(part):
            score += [f"--{part.replace('_', '-')}", cfg["score"][part]]
    speed = _flags(cfg["score"], SPEED_KEYS)
    if cfg["score"].get("before_training", True):
        steps.append(("score-before", out / "score-before" / "summary.json",
                      score + speed + ["--out", str(out / "score-before")]))
    train = cfg["train"]
    seed = cond.get("seed", train.get("seed", 0))
    steps.append(("adapt", out / "train" / "final.pt",
                  py + ["vitok.adapt", "--model", model, *revision, "--base-tokenizer", cfg["base_tokenizer"],
                        "--tokenizer", cond["tokenizer"], "--shards", *cfg["shards"], "--held-out", *cfg["held_out"],
                        "--out", str(out / "train"), "--seed", str(seed), *_flags(train, TRAIN_KEYS),
                        *(["--grad-checkpointing"] if train.get("grad_checkpointing") else []),
                        *(["--snapshot-steps", *map(str, train["snapshot_steps"])]
                          if train.get("snapshot_steps") else [])]))
    snaps = [*(f"step_{s:06d}" for s in train.get("snapshot_steps", []) if cfg["score"].get("snapshots")), "final"]
    for snap in snaps:
        weights = out / "train" / ("final.pt" if snap == "final" else f"snapshots/{snap}.pt")
        steps.append((f"score-{snap}", out / f"score-{snap}" / "summary.json",
                      score + (speed if snap == "final" else []) +
                      ["--snapshot", str(weights), "--out", str(out / f"score-{snap}")]))
    return steps


def run_condition(cfg: dict, name: str, work: Path, deadline: float, run: Callable) -> list[str]:
    """Run every step not done yet; returns the steps run. Raises Incomplete at the deadline."""
    ran = []
    for step, done, argv in commands(cfg, name, work):
        if done.exists():
            continue
        left = (deadline - time.monotonic()) / 60
        if left <= 0:
            raise Incomplete(step)
        if step != "extend":
            argv = [*argv, "--deadline-minutes", f"{left:.2f}"]
        (work / name).mkdir(parents=True, exist_ok=True)
        code = run(argv, work / name / f"{step}.log")
        if code == EXIT_INCOMPLETE:
            raise Incomplete(step)
        if code != 0:
            raise RuntimeError(f"{step} exited with code {code}; log {work / name / f'{step}.log'}")
        if not done.exists():
            raise RuntimeError(f"{step} exited 0 but {done} is missing")
        ran.append(step)
    return ran


def subprocess_run(argv: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(argv)}\n")
        f.flush()
        return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, check=False).returncode


def main(argv=None, run: Callable = subprocess_run) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--deadline-minutes", type=float, required=True, help="wall time this queue may use")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    unknown = [c for c in args.conditions if c not in cfg["conditions"]]
    if unknown:
        raise SystemExit(f"conditions {unknown} are not in {args.config}")
    deadline = time.monotonic() + 60 * args.deadline_minutes
    outcome = {}
    for name in args.conditions:
        try:
            outcome[name] = {"status": "done", "ran": run_condition(cfg, name, args.work, deadline, run)}
        except Incomplete as e:
            outcome[name] = {"status": "incomplete", "at": str(e)}
            for rest in args.conditions[args.conditions.index(name) + 1:]:
                outcome[rest] = {"status": "not started"}
            break
        except Exception as e:  # continue past a failing condition; the summary lists it
            outcome[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    args.work.mkdir(parents=True, exist_ok=True)
    tag = "-".join(args.conditions)
    (args.work / f"queue-{tag}.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2), flush=True)
    if any(o["status"] == "failed" for o in outcome.values()):
        return 1
    return EXIT_INCOMPLETE if any(o["status"] != "done" for o in outcome.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
