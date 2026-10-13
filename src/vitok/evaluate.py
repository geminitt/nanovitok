"""A queue of evaluations: each job scores one model (optionally with trained weights) on a named group of sets.

    python -m vitok.evaluate --plan plan.json --jobs original multisyllable-1000-final --deadline-minutes 60

The plan (JSON) holds the scoring options shared by every job ("score": batch_size, dtype), the set groups
("sets": group -> {"lm": {name: file}, "mc": {name: file}}), the generation-speed settings ("speed": docs, prompts,
new_tokens) and the jobs ("jobs": name -> model, optional revision and snapshot, tokenizer, set group, whether to
measure speed, output folder). Every job is a `vitok.score` subprocess, skipped when its summary exists; scoring
resumes item by item, so rerunning the command continues where it stopped. A failing job does not stop the queue; a
job stopped by the deadline does. Logs (`<job>.log`) and the outcome (`evaluate-<jobs>.json`) go next to the plan.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from vitok.adapt import EXIT_INCOMPLETE
from vitok.pipeline import subprocess_run

JOB_KEYS = {"model", "tokenizer", "sets", "out"}


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in ("score", "sets", "jobs") if k not in plan]
    if missing:
        raise ValueError(f"{path} lacks {missing}")
    for name, job in plan["jobs"].items():
        if JOB_KEYS - set(job):
            raise ValueError(f"job {name} lacks {sorted(JOB_KEYS - set(job))}")
        if job["sets"] not in plan["sets"]:
            raise ValueError(f"job {name}: unknown set group {job['sets']!r}")
        if job.get("speed") and "speed" not in plan:
            raise ValueError(f"job {name} measures speed but the plan has no speed settings")
    return plan


def command(plan: dict, name: str) -> list[str]:
    job, score = plan["jobs"][name], plan["score"]
    argv = [sys.executable, "-m", "vitok.score", "--model", job["model"], "--tokenizer", job["tokenizer"]]
    if job.get("revision"):
        argv += ["--revision", job["revision"]]
    if job.get("snapshot"):
        argv += ["--snapshot", job["snapshot"]]
    for kind in ("lm", "mc"):
        sets = plan["sets"][job["sets"]].get(kind)
        if sets:
            argv += [f"--{kind}", *(f"{n}={p}" for n, p in sets.items())]
    if job.get("speed"):
        s = plan["speed"]
        argv += ["--speed-docs", s["docs"], "--speed-prompts", str(s["prompts"]), "--new-tokens", str(s["new_tokens"])]
    for k in ("batch_size", "dtype"):
        if k in score:
            argv += [f"--{k.replace('_', '-')}", str(score[k])]
    return argv + ["--out", job["out"]]


def main(argv=None, run: Callable[[list[str], Path], int] = subprocess_run) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--deadline-minutes", type=float, required=True)
    args = ap.parse_args(argv)
    plan = load_plan(args.plan)
    unknown = [j for j in args.jobs if j not in plan["jobs"]]
    if unknown:
        raise SystemExit(f"jobs {unknown} are not in {args.plan}")
    deadline = time.monotonic() + 60 * args.deadline_minutes
    outcome: dict[str, dict] = {}
    for i, name in enumerate(args.jobs):
        if (Path(plan["jobs"][name]["out"]) / "summary.json").exists():
            outcome[name] = {"status": "done"}
            continue
        left = (deadline - time.monotonic()) / 60
        code = EXIT_INCOMPLETE if left <= 0 else run(
            [*command(plan, name), "--deadline-minutes", f"{left:.2f}"], args.plan.parent / f"{name}.log")
        if code == EXIT_INCOMPLETE:
            outcome[name] = {"status": "incomplete"}
            outcome |= {rest: {"status": "not started"} for rest in args.jobs[i + 1:]}
            break
        done = (Path(plan["jobs"][name]["out"]) / "summary.json").exists()
        outcome[name] = {"status": "done"} if code == 0 and done else \
            {"status": "failed", "error": f"exit code {code}" + ("" if done else ", no summary")}
    tag = "-".join(args.jobs)
    (args.plan.parent / f"evaluate-{tag}.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2), flush=True)
    if any(o["status"] == "failed" for o in outcome.values()):
        return 1
    return EXIT_INCOMPLETE if any(o["status"] != "done" for o in outcome.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
