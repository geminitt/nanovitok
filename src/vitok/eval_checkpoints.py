"""Evaluate every trained checkpoint found under a directory (for example on the val shard).

    python -m vitok.eval_checkpoints --runs kaggle/outputs --docs val_docs.jsonl --variants clean \
        --nanochat third_party/nanochat --out results/val

A run is a directory named {condition}_d{depth}_s{seed} holding base_checkpoints/d{depth}/model_*.pt and
the tokenizer/ it was trained with, as `vitok.kaggle_run` leaves them. Each run is scored by `vitok.eval`
in its own process, in fp16 like the Kaggle runs, and a finished result is not recomputed.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

RUN_RE = re.compile(r"(?:bpe|super)-nf[cd]_d(?P<depth>\d+)_s\d+")


def find_runs(root: Path) -> dict[str, Path]:
    runs = {}
    for ckpt in sorted(root.rglob("base_checkpoints/d*/model_*.pt")):
        run_dir = ckpt.parents[2]
        if RUN_RE.fullmatch(run_dir.name):
            assert runs.get(run_dir.name, run_dir) == run_dir, f"two copies of {run_dir.name}"
            runs[run_dir.name] = run_dir
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="searched recursively for runs")
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["clean"])
    ap.add_argument("--nanochat", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="run tags to evaluate (default: all)")
    ap.add_argument("--batch-size", type=int, default=8, help="16 overflows a 6 GB GPU at 1,024 tokens")
    ap.add_argument("--dtype", default="float16", help="NANOCHAT_DTYPE; float16 matches the Kaggle T4 runs")
    args = ap.parse_args()

    runs = find_runs(args.runs)
    if args.only:
        runs = {t: d for t, d in runs.items() if t in args.only}
    assert runs, f"no checkpoints under {args.runs}"
    args.out.mkdir(parents=True, exist_ok=True)
    for tag, run_dir in sorted(runs.items()):
        out = args.out / f"{tag}.json"
        if out.exists():
            print(f"[{tag}] done already", flush=True)
            continue
        assert (run_dir / "tokenizer" / "tokenizer.json").exists(), f"{run_dir}/tokenizer is missing"
        env = {**os.environ, "NANOCHAT_BASE_DIR": str(run_dir.resolve()), "NANOCHAT_DTYPE": args.dtype,
               "PYTHONPATH": os.pathsep.join([str(args.nanochat.resolve()), os.environ.get("PYTHONPATH", "")])}
        cmd = [sys.executable, "-m", "vitok.eval", "--test", str(args.docs.resolve()),
               f"--model-tag=d{RUN_RE.fullmatch(tag)['depth']}", "--variants", *args.variants,
               "--batch-size", str(args.batch_size), "--out", str(out.resolve())]
        with open(args.out / f"{tag}.log", "w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"[{tag}] " + (args.out / f"{tag}.log").read_text().strip().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
