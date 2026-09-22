"""Train + evaluate a queue of conditions on one GPU (used by kaggle/notebooks/02_train_eval.ipynb).

    python -m vitok.kaggle_run --gpu 0 --conditions bpe-nfc bpe-nfd --depth 6 --seed 0 \
        --data /kaggle/input/vitok-data --nanochat /kaggle/working/nanochat --work /kaggle/working

Each run gets its own NANOCHAT_BASE_DIR, so two queues can share the data without clashing.
A run whose final checkpoint already exists is not retrained (only re-evaluated if needed).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from vitok.conditions import load_cpt, train_args


def run_one(args, cond: str, log) -> None:
    from vitok.hf_tokenizer import write_token_bytes

    tag = f"{cond}_d{args.depth}_s{args.seed}"
    run_dir = args.work / "runs" / tag
    result = args.work / "results" / f"{tag}.json"
    tok_dir = run_dir / "tokenizer"
    if not tok_dir.exists():
        shutil.copytree(args.data / args.tokenizers / cond, tok_dir)
        write_token_bytes(tok_dir)
    data_link = run_dir / "base_data_climbmix"
    if not data_link.exists():
        data_link.symlink_to(args.data / "shards", target_is_directory=True)

    cfg = train_args(load_cpt(args.data / args.compression), cond, args.depth, args.device_batch_size)
    if args.num_iterations:
        cfg["num-iterations"] = args.num_iterations
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(args.gpu),
           "NANOCHAT_BASE_DIR": str(run_dir),
           "NANOCHAT_DTYPE": "float16",
           "NANOCHAT_SEED": str(args.seed),
           "PYTHONPATH": os.pathsep.join([str(args.nanochat), os.environ.get("PYTHONPATH", "")])}
    if args.no_compile:
        env["TORCHDYNAMO_DISABLE"] = "1"

    ckpt_dir = run_dir / "base_checkpoints" / f"d{args.depth}"
    final = ckpt_dir / f"model_{cfg['num-iterations']:06d}.pt"
    if not final.exists():
        cmd = [sys.executable, "-m", "scripts.base_train", *[f"--{k}={v}" for k, v in cfg.items()],
               "--run=dummy", f"--model-tag=d{args.depth}", "--core-metric-every=-1", "--sample-every=-1",
               f"--eval-every={args.eval_every}", f"--eval-tokens={args.eval_tokens}"]
        log(f"[{tag}] train: {' '.join(cmd[3:])}")
        with open(run_dir / "train.log", "w") as f:
            subprocess.run(cmd, cwd=args.nanochat, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    for p in ckpt_dir.glob("optim_*.pt"):  # optimizer state is only needed for resuming
        p.unlink()

    if not result.exists():
        cmd = [sys.executable, "-m", "vitok.eval", "--test", str(args.data / "test.jsonl"),
               "--pairs", str(args.data / "minimal_pairs.jsonl"), f"--model-tag=d{args.depth}",
               "--out", str(result)]
        log(f"[{tag}] eval")
        with open(run_dir / "eval.log", "w") as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    log(f"[{tag}] done -> {result}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--tokenizers", default="tokenizers")
    ap.add_argument("--compression", default="compression.json")
    ap.add_argument("--nanochat", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--num-iterations", type=int, default=None, help="override, for smoke tests")
    ap.add_argument("--device-batch-size", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-tokens", type=int, default=2_000_000)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    (args.work / "results").mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    for cond in args.conditions:
        (args.work / "runs" / f"{cond}_d{args.depth}_s{args.seed}").mkdir(parents=True, exist_ok=True)
        run_one(args, cond, log)


if __name__ == "__main__":
    main()
