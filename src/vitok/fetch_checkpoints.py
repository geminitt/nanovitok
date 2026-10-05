"""Download the final checkpoint of every run from the Kaggle notebook version that trained it.

    python -m vitok.fetch_checkpoints --out kaggle/outputs

Each session of notebook 02 saved its runs in the output of one notebook version (RUNS below). The
Kaggle CLI always serves the latest version, so this uses kagglehub, which can address an older one.
Every run directory also gets the tokenizer it was trained with (copied from vitok-data, plus nanochat's
token_bytes.pt), as `vitok.kaggle_run` leaves it, so `vitok.eval_checkpoints` can score it.

A file already in place is skipped, so an interrupted download resumes by rerunning the command; files
are moved into place only once complete. Needs a Kaggle API token (~/.kaggle/access_token).
"""

import argparse
import shutil
import time
from pathlib import Path

from vitok.conditions import BASE_SEQ, BUDGET_TOKENS, SEQS_PER_STEP

NOTEBOOK = "spritker/tokenizer-vietnamese-train-eval"
RUNS = {  # notebook version -> runs saved in its output
    4: ["bpe-nfc_d6_s0", "bpe-nfd_d6_s0", "super-nfc_d6_s0", "super-nfd_d6_s0"],
    5: ["bpe-nfc_d6_s1", "super-nfc_d6_s1"],
    6: ["bpe-nfc_d8_s0", "bpe-nfd_d8_s0", "super-nfc_d8_s0", "super-nfd_d8_s0"],
    8: ["bpe-nfc_d8_s1", "super-nfc_d8_s1"],
    9: ["bpe-nfc_d10_s0", "super-nfc_d10_s0"],
}


def checkpoint_files(tag: str) -> list[str]:
    """Paths of a run's final checkpoint inside the notebook output."""
    depth = int(tag.split("_d")[1].split("_")[0])
    step = BUDGET_TOKENS[depth] // (SEQS_PER_STEP * BASE_SEQ)
    return [f"runs/{tag}/base_checkpoints/d{depth}/{name}_{step:06d}.{ext}"
            for name, ext in (("meta", "json"), ("model", "pt"))]


def kaggle_download(version: int, rel: str, stage: Path, attempts: int = 3) -> Path:
    import kagglehub

    for attempt in range(1, attempts + 1):
        try:
            path = Path(kagglehub.notebook_output_download(f"{NOTEBOOK}/versions/{version}", path=rel,
                                                           output_dir=str(stage)))
            if not path.is_file():
                raise FileNotFoundError(f"kagglehub returned {path}, which is not a file")
            return path
        except Exception as e:
            if attempt == attempts:
                raise
            wait = 30 * attempt
            print(f"v{version} {rel}: {e!r}, retrying in {wait}s", flush=True)
            time.sleep(wait)


def fetch(out: Path, tokenizers: Path, only: list[str] | None = None, download=kaggle_download) -> list[str]:
    """Put every run's checkpoint and tokenizer under out/results-v{version}/; return the failures."""
    from vitok.hf_tokenizer import write_token_bytes

    stage = out / ".fetch-staging"
    failed = []
    for version, tags in RUNS.items():
        for tag in tags:
            if only and tag not in only:
                continue
            run_dir = out / f"results-v{version}" / "runs" / tag
            for rel in checkpoint_files(tag):
                dst = out / f"results-v{version}" / rel
                if dst.exists():
                    print(f"v{version} {rel}: already in place", flush=True)
                    continue
                try:
                    src = download(version, rel, stage / f"v{version}")
                except Exception as e:  # keep going, list the failures at the end
                    print(f"v{version} {rel}: FAILED {e!r}", flush=True)
                    failed.append(rel)
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(src, dst)
                assert dst.is_file() and not Path(src).exists()
                print(f"v{version} {rel}: {dst.stat().st_size / 1e6:.0f} MB", flush=True)
            tok_dir = run_dir / "tokenizer"
            if not (tok_dir / "token_bytes.pt").exists():
                cond = tag.split("_d")[0]
                shutil.copytree(tokenizers / cond, tok_dir, dirs_exist_ok=True)
                write_token_bytes(tok_dir)
    if not failed:
        shutil.rmtree(stage, ignore_errors=True)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="kaggle/outputs")
    ap.add_argument("--tokenizers", type=Path, default=None,
                    help="the tokenizers the runs used (default: OUT/vitok-data/tokenizers-16k)")
    ap.add_argument("--only", nargs="*", default=None, help="run tags to fetch (default: all)")
    args = ap.parse_args()
    failed = fetch(args.out, args.tokenizers or args.out / "vitok-data" / "tokenizers-16k", args.only)
    if failed:
        raise SystemExit(f"{len(failed)} files failed, rerun to retry: " + ", ".join(failed))
    print("all checkpoints in place", flush=True)


if __name__ == "__main__":
    main()
