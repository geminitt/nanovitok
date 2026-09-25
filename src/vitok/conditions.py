"""Training configuration per (condition, depth), implementing the equal-text design.

Every condition trains for the same number of steps with the same number of sequences per step.
The context length in tokens is scaled by chars-per-token so every context holds about the same
amount of text as bpe-nfc's BASE_SEQ tokens. LR and weight decay are pinned to bpe-nfc's batch.

    python -m vitok.conditions --compression compression.json --condition super-nfc --depth 8
"""

import argparse
import json
from pathlib import Path

BASE_CONDITION = "bpe-nfc"
BASE_SEQ = 1024
SEQS_PER_STEP = 64
# Text budget expressed in bpe-nfc tokens, about 20 per non-embedding parameter.
BUDGET_TOKENS = {6: 250_000_000, 8: 500_000_000, 10: 1_000_000_000}
# Per-GPU micro batch (sequences) on a 16GB T4; must divide SEQS_PER_STEP. Halve on OOM.
DEVICE_BATCH = {6: 32, 8: 32, 10: 16}


def seq_len(cpt: dict, condition: str) -> int:
    raw = BASE_SEQ * cpt[BASE_CONDITION] / cpt[condition]
    return max(8, round(raw / 8) * 8)


def train_args(cpt: dict, condition: str, depth: int, device_batch: int | None = None) -> dict:
    t = seq_len(cpt, condition)
    base_batch = SEQS_PER_STEP * BASE_SEQ
    return {
        "depth": depth,
        "max-seq-len": t,
        "device-batch-size": device_batch or DEVICE_BATCH[depth],
        "total-batch-size": SEQS_PER_STEP * t,
        "scaling-batch-size": base_batch,
        "num-iterations": BUDGET_TOKENS[depth] // base_batch,
        # full attention in every layer: without FlashAttention 3 (not on T4), nanochat emulates sliding
        # windows with an explicit SDPA mask, which still does the full T^2 work and loses the is_causal path
        "window-pattern": "L",
    }


def load_cpt(compression_json: Path) -> dict:
    data = json.loads(Path(compression_json).read_text(encoding="utf-8"))
    return {k: v["chars_per_token"] for k, v in data.items() if isinstance(v, dict)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compression", type=Path, required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--device-batch-size", type=int, default=None)
    args = ap.parse_args()
    cfg = train_args(load_cpt(args.compression), args.condition, args.depth, args.device_batch_size)
    print(" ".join(f"--{k}={v}" for k, v in cfg.items()))


if __name__ == "__main__":
    main()
