"""Evaluate one trained checkpoint: per-document nats on clean / diacritic-stripped test text,
and log-probabilities for minimal pairs. Bits per NFC character are computed in vitok.analysis
over the documents every condition could score.

    NANOCHAT_BASE_DIR=runs/super-nfc_d8 python -m vitok.eval \
        --test test.jsonl --pairs minimal_pairs.jsonl --model-tag d8 --out results/super-nfc_d8_s0.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from vitok.text import strip_diacritics, strip_diacritics_partial

VARIANTS = {
    "clean": lambda s: s,
    "strip50": lambda s: strip_diacritics_partial(s, 0.5, seed=0),
    "strip100": strip_diacritics,
}


@torch.no_grad()
def sequence_nats(model, tokenizer, texts: list[str], max_len: int, batch_size: int = 16) -> list[float | None]:
    """Total next-token loss (nats) of each text given BOS. None if the text exceeds max_len tokens.

    Batches are right-padded; causal attention means padding never influences real positions,
    and padded targets are ignored (-1)."""
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(texts, prepend=bos)
    out: list[float | None] = [None] * len(texts)
    order = sorted((i for i in range(len(texts)) if 2 <= len(ids[i]) <= max_len + 1), key=lambda i: len(ids[i]))
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        T = max(len(ids[i]) for i in chunk) - 1
        x = torch.full((len(chunk), T), bos, dtype=torch.long)
        y = torch.full((len(chunk), T), -1, dtype=torch.long)
        for r, i in enumerate(chunk):
            seq = torch.tensor(ids[i], dtype=torch.long)
            x[r, :len(seq) - 1] = seq[:-1]
            y[r, :len(seq) - 1] = seq[1:]
        if T == 1:  # nanochat's training forward needs T > 1; pad one ignored position
            x = torch.cat([x, x[:, :1]], dim=1)
            y = torch.cat([y, torch.full_like(y[:, :1], -1)], dim=1)
        loss = model(x.to(device), y.to(device), loss_reduction="none").view(len(chunk), -1)
        for r, i in enumerate(chunk):
            out[i] = loss[r].double().sum().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from nanochat.checkpoint_manager import build_model, find_last_step
    from nanochat.common import get_base_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(get_base_dir(), "base_checkpoints", args.model_tag)
    step = args.step if args.step is not None else find_last_step(ckpt_dir)
    model, tokenizer, meta = build_model(ckpt_dir, step, device, "eval")
    max_len = meta["model_config"]["sequence_len"]

    docs = [json.loads(line) for line in args.test.read_text(encoding="utf-8").splitlines()]
    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines()]
    t0 = time.time()
    result = {"step": step, "max_seq_len": max_len, "user_config": meta.get("user_config"),
              "doc_ids": [d["id"] for d in docs], "docs": {}}
    for name, fn in VARIANTS.items():
        texts = [fn(d["text"]) for d in docs]
        result["docs"][name] = {
            "chars": [len(t) for t in texts],
            "nats": sequence_nats(model, tokenizer, texts, max_len, args.batch_size),
        }
    good = sequence_nats(model, tokenizer, [p["good"] for p in pairs], max_len, args.batch_size)
    bad = sequence_nats(model, tokenizer, [p["bad"] for p in pairs], max_len, args.batch_size)
    result["pairs"] = {"ids": [p["id"] for p in pairs], "good_nats": good, "bad_nats": bad}
    result["eval_seconds"] = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result), encoding="utf-8")
    skipped = sum(n is None for n in result["docs"]["clean"]["nats"])
    print(f"wrote {args.out} | docs skipped (too long): {skipped}/{len(docs)} | {result['eval_seconds']:.0f}s")


if __name__ == "__main__":
    main()
