"""Adaptation training: continue training a (possibly extended) causal LM on Vietnamese text, for one condition.

    python -m vitok.adapt --model runs/.../multisyllable-model --base-tokenizer qwen3/tokenizer.json \
        --tokenizer runs/.../multisyllable/tokenizer.json --shards data/shards/shard_000*.parquet \
        --held-out test.jsonl val_docs.jsonl --steps 2000 --batch 32 --seq-len 512 --out runs/.../multisyllable-train

What is trained (Yamaguchi et al.'s small-budget recipe, with the In-Place lesson of keeping old rows frozen): the
embedding rows of the new tokens (the ids the tokenizer has and the base tokenizer does not) and the chosen
transformer layers (default the bottom 2 and top 2). With tied embeddings the same rows are the new tokens' output
vectors. Every other weight, including every old embedding row, stays bit-identical. A condition without new tokens
(`base`) trains the same layers and no embedding row.

Equal compute: every condition takes the same number of steps of `batch` rows of `seq_len` tokens. Each step logs the
characters its tokens cover, so results can be plotted against compute and against text read.

Long-run safety: the run checkpoints every `--save-every` steps, on SIGTERM and before `--deadline-minutes`, and the
same command resumes from the latest checkpoint (model trainables, optimizer, gradient scaler, RNG, data position). A
checkpoint whose run signature differs from the command's is refused, never overwritten. Snapshots of the trainable
weights at `--snapshot-steps` give the recovery curve; `load_snapshot` puts one back into the starting model.
"""

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from vitok import cpt_data

EXIT_INCOMPLETE = 3  # stopped by the deadline or SIGTERM with a checkpoint saved; rerun the command to resume
EMBED = "model.embed_tokens.weight"


def input_weight(model) -> torch.Tensor:
    """The input embedding matrix (also the output head when tied)."""
    return model.get_input_embeddings().weight


def new_token_ids(base: Tokenizer, new: Tokenizer) -> range:
    """The ids the new tokenizer adds over the base; they must form one block after the old ones."""
    old = base.get_vocab(with_added_tokens=True)
    ids = sorted(i for t, i in new.get_vocab(with_added_tokens=True).items() if t not in old)
    if not ids:
        return range(0)
    if ids != list(range(ids[0], ids[-1] + 1)) or ids[0] <= max(old.values()):
        raise ValueError(f"new token ids must be one block after the old ones, got {ids[0]}..{ids[-1]} ({len(ids)})")
    return range(ids[0], ids[-1] + 1)


def resolve_layers(spec: str, n_layers: int) -> list[int]:
    """"0,1,-2,-1" -> [0, 1, n-2, n-1]; refuses out-of-range or repeated layers."""
    out = []
    for part in spec.split(","):
        i = int(part)
        j = i + n_layers if i < 0 else i
        if not 0 <= j < n_layers or j in out:
            raise ValueError(f"layer {i} is out of range for {n_layers} layers or repeated")
        out.append(j)
    return sorted(out)


def set_trainable(model, new_ids: range, layers: list[int]) -> dict[str, torch.nn.Parameter]:
    """Freeze everything, then unfreeze the chosen layers and (if there are new ids) the embedding, whose gradient is
    masked to the new rows. Returns the trainable parameters by name."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = {}
    for name, p in model.named_parameters():
        if any(name.startswith(f"model.layers.{i}.") for i in layers):
            p.requires_grad_(True)
            trainable[name] = p
    emb = model.get_input_embeddings().weight
    if len(new_ids):
        if new_ids.stop > emb.shape[0]:
            raise ValueError(f"new ids end at {new_ids.stop - 1} but the model has {emb.shape[0]} embedding rows")
        emb.requires_grad_(True)
        mask = torch.zeros(emb.shape[0], 1, dtype=emb.dtype, device=emb.device)
        mask[new_ids.start:new_ids.stop] = 1
        emb.register_hook(lambda g: g * mask)
        trainable[EMBED] = emb
    assert all(p.requires_grad for p in trainable.values())
    assert sum(p.requires_grad for p in model.parameters()) == len(trainable), "a tied weight was counted twice"
    return trainable


def make_optimizer(trainable: dict, lr: float, embed_lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW with weight decay on the layers' matrices only (never on norms or embedding rows); each group keeps its
    peak learning rate in "peak" for the schedule."""
    groups: list[dict[str, Any]] = [
              {"params": [p for n, p in trainable.items() if n != EMBED and p.ndim >= 2],
               "weight_decay": weight_decay, "peak": lr},
              {"params": [p for n, p in trainable.items() if n != EMBED and p.ndim < 2],
               "weight_decay": 0.0, "peak": lr},
              {"params": [p for n, p in trainable.items() if n == EMBED], "weight_decay": 0.0, "peak": embed_lr}]
    groups = [g for g in groups if g["params"]]
    assert sum(len(g["params"]) for g in groups) == len(trainable)
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def precision(dtype: str) -> tuple[torch.dtype | None, bool]:
    """(autocast dtype or None for fp32, whether the loss is scaled): fp16 needs loss scaling, bf16 does not."""
    return {"fp32": (None, False), "fp16": (torch.float16, True), "bf16": (torch.bfloat16, False)}[dtype]


def lr_at(step: int, steps: int, peak: float, warmup: int, min_ratio: float) -> float:
    """Linear warmup to `peak`, then cosine decay to `peak * min_ratio` at the last step."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - 1 - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def trainable_state(trainable: dict, new_ids: range) -> dict[str, torch.Tensor]:
    """CPU copies of the trainable weights; the embedding is stored as its new rows only."""
    out = {}
    for name, p in trainable.items():
        t = p.detach()[new_ids.start:new_ids.stop] if name == EMBED else p.detach()
        out[name] = t.to("cpu", copy=True)
    return out


@torch.no_grad()
def load_snapshot(model, path: Path) -> dict:
    """Put a snapshot's (or checkpoint's) trainable weights into the starting model, in place; returns its metadata."""
    snap = torch.load(path, map_location="cpu", weights_only=False)
    params = dict(model.named_parameters())
    rows = range(*snap["new_ids"])
    for name, t in snap["params"].items():
        if name not in params:
            raise ValueError(f"snapshot weight {name} is not in the model")
        target = params[name][rows.start:rows.stop] if name == EMBED else params[name]
        if target.shape != t.shape:
            raise ValueError(f"snapshot weight {name} has shape {tuple(t.shape)}, the model {tuple(target.shape)}")
        target.copy_(t.to(target.dtype))
    return {k: v for k, v in snap.items() if k != "params"}


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def signature(args, tok_sha: str, held_sha: list[str]) -> dict:
    """Everything that changes what the run computes; a checkpoint resumes only under the same signature."""
    return {"model": str(args.model), "revision": args.revision, "tokenizer_sha256": tok_sha,
            "shards": [[p.name, p.stat().st_size] for p in args.shards], "held_out_sha256": held_sha,
            "steps": args.steps, "batch": args.batch, "micro_batch": args.micro_batch, "seq_len": args.seq_len,
            "lr": args.lr, "embed_lr": args.embed_lr, "warmup": args.warmup, "min_lr_ratio": args.min_lr_ratio,
            "weight_decay": args.weight_decay, "layers": args.layers, "seed": args.seed, "dtype": args.dtype,
            "grad_clip": args.grad_clip, "grad_checkpointing": args.grad_checkpointing}


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _set_rng_state(s: dict):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _truncate_log(path: Path, upto_step: int):
    """Keep the log lines of steps the checkpoint covers; drop later lines and a line torn by a crash."""
    if not path.exists():
        return
    keep = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("step", -1) < upto_step:
            keep.append(line)
    path.write_text("".join(x + "\n" for x in keep), encoding="utf-8")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="Hugging Face id or folder of the starting model")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--base-tokenizer", type=Path, required=True, help="the original tokenizer.json")
    ap.add_argument("--tokenizer", type=Path, required=True, help="this condition's tokenizer.json")
    ap.add_argument("--shards", type=Path, nargs="+", required=True, help="parquet files with a text column, in order")
    ap.add_argument("--held-out", type=Path, nargs="+", required=True,
                    help="jsonl files whose texts must not be trained on")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True, help="rows per step (equal across conditions)")
    ap.add_argument("--micro-batch", type=int, default=None, help="rows per forward pass (default: --batch)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4, help="peak learning rate of the transformer layers")
    ap.add_argument("--embed-lr", type=float, default=None, help="peak learning rate of the new rows (default: --lr)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="on the layers' matrices (never on embedding rows)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--layers", default="0,1,-2,-1", help="trainable layers; negative counts from the top")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32", help="autocast dtype on CUDA")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eos-token", default="<|endoftext|>")
    ap.add_argument("--encode-batch", type=int, default=256, help="documents tokenized at a time (memory only)")
    ap.add_argument("--save-every", type=int, default=100, help="checkpoint every this many steps")
    ap.add_argument("--snapshot-steps", type=int, nargs="*", default=[],
                    help="keep the trainable weights after these steps")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--deadline-minutes", type=float, default=None,
                    help="checkpoint and stop after this much wall time")
    ap.add_argument("--max-bad-steps", type=int, default=20, help="abort after this many consecutive non-finite losses")
    args = ap.parse_args(argv)
    args.micro_batch = args.micro_batch or args.batch
    args.embed_lr = args.embed_lr or args.lr
    if args.steps < 1 or args.batch < 1 or args.batch % args.micro_batch:
        raise SystemExit(f"--steps and --batch must be positive and --batch ({args.batch}) a multiple of "
                         f"--micro-batch ({args.micro_batch})")
    if args.dtype != "fp32" and not args.device.startswith("cuda"):
        raise SystemExit(f"--dtype {args.dtype} needs a CUDA device")
    if any(not 1 <= s <= args.steps for s in args.snapshot_steps):
        raise SystemExit(f"--snapshot-steps must lie in 1..{args.steps}")
    return args


def train(args) -> int:
    from transformers import AutoModelForCausalLM

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "snapshots").mkdir(exist_ok=True)
    ckpt_path, log_path = args.out / "checkpoint.pt", args.out / "train_log.jsonl"
    base_tok, tok = Tokenizer.from_file(str(args.base_tokenizer)), Tokenizer.from_file(str(args.tokenizer))
    new_ids = new_token_ids(base_tok, tok)
    eos = tok.token_to_id(args.eos_token)
    if eos is None:
        raise SystemExit(f"the tokenizer has no {args.eos_token!r}")
    sig = signature(args, _sha256(args.tokenizer), [_sha256(p) for p in args.held_out])

    ckpt = None
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["signature"] != sig:
            diff = sorted(k for k in sig.keys() | ckpt["signature"].keys() if sig.get(k) != ckpt["signature"].get(k))
            raise SystemExit(f"{ckpt_path} belongs to another run (differs in {diff}); use a new --out")
    elif (args.out / "final.pt").exists():
        print(f"{args.out} already holds a finished run (final.pt)", flush=True)
        return 0

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32)
    model.to(args.device).train()
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vocab_rows = input_weight(model).shape[0]
    if tok.get_vocab_size(with_added_tokens=True) > vocab_rows:
        raise SystemExit(f"the tokenizer has more ids than the model's {vocab_rows} embedding rows; extend it first")
    head = model.get_output_embeddings()
    if len(new_ids) and head is not None and head.weight.data_ptr() != input_weight(model).data_ptr():
        raise SystemExit("an untied output head is not supported: its new rows would stay at their initial values")
    layers = resolve_layers(args.layers, model.config.num_hidden_layers)
    trainable = set_trainable(model, new_ids, layers)
    # a fixed sample of old rows, compared at every save: frozen rows must stay bit-identical
    emb = input_weight(model)
    probe_rows = torch.linspace(0, (new_ids.start or vocab_rows) - 1, 256).long().unique()
    probe = emb.detach()[probe_rows].clone()

    opt = make_optimizer(trainable, args.lr, args.embed_lr, args.weight_decay)
    use_cuda = args.device.startswith("cuda")
    amp_dtype, scale_loss = precision(args.dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=scale_loss)

    step, tokens, chars, data_state = 0, 0, 0, None
    if ckpt is not None:
        load_snapshot(model, ckpt_path)
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        _set_rng_state(ckpt["rng"])
        step, tokens, chars, data_state = ckpt["step"], ckpt["tokens"], ckpt["chars"], ckpt["data_state"]
        assert torch.equal(emb.detach()[probe_rows], probe), "old embedding rows differ after loading the checkpoint"
        print(f"resumed at step {step} ({tokens:,} tokens, {chars:,} chars)", flush=True)
    _truncate_log(log_path, step)

    held = cpt_data.held_out_digests(t for p in args.held_out for t in cpt_data.jsonl_texts(p))
    docs = cpt_data.training_docs(cpt_data.parquet_texts(args.shards), held)
    docs = islice(docs, data_state["docs_read"] if data_state else 0, None)
    batches = cpt_data.packed_batches(tok, docs, eos, args.seq_len, args.batch, encode_batch=args.encode_batch,
                                      state=data_state)

    stop = {"signal": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("signal", True))
    started, bad = time.monotonic(), 0
    last_state = data_state

    def save(path: Path, kind: str):
        assert torch.equal(emb.detach()[probe_rows], probe), "frozen old embedding rows changed during training"
        obj = {"params": trainable_state(trainable, new_ids), "new_ids": [new_ids.start, new_ids.stop],
               "step": step, "tokens": tokens, "chars": chars, "signature": sig, "kind": kind}
        if kind == "checkpoint":
            obj |= {"optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "rng": _rng_state(),
                    "data_state": last_state}
        _atomic_save(obj, path)

    with open(log_path, "a", encoding="utf-8") as log:
        while step < args.steps:
            t0 = time.monotonic()
            try:
                b = next(batches)
            except StopIteration:
                raise SystemExit(f"training text ran out at step {step} of {args.steps}; add shards") from None
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, g["peak"], args.warmup, args.min_lr_ratio)
            x = torch.from_numpy(b.input_ids).to(args.device)
            assert int(x.max()) < vocab_rows and int(x.min()) >= 0
            loss_sum = 0.0
            for rows in x.split(args.micro_batch):
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_cuda and amp_dtype is not None):
                    loss = model(input_ids=rows, labels=rows).loss
                scaler.scale(loss * rows.shape[0] / args.batch).backward()
                loss_sum += float(loss.detach()) * rows.shape[0] / args.batch
            scaler.unscale_(opt)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(list(trainable.values()), args.grad_clip))
            scaler.step(opt)  # skipped by the scaler when gradients overflowed
            scaler.update()
            opt.zero_grad(set_to_none=True)
            finite = math.isfinite(loss_sum) and math.isfinite(grad_norm)
            bad = 0 if finite else bad + 1
            step += 1
            tokens += b.input_ids.size
            chars += b.chars
            last_state = b.state
            if step % args.log_every == 0 or step == args.steps or not finite:
                rec = {"step": step - 1, "loss": loss_sum, "grad_norm": grad_norm, "lr": opt.param_groups[0]["lr"],
                       "tokens": tokens, "chars": chars, "step_seconds": time.monotonic() - t0,
                       "elapsed_seconds": time.monotonic() - started, "finite": finite,
                       "max_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if use_cuda else None}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                os.fsync(log.fileno())
                print(json.dumps(rec), flush=True)
            if bad > args.max_bad_steps:  # after logging, so the log shows the step that ends the run
                raise SystemExit(f"{bad} consecutive steps with a non-finite loss or gradient at step {step - 1}")
            if step in args.snapshot_steps:
                save(args.out / "snapshots" / f"step_{step:06d}.pt", "snapshot")
            over = args.deadline_minutes is not None and time.monotonic() - started > 60 * args.deadline_minutes
            if step % args.save_every == 0 or stop["signal"] or over:
                save(ckpt_path, "checkpoint")
            if (stop["signal"] or over) and step < args.steps:
                print(f"stopped at step {step} ({'SIGTERM' if stop['signal'] else 'deadline'}); "
                      f"rerun the same command to resume", flush=True)
                return EXIT_INCOMPLETE
    save(args.out / "final.pt", "final")
    (args.out / "manifest.json").write_text(json.dumps(manifest(args, sig, new_ids, step, tokens, chars), indent=2),
                                            encoding="utf-8")
    ckpt_path.unlink(missing_ok=True)  # only after the final weights and manifest are written
    return 0


def manifest(args, sig, new_ids, step, tokens, chars) -> dict:
    import subprocess
    from importlib.metadata import version

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                                cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"signature": sig, "new_tokens": len(new_ids), "steps": step, "tokens": tokens, "chars": chars,
            "git_commit": commit, "packages": {p: version(p) for p in ("torch", "transformers", "tokenizers")},
            "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
            "python": sys.version.split()[0], "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def main(argv=None):
    sys.exit(train(parse_args(argv)))


if __name__ == "__main__":
    main()
