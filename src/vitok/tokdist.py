"""Token Distillation initialization of the new tokens' rows (Dobler, Elliott and de Melo, ICLR 2026; PLAN.md).

    python -m vitok.tokdist --model runs/.../multisyllable-mean --base-tokenizer base.json \
        --tokenizer multisyllable-1000.json --stream runs/mixed-data-2026-10-21/stream --out runs/.../tokdist

Objective (official code, github.com/konstantinjdobler/token-distillation, commit 35702b5, `train_loop.py`): the
frozen model reads a snippet twice, once in the base tokenization (teacher, no gradient) and once with one
occurrence of a new token's pieces merged into its id (student); the loss is the mean squared error between the
last-layer hidden states at aligned positions (a merged token is matched to its last piece). With `--ce` (the
paper's `CE-auto-weighted`, its regularizer for tied embeddings) the student's next-token cross-entropy is added with
the weight mse / ce of that batch. Only the new rows learn: they are a separate fp32 parameter, so every other
weight stays bit-identical. AdamW (no weight decay), constant learning rate, one epoch over the snippets in a seeded
order; the batch drops its last incomplete part, as the official loader.

Snippets: the stream's documents, in order, encoded by both tokenizers; each place where the new tokenizer emits a
new token gives a window of `--snippet-len` base tokens holding that token's pieces (the window centered on them
where the document allows), at most `--per-doc` per token and document and `--snippets` per token. A token without
any snippet gets no MSE gradient; without `--ce` it keeps its starting row, with `--ce` it still moves as an output
candidate of the student positions (counted in the manifest).

The starting model is the mean-initialized extension (`vitok.extend`); the output folder `model/` is that model with
the trained rows, loadable like any extension. Training checkpoints every `--save-every` batches, on SIGTERM and
before `--deadline-minutes`; the same command resumes (a changed signature is refused).
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from vitok.adapt import EXIT_INCOMPLETE, _atomic_save, _sha256, hold_lock, input_weight, new_token_ids
from vitok.extend import new_token_pieces


def find_snippets(texts: Iterable[str], base: Tokenizer, new: Tokenizer, pieces: dict[int, list[int]], limit: int,
                  snippet_len: int, per_doc: int, encode_batch: int = 64) -> tuple[dict[int, list[dict]], dict]:
    """(token id -> up to `limit` snippets {"doc", "ids" (base ids), "at" (first piece in `ids`)} in stream order,
    counts). An occurrence where the base tokenization has no token boundary at the new token's start or does not
    spell its pieces there (punctuation joined to a word by the base pre-tokenizer) is skipped and counted."""
    out: dict[int, list[dict]] = {i: [] for i in pieces}
    stats = {"documents": 0, "misaligned": 0, "too_long": 0}
    first = min(pieces) if pieces else 0
    open_ = set(pieces)
    batch: list[str] = []
    doc = 0

    def scan(texts_: list[str], start_doc: int):
        stats["documents"] += len(texts_)
        for d, (b, n) in enumerate(zip(base.encode_batch(texts_, add_special_tokens=False),
                                       new.encode_batch(texts_, add_special_tokens=False), strict=True)):
            ids = np.asarray(n.ids)
            hits = np.flatnonzero(ids >= first)
            if not len(hits):
                continue
            base_ids, starts = b.ids, np.asarray([s for s, _ in b.offsets])
            per: dict[int, int] = {}
            for h in hits:
                tid = int(ids[h])
                if tid not in open_ or per.get(tid, 0) >= per_doc:
                    continue
                k = len(pieces[tid])
                p = int(np.searchsorted(starts, n.offsets[h][0]))
                if p >= len(starts) or starts[p] != n.offsets[h][0] or base_ids[p:p + k] != pieces[tid]:
                    stats["misaligned"] += 1
                    continue
                s = max(0, min(p - (snippet_len - k) // 2, len(base_ids) - snippet_len))
                if p + k > s + snippet_len:
                    stats["too_long"] += 1
                    continue
                out[tid].append({"doc": start_doc + d, "ids": base_ids[s:s + snippet_len], "at": p - s})
                per[tid] = per.get(tid, 0) + 1
                if len(out[tid]) == limit:
                    open_.discard(tid)

    for text in texts:
        batch.append(text)
        if len(batch) == encode_batch:
            scan(batch, doc)
            doc += len(batch)
            batch = []
            if not open_:
                return out, stats
    if batch:
        scan(batch, doc)
    return out, stats


def merge(snippet: list[int], at: int, k: int, new_id: int) -> tuple[list[int], list[int]]:
    """Student ids and, for each student position, its teacher position (a merged token -> its last piece)."""
    student = snippet[:at] + [new_id] + snippet[at + k:]
    align = list(range(at)) + [at + k - 1] + list(range(at + k, len(snippet)))
    return student, align


def examples(snips: dict[int, list[dict]], pieces: dict[int, list[int]], seed: int) -> list[dict]:
    """Every (student, align, teacher) example, in the seeded order of the one epoch."""
    out = []
    for tid in sorted(snips):
        for s in snips[tid]:
            student, align = merge(s["ids"], s["at"], len(pieces[tid]), tid)
            out.append({"token": tid, "student": student, "align": align, "teacher": s["ids"]})
    random.Random(seed).shuffle(out)
    return out


def loss_fn(model, w_old: torch.Tensor, new: torch.Tensor, first: int, batch: list[dict], device: str, ce: bool,
            amp: dict) -> tuple[torch.Tensor, dict]:
    """The batch loss (MSE, plus the auto-weighted cross-entropy with `ce`) and its parts."""
    def embed(ids: torch.Tensor) -> torch.Tensor:
        out = w_old[ids.clamp(max=w_old.shape[0] - 1)].float()
        return torch.where((ids >= first)[..., None], new[(ids - first).clamp(min=0, max=new.shape[0] - 1)], out)

    def pad(seqs, m):
        return torch.tensor([q + [0] * (m - len(q)) for q in seqs], device=device)

    n_s, n_t = max(len(e["student"]) for e in batch), max(len(e["teacher"]) for e in batch)
    student, teacher = pad([e["student"] for e in batch], n_s), pad([e["teacher"] for e in batch], n_t)
    align = pad([e["align"] for e in batch], n_s)
    valid = torch.tensor([[1] * len(e["student"]) + [0] * (n_s - len(e["student"])) for e in batch],
                         device=device).bool()
    t_mask = torch.tensor([[1] * len(e["teacher"]) + [0] * (n_t - len(e["teacher"])) for e in batch], device=device)
    s_mask = valid.long()
    with torch.autocast(**amp):
        with torch.no_grad():
            t_h = model.model(inputs_embeds=embed(teacher).to(w_old.dtype),
                              attention_mask=t_mask).last_hidden_state.float()
        s_h = model.model(inputs_embeds=embed(student).to(w_old.dtype), attention_mask=s_mask).last_hidden_state.float()
        target = t_h.gather(1, align[..., None].expand(-1, -1, t_h.shape[-1]))
        mse = F.mse_loss(s_h[valid], target[valid])
        parts = {"mse": float(mse.detach())}
        loss = mse
        if ce:
            w_all = torch.cat([w_old[:first].float(), new], 0)  # new id i -> row i
            logits = s_h[:, :-1] @ w_all.T
            tgt = student[:, 1:].masked_fill(~valid[:, 1:], -100)
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
            parts["ce"] = float(ce_loss.detach())
            loss = mse + ce_loss * (parts["mse"] / parts["ce"])
    return loss, parts


def signature(args, model_sha: str) -> dict:
    return {"model_sha256": model_sha, "base_tokenizer_sha256": _sha256(args.base_tokenizer),
            "tokenizer_sha256": _sha256(args.tokenizer), "stream": _sha256(args.stream / "stream_manifest.json"),
            "snippets": args.snippets, "snippet_len": args.snippet_len, "per_doc": args.per_doc,
            "batch": args.batch, "lr": args.lr, "ce": args.ce, "seed": args.seed, "dtype": args.dtype,
            "max_docs": args.max_docs}


def model_sha256(folder: Path) -> str:
    """One digest over the starting model's weight files (names and bytes)."""
    h = hashlib.sha256()
    for path in sorted(folder.glob("*.safetensors")):
        h.update(path.name.encode() + bytes.fromhex(_sha256(path)))
    return h.hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True, help="the mean-initialized extension (vitok.extend)")
    ap.add_argument("--base-tokenizer", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--stream", type=Path, required=True, help="training stream folder (vitok.mixed_stream)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snippets", type=int, default=100, help="snippets per token (official README: 100)")
    ap.add_argument("--snippet-len", type=int, default=50, help="base tokens per snippet (paper: 50)")
    ap.add_argument("--per-doc", type=int, default=5, help="snippets per token and document")
    ap.add_argument("--max-docs", type=int, default=None, help="search only the first N documents")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ce", action=argparse.BooleanOptionalAction, default=True,
                    help="add the auto-weighted cross-entropy (CE-auto-weighted, default on)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16", help="autocast dtype on CUDA")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-every", type=int, default=200, help="checkpoint every this many batches")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--deadline-minutes", type=float, default=None)
    args = ap.parse_args(argv)
    if min(args.snippets, args.snippet_len, args.per_doc, args.batch, args.save_every, args.log_every) < 1:
        raise SystemExit("counts and sizes must be positive")
    if args.dtype != "fp32" and not args.device.startswith("cuda"):
        args.dtype = "fp32"  # autocast is CUDA only here; CPU runs in fp32
    return args


def run(args) -> int:
    from transformers import AutoModelForCausalLM

    from vitok.mixed_stream import mixed_records

    args.out.mkdir(parents=True, exist_ok=True)
    lock = hold_lock(args.out)  # noqa: F841
    if (args.out / "model" / "tokdist_manifest.json").exists():
        print(f"{args.out} already holds a finished run", flush=True)
        return 0
    sig = signature(args, model_sha256(args.model))
    base, new_tok = Tokenizer.from_file(str(args.base_tokenizer)), Tokenizer.from_file(str(args.tokenizer))
    pieces = new_token_pieces(base, new_tok)
    ids = new_token_ids(base, new_tok)
    if not pieces or sorted(pieces) != list(ids):
        raise SystemExit("the tokenizer adds no new tokens over the base, or not one block of them")
    first = ids.start

    snip_path, ckpt_path = args.out / "snippets.json", args.out / "checkpoint.pt"
    log_path = args.out / "train_log.jsonl"
    if snip_path.exists():
        saved = json.loads(snip_path.read_text())
        if saved["signature"] != sig:
            raise SystemExit(f"{snip_path} belongs to another run; use a new --out")
        snips = {int(k): v for k, v in saved["snippets"].items()}
    else:
        t = time.monotonic()
        texts = (r["text"] for r in mixed_records(args.stream))
        if args.max_docs is not None:
            texts = (x for _, x in zip(range(args.max_docs), texts, strict=False))
        snips, stats = find_snippets(texts, base, new_tok, pieces, args.snippets, args.snippet_len, args.per_doc)
        tmp = snip_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"signature": sig, "seconds": time.monotonic() - t, "stats": stats,
                                   "snippets": {str(k): v for k, v in snips.items()}}))
        os.replace(tmp, snip_path)
    search = json.loads(snip_path.read_text())
    search_stats = {"seconds": search["seconds"], **search["stats"]}
    data = examples(snips, pieces, args.seed)
    if len(data) < args.batch:
        raise SystemExit(f"only {len(data)} snippets, fewer than one batch of {args.batch}")
    n_batches = len(data) // args.batch

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if args.dtype == "bf16"
                                                 else torch.float32)
    model.to(args.device).eval().requires_grad_(False)
    w_old = input_weight(model)
    head = model.get_output_embeddings()
    if head is None or head.weight.data_ptr() != w_old.data_ptr():
        raise SystemExit("expected tied embeddings")
    if ids.stop > w_old.shape[0]:
        raise SystemExit(f"the model has {w_old.shape[0]} rows; extend it first")
    start_rows = w_old.detach()[first:ids.stop].float().clone()
    new = torch.nn.Parameter(start_rows.clone())
    opt = torch.optim.AdamW([new], lr=args.lr, weight_decay=0.0)
    done = 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ck["signature"] != sig:
            raise SystemExit(f"{ckpt_path} belongs to another run; use a new --out")
        with torch.no_grad():
            new.copy_(ck["new"].to(args.device))
        opt.load_state_dict(ck["optimizer"])
        done = ck["batches"]
        print(f"resumed at batch {done} of {n_batches}", flush=True)
    if log_path.exists():  # keep the lines the checkpoint covers
        keep = []
        for line in log_path.read_text().splitlines():
            try:
                if json.loads(line)["batch"] < done:
                    keep.append(line)
            except (json.JSONDecodeError, KeyError):
                continue
        log_path.write_text("".join(x + "\n" for x in keep))

    probe_rows = torch.linspace(0, first - 1, 256).long().unique()
    probe = w_old.detach()[probe_rows].clone()
    use_bf16 = args.device.startswith("cuda") and args.dtype == "bf16"
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
    stop = {"signal": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("signal", True))
    started = time.monotonic()

    def save():
        _atomic_save({"new": new.detach().cpu(), "optimizer": opt.state_dict(), "batches": done, "signature": sig},
                     ckpt_path)

    with open(log_path, "a", encoding="utf-8") as log:
        while done < n_batches:
            batch = data[done * args.batch:(done + 1) * args.batch]
            loss, parts = loss_fn(model, w_old, new, first, batch, args.device, args.ce, amp)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at batch {done}: {parts}")
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            done += 1
            if done % args.log_every == 0 or done == n_batches:
                rec = {"batch": done - 1, "loss": float(loss.detach()), **parts,
                       "new_row_norm": float(new.detach().norm(dim=1).mean()),
                       "elapsed_seconds": time.monotonic() - started}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                os.fsync(log.fileno())
                print(json.dumps(rec), flush=True)
            over = args.deadline_minutes is not None and time.monotonic() - started > 60 * args.deadline_minutes
            if done % args.save_every == 0 or stop["signal"] or over:
                save()
            if (stop["signal"] or over) and done < n_batches:
                print(f"stopped at batch {done}; rerun the same command to resume", flush=True)
                return EXIT_INCOMPLETE

    if not torch.equal(w_old.detach()[probe_rows], probe):
        raise AssertionError("old rows changed during Token Distillation")
    trained = sorted(t for t, s in snips.items() if s)
    with torch.no_grad():
        final = new.detach()
        moved = (final - start_rows).abs().amax(dim=1)
        rows_trained = torch.tensor([t - first for t in trained], device=final.device, dtype=torch.long)
        untrained = torch.tensor([t - first for t in ids if t not in set(trained)], device=final.device,
                                 dtype=torch.long)
        if len(trained) and not bool((moved[rows_trained] > 0).all()):
            raise AssertionError("a token with snippets kept its starting row")
        # without CE a row gets a gradient only from its own snippets; with CE every new row is an output candidate
        # of every student position (tied embeddings), so rows without snippets move through the softmax too
        untrained_moved = int((moved[untrained] > 0).sum()) if len(untrained) else 0
        if untrained_moved and not args.ce:
            raise AssertionError("a token without snippets moved without CE")
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)  # saved in fp32
        emb = input_weight(model)
        before = emb.detach()[probe_rows.cpu()].clone()
        emb[first:ids.stop] = final.cpu()
        assert torch.equal(emb.detach()[probe_rows.cpu()], before)
    model_dir = args.out / "model"
    model.save_pretrained(model_dir)
    shutil.copy(args.tokenizer, model_dir / "tokenizer.json")
    norms = final.norm(dim=1)
    counts = [len(snips[t]) for t in ids]
    manifest = {"signature": sig, "new_tokens": len(ids), "tokens_with_snippets": len(trained),
                "snippets": sum(counts), "snippets_per_token": {"min": min(counts), "median": float(np.median(counts)),
                                                                "max": max(counts)},
                "batches": n_batches, "first_new_id": first, "search": search_stats,
                "tokens_without_snippets_moved_by_ce": untrained_moved,
                "new_row_norm": {"start": float(start_rows.norm(dim=1).mean()), "final": float(norms.mean()),
                                 "max": float(norms.max())},
                "old_row_norm_mean": float(w_old.detach()[:first].float().norm(dim=1).mean()),
                "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
                "seconds": time.monotonic() - started, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    assert all(math.isfinite(v) for v in manifest["new_row_norm"].values())
    (model_dir / "tokdist_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ckpt_path.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


def main(argv=None):
    sys.exit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
