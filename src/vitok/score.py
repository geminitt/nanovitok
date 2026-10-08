"""Evaluation of one model (one condition, optionally one training snapshot): every measurement the question needs.

    python -m vitok.score --model runs/.../multisyllable-model --tokenizer runs/.../multisyllable/tokenizer.json \
        --snapshot runs/.../multisyllable-train/final.pt --vi-docs test.jsonl --en-docs en.jsonl \
        --belebele belebele_vie_Latn.jsonl --out runs/.../score-multisyllable-final

Parts, each written item by item to its own jsonl (so a stopped run resumes where it stopped):
- vi_docs / en_docs: total nats of every document's tokens given an end-of-text token as context, and its NFC
  characters. bpc = sum of nats / (ln 2 * sum of chars), comparable across tokenizers because the characters are the
  same; documents are kept by index and digest so conditions pair on the same documents.
- belebele: zero-shot multiple choice by log-likelihood. Each answer is scored as a continuation of
  "{passage}\nCâu hỏi: {question}\nTrả lời:"; the prediction is the answer with the highest log-likelihood per
  character (acc_char, the protocol's measure) and, reported beside it, the highest total log-likelihood (acc_sum).
- generation: greedy continuation of the first `--speed-prompt-chars` characters of `--speed-prompts` documents for
  exactly `--new-tokens` tokens (special tokens suppressed, so every model does the same number of decode steps):
  prefill and decode times, generated characters per second, peak memory, and how repetitive the text is (share of
  4-syllable windows seen earlier in the same text).
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch
from tokenizers import Tokenizer

from vitok import cpt_data
from vitok.adapt import EXIT_INCOMPLETE, input_weight, load_snapshot
from vitok.data import truncate_chars

PROMPT = "{passage}\nCâu hỏi: {question}\nTrả lời:"  # Vietnamese "Question:" / "Answer:" (experimental input)
SYLLABLE = re.compile(r"\w+")


def sum_logprobs(model, seqs: list[list[int]], starts: list[int], batch_size: int, amp_dtype=None) -> list[float]:
    """For each sequence, the summed log-probability of its tokens from position starts[i] on, each given all earlier
    tokens. Batches are right-padded; causal attention keeps padding from touching real positions."""
    assert len(seqs) == len(starts) and all(1 <= s < len(q) for q, s in zip(seqs, starts, strict=True))
    device = next(model.parameters()).device
    out = [0.0] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for b in range(0, len(order), batch_size):
        chunk = order[b:b + batch_size]
        T = max(len(seqs[i]) for i in chunk)
        x = torch.zeros(len(chunk), T, dtype=torch.long)
        mask = torch.zeros(len(chunk), T, dtype=torch.long)
        for r, i in enumerate(chunk):
            x[r, :len(seqs[i])] = torch.tensor(seqs[i])
            mask[r, :len(seqs[i])] = 1
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(input_ids=x.to(device), attention_mask=mask.to(device)).logits
        for r, i in enumerate(chunk):
            n, s = len(seqs[i]), starts[i]
            lp = torch.log_softmax(logits[r, s - 1:n - 1].float(), dim=-1)
            value = float(lp.gather(1, x[r, s:n].to(device)[:, None]).sum())
            if not math.isfinite(value):  # an fp16 overflow is a numerical failure, not a score
                raise FloatingPointError(f"sequence {i}: log-probability {value}")
            out[i] = value
        del logits
    return out


def repetition(text: str, n: int = 4) -> float:
    """Share of n-syllable windows that already occurred earlier in the text (0 = none repeated)."""
    words = SYLLABLE.findall(text.lower())
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    seen, repeated = set(), 0
    for g in grams:
        repeated += g in seen
        seen.add(g)
    return repeated / len(grams)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate(model, tok: Tokenizer, prompt: str, new_tokens: int, banned: list[int], amp_dtype=None) -> dict:
    """Greedy decoding of exactly new_tokens tokens with a KV cache, timed; banned ids are never chosen."""
    device = next(model.parameters()).device
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids], device=device)
    ban = torch.tensor(banned, device=device, dtype=torch.long)
    out = []
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        _sync(device)
        t0 = time.perf_counter()
        res = model(input_ids=ids, use_cache=True)
        logits, past = res.logits[:, -1].float(), res.past_key_values
        logits[:, ban] = -math.inf
        nxt = logits.argmax(-1, keepdim=True)
        _sync(device)
        t1 = time.perf_counter()
        for _ in range(new_tokens - 1):
            out.append(int(nxt))
            res = model(input_ids=nxt, past_key_values=past, use_cache=True)
            logits, past = res.logits[:, -1].float(), res.past_key_values
            logits[:, ban] = -math.inf
            nxt = logits.argmax(-1, keepdim=True)
        out.append(int(nxt))
        _sync(device)
        t2 = time.perf_counter()
    text = tok.decode(out, skip_special_tokens=False)
    assert len(out) == new_tokens and not set(out) & set(banned)
    return {"prompt_tokens": ids.shape[1], "prompt_chars": len(prompt), "new_tokens": new_tokens,
            "new_chars": len(text), "prefill_seconds": t1 - t0, "decode_seconds": t2 - t1,
            "repetition_4": repetition(text), "text": text}


def _done(path: Path) -> dict[int, dict]:
    """Records already written (a line torn by a crash is dropped and redone)."""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[rec["i"]] = rec
    return done


def _append(path: Path, recs: list[dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _rewrite(path: Path, done: dict[int, dict]):
    """Drop a torn line before appending after it."""
    path.write_text("".join(json.dumps(done[i], ensure_ascii=False) + "\n" for i in sorted(done)), encoding="utf-8")


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Deadline(Exception):
    pass


def score_docs(model, tok, eos, docs: list[str], path: Path, batch_size, amp_dtype, deadline):
    done = _done(path)
    _rewrite(path, done)
    todo = [i for i in range(len(docs)) if i not in done]
    step = batch_size * 8
    for b in range(0, len(todo), step):
        part = todo[b:b + step]
        seqs = [[eos] + tok.encode(docs[i], add_special_tokens=False).ids for i in part]
        lps = sum_logprobs(model, seqs, [1] * len(part), batch_size, amp_dtype)
        _append(path, [{"i": i, "digest": cpt_data.digest(docs[i]).hex(), "chars": len(docs[i]),
                        "tokens": len(s) - 1, "nats": -lp} for i, s, lp in zip(part, seqs, lps, strict=True)])
        if deadline():  # checked after a unit, so every call makes progress
            raise Deadline
    done = _done(path)
    assert sorted(done) == list(range(len(docs)))
    recs = [done[i] for i in range(len(docs))]
    nats, chars = sum(r["nats"] for r in recs), sum(r["chars"] for r in recs)
    return {"docs": len(recs), "chars": chars, "tokens": sum(r["tokens"] for r in recs),
            "bpc": nats / (math.log(2) * chars), "chars_per_token": chars / sum(r["tokens"] for r in recs)}


def score_belebele(model, tok, items: list[dict], path: Path, batch_size, amp_dtype, deadline):
    done = _done(path)
    _rewrite(path, done)
    todo = [i for i in range(len(items)) if i not in done]
    for b in range(0, len(todo), batch_size * 2):
        part = todo[b:b + batch_size * 2]
        seqs, starts, chars = [], [], []
        for i in part:
            it = items[i]
            ctx = tok.encode(PROMPT.format(passage=it["flores_passage"], question=it["question"]),
                             add_special_tokens=False).ids
            for k in range(1, 5):
                answer = " " + it[f"mc_answer{k}"]
                seqs.append(ctx + tok.encode(answer, add_special_tokens=False).ids)
                starts.append(len(ctx))
                chars.append(len(answer))
        lps = sum_logprobs(model, seqs, starts, batch_size, amp_dtype)
        recs = []
        for j, i in enumerate(part):
            lp, ch = lps[4 * j:4 * j + 4], chars[4 * j:4 * j + 4]
            gold = int(items[i]["correct_answer_num"]) - 1
            by_char = [v / c for v, c in zip(lp, ch, strict=True)]
            recs.append({"i": i, "logprobs": lp, "chars": ch, "gold": gold,
                         "correct_sum": max(range(4), key=lp.__getitem__) == gold,
                         "correct_char": max(range(4), key=by_char.__getitem__) == gold})
        _append(path, recs)
        if deadline():
            raise Deadline
    done = _done(path)
    assert sorted(done) == list(range(len(items)))
    n = len(done)
    return {"items": n, "acc_char": sum(r["correct_char"] for r in done.values()) / n,
            "acc_sum": sum(r["correct_sum"] for r in done.values()) / n}


def score_generation(model, tok, prompts: list[str], path: Path, new_tokens, amp_dtype, deadline):
    device = next(model.parameters()).device
    banned = sorted(i for i, t in tok.get_added_tokens_decoder().items() if t.special)
    done = _done(path)
    _rewrite(path, done)
    if len(done) < len(prompts):
        generate(model, tok, prompts[0], min(8, new_tokens), banned, amp_dtype)  # warm-up, not recorded
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    for i in range(len(prompts)):
        if i in done:
            continue
        rec = generate(model, tok, prompts[i], new_tokens, banned, amp_dtype)
        rec["max_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        _append(path, [{"i": i, **rec}])
        if deadline():
            raise Deadline
    recs = [_done(path)[i] for i in range(len(prompts))]
    decode = sum(r["decode_seconds"] for r in recs)
    prefill = sum(r["prefill_seconds"] for r in recs)
    return {"prompts": len(recs), "new_tokens": new_tokens,
            "decode_chars_per_second": sum(r["new_chars"] for r in recs) / decode,
            "decode_tokens_per_second": sum(r["new_tokens"] - 1 for r in recs) / decode,
            "prefill_chars_per_second": sum(r["prompt_chars"] for r in recs) / prefill,
            "generated_chars_per_token": sum(r["new_chars"] for r in recs) / sum(r["new_tokens"] for r in recs),
            "repetition_4_mean": sum(r["repetition_4"] for r in recs) / len(recs),
            "max_memory_gib": max((r["max_memory_gib"] or 0) for r in recs) or None}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="Hugging Face id or folder of the starting model")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, default=None, help="trained weights from vitok.adapt (none: as loaded)")
    ap.add_argument("--vi-docs", type=Path, default=None, help="jsonl with a text field (the test documents)")
    ap.add_argument("--en-docs", type=Path, default=None, help="jsonl with a text field (English documents)")
    ap.add_argument("--belebele", type=Path, default=None, help="jsonl of Belebele vie_Latn items")
    ap.add_argument("--speed-prompts", type=int, default=0, help="generation prompts, taken from --vi-docs")
    ap.add_argument("--speed-prompt-chars", type=int, default=500)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32", help="autocast dtype on CUDA")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eos-token", default="<|endoftext|>")
    ap.add_argument("--deadline-minutes", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.dtype != "fp32" and not args.device.startswith("cuda"):
        raise SystemExit(f"--dtype {args.dtype} needs a CUDA device")
    if args.speed_prompts and not args.vi_docs:
        raise SystemExit("--speed-prompts takes its prompts from --vi-docs")
    if args.speed_prompts and args.new_tokens < 2:
        raise SystemExit("--new-tokens must be at least 2")
    return args


def _texts(path: Path) -> list[str]:
    texts = cpt_data.jsonl_texts(path)
    if not texts or any(t != unicodedata.normalize("NFC", t) for t in texts):
        raise SystemExit(f"{path} must hold NFC texts (and at least one)")
    return texts


def run(args) -> int:
    from transformers import AutoModelForCausalLM

    args.out.mkdir(parents=True, exist_ok=True)
    sig = {"model": str(args.model), "revision": args.revision, "tokenizer_sha256": _sha256(args.tokenizer),
           "snapshot_sha256": _sha256(args.snapshot), "vi_docs_sha256": _sha256(args.vi_docs),
           "en_docs_sha256": _sha256(args.en_docs), "belebele_sha256": _sha256(args.belebele), "prompt": PROMPT,
           "speed_prompts": args.speed_prompts, "speed_prompt_chars": args.speed_prompt_chars,
           "new_tokens": args.new_tokens, "batch_size": args.batch_size, "dtype": args.dtype,
           "device": args.device}
    sig_path = args.out / "signature.json"
    if sig_path.exists() and json.loads(sig_path.read_text()) != sig:
        old = json.loads(sig_path.read_text())
        raise SystemExit(f"{args.out} holds another evaluation (differs in "
                         f"{sorted(k for k in sig.keys() | old.keys() if sig.get(k) != old.get(k))}); use a new --out")
    sig_path.write_text(json.dumps(sig, indent=2), encoding="utf-8")

    tok = Tokenizer.from_file(str(args.tokenizer))
    eos = tok.token_to_id(args.eos_token)
    if eos is None:
        raise SystemExit(f"the tokenizer has no {args.eos_token!r}")
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.float32)
    meta = load_snapshot(model, args.snapshot) if args.snapshot else None
    model.to(args.device).eval()
    if tok.get_vocab_size(with_added_tokens=True) > input_weight(model).shape[0]:
        raise SystemExit("the tokenizer has more ids than the model has embedding rows")
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.dtype) if args.device.startswith("cuda") else None
    started = time.monotonic()
    deadline = lambda: args.deadline_minutes is not None and time.monotonic() - started > 60 * args.deadline_minutes
    summary: dict[str, object] = {"signature": sig, "snapshot": meta}
    try:
        if args.vi_docs:
            vi = _texts(args.vi_docs)
            summary["vi"] = score_docs(model, tok, eos, vi, args.out / "vi_docs.jsonl", args.batch_size, amp, deadline)
        if args.en_docs:
            summary["en"] = score_docs(model, tok, eos, _texts(args.en_docs), args.out / "en_docs.jsonl",
                                       args.batch_size, amp, deadline)
        if args.belebele:
            items = [json.loads(line) for line in args.belebele.read_text(encoding="utf-8").splitlines() if line]
            summary["belebele"] = score_belebele(model, tok, items, args.out / "belebele.jsonl", args.batch_size,
                                                 amp, deadline)
        if args.speed_prompts:
            prompts = [truncate_chars(t, args.speed_prompt_chars) for t in vi[:args.speed_prompts]]
            summary["generation"] = score_generation(model, tok, prompts, args.out / "generation.jsonl",
                                                     args.new_tokens, amp, deadline)
    except Deadline:
        print("stopped by the deadline; rerun the same command to resume", flush=True)
        return EXIT_INCOMPLETE
    summary["device_name"] = torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu"
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "signature"}, indent=2), flush=True)
    return 0


def main(argv=None):
    sys.exit(run(parse_args(argv)))


if __name__ == "__main__":
    main()
