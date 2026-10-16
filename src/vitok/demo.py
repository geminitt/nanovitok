"""Side-by-side generation demo: the logic behind `vitok.demo_app` (Streamlit), free of any UI code.

A trained condition is the extended Qwen3 model plus its final snapshot (`extended/` + `final.pt`) and its tokenizer.
`stream_ids` is `score.generate`'s greedy decoding, yielding each token id as it is chosen. `race` runs several models
at the same time, one thread each; they share one GPU and one CPU, so the speed it sees is not a fair measurement.
`benchmark` runs them one after the other, and its numbers are the comparable ones.
"""

import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from vitok import score
from vitok.adapt import input_weight, load_snapshot


def load_condition(model_dir: Path, tokenizer: Path, snapshot: Path | None, device: str, dtype: torch.dtype):
    """(model in eval mode on `device` in `dtype`, tokenizer); the snapshot's weights are applied in fp32 first."""
    tok = Tokenizer.from_file(str(tokenizer))
    model: Any = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    if snapshot is not None:
        load_snapshot(model, snapshot)
    if tok.get_vocab_size(with_added_tokens=True) > input_weight(model).shape[0]:
        raise SystemExit("the tokenizer has more ids than the model has embedding rows")
    return model.to(device=device, dtype=dtype).eval(), tok


def banned_ids(tok: Tokenizer) -> list[int]:
    """Special tokens are never generated, so every model takes exactly the requested number of decode steps."""
    return sorted(i for i, t in tok.get_added_tokens_decoder().items() if t.special)


@torch.no_grad()
def stream_ids(model, tok: Tokenizer, prompt: str, new_tokens: int, banned: list[int]) -> Iterator[int]:
    """Greedy decoding with a KV cache: the same choices as `score.generate`, one token id at a time."""
    if new_tokens < 1:
        raise ValueError(f"new_tokens must be at least 1, got {new_tokens}")
    if not prompt:
        raise ValueError("the prompt is empty")
    device = next(model.parameters()).device
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False).ids], device=device)
    ban = torch.tensor(banned, device=device, dtype=torch.long)

    def choose(res):
        logits = res.logits[:, -1].float()
        logits[:, ban] = float("-inf")
        return logits.argmax(-1, keepdim=True), res.past_key_values

    nxt, past = choose(model(input_ids=ids, use_cache=True))
    for _ in range(new_tokens - 1):
        yield int(nxt)
        nxt, past = choose(model(input_ids=nxt, past_key_values=past, use_cache=True))
    yield int(nxt)


def summarize(new_tokens: int, new_chars: int, decode_seconds: float) -> dict:
    """The numbers shown for one generation; decode time excludes the prompt and the first token."""
    if decode_seconds <= 0:
        raise ValueError(f"decode_seconds must be positive, got {decode_seconds}")
    return {"new_tokens": new_tokens, "new_chars": new_chars, "chars_per_token": new_chars / new_tokens,
            "chars_per_second": new_chars / decode_seconds, "tokens_per_second": new_tokens / decode_seconds}


def race(models: dict[str, tuple], prompt: str, new_tokens: int,
         on_text: Callable[[str, str], None]) -> dict[str, dict]:
    """Generate with every model at the same time, one thread each. After each token `on_text(name, text so far)` is
    called from that model's thread. Returns each model's text and counts; a failure in any thread is raised here,
    after the others have finished."""
    results: dict[str, dict] = {}
    errors: dict[str, BaseException] = {}

    def work(name: str, model, tok: Tokenizer):
        try:
            ids: list[int] = []
            first = last = 0.0
            for i in stream_ids(model, tok, prompt, new_tokens, banned_ids(tok)):
                ids.append(i)
                last = time.perf_counter()
                first = first or last
                on_text(name, tok.decode(ids, skip_special_tokens=False))
            text = tok.decode(ids, skip_special_tokens=False)
            results[name] = {"text": text, "new_tokens": len(ids), "new_chars": len(text),
                             "decode_seconds": last - first}
        except BaseException as e:  # noqa: BLE001  (re-raised below, from the caller's thread)
            errors[name] = e

    threads = [threading.Thread(target=work, args=(n, m, t), name=f"race-{n}") for n, (m, t) in models.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise next(iter(errors.values()))
    return results


def race_events(models: dict[str, tuple], prompt: str, new_tokens: int) -> Iterator[tuple[str, Any]]:
    """`race` as a stream for a UI loop: ("update", (name, text so far)) for every token, then ("done", results).
    The race runs in a background thread; its failure is raised from here once the queued updates are delivered."""
    events: queue.Queue = queue.Queue()
    outcome: dict[str, object] = {}

    def run():
        try:
            outcome["results"] = race(models, prompt, new_tokens, on_text=lambda n, t: events.put(("update", (n, t))))
        except BaseException as e:  # noqa: BLE001  (re-raised in the consuming thread)
            outcome["error"] = e
        events.put(("end", None))

    threading.Thread(target=run, name="race-events", daemon=True).start()
    while (event := events.get())[0] != "end":
        yield event
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    yield "done", outcome["results"]


def benchmark(models: dict[str, tuple], prompt: str, new_tokens: int) -> dict[str, dict]:
    """The models one after the other, each warmed up first: comparable speeds, plus `summarize`'s numbers."""
    out = {}
    for name, (model, tok) in models.items():
        banned = banned_ids(tok)
        score.generate(model, tok, prompt, min(8, new_tokens), banned)  # warm-up, not recorded
        rec = score.generate(model, tok, prompt, new_tokens, banned)
        out[name] = {"text": rec["text"]} | summarize(rec["new_tokens"], rec["new_chars"], rec["decode_seconds"])
    return out
