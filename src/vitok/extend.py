"""Embedding extension: give a pretrained causal LM one embedding row per token the retrofit added.

    python -m vitok.extend --model Qwen/Qwen3-0.6B --revision <sha> --tokenizer runs/.../multisyllable/tokenizer.json \
        --out runs/.../multisyllable-model

Every old row is copied unchanged. A new token's row is the mean of the rows of the pieces the base tokenizer splits it
into (the subword-mean initialization of continued-BPE work). With tied input and output embeddings, as in Qwen3-0.6B,
the same row is also the token's output vector. The embedding matrix grows only when the new ids do not fit in the
rows the checkpoint already has (Qwen3 pads its matrix to 151,936 rows).

The checks it runs on itself: the old rows are bit-identical, every new row equals its mean, and for text that the
new tokenizer encodes exactly like the base, the logits over the old vocabulary are unchanged.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

# A grown output matrix changes the shape of the logits matmul, so BLAS sums in another order: on Qwen3-0.6B (fp32,
# CPU) the old-vocabulary logits then differ by 1.2e-5 at most (2026-10-05). Larger differences mean a real change.
LOGITS_TOLERANCE = 1e-4


def new_token_pieces(base: Tokenizer, new: Tokenizer) -> dict[int, list[int]]:
    """New token id -> the base ids its byte-level string splits into under the base's merges.

    BPE on the token's own string, without pre-tokenization: base merges never cross a space, so a multi-word token
    splits into the pieces of each of its words.
    """
    base_vocab = base.get_vocab(with_added_tokens=True)
    pieces = {}
    for token, i in new.get_vocab(with_added_tokens=True).items():
        if token in base_vocab:
            if base_vocab[token] != i:
                raise ValueError(f"token {token!r} moved from id {base_vocab[token]} to {i}")
            continue
        ids = [t.id for t in base.model.tokenize(token)]
        if not ids or "".join(base.id_to_token(j) or "" for j in ids) != token:
            raise ValueError(f"new token {token!r} does not split into base tokens")
        pieces[i] = ids
    return pieces


@torch.no_grad()
def extend(model, pieces: dict[int, list[int]], pad_to: int = 64):
    """Grow the input embedding (and a tied or untied output head) for the new ids, in place; returns the model."""
    emb = model.get_input_embeddings()
    old_rows = emb.weight.shape[0]
    need = max(pieces, default=-1) + 1
    if need > old_rows:
        model.resize_token_embeddings(need, pad_to_multiple_of=pad_to, mean_resizing=False)
        emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    tied = head is not None and head.weight.data_ptr() == emb.weight.data_ptr()
    for i, ids in pieces.items():
        emb.weight[i] = emb.weight[ids].float().mean(dim=0).to(emb.weight.dtype)
        if head is not None and not tied:
            head.weight[i] = head.weight[ids].float().mean(dim=0).to(head.weight.dtype)
    return model


@torch.no_grad()
def check(original, extended, pieces, base: Tokenizer, new: Tokenizer, texts: list[str]) -> dict:
    """The self-checks of the module docstring, as numbers (all must be zero or true)."""
    old_rows = original.get_input_embeddings().weight.shape[0]
    old_vocab = base.get_vocab_size(with_added_tokens=True)
    e_old, e_new = original.get_input_embeddings().weight, extended.get_input_embeddings().weight
    keep = [i for i in range(old_rows) if i not in pieces]
    new_mean = [float((e_new[i] - e_new[ids].float().mean(0).to(e_new.dtype)).abs().max()) for i, ids in pieces.items()]
    report = {"old_rows_max_abs_diff": float((e_new[keep] - e_old[keep]).abs().max()),
              "new_rows_mean_max_abs_diff": max(new_mean, default=0.0),
              "tied": extended.get_output_embeddings().weight.data_ptr() == e_new.data_ptr(),
              "texts_checked": 0, "logits_max_abs_diff": 0.0, "same_argmax": True}
    for text in texts:
        ids = base.encode(text, add_special_tokens=False).ids
        if not ids or new.encode(text, add_special_tokens=False).ids != ids:
            continue  # only text the new tokenizer leaves alone
        x = torch.tensor([ids])
        a = original(x).logits[..., :old_vocab].float()
        b = extended(x).logits[..., :old_vocab].float()
        report["texts_checked"] += 1
        report["logits_max_abs_diff"] = max(report["logits_max_abs_diff"], float((a - b).abs().max()))
        report["same_argmax"] &= bool((a.argmax(-1) == b.argmax(-1)).all())
    return report


def main():
    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face id or local folder of the base model")
    ap.add_argument("--revision", default=None, help="pin the base model to this commit")
    ap.add_argument("--base-tokenizer", type=Path, required=True, help="the base tokenizer.json")
    ap.add_argument("--tokenizer", type=Path, required=True, help="the retrofitted tokenizer.json")
    ap.add_argument("--check-texts", type=Path, required=True, help="jsonl with a text field (English and code too)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base, new = Tokenizer.from_file(str(args.base_tokenizer)), Tokenizer.from_file(str(args.tokenizer))
    pieces = new_token_pieces(base, new)
    load = dict(revision=args.revision, torch_dtype=torch.float32)
    original = AutoModelForCausalLM.from_pretrained(args.model, **load).eval()
    extended = extend(AutoModelForCausalLM.from_pretrained(args.model, **load).eval(), pieces)
    texts = [json.loads(line)["text"] for line in args.check_texts.read_text(encoding="utf-8").splitlines()]
    report = check(original, extended, pieces, base, new, texts)
    if report["old_rows_max_abs_diff"] != 0 or report["new_rows_mean_max_abs_diff"] > 1e-6 \
            or report["logits_max_abs_diff"] > LOGITS_TOLERANCE or not report["same_argmax"] \
            or report["texts_checked"] == 0:
        raise SystemExit(f"embedding extension failed its checks: {report}")
    args.out.mkdir(parents=True, exist_ok=True)
    extended.save_pretrained(args.out)
    shutil.copy(args.tokenizer, args.out / "tokenizer.json")
    manifest = {"model": args.model, "revision": args.revision, "tokenizer": str(args.tokenizer),
                "new_tokens": len(pieces), "embedding_rows": extended.get_input_embeddings().weight.shape[0],
                "checks": report, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (args.out / "extend_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
