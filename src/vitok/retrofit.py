"""K1: retrofit Vietnamese tokens onto an existing byte-level BPE tokenizer (Qwen3's), keeping every old token.

    python -m vitok.retrofit --base qwen3/tokenizer.json --corpus vi.txt --kind multi --n-new 4000 \
        --docs test.jsonl --out runs/k1/c2

The base tokenizer is kept as it is: its merges, vocabulary, special tokens, normalizer and pre-tokenizer. New merges
continue its BPE on Vietnamese text (as in continued BPE training) and are appended, with new ids after the largest
existing id, so the base model's embedding rows keep their meaning.

- kind "single": merges inside a word only (the base pre-tokenizer is untouched), condition C1.
- kind "multi": one extra pre-tokenizer alternative, placed first, groups runs of letter-only words joined by single
  spaces, so a merge may join whole words; digits, punctuation and other whitespace keep the base's splits. C2.

The report says what the new tokens buy (chars per token vs the base), whether every new token is reachable (its own
text encodes to it), and whether text without Vietnamese words still gets the base's ids.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from tokenizers import Regex, Tokenizer, pre_tokenizers

from vitok import superbpe

KINDS = ("single", "multi")
# letter-only words joined by single spaces, with the base's optional leading non-letter character
WORD_RUN = r"[^\r\n\p{L}\p{N}]?\p{L}+(?: \p{L}+)+"


def base_parts(base: dict) -> tuple[dict, list[tuple[str, str]], str]:
    """Vocabulary, merges and pre-tokenizer regex of a byte-level BPE tokenizer.json (Split + ByteLevel)."""
    model, pre = base.get("model", {}), base.get("pre_tokenizer") or {}
    if model.get("type") != "BPE":
        raise ValueError(f"expected a BPE model, got {model.get('type')!r}")
    steps = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else []
    splits = [p for p in steps if p.get("type") == "Split"]
    if len(splits) != 1 or not any(p.get("type") == "ByteLevel" for p in steps):
        raise ValueError("expected a pre-tokenizer of one regex Split followed by ByteLevel")
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in model["merges"]]
    return dict(model["vocab"]), merges, splits[0]["pattern"]["Regex"]


def regex_for(kind: str, base_regex: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}: expected one of {KINDS}")
    return base_regex if kind == "single" else WORD_RUN + "|" + base_regex


def _pre_tokenizer(regex: str):
    return pre_tokenizers.Sequence([pre_tokenizers.Split(pattern=Regex(regex), behavior="isolated", invert=False),
                                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])


def _texts(path: Path):
    """Lines of a text file, or the documents of a jsonl file with a "text" field (each ended by a blank line)."""
    with open(path, encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                if line.strip():
                    yield json.loads(line)["text"] + "\n\n"
        else:
            yield from f


def learn_merges(base: dict, corpus: list[Path], kind: str, n_new: int) -> list[tuple[str, str]]:
    """The next n_new merges of the base BPE on the corpus (normalized like the base), in learning order."""
    vocab, merges, base_regex = base_parts(base)
    normalizer = Tokenizer.from_str(json.dumps(base)).normalizer
    with tempfile.TemporaryDirectory() as tmp:
        normalized = Path(tmp) / "corpus.txt"
        with open(normalized, "w", encoding="utf-8") as out:
            for path in corpus:
                for text in _texts(path):
                    out.write(normalizer.normalize_str(text) if normalizer else text)
        ids = superbpe.encode_corpus([str(normalized)], vocab, merges, _pre_tokenizer(regex_for(kind, base_regex)))
    _, new = superbpe.train_stage2(ids, vocab, len(vocab) + n_new, log_every=0)
    return new


def retrofit(base: dict, new_merges: list[tuple[str, str]], kind: str) -> dict:
    """A copy of the base tokenizer.json with the new merges appended and the kind's pre-tokenizer."""
    vocab, merges, base_regex = base_parts(base)
    out = copy.deepcopy(base)
    model = out["model"]
    # `tokenizers` re-numbers an added token that is not in the model vocabulary from the vocabulary's size, so
    # appending tokens would shift every special token's id; pinning them in the vocabulary keeps their ids
    for t in base.get("added_tokens", []):
        owner = next((k for k, i in model["vocab"].items() if i == t["id"]), None)
        if owner not in (None, t["content"]):
            raise ValueError(f"added token {t['content']!r} has id {t['id']}, which the vocabulary gives to {owner!r}")
        model["vocab"][t["content"]] = t["id"]
    next_id = 1 + max(model["vocab"].values())
    known = set(vocab)
    for a, b in new_merges:
        if a not in known or b not in known:
            raise ValueError(f"merge {a!r} + {b!r} uses a token that is neither in the base nor made by an earlier merge")
        if a + b not in known:
            model["vocab"][a + b] = next_id
            next_id += 1
            known.add(a + b)
    as_str = model["merges"] and isinstance(model["merges"][0], str)
    model["merges"] = [*model["merges"], *(f"{a} {b}" if as_str else [a, b] for a, b in new_merges)]
    if kind == "multi":
        for step in out["pre_tokenizer"]["pretokenizers"]:
            if step.get("type") == "Split":
                step["pattern"] = {"Regex": regex_for(kind, base_regex)}
    else:
        regex_for(kind, base_regex)  # validates the kind
    added = {t["content"] for t in base.get("added_tokens", [])}
    new_ids = [i for t, i in model["vocab"].items() if t not in vocab and t not in added]
    assert len(set(model["vocab"].values())) == len(model["vocab"])
    assert sorted(new_ids) == list(range(next_id - len(new_ids), next_id)), "new ids must follow the old ones"
    return out


def _byte_decoder() -> dict[str, int]:
    """GPT-2's byte-level alphabet, character -> byte (the inverse of the mapping ByteLevel encodes with)."""
    keep = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars, n = list(keep), 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(keep, chars)}


# built at import: mutation testing cannot reach it here, so tests check the table itself
BYTE_DECODER = _byte_decoder()


def token_bytes(token: str) -> bytes:
    return bytes(BYTE_DECODER[c] for c in token)


def measure(new: Tokenizer, base: Tokenizer, docs: list[str], others: dict[str, list[str]]) -> dict:
    """What the retrofit buys and whether it keeps its promises (see the module docstring).

    A new token that is whole characters is reachable when its own text encodes to exactly it. A token that ends or
    starts inside a character (BPE on bytes learns such pieces) cannot be typed on its own; it is only counted, and
    whether it occurs at all shows in the usage count on the documents.
    """
    base_vocab = base.get_vocab(with_added_tokens=True)
    new_tokens = {t: i for t, i in new.get_vocab(with_added_tokens=True).items() if t not in base_vocab}
    enc = lambda tok, texts: [e.ids for e in tok.encode_batch(texts, add_special_tokens=False)]
    chars = sum(map(len, docs))
    n_base, n_new = (sum(map(len, enc(t, docs))) for t in (base, new))
    used = {i for ids in enc(new, docs) for i in ids}
    whole, partial = {}, {}
    for t, i in new_tokens.items():
        try:
            whole[i] = token_bytes(t).decode("utf-8")
        except UnicodeDecodeError:
            partial[i] = t
    reachable = sum(new.encode(text, add_special_tokens=False).ids == [i] for i, text in whole.items())
    report = {
        "docs": len(docs), "chars": chars, "new_tokens": len(new_tokens),
        "chars_per_token_base": chars / n_base, "chars_per_token_new": chars / n_new,
        "token_change": n_new / n_base - 1,
        "round_trip": sum(new.decode(ids) == d for ids, d in zip(enc(new, docs), docs)),
        "new_tokens_whole_chars": len(whole), "new_tokens_reachable": reachable,
        "new_tokens_partial_chars": len(partial),
        "new_tokens_used_on_docs": len(used & set(new_tokens.values())),
        "identical_ids": {name: sum(a == b for a, b in zip(enc(base, texts), enc(new, texts))) for name, texts in others.items()},
        "identical_ids_of": {name: len(texts) for name, texts in others.items()},
    }
    assert 0 <= report["new_tokens_used_on_docs"] <= report["new_tokens"] and report["round_trip"] <= len(docs)
    return report


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(args, n_merges: int) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                                cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"base": str(args.base), "base_sha256": _sha256(args.base),
            "corpus": [{"path": str(p), "sha256": _sha256(p), "bytes": p.stat().st_size} for p in args.corpus],
            "kind": args.kind, "n_new_requested": args.n_new, "merges_learned": n_merges,
            "git_commit": commit, "packages": {p: version(p) for p in ("tokenizers", "numpy")},
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="the base tokenizer.json (e.g. Qwen3's)")
    ap.add_argument("--corpus", type=Path, nargs="+", required=True,
                    help="Vietnamese text (.txt lines, or .jsonl with a text field) to learn merges on")
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--n-new", type=int, required=True)
    ap.add_argument("--docs", type=Path, default=None, help="jsonl with a text field, for the report (held out)")
    ap.add_argument("--others", type=Path, nargs="*", default=[], help="non-Vietnamese files whose ids must not change")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.n_new < 0:
        raise SystemExit("--n-new must be >= 0")
    base = json.loads(args.base.read_text(encoding="utf-8"))
    new_merges = learn_merges(base, args.corpus, args.kind, args.n_new) if args.n_new else []
    out = retrofit(base, new_merges, args.kind)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tokenizer.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(manifest(args, len(new_merges)), indent=2), encoding="utf-8")
    if args.docs:
        docs = [json.loads(line)["text"] for line in args.docs.read_text(encoding="utf-8").splitlines()]
        others = {p.name: [p.read_text(encoding="utf-8")] for p in args.others}
        report = measure(Tokenizer.from_str(json.dumps(out)), Tokenizer.from_str(json.dumps(base)), docs, others)
        (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
