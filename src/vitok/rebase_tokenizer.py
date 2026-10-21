"""Move a retrofitted tokenizer onto another tokenizer file with the same vocabulary and merges (PLAN.md, Stage 1).

    python -m vitok.rebase_tokenizer --target qwen3-base/tokenizer.json --old-base base.json \
        --retrofit multisyllable-1000.json --out multisyllable-1000.json

The token-type tokenizers were built on the post-trained Qwen3-0.6B file; Qwen3-0.6B-Base has the same byte-level
vocabulary and merges but four fewer added tokens (151665-151668). The rebased file is the target file with the
retrofit's new vocabulary entries, appended merges and pre-tokenizer; its added tokens are the target's. New-token
ids stay as they were, so ids the target does not use (here 151665-151668) stay unused.
"""

import argparse
import copy
import json
from pathlib import Path


def _merges(model: dict) -> list[str]:
    """Merges as "a b" strings (tokenizers writes either strings or pairs)."""
    return [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]


def rebase(target: dict, old_base: dict, retrofit: dict) -> dict:
    """The target tokenizer with the retrofit's additions over `old_base`; refuses files that do not line up."""
    tv, ov, rv = target["model"]["vocab"], old_base["model"]["vocab"], retrofit["model"]["vocab"]
    if tv != ov or _merges(target["model"]) != _merges(old_base["model"]):
        raise ValueError("the target and the old base differ in vocabulary or merges")
    for key in ("normalizer", "post_processor", "decoder"):
        if not target[key] == old_base[key] == retrofit[key]:
            raise ValueError(f"{key} differs between the files")
    if any(rv.get(t) != i for t, i in ov.items()):
        raise ValueError("the retrofit moved or dropped a token of the old base")
    old_merges, new_merges = _merges(old_base["model"]), _merges(retrofit["model"])
    if new_merges[:len(old_merges)] != old_merges:
        raise ValueError("the retrofit does not extend the old base's merges")
    old_added = {t["id"]: t["content"] for t in old_base["added_tokens"]}
    target_added = {t["id"]: t["content"] for t in target["added_tokens"]}
    if any(old_added.get(i) != c for i, c in target_added.items()):
        raise ValueError("an added token of the target is not in the old base at the same id")
    added_only_old = set(old_added.values()) - set(target_added.values())
    new_vocab = {t: i for t, i in rv.items() if t not in ov and t not in old_added.values()}
    top_old = max([*ov.values(), *old_added])
    if any(i <= top_old for i in new_vocab.values()):
        raise ValueError("a new token id falls inside the old base's ids")
    vocab = dict(tv)
    vocab.update({c: i for i, c in target_added.items() if c in rv})  # added tokens the retrofit kept in its vocab
    vocab.update(new_vocab)
    if len(set(vocab.values())) != len(vocab) or any(c in vocab for c in added_only_old):
        raise AssertionError("rebased vocabulary has a duplicate id or an added token the target lacks")
    out = copy.deepcopy(target)
    out["pre_tokenizer"] = copy.deepcopy(retrofit["pre_tokenizer"])
    out["model"]["vocab"] = vocab
    added: list = new_merges[len(old_merges):]
    if target["model"]["merges"] and not isinstance(target["model"]["merges"][0], str):
        added = [m.split(" ") for m in added]  # keep the target's form: one file cannot mix the two
        assert all(len(m) == 2 for m in added), "a byte-level merge holds exactly one space"
    out["model"]["merges"] = copy.deepcopy(target["model"]["merges"]) + added
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", type=Path, required=True, help="tokenizer.json of the new source model")
    ap.add_argument("--old-base", type=Path, required=True, help="tokenizer.json the retrofit was built on")
    ap.add_argument("--retrofit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    out = rebase(load(args.target), load(args.old_base), load(args.retrofit))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
