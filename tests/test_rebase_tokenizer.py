"""Rebased tokenizer contracts: the target file plus the retrofit's additions at unchanged ids, the target's added
tokens only, and refusal of files that do not line up."""

import copy
import json

import pytest
from tokenizers import Tokenizer

from vitok import rebase_tokenizer

BYTES = {c: i for i, c in enumerate("abcdĠ")}
PRE_BASE = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": True}
PRE_NEW = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}


def tok(vocab, merges, added, pre=PRE_BASE):
    return {"version": "1.0", "truncation": None, "padding": None,
            "added_tokens": [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                              "normalized": False, "special": True} for i, c in added],
            "normalizer": None, "pre_tokenizer": pre, "post_processor": None,
            "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
            "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                      "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                      "vocab": dict(vocab), "merges": list(merges)}}


VOCAB = BYTES | {"ab": 5, "cd": 6}
MERGES = ["a b", "c d"]
OLD_ADDED = [(7, "<|end|>"), (8, "<think>")]  # the post-trained file: one more added token
TARGET_ADDED = [(7, "<|end|>")]
NEW = {"abĠcd": 10, "Ġcd": 9}  # ids after the old base's last id (8)


def files():
    old = tok(VOCAB, MERGES, OLD_ADDED)
    target = tok(VOCAB, [m.split() for m in MERGES], TARGET_ADDED)  # pair form: same merges
    retro = tok(VOCAB | {c: i for i, c in OLD_ADDED} | NEW, MERGES + ["Ġ cd", "ab Ġcd"], OLD_ADDED, PRE_NEW)
    return target, old, retro


def test_the_rebased_file_is_the_target_plus_the_additions_at_unchanged_ids():
    target, old, retro = files()
    out = rebase_tokenizer.rebase(target, old, retro)
    assert out["added_tokens"] == target["added_tokens"] and out["pre_tokenizer"] == PRE_NEW
    assert out["model"]["vocab"] == VOCAB | {"<|end|>": 7} | NEW  # "<think>" (id 8) is gone, id 8 unused
    assert rebase_tokenizer._merges(out["model"]) == MERGES + ["Ġ cd", "ab Ġcd"]
    assert files() == (target, old, retro)  # inputs untouched
    a = Tokenizer.from_str(json.dumps(out))
    b = Tokenizer.from_str(json.dumps(retro))
    for text in ["ab cd", "abab cd cd", "dcba", "ab  cd"]:
        assert a.encode(text).ids == b.encode(text).ids
    assert a.encode("<think>").ids != b.encode("<think>").ids  # the dropped added token no longer matches


def _break(kind):
    target, old, retro = files()
    if kind == "vocab":
        target["model"]["vocab"]["ab"] = 4
    elif kind == "merges":
        target["model"]["merges"] = target["model"]["merges"][:1]
    elif kind == "decoder":
        retro["decoder"] = None
    elif kind == "moved":
        retro["model"]["vocab"]["cd"] = 11
    elif kind == "not_extended":
        retro["model"]["merges"] = ["c d", "a b", "Ġ cd", "ab Ġcd"]
    elif kind == "added":
        target["added_tokens"][0]["content"] = "<|other|>"
    elif kind == "inside":
        retro["model"]["vocab"]["Ġcd"] = 8
        del retro["model"]["vocab"]["<think>"]
    return target, old, retro


@pytest.mark.parametrize("kind", ["vocab", "merges", "decoder", "moved", "not_extended", "added", "inside"])
def test_files_that_do_not_line_up_are_refused(kind):
    with pytest.raises(ValueError):
        rebase_tokenizer.rebase(*_break(kind))


def test_the_command_writes_the_rebased_file(tmp_path):
    paths = []
    for name, data in zip(["target", "old", "retro"], files(), strict=True):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(copy.deepcopy(data)))
        paths.append(p)
    out = tmp_path / "sub" / "out.json"
    rebase_tokenizer.main(["--target", str(paths[0]), "--old-base", str(paths[1]), "--retrofit", str(paths[2]),
                           "--out", str(out)])
    assert json.loads(out.read_text()) == rebase_tokenizer.rebase(*files())
