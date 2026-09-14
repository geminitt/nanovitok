import numpy as np
import pytest

from conftest import SENTENCES
from vitok.conditions import BASE_SEQ, SEQS_PER_STEP, train_args
from vitok.minimal_pairs import build_pairs
from vitok.stats import mcnemar_exact, paired_bootstrap_bpc
from vitok.text import strip_diacritics

CPT = {"bpe-nfc": 4.0, "bpe-nfd": 3.6, "super-nfc": 5.0, "super-nfd": 4.5}


def test_equal_text_design():
    configs = {c: train_args(CPT, c, depth=8) for c in CPT}
    # same steps, same LR scaling batch, same sequences per step
    assert len({cfg["num-iterations"] for cfg in configs.values()}) == 1
    assert len({cfg["scaling-batch-size"] for cfg in configs.values()}) == 1
    for c, cfg in configs.items():
        assert cfg["total-batch-size"] == SEQS_PER_STEP * cfg["max-seq-len"]
        assert cfg["total-batch-size"] % (cfg["device-batch-size"] * cfg["max-seq-len"]) == 0
        chars_per_context = cfg["max-seq-len"] * CPT[c]
        assert chars_per_context == pytest.approx(BASE_SEQ * CPT["bpe-nfc"], rel=0.01)
    assert configs["super-nfc"]["max-seq-len"] < configs["bpe-nfc"]["max-seq-len"]


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    chars = rng.integers(200, 2000, size=300)
    base = chars * rng.uniform(0.9, 1.1, size=300)
    res = paired_bootstrap_bpc(base * 0.97, base, chars, n=2000)
    assert res["diff"] < 0 and res["significant"]
    same = paired_bootstrap_bpc(base, base, chars, n=500)
    assert not same["significant"]


def test_mcnemar():
    a = [True] * 30 + [False] * 5 + [True] * 50
    b = [False] * 30 + [True] * 5 + [True] * 50
    res = mcnemar_exact(a, b)
    assert res["a_only"] == 30 and res["b_only"] == 5
    assert res["p_value"] < 0.001
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_minimal_pairs():
    docs = [" ".join(SENTENCES)]
    counts = {}
    import re
    for w in re.findall(r"[^\W\d_]+", docs[0] + " mà má mả ma hòa hóa"):
        counts[w.lower()] = 100
    pairs = build_pairs(docs, counts, n=10, min_chars=10)
    assert pairs
    for p in pairs:
        assert p["good"] != p["bad"]
        assert strip_diacritics(p["good"]) == strip_diacritics(p["bad"])  # only the tone changed
        assert p["to"].lower() in counts
