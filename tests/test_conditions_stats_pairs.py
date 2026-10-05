import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import SENTENCES
from vitok.conditions import BASE_SEQ, SEQS_PER_STEP, train_args
from vitok.minimal_pairs import build_pairs
from vitok.stats import bpc, mcnemar_exact, paired_bootstrap_bpc, t_cdf, t_quantile
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
    # the relative difference is A's bpc over B's, minus one, with an interval around it
    assert res["rel_diff"] == pytest.approx(-0.03)
    assert res["rel_ci95"][0] <= res["rel_diff"] <= res["rel_ci95"][1] < 0


def test_paired_bootstrap_interval_is_the_document_resampling_interval():
    """The 95% interval brackets the estimate, has the width the sampling noise implies, and is reproducible."""
    rng = np.random.default_rng(1)
    n_docs = 400
    chars = rng.integers(500, 1500, size=n_docs).astype(float)
    nats_b = chars * 0.7 * rng.uniform(0.95, 1.05, size=n_docs)
    nats_a = nats_b + chars * rng.normal(0.002, 0.02, size=n_docs)  # noisy per-document differences
    res = paired_bootstrap_bpc(nats_a, nats_b, chars, n=4000, seed=3)
    assert res["ci95"][0] < res["diff"] < res["ci95"][1]
    assert res["ci95"][0] < res["median"] < res["ci95"][1]
    # delta method for a ratio of sums: SE of sum(d) / (ln2 sum(c)) with d = nats_a - nats_b
    d, c = (nats_a - nats_b) / math.log(2), chars
    ratio = d.sum() / c.sum()
    se = math.sqrt(n_docs * np.var(d - ratio * c, ddof=1)) / c.sum()
    assert res["ci95"][1] - res["ci95"][0] == pytest.approx(2 * 1.96 * se, rel=0.15)
    assert paired_bootstrap_bpc(nats_a, nats_b, chars, n=4000, seed=3) == res      # same seed, same answer
    assert paired_bootstrap_bpc(nats_a, nats_b, chars, n=4000, seed=4)["ci95"] != res["ci95"]
    one = paired_bootstrap_bpc(nats_a[:1], nats_b[:1], chars[:1], n=50)  # one document: the interval is a point
    assert one["ci95"][0] == pytest.approx(one["diff"]) == pytest.approx(one["ci95"][1])


@pytest.mark.parametrize("args", [([1.0, 2.0], [1.0, 2.0], [3.0]), ([1.0], [1.0, 2.0], [3.0, 4.0]), ([], [], [])])
def test_paired_bootstrap_refuses_unaligned_input(args):
    with pytest.raises(AssertionError):
        paired_bootstrap_bpc(*map(np.asarray, args), n=10)


@pytest.mark.parametrize("nats,chars,ok", [([2.0, 4.0], [3, 5], True), ([2.0, 4.0], [3], False), ([2.0], [0], False),
                                           ([], [], False)])
def test_bpc_is_total_bits_over_total_chars(nats, chars, ok):
    if ok:
        assert bpc(nats, chars) == pytest.approx(sum(nats) / math.log(2) / sum(chars))
    else:
        with pytest.raises(AssertionError):
            bpc(nats, chars)


def _exact_two_sided(n01, n10):
    """Reference: total probability, under Binomial(n, 1/2), of every outcome no more likely than the observed one."""
    from fractions import Fraction
    n = n01 + n10
    prob = [Fraction(math.comb(n, i), 2 ** n) for i in range(n + 1)]
    return float(sum(q for q in prob if q <= prob[n01])) if n else 1.0


@pytest.mark.parametrize("n01,n10", [(30, 5), (5, 30), (3, 11), (11, 3), (9, 4), (7, 7), (1, 0), (0, 0), (40, 38)])
def test_mcnemar_is_the_exact_binomial_test(n01, n10):
    both = 50
    a = [True] * n01 + [False] * n10 + [True] * both
    b = [False] * n01 + [True] * n10 + [True] * both
    for kind in (bool, int):  # outcomes may come as booleans or as 0/1
        res = mcnemar_exact([kind(v) for v in a], [kind(v) for v in b])
        assert (res["a_only"], res["b_only"]) == (n01, n10)
        assert res["p_value"] == pytest.approx(_exact_two_sided(n01, n10), rel=1e-12)


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


@pytest.mark.parametrize("df,expected", [(1, 12.706), (2, 4.303), (10, 2.228), (1000, 1.962)])
def test_t_quantile_matches_the_table(df, expected):
    assert t_quantile(0.975, df) == pytest.approx(expected, abs=1e-3)
    assert t_cdf(t_quantile(0.975, df), df) == pytest.approx(0.975, abs=1e-9)


@settings(max_examples=300, deadline=None)
@given(st.one_of(st.floats(min_value=-60, max_value=60, allow_nan=False),
                 st.floats(min_value=-1e-5, max_value=1e-5, allow_nan=False)))
def test_t_cdf_is_the_student_t_cdf(x):
    # closed forms exist for 1 and 2 degrees of freedom (Cauchy, and x / (2 sqrt(2 + x^2))); tiny |x| included,
    # where df / (df + x^2) rounds next to 1 (the 2026-10-05 precision bug: 3.2e-9 error at x = 1e-8)
    assert t_cdf(x, 1) == pytest.approx(0.5 + math.atan(x) / math.pi, abs=1e-12)
    assert t_cdf(x, 2) == pytest.approx(0.5 + x / (2 * math.sqrt(2 + x * x)), abs=1e-12)
    for df in (3, 10, 1000):
        assert t_cdf(-x, df) == pytest.approx(1 - t_cdf(x, df), abs=1e-10)
        assert t_cdf(x, df) <= t_cdf(x + 0.5, df) + 1e-12
