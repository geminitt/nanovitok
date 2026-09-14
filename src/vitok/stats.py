"""Statistics used in the analysis: bits per character, paired bootstrap, exact McNemar."""

import math

import numpy as np


def bpc(nats, chars) -> float:
    return float(np.sum(nats) / (math.log(2) * np.sum(chars)))


def paired_bootstrap_bpc(nats_a, nats_b, chars, n: int = 10_000, seed: int = 0) -> dict:
    """CI of bpc(A) - bpc(B) resampling documents; both models scored on the same documents."""
    nats_a, nats_b, chars = map(np.asarray, (nats_a, nats_b, chars))
    rng = np.random.default_rng(seed)
    diffs = []
    for start in range(0, n, 500):  # chunked to bound memory
        idx = rng.integers(0, len(chars), size=(min(500, n - start), len(chars)))
        ca = chars[idx].sum(axis=1) * math.log(2)
        diffs.append((nats_a[idx].sum(axis=1) - nats_b[idx].sum(axis=1)) / ca)
    diffs = np.concatenate(diffs)
    lo, mid, hi = np.percentile(diffs, [2.5, 50, 97.5])
    return {"diff": bpc(nats_a, chars) - bpc(nats_b, chars), "ci95": [float(lo), float(hi)],
            "median": float(mid), "significant": bool(lo > 0 or hi < 0)}


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Two-sided exact McNemar test on paired binary outcomes."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))  # A right, B wrong
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    k = min(n01, n10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n01, "b_only": n10, "p_value": p}
