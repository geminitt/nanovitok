"""Statistics used in the analysis: bits per character, paired bootstrap, exact McNemar."""

import math

import numpy as np


def bpc(nats, chars) -> float:
    return float(np.sum(nats) / (math.log(2) * np.sum(chars)))


def paired_bootstrap_bpc(nats_a, nats_b, chars, n: int = 10_000, seed: int = 0) -> dict:
    """CI of bpc(A) - bpc(B) resampling documents; both models scored on the same documents.

    Also returns the difference relative to bpc(B), with its own percentile interval, for the
    pre-registered "not worse by more than 1%" margins (a non-inferiority test: A is not worse
    than B by more than m when the upper end of the relative interval is below m).
    """
    nats_a, nats_b, chars = map(np.asarray, (nats_a, nats_b, chars))
    rng = np.random.default_rng(seed)
    diffs, rels = [], []
    for start in range(0, n, 500):  # chunked to bound memory
        idx = rng.integers(0, len(chars), size=(min(500, n - start), len(chars)))
        sa, sb = nats_a[idx].sum(axis=1), nats_b[idx].sum(axis=1)
        diffs.append((sa - sb) / (chars[idx].sum(axis=1) * math.log(2)))
        rels.append(sa / sb - 1)  # same characters on both sides, so the ratio of nats is the ratio of bpc
    diffs, rels = np.concatenate(diffs), np.concatenate(rels)
    lo, mid, hi = np.percentile(diffs, [2.5, 50, 97.5])
    rlo, rhi = np.percentile(rels, [2.5, 97.5])
    return {"diff": bpc(nats_a, chars) - bpc(nats_b, chars), "ci95": [float(lo), float(hi)],
            "median": float(mid), "significant": bool(lo > 0 or hi < 0),
            "rel_diff": float(np.sum(nats_a) / np.sum(nats_b) - 1), "rel_ci95": [float(rlo), float(rhi)]}


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Two-sided exact McNemar test on paired binary outcomes."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))  # A right, B wrong
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    k = min(n01, n10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n01, "b_only": n10, "p_value": p}


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), by the continued fraction of Numerical Recipes (6.4)."""
    if x <= 0 or x >= 1:
        return float(x >= 1)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x > (a + 1) / (a + b + 2):  # the fraction converges fast only on this side; use the symmetry otherwise
        return 1 - _betainc(b, a, 1 - x)
    tiny, c, d = 1e-300, 1.0, 1 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    f = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 + num * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + num / c
            c = c if abs(c) > tiny else tiny
            f *= c * d
        if abs(c * d - 1) < 1e-15:
            break
    return front * f / a


def t_cdf(x: float, df: int) -> float:
    """CDF of Student's t with `df` degrees of freedom."""
    tail = 0.5 * _betainc(df / 2, 0.5, df / (df + x * x))
    return 1 - tail if x >= 0 else tail


def t_quantile(p: float, df: int) -> float:
    """Inverse of t_cdf, by bisection (t_quantile(0.975, 2) = 4.303)."""
    lo, hi = -1e6, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2
