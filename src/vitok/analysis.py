"""Aggregate eval results into the pre-registered comparisons (docs/analysis_plan.md).

    python -m vitok.analysis --results results --compression compression.json --out results/summary.md

Result files are named {condition}_d{depth}_s{seed}.json (written by vitok.eval).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from vitok.stats import bpc, mcnemar_exact, paired_bootstrap_bpc

NAME_RE = re.compile(r"(?P<cond>(?:bpe|super)-nf[cd])_d(?P<depth>\d+)_s(?P<seed>\d+)\.json$")

# (A, B, variant, hypothesis): report bpc(A) - bpc(B); negative favours A.
COMPARISONS = [
    ("super-nfc", "bpe-nfc", "clean", "H1"),
    ("super-nfd", "bpe-nfd", "clean", "H1"),
    ("bpe-nfd", "bpe-nfc", "strip100", "H2"),
    ("bpe-nfd", "bpe-nfc", "strip50", "H2"),
    ("bpe-nfd", "bpe-nfc", "clean", "H2 cost"),
    ("super-nfd", "super-nfc", "strip100", "H2"),
    ("super-nfd", "super-nfc", "clean", "H2 cost"),
]


def load(results_dir: Path) -> dict:
    runs = {}
    for path in sorted(results_dir.glob("*.json")):
        m = NAME_RE.search(path.name)
        if m:
            runs[(m["cond"], int(m["depth"]), int(m["seed"]))] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def common_docs(runs: list[dict], variant: str) -> np.ndarray:
    """Indices of documents every run could score (none exceeded its context)."""
    ok = np.ones(len(runs[0]["docs"][variant]["nats"]), bool)
    for r in runs:
        ok &= np.array([n is not None for n in r["docs"][variant]["nats"]])
    return np.flatnonzero(ok)


def arr(run, variant, idx):
    return np.array([run["docs"][variant]["nats"][i] for i in idx], float)


def summarize(runs: dict, cpt: dict) -> str:
    lines = ["# Results summary", ""]
    depths = sorted({d for _, d, _ in runs})
    for depth in depths:
        at = {k: v for k, v in runs.items() if k[1] == depth}
        lines += [f"## d{depth}", "", "| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token |",
                  "|---|---|---|---|---|---|"]
        idx = {v: common_docs(list(at.values()), v) for v in ("clean", "strip50", "strip100")}
        for (cond, _, seed), r in sorted(at.items()):
            chars = {v: np.array(r["docs"][v]["chars"])[idx[v]] for v in idx}
            vals = [bpc(arr(r, v, idx[v]), chars[v]) for v in ("clean", "strip50", "strip100")]
            p = r["pairs"]
            acc = np.mean([g is not None and b is not None and g < b for g, b in zip(p["good_nats"], p["bad_nats"])])
            lines.append(f"| {cond} s{seed} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {acc:.3f} | {cpt.get(cond, float('nan')):.3f} |")
        lines += ["", f"Documents scored by every run: clean {len(idx['clean'])}, strip100 {len(idx['strip100'])}", "",
                  "| hypothesis | A − B | variant | Δbpc | 95% CI | significant |", "|---|---|---|---|---|---|"]
        for a, b, variant, hyp in COMPARISONS:
            ra, rb = at.get((a, depth, 0)), at.get((b, depth, 0))
            if ra is None or rb is None:
                continue
            i = idx[variant]
            chars = np.array(ra["docs"][variant]["chars"])[i]
            res = paired_bootstrap_bpc(arr(ra, variant, i), arr(rb, variant, i), chars)
            lines.append(f"| {hyp} | {a} − {b} | {variant} | {res['diff']:+.4f} | "
                         f"[{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}] | {'yes' if res['significant'] else 'no'} |")
        # minimal pairs, McNemar for the same comparisons on clean text
        lines += ["", "| pairs | A − B | A only right | B only right | p |", "|---|---|---|---|---|"]
        for a, b, variant, hyp in COMPARISONS:
            ra, rb = at.get((a, depth, 0)), at.get((b, depth, 0))
            if variant != "clean" or ra is None or rb is None:
                continue
            ok = lambda r: [g is not None and x is not None and g < x for g, x in zip(r["pairs"]["good_nats"], r["pairs"]["bad_nats"])]
            t = mcnemar_exact(ok(ra), ok(rb))
            lines.append(f"| {hyp} | {a} − {b} | {t['a_only']} | {t['b_only']} | {t['p_value']:.3g} |")
        seeds = [k for k in at if k[2] > 0]
        for cond, _, seed in seeds:
            r0, r1 = at.get((cond, depth, 0)), at[(cond, depth, seed)]
            if r0:
                i = idx["clean"]
                ch = np.array(r0["docs"]["clean"]["chars"])[i]
                res = paired_bootstrap_bpc(arr(r1, "clean", i), arr(r0, "clean", i), ch)
                lines.append(f"\nSeed noise {cond}: s{seed} − s0 = {res['diff']:+.4f} bpc")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--compression", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.compression.read_text(encoding="utf-8"))
    cpt = {k: v["chars_per_token"] for k, v in data.items() if isinstance(v, dict)}
    text = summarize(load(args.results), cpt)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
