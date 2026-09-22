"""Aggregate eval results into the pre-registered comparisons (docs/analysis_plan.md).

    python -m vitok.analysis --results kaggle/outputs --compression kaggle/outputs/vitok-data/compression-16k.json \
        --out results/summary.md --figures figures

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
    """Every result file under `results_dir`, so one directory of per-session downloads also works."""
    runs = {}
    for path in sorted(results_dir.rglob("*.json")):
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


def summarize(runs: dict, compression: dict) -> str:
    lines = ["# Results summary", ""]
    depths = sorted({d for _, d, _ in runs})
    for depth in depths:
        at = {k: v for k, v in runs.items() if k[1] == depth}
        lines += [f"## d{depth}", "",
                  "| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token | tokens/syllable | superword share |",
                  "|---|---|---|---|---|---|---|---|"]
        idx = {v: common_docs(list(at.values()), v) for v in ("clean", "strip50", "strip100")}
        for (cond, _, seed), r in sorted(at.items()):
            chars = {v: np.array(r["docs"][v]["chars"])[idx[v]] for v in idx}
            vals = [bpc(arr(r, v, idx[v]), chars[v]) for v in ("clean", "strip50", "strip100")]
            p = r["pairs"]
            acc = np.mean([g is not None and b is not None and g < b for g, b in zip(p["good_nats"], p["bad_nats"])])
            c = compression.get(cond, {})
            lines.append(f"| {cond} s{seed} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {acc:.3f} | "
                         f"{c.get('chars_per_token', float('nan')):.3f} | {c.get('tokens_per_syllable', float('nan')):.3f} | "
                         f"{c.get('superword_token_share', float('nan')):.1%} |")
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


def sensitivity(runs: dict) -> str:
    """Does a conclusion survive dropping the longest documents, or splitting them by length?

    Every subset keeps the pairing (both runs score the same documents), so this only asks whether
    one slice of the test set drives the result.
    """
    lines = ["## Sensitivity of H1 (bpc clean, seed 0)", "",
             "| A − B | depth | all docs | without the 5% longest | short half | long half | docs favouring A |",
             "|---|---|---|---|---|---|---|"]
    for depth in sorted({d for _, d, _ in runs}):
        at = {k: v for k, v in runs.items() if k[1] == depth and k[2] == 0}
        idx = common_docs(list(at.values()), "clean")
        for a, b, variant, hyp in COMPARISONS:
            if variant != "clean" or hyp != "H1" or (a, depth, 0) not in at or (b, depth, 0) not in at:
                continue
            ra, rb = at[(a, depth, 0)], at[(b, depth, 0)]
            chars = np.array(ra["docs"]["clean"]["chars"])[idx]
            na, nb = arr(ra, "clean", idx), arr(rb, "clean", idx)
            order = np.argsort(chars)
            subsets = {"all": np.arange(len(idx)), "no_top5pct": order[: int(len(idx) * 0.95)],
                       "short": order[: len(idx) // 2], "long": order[len(idx) // 2:]}
            cells = [f"{bpc(na[s], chars[s]) - bpc(nb[s], chars[s]):+.4f}" for s in subsets.values()]
            favour_a = int(np.sum(na < nb))
            lines.append(f"| {a} − {b} | d{depth} | " + " | ".join(cells) +
                         f" | {favour_a}/{len(idx)} |")
    return "\n".join(lines) + "\n"


def nonembedding_params(user_config: dict) -> int:
    """12 * n_embd^2 * n_layer, the way nanochat sizes a model. Matches the `transformer_matrices`
    line of train.log to within 0.001%; embeddings are excluded because every condition shares them."""
    base = user_config["depth"] * user_config["aspect_ratio"]
    head = user_config["head_dim"]
    n_embd = -(-base // head) * head
    return 12 * n_embd * n_embd * user_config["depth"]


def scaling(runs: dict, figures: Path | None = None) -> str:
    """H3: how each comparison changes with model size, against the seed spread as the error bar."""
    depths = sorted({d for _, d, _ in runs})
    params = {d: nonembedding_params(next(r for (_, dd, _), r in runs.items() if dd == d)["user_config"])
              for d in depths}
    bpc_of, noise = {}, {}
    for (cond, depth, seed), r in runs.items():
        i = common_docs([v for (c, d, s), v in runs.items() if d == depth], "clean")
        chars = np.array(r["docs"]["clean"]["chars"])[i]
        bpc_of[(cond, depth, seed)] = bpc(arr(r, "clean", i), chars)
    for (cond, depth, seed) in runs:
        if seed > 0 and (cond, depth, 0) in bpc_of:
            noise[(cond, depth)] = abs(bpc_of[(cond, depth, seed)] - bpc_of[(cond, depth, 0)])

    lines = ["## H3: effect vs model size (bpc clean, seed 0)", "",
             "| A − B | " + " | ".join(f"d{d}" for d in depths) + " |",
             "|---" * (len(depths) + 1) + "|"]
    for a, b, variant, hyp in COMPARISONS:
        if variant != "clean" or hyp != "H1":
            continue
        cells = []
        for d in depths:
            if (a, d, 0) in bpc_of and (b, d, 0) in bpc_of:
                bar = max(noise.get((a, d), 0), noise.get((b, d), 0))
                cells.append(f"{bpc_of[(a, d, 0)] - bpc_of[(b, d, 0)]:+.4f} ± {bar:.4f}" if bar else
                             f"{bpc_of[(a, d, 0)] - bpc_of[(b, d, 0)]:+.4f}")
            else:
                cells.append("—")
        lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines += ["", "± is the spread between the two seeds of the same condition (a lower bound on run-to-run",
              "noise: the seed only changes weight init, the data order is identical).", "",
              "| depth | non-embedding params | " + " | ".join(sorted({c for c, _, _ in runs})) + " |",
              "|---" * (len(set(c for c, _, _ in runs)) + 2) + "|"]
    conds = sorted({c for c, _, _ in runs})
    for d in depths:
        cells = [f"{bpc_of[(c, d, 0)]:.4f}" if (c, d, 0) in bpc_of else "—" for c in conds]
        lines.append(f"| d{d} | {params[d]:,} | " + " | ".join(cells) + " |")
    if figures is not None:
        lines += ["", f"Figure: {_plot_scaling(bpc_of, noise, params, conds, depths, figures)}"]
    return "\n".join(lines) + "\n"


def _plot_scaling(bpc_of, noise, params, conds, depths, figures: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for cond in conds:
        xs = [params[d] for d in depths if (cond, d, 0) in bpc_of]
        ys = [bpc_of[(cond, d, 0)] for d in depths if (cond, d, 0) in bpc_of]
        left.plot(xs, ys, marker="o", label=cond)
    left.set(xscale="log", xlabel="non-embedding parameters", ylabel="bpc (clean)", title="bpc vs model size")
    left.legend(fontsize=8)
    for a, b, variant, hyp in COMPARISONS:
        if variant != "clean" or hyp != "H1":
            continue
        pts = [(params[d], bpc_of[(a, d, 0)] - bpc_of[(b, d, 0)], max(noise.get((a, d), 0), noise.get((b, d), 0)))
               for d in depths if (a, d, 0) in bpc_of and (b, d, 0) in bpc_of]
        if pts:
            right.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                           marker="o", capsize=3, label=f"{a} − {b}")
    right.axhline(0, color="black", lw=0.8)
    right.set(xscale="log", xlabel="non-embedding parameters", ylabel="Δ bpc (SuperBPE − BPE)",
              title="H3: SuperBPE advantage vs size")
    right.legend(fontsize=8)
    for ax in (left, right):  # a handful of sizes: label the points themselves, not decades
        ax.set_xticks([params[d] for d in depths], [f"d{d}\n{params[d] / 1e6:.1f}M" for d in depths])
        ax.set_xticks([], minor=True)
    fig.tight_layout()
    path = figures / "h3_scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--compression", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", type=Path, default=None, help="write the H3 figure here (needs matplotlib)")
    args = ap.parse_args()
    data = json.loads(args.compression.read_text(encoding="utf-8"))
    compression = {k: v for k, v in data.items() if isinstance(v, dict)}
    runs = load(args.results)
    text = summarize(runs, compression) + "\n" + sensitivity(runs)
    if len({d for _, d, _ in runs}) > 1:
        text += "\n" + scaling(runs, args.figures)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
