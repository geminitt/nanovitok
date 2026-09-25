"""Aggregate eval results into the pre-registered comparisons (see README, Design).

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
    ("super-nfd", "super-nfc", "strip50", "H2"),
    ("super-nfd", "super-nfc", "clean", "H2 cost"),
]
VARIANTS = ("clean", "strip50", "strip100")
MARGIN = 0.01            # pre-registered: "not worse by more than 1%" (relative bpc)
MIN_TOKEN_REDUCTION = 0.15
VAL_BPB_RE = re.compile(r"Validation bpb: ([\d.]+)")


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


def chars_of(run, variant, idx):
    return np.array(run["docs"][variant]["chars"])[idx]


def seed_noise(at: dict, variant: str, idx) -> float | None:
    """Largest |bpc(s1) - bpc(s0)| among the conditions run with two seeds, on `variant`.

    A lower bound on run-to-run noise: the seed only changes weight init, the data order is identical.
    None when no condition at this depth has a second seed.
    """
    spreads = []
    for (cond, depth, seed), r1 in at.items():
        r0 = at.get((cond, depth, 0))
        if seed > 0 and r0 is not None:
            ch = chars_of(r0, variant, idx)
            spreads.append(abs(bpc(arr(r1, variant, idx), ch) - bpc(arr(r0, variant, idx), ch)))
    return max(spreads) if spreads else None


def comparisons(runs: dict) -> list[dict]:
    """Every pre-registered comparison at every depth, on seed 0, with its bootstrap CI and seed noise."""
    rows = []
    for depth in sorted({d for _, d, _ in runs}):
        at = {k: v for k, v in runs.items() if k[1] == depth}
        idx = {v: common_docs(list(at.values()), v) for v in VARIANTS}
        for a, b, variant, hyp in COMPARISONS:
            ra, rb = at.get((a, depth, 0)), at.get((b, depth, 0))
            if ra is None or rb is None:
                continue
            i = idx[variant]
            res = paired_bootstrap_bpc(arr(ra, variant, i), arr(rb, variant, i), chars_of(ra, variant, i))
            rows.append({"depth": depth, "a": a, "b": b, "variant": variant, "hyp": hyp, **res,
                         "seed_noise": seed_noise(at, variant, i)})
    return rows


def beyond_noise(row) -> str:
    if row["seed_noise"] is None:
        return "no second seed"
    return "yes" if abs(row["diff"]) > row["seed_noise"] else "no"


def summarize(runs: dict, compression: dict, rows: list[dict]) -> str:
    lines = ["# Results summary", ""]
    depths = sorted({d for _, d, _ in runs})
    for depth in depths:
        at = {k: v for k, v in runs.items() if k[1] == depth}
        lines += [f"## d{depth}", "",
                  "| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token | tokens/syllable | superword share |",
                  "|---|---|---|---|---|---|---|---|"]
        idx = {v: common_docs(list(at.values()), v) for v in VARIANTS}
        for (cond, _, seed), r in sorted(at.items()):
            vals = [bpc(arr(r, v, idx[v]), chars_of(r, v, idx[v])) for v in VARIANTS]
            p = r["pairs"]
            acc = np.mean([g is not None and b is not None and g < b for g, b in zip(p["good_nats"], p["bad_nats"])])
            c = compression.get(cond, {})
            lines.append(f"| {cond} s{seed} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {acc:.3f} | "
                         f"{c.get('chars_per_token', float('nan')):.3f} | {c.get('tokens_per_syllable', float('nan')):.3f} | "
                         f"{c.get('superword_token_share', float('nan')):.1%} |")
        lines += ["", f"Documents scored by every run: " + ", ".join(f"{v} {len(idx[v])}" for v in VARIANTS), ""]
        noise = {v: seed_noise(at, v, idx[v]) for v in VARIANTS}
        if any(n is not None for n in noise.values()):
            lines += ["Seed noise (largest |s1 − s0| among conditions with two seeds; a lower bound): " +
                      ", ".join(f"{v} {n:.4f}" for v, n in noise.items() if n is not None), ""]
        lines += ["| hypothesis | A − B | variant | Δbpc | 95% CI | Δ relative [95% CI] | CI excludes 0 | |Δ| > seed noise |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in (r for r in rows if r["depth"] == depth):
            lines.append(f"| {row['hyp']} | {row['a']} − {row['b']} | {row['variant']} | {row['diff']:+.4f} | "
                         f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] | "
                         f"{row['rel_diff']:+.2%} [{row['rel_ci95'][0]:+.2%}, {row['rel_ci95'][1]:+.2%}] | "
                         f"{'yes' if row['significant'] else 'no'} | {beyond_noise(row)} |")
        # minimal pairs, McNemar for the same comparisons on clean text
        lines += ["", "| pairs | A − B | A only right | B only right | p |", "|---|---|---|---|---|"]
        for a, b, variant, hyp in COMPARISONS:
            ra, rb = at.get((a, depth, 0)), at.get((b, depth, 0))
            if variant != "clean" or ra is None or rb is None:
                continue
            ok = lambda r: [g is not None and x is not None and g < x for g, x in zip(r["pairs"]["good_nats"], r["pairs"]["bad_nats"])]
            t = mcnemar_exact(ok(ra), ok(rb))
            lines.append(f"| {hyp} | {a} − {b} | {t['a_only']} | {t['b_only']} | {t['p_value']:.3g} |")
        lines.append("")
    return "\n".join(lines)


def direction(row) -> str:
    """Which way a comparison goes once both the document bootstrap and the seed noise are accounted for."""
    if not row["significant"] or beyond_noise(row) == "no":
        return "not resolved"
    return "A lower" if row["diff"] < 0 else "A higher"


def verdicts(rows: list[dict], reductions: dict, runs: dict, wordhood: dict | None = None) -> str:
    """The pre-registered hypotheses, decided by the rules committed before training (README, Design)."""
    lines = ["## Pre-registered verdicts", "",
             "A comparison counts as resolved only when its 95% CI excludes 0 **and** |Δ| exceeds the seed noise "
             "(where a second seed exists).", ""]
    # H1: compression, then non-inferiority at every depth
    met = all(reductions.get(n, 0) >= MIN_TOKEN_REDUCTION for n in ("nfc", "nfd"))
    lines.append(f"**H1** — token reduction on the val shard: NFC {reductions.get('nfc', float('nan')):.1%}, "
                 f"NFD {reductions.get('nfd', float('nan')):.1%} (threshold {MIN_TOKEN_REDUCTION:.0%}): "
                 f"{'met' if met else 'not met'}.")
    h1 = [r for r in rows if r["hyp"] == "H1"]
    for r in h1:
        ok = r["rel_ci95"][1] < MARGIN
        lines.append(f"- d{r['depth']} {r['a']} − {r['b']}: {r['rel_diff']:+.2%} "
                     f"[{r['rel_ci95'][0]:+.2%}, {r['rel_ci95'][1]:+.2%}] → not worse by more than {MARGIN:.0%}: "
                     f"{'yes' if ok else 'no'}")
    h1_ok = met and bool(h1) and all(r["rel_ci95"][1] < MARGIN for r in h1)
    lines += [f"- Verdict: **{'supported' if h1_ok else 'not supported'}**.", ""]

    # H2: NFD lower on stripped text, and a clean-text cost within the margin
    lines.append("**H2** — NFD tokenizers give lower bpc on diacritic-stripped text, at a clean-text cost within "
                 f"{MARGIN:.0%}:")
    h2 = [r for r in rows if r["hyp"] == "H2"]
    dirs = [direction(r) for r in h2]
    for r, d in zip(h2, dirs):
        label = {"A lower": "NFD better", "A higher": "NFD worse"}.get(d, d)
        lines.append(f"- d{r['depth']} {r['a']} − {r['b']}, {r['variant']}: {r['diff']:+.4f} → {label}")
    cost = [r for r in rows if r["hyp"] == "H2 cost"]
    for r in cost:
        lines.append(f"- d{r['depth']} {r['a']} − {r['b']}, clean (cost): {r['rel_diff']:+.2%} "
                     f"[{r['rel_ci95'][0]:+.2%}, {r['rel_ci95'][1]:+.2%}] → within {MARGIN:.0%}: "
                     f"{'yes' if r['rel_ci95'][1] < MARGIN else 'no'}")
    resolved = [d for d in dirs if d != "not resolved"]
    if not resolved:
        h2_verdict = "not supported (no comparison resolved)"
    elif all(d == "A lower" for d in resolved):
        h2_verdict = "supported" if all(r["rel_ci95"][1] < MARGIN for r in cost) else "supported on stripped text, cost above the margin"
    elif all(d == "A higher" for d in resolved):
        h2_verdict = "contradicted (NFD worse wherever resolved)"
    else:
        h2_verdict = "not supported (resolved comparisons point in opposite directions)"
    lines += [f"- Verdict: **{h2_verdict}**.", ""]

    # H3: sign of the SuperBPE effect by size, for each seed that exists
    lines.append("**H3** — Δbpc clean, super-nfc − bpe-nfc, by depth and seed:")
    for depth in sorted({d for _, d, _ in runs}):
        cells = []
        for seed in sorted({s for (c, d, s) in runs if d == depth}):
            dl = delta(runs, "super-nfc", "bpe-nfc", depth, seed)
            if dl is not None:
                cells.append(f"s{seed} {dl:+.4f}")
        lines.append(f"- d{depth}: " + ", ".join(cells))
    lines += ["- Three sizes and at most two seeds: a trend, not a law.", ""]

    # H4: exploratory
    if wordhood:
        lines.append("**H4** (exploratory, no test) — superwords of 2–4 syllables that are exactly one underthesea word:")
        for cond in ("super-nfc", "super-nfd"):
            w = wordhood.get(cond)
            if w and "frequency_matched_baseline" in w:
                lines.append(f"- {cond}: {w['match_rate']:.1%}; frequency-matched baseline "
                             f"{w['frequency_matched_baseline']:.1%}; all adjacent syllables {w['baseline_rate']:.1%}; "
                             f"{w['share_outside_2_4']:.1%} of superword occurrences span another number of syllables")
        lines.append("")
    return "\n".join(lines) + "\n"


def delta(runs, a, b, depth, seed) -> float | None:
    """bpc clean of A minus B for one seed, on the documents every run at this depth scored."""
    ra, rb = runs.get((a, depth, seed)), runs.get((b, depth, seed))
    if ra is None or rb is None:
        return None
    i = common_docs([v for (c, d, s), v in runs.items() if d == depth], "clean")
    ch = chars_of(ra, "clean", i)
    return bpc(arr(ra, "clean", i), ch) - bpc(arr(rb, "clean", i), ch)


def nanochat_val_bpb(results_dir: Path) -> dict:
    """Final 'Validation bpb' of every training log, keyed like the result files."""
    out = {}
    for log in sorted(results_dir.rglob("runs/*/train.log")):
        m = re.fullmatch(r"(?P<cond>(?:bpe|super)-nf[cd])_d(?P<depth>\d+)_s(?P<seed>\d+)", log.parent.name)
        values = VAL_BPB_RE.findall(log.read_text(encoding="utf-8", errors="replace"))
        if m and values:
            out[(m["cond"], int(m["depth"]), int(m["seed"]))] = float(values[-1])
    return out


def val_vs_test(runs: dict, val_bpb: dict, compression: dict) -> str:
    """nanochat's validation bpb next to our test bpc, for the NFC pair (bytes per char are equal there)."""
    lines = ["## nanochat validation bpb vs test bpc (super-nfc − bpe-nfc, seed 0)", "",
             "| depth | val bpb bpe-nfc | val bpb super-nfc | val relative | test bpc relative |", "|---|---|---|---|---|"]
    for depth in sorted({d for _, d, _ in runs}):
        vb, vs = val_bpb.get(("bpe-nfc", depth, 0)), val_bpb.get(("super-nfc", depth, 0))
        dl = delta(runs, "super-nfc", "bpe-nfc", depth, 0)
        rb = runs.get(("bpe-nfc", depth, 0))
        if vb is None or vs is None or dl is None:
            continue
        i = common_docs([v for (c, d, s), v in runs.items() if d == depth], "clean")
        base = bpc(arr(rb, "clean", i), chars_of(rb, "clean", i))
        lines.append(f"| d{depth} | {vb:.4f} | {vs:.4f} | {vs / vb - 1:+.2%} | {dl / base:+.2%} |")
    cpt = {c: compression.get(c, {}).get("chars_per_token") for c in ("bpe-nfc", "super-nfc")}
    extra = f"{cpt['super-nfc'] / cpt['bpe-nfc'] - 1:.0%}" if all(cpt.values()) else "more"
    lines += ["", "nanochat evaluates a fixed number of tokens of the val shard, so the two tokenizers are scored on",
              f"different documents (SuperBPE covers {extra} more text), unpaired. Only the paired test bpc is used for",
              "conclusions; the table shows how far the two disagree. NFD runs are left out: their bpb counts NFD bytes.", ""]
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


def val_sensitivity(val_runs: dict, test_rows: list[dict]) -> str:
    """The pre-registered sensitivity check: H1 on the val shard, scored per document like the test set."""
    lines = ["## Sensitivity of H1 on the val shard (seed 0)", "",
             "| A − B | depth | val docs | val Δbpc | val relative [95% CI] | test relative | same sign |",
             "|---|---|---|---|---|---|---|"]
    test = {(r["depth"], r["a"], r["b"]): r for r in test_rows if r["hyp"] == "H1"}
    for depth in sorted({d for _, d, _ in val_runs}):
        at = {k: v for k, v in val_runs.items() if k[1] == depth}
        idx = common_docs(list(at.values()), "clean")
        for a, b, variant, hyp in COMPARISONS:
            ra, rb = at.get((a, depth, 0)), at.get((b, depth, 0))
            if hyp != "H1" or ra is None or rb is None:
                continue
            res = paired_bootstrap_bpc(arr(ra, "clean", idx), arr(rb, "clean", idx), chars_of(ra, "clean", idx))
            t = test.get((depth, a, b))
            same = "—" if t is None else ("yes" if (res["diff"] > 0) == (t["diff"] > 0) else "no")
            lines.append(f"| {a} − {b} | d{depth} | {len(idx)} | {res['diff']:+.4f} | {res['rel_diff']:+.2%} "
                         f"[{res['rel_ci95'][0]:+.2%}, {res['rel_ci95'][1]:+.2%}] | "
                         f"{'—' if t is None else format(t['rel_diff'], '+.2%')} | {same} |")
    return "\n".join(lines) + "\n"


def nonembedding_params(user_config: dict) -> int:
    """12 * n_embd^2 * n_layer, the way nanochat sizes a model. Matches the `transformer_matrices`
    line of train.log to within 0.001%; embeddings are excluded because every condition shares them."""
    base = user_config["depth"] * user_config["aspect_ratio"]
    head = user_config["head_dim"]
    n_embd = -(-base // head) * head
    return 12 * n_embd * n_embd * user_config["depth"]


def scaling(runs: dict, figures: Path | None = None) -> str:
    """H3: how each comparison changes with model size, with the second seed where one exists."""
    depths = sorted({d for _, d, _ in runs})
    params = {d: nonembedding_params(next(r for (_, dd, _), r in runs.items() if dd == d)["user_config"])
              for d in depths}
    bpc_of = {}
    for (cond, depth, seed), r in runs.items():
        i = common_docs([v for (c, d, s), v in runs.items() if d == depth], "clean")
        bpc_of[(cond, depth, seed)] = bpc(arr(r, "clean", i), chars_of(r, "clean", i))

    def effect(a, b, d, seed):
        return bpc_of[(a, d, seed)] - bpc_of[(b, d, seed)] if (a, d, seed) in bpc_of and (b, d, seed) in bpc_of else None

    lines = ["## H3: effect vs model size (bpc clean)", "",
             "| A − B | " + " | ".join(f"d{d}" for d in depths) + " |",
             "|---" * (len(depths) + 1) + "|"]
    for a, b, variant, hyp in COMPARISONS:
        if variant != "clean" or hyp != "H1":
            continue
        cells = []
        for d in depths:
            e0, e1 = effect(a, b, d, 0), effect(a, b, d, 1)
            cells.append("—" if e0 is None else f"{e0:+.4f}" + (f" (s1: {e1:+.4f})" if e1 is not None else ""))
        lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines += ["", "Seed 0, with the same difference for seed 1 in brackets where both conditions have a second seed.",
              "The seed only changes weight init (the data order is identical), so the gap between the two is a",
              "lower bound on run-to-run noise.", "",
              "| depth | non-embedding params | " + " | ".join(sorted({c for c, _, _ in runs})) + " |",
              "|---" * (len(set(c for c, _, _ in runs)) + 2) + "|"]
    conds = sorted({c for c, _, _ in runs})
    for d in depths:
        cells = [f"{bpc_of[(c, d, 0)]:.4f}" if (c, d, 0) in bpc_of else "—" for c in conds]
        lines.append(f"| d{d} | {params[d]:,} | " + " | ".join(cells) + " |")
    if figures is not None:
        lines += ["", f"Figure: {_plot_scaling(bpc_of, effect, params, conds, depths, figures)}"]
    return "\n".join(lines) + "\n"


def _plot_scaling(bpc_of, effect, params, conds, depths, figures: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for cond in conds:
        xs = [params[d] for d in depths if (cond, d, 0) in bpc_of]
        ys = [bpc_of[(cond, d, 0)] for d in depths if (cond, d, 0) in bpc_of]
        left.plot(xs, ys, marker="o", label=cond)
    left.set(xscale="log", xlabel="non-embedding parameters", ylabel="bpc (clean, seed 0)", title="bpc vs model size")
    left.legend(fontsize=8)
    for a, b, variant, hyp in COMPARISONS:
        if variant != "clean" or hyp != "H1":
            continue
        pts = [(params[d], effect(a, b, d, 0)) for d in depths if effect(a, b, d, 0) is not None]
        if not pts:
            continue
        line, = right.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{a} − {b}, seed 0")
        seed1 = [(params[d], effect(a, b, d, 1)) for d in depths if effect(a, b, d, 1) is not None]
        if seed1:
            right.plot([p[0] for p in seed1], [p[1] for p in seed1], ls="none", marker="o", mfc="none",
                       color=line.get_color(), label=f"{a} − {b}, seed 1")
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
    ap.add_argument("--wordhood", type=Path, default=None, help="wordhood JSON from vitok.wordhood (H4)")
    ap.add_argument("--val-results", type=Path, default=None, help="per-document val results from vitok.eval_checkpoints")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", type=Path, default=None, help="write the H3 figure here (needs matplotlib)")
    args = ap.parse_args()
    data = json.loads(args.compression.read_text(encoding="utf-8"))
    compression = {k: v for k, v in data.items() if isinstance(v, dict)}
    reductions = {n: data.get(f"token_reduction_{n}", float("nan")) for n in ("nfc", "nfd")}
    wordhood = json.loads(args.wordhood.read_text(encoding="utf-8")) if args.wordhood else None
    runs = load(args.results)
    rows = comparisons(runs)
    text = (summarize(runs, compression, rows) + "\n" + verdicts(rows, reductions, runs, wordhood) + "\n"
            + sensitivity(runs) + "\n" + val_vs_test(runs, nanochat_val_bpb(args.results), compression))
    if args.val_results:
        text += "\n" + val_sensitivity(load(args.val_results), rows)
    if len({d for _, d, _ in runs}) > 1:
        text += "\n" + scaling(runs, args.figures)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
