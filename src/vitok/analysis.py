"""Aggregate eval results into the pre-registered comparisons (see README, Design).

    python -m vitok.analysis --results kaggle/outputs --compression kaggle/outputs/vitok-data/compression-16k.json \
        --out results/summary.md --figures figures

Result files are named {condition}_d{depth}_s{seed}.json (written by vitok.eval).
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from vitok.stats import bpc, paired_bootstrap_bpc, t_quantile

NAME_RE = re.compile(r"(?P<cond>(?:bpe|super)-nfc)_d(?P<depth>\d+)_s(?P<seed>\d+)\.json$")

# (A, B, variant, hypothesis): report bpc(A) - bpc(B); negative favours A.
COMPARISONS = [("super-nfc", "bpe-nfc", "clean", "H1")]
VARIANTS = ("clean",)
MARGIN = 0.01            # pre-registered: "not worse by more than 1%" (relative bpc)
MIN_TOKEN_REDUCTION = 0.15
VAL_BPB_RE = re.compile(r"Validation bpb: ([\d.]+)")


def load(results_dir: Path) -> dict:
    """Every result file under `results_dir`, so one directory of per-session downloads also works."""
    runs: dict[tuple[str, int, int], dict] = {}
    seen: dict[tuple[str, int, int], Path] = {}
    for path in sorted(results_dir.rglob("*.json")):
        m = NAME_RE.search(path.name)
        if m:
            key = (m["cond"], int(m["depth"]), int(m["seed"]))
            # two files for one run (say, test and val results under one root) must not silently replace each other
            if key in seen:
                raise ValueError(f"two result files for {path.name}: {seen[key]} and {path}")
            seen[key] = path
            runs[key] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def common_docs(runs: list[dict], variant: str) -> np.ndarray:
    """Indices of documents every run could score (none exceeded its context).

    Paired comparisons index every run with these positions, so all runs must hold the same documents in the same
    order with the same character counts; result files that disagree are refused.
    """
    first = runs[0]
    for r in runs[1:]:
        if r.get("doc_ids") != first.get("doc_ids") or r["docs"][variant]["chars"] != first["docs"][variant]["chars"]:
            raise ValueError(f"result files disagree on the documents or their lengths ({variant}); "
                             "they cannot be paired")
    ok = np.ones(len(first["docs"][variant]["nats"]), bool)
    for r in runs:
        ok &= np.array([n is not None for n in r["docs"][variant]["nats"]])
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        raise ValueError(f"no document was scored by every run ({variant})")
    return idx


def arr(run, variant, idx):
    return np.array([run["docs"][variant]["nats"][i] for i in idx], float)


def chars_of(run, variant, idx):
    return np.array(run["docs"][variant]["chars"])[idx]


def seed_spreads(at: dict, variant: str, idx) -> list[float]:
    """bpc(s1) - bpc(s0) for every condition run with two seeds at this depth, on `variant`."""
    out = []
    for (cond, depth, seed), r1 in sorted(at.items()):
        r0 = at.get((cond, depth, 0))
        if seed > 0 and r0 is not None:
            ch = chars_of(r0, variant, idx)
            out.append(bpc(arr(r1, variant, idx), ch) - bpc(arr(r0, variant, idx), ch))
    return out


def noise_threshold(spreads: list[float], level: float = 0.95) -> float | None:
    """How large a difference between two single runs must be before run-to-run noise cannot explain it.

    With run noise of standard deviation s per run, a seed spread s1 - s0 and a difference between two
    conditions A - B both carry noise of standard deviation sqrt(2) s. The spreads therefore estimate the
    noise of a difference directly, sigma_hat = sqrt(mean(spread^2)), with one degree of freedom per spread.
    Estimated from so few, the noise calls for Student's t instead of the normal 1.96: the threshold is
    t_(1 - alpha/2, k) * sigma_hat, 4.30 * sigma_hat for k = 2. Comparing |Delta| with a single spread
    instead would flag a third of pure-noise differences.

    Two assumptions make this an approximation of unknown direction. The data order is identical in every
    run, so data-order noise is left out (the threshold is too low by that much). And every condition is
    initialized from the same seed with the same parameter shapes, so A and B start from identical weights:
    that shared part cancels in A - B, which the spreads (two seeds of one condition) do not reflect (the
    threshold is too high by that much). The spreads also come from the NFC conditions only. None when no
    condition has a second seed.
    """
    if not spreads:
        return None
    sigma = math.sqrt(sum(d * d for d in spreads) / len(spreads))
    threshold = t_quantile(1 - (1 - level) / 2, len(spreads)) * sigma
    assert threshold >= 0
    return threshold


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
            spreads = seed_spreads(at, variant, i)
            rows.append({"depth": depth, "a": a, "b": b, "variant": variant, "hyp": hyp, **res,
                         "seed_spreads": spreads, "noise_threshold": noise_threshold(spreads)})
    return rows


def beyond_noise(row) -> str:
    if row["noise_threshold"] is None:
        return "no second seed"
    return "yes" if abs(row["diff"]) > row["noise_threshold"] else "no"


def summarize(runs: dict, compression: dict, rows: list[dict]) -> str:
    lines = ["# From-scratch NFC pilot", "",
             "Pilot evidence under the original equal-text design; see README for the Qwen3 retrofit study.", ""]
    for depth in sorted({d for _, d, _ in runs}):
        at = {k: v for k, v in runs.items() if k[1] == depth}
        idx = common_docs(list(at.values()), "clean")
        lines += [f"## d{depth}", "", "| run | bpc | chars/token | tokens/syllable | superword share |",
                  "|---|---:|---:|---:|---:|"]
        for (cond, _, seed), run in sorted(at.items()):
            value = bpc(arr(run, "clean", idx), chars_of(run, "clean", idx))
            c = compression.get(cond, {})
            lines.append(f"| {cond} s{seed} | {value:.4f} | {c.get('chars_per_token', float('nan')):.3f} | "
                         f"{c.get('tokens_per_syllable', float('nan')):.3f} | "
                         f"{c.get('superword_token_share', float('nan')):.1%} |")
        lines += ["", f"Documents scored by every run: {len(idx)}.", ""]
        spreads = seed_spreads(at, "clean", idx)
        if spreads:
            lines += ["Seed spreads (s1 − s0): " + ", ".join(f"{v:+.4f}" for v in spreads) +
                      f"; approximate t-based noise threshold {noise_threshold(spreads):.4f}.", ""]
        lines += ["| A − B | Δbpc | 95% interval | Relative difference [95% interval] | "
                  "Interval excludes 0 | Beyond seed noise |",
                  "|---|---:|---|---|---|---|"]
        for row in (r for r in rows if r["depth"] == depth):
            lines.append(f"| {row['a']} − {row['b']} | {row['diff']:+.4f} | "
                         f"[{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] | "
                         f"{row['rel_diff']:+.2%} [{row['rel_ci95'][0]:+.2%}, {row['rel_ci95'][1]:+.2%}] | "
                         f"{'yes' if row['significant'] else 'no'} | {beyond_noise(row)} |")
        lines.append("")
    return "\n".join(lines)


def direction(row) -> str:
    """Which way a comparison goes once both the document bootstrap and the seed noise are accounted for."""
    if not row["significant"] or beyond_noise(row) == "no":
        return "not resolved"
    return "A lower" if row["diff"] < 0 else "A higher"


def verdicts(rows: list[dict], reductions: dict, runs: dict) -> str:
    """Retained NFC pilot decision rules fixed before its from-scratch runs."""
    lines = ["## NFC pilot quality margin", ""]
    reduction = reductions.get("nfc", float("nan"))
    met = reduction >= MIN_TOKEN_REDUCTION
    lines.append(f"Token reduction: NFC {reduction:.1%} (threshold {MIN_TOKEN_REDUCTION:.0%}): "
                 f"{'met' if met else 'not met'}.")
    h1 = [r for r in rows if r["hyp"] == "H1"]
    for r in h1:
        ok = r["rel_ci95"][1] < MARGIN
        lines.append(f"- d{r['depth']} {r['a']} − {r['b']}: {r['rel_diff']:+.2%} "
                     f"[{r['rel_ci95'][0]:+.2%}, {r['rel_ci95'][1]:+.2%}] → not worse by more than {MARGIN:.0%}: "
                     f"{'yes' if ok else 'no'}")
    passed = met and bool(h1) and all(r["rel_ci95"][1] < MARGIN for r in h1)
    lines += [f"- NFC pilot criterion: **{'met' if passed else 'not met'}**.", "", "## NFC pilot size trend", ""]
    for depth in sorted({d for _, d, _ in runs}):
        cells = []
        for seed in sorted({s for _, d, s in runs if d == depth}):
            value = delta(runs, "super-nfc", "bpe-nfc", depth, seed)
            if value is not None:
                cells.append(f"s{seed} {value:+.4f}")
        lines.append(f"- d{depth}: " + ", ".join(cells))
    lines += ["- Three sizes and at most two seeds: a trend, not a law.", ""]
    return "\n".join(lines)


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
        m = re.fullmatch(r"(?P<cond>(?:bpe|super)-nfc)_d(?P<depth>\d+)_s(?P<seed>\d+)", log.parent.name)
        values = VAL_BPB_RE.findall(log.read_text(encoding="utf-8", errors="replace"))
        if m and values:
            out[(m["cond"], int(m["depth"]), int(m["seed"]))] = float(values[-1])
    return out


def val_vs_test(runs: dict, val_bpb: dict, compression: dict) -> str:
    """nanochat's validation bpb next to our test bpc, for the NFC pair (bytes per char are equal there)."""
    lines = ["## nanochat validation bpb vs test bpc (super-nfc − bpe-nfc, seed 0)", "",
             "| depth | val bpb bpe-nfc | val bpb super-nfc | val relative | test bpc relative |",
             "|---|---|---|---|---|"]
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
              f"different documents (SuperBPE covers {extra} more text), unpaired. Only the paired test bpc is "
              "used for",
              "conclusions; the table shows how far the two disagree.", ""]
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
            order = np.argsort(chars, kind="stable")  # ties broken by document order, identical on every machine
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
        for a, b, _variant, hyp in COMPARISONS:
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
    assert n_embd >= base and n_embd % head == 0
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
        if (a, d, seed) not in bpc_of or (b, d, seed) not in bpc_of:
            return None
        return bpc_of[(a, d, seed)] - bpc_of[(b, d, seed)]

    lines = ["## NFC pilot effect vs model size (bpc clean)", "",
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
              title="NFC pilot: SuperBPE effect vs size")
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
    ap.add_argument("--val-results", type=Path, default=None,
                    help="per-document val results from vitok.eval_checkpoints")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", type=Path, default=None, help="write the H3 figure here (needs matplotlib)")
    args = ap.parse_args()
    data = json.loads(args.compression.read_text(encoding="utf-8"))
    compression = {k: v for k, v in data.items() if isinstance(v, dict)}
    reductions = {n: data.get(f"token_reduction_{n}", float("nan")) for n in ("nfc",)}
    runs = load(args.results)
    rows = comparisons(runs)
    text = (summarize(runs, compression, rows) + "\n" + verdicts(rows, reductions, runs) + "\n"
            + sensitivity(runs) + "\n" + val_vs_test(runs, nanochat_val_bpb(args.results), compression))
    if args.val_results:
        text += "\n" + val_sensitivity(load(args.val_results), rows)
    if len({d for _, d, _ in runs}) > 1:
        text += "\n" + scaling(runs, args.figures)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
