"""Report figures and tables of the token-type experiment, from `vitok.cpt_report`'s results.json.

    python -m vitok.cpt_figures --results results/token-type/results.json --out results/token-type

Static figures for a printed report (PNG, light surface). Colors follow the dataviz reference palette's first three
categorical slots, validated all-pairs for color-vision deficiency (worst ΔE 9.2); aqua sits below 3:1 contrast, so
every series is also direct-labeled and drawn with its own line style, and every figure has a table in `report.md`.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"base": "#2a78d6", "syllable-1000": "#eb6834", "multisyllable-1000": "#1baf7a"}
STYLES = {"base": "-", "syllable-1000": "--", "multisyllable-1000": ":"}
LABELS = {"base": "base (no new tokens)", "syllable-1000": "syllable (+1,000)",
          "multisyllable-1000": "multisyllable (+1,000)"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DOMAINS = {"fineweb2": "Web (FineWeb-2)", "wikipedia": "Wikipedia", "legal": "Legal", "flores": "FLORES+",
           "english": "English (web)"}


def style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))  # narrow small multiples: few x labels


def label_ends(ax, ends: dict[str, tuple[float, float]]):
    """Direct labels at the line ends, spread to a minimum vertical gap so converging lines stay readable."""
    lo, hi = ax.get_ylim()
    gap = 0.07 * (hi - lo)
    placed: list[float] = []
    for cond, (x, y) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        y_text = max(y, placed[-1] + gap) if placed else y
        placed.append(y_text)
        ax.annotate(cond.split("-")[0], (x, y), xytext=(x, y_text), textcoords="data", va="center", fontsize=8,
                    color=INK2, xycoords="data", annotation_clip=False,
                    arrowprops=None if abs(y_text - y) < 1e-12 else {"arrowstyle": "-", "color": GRID, "lw": 0.6})
        ax.texts[-1].set_x(x)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.12 * (ax.get_xlim()[1] - ax.get_xlim()[0]))


def legend_below(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK2, ncols=len(labels), loc="lower center",
               bbox_to_anchor=(0.5, -0.02))


def save(fig, path: Path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200, metadata={"Software": None})
    plt.close(fig)


def learning_curves(curves: dict, out: Path):
    """Dev bpc against training tokens and against characters read: two panels, one y-scale each (no dual axis)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, xlabel in ((axes[0], "tokens", "training tokens (millions)"),
                            (axes[1], "chars", "characters read (millions)")):
        ends = {}
        for c, cur in curves.items():
            pts = [(p[key] / 1e6, p["bpc"]) for p in cur["dev"]]
            ax.plot(*zip(*pts, strict=True), color=COLORS[c], linestyle=STYLES[c], linewidth=2, label=LABELS[c])
            ends[c] = pts[-1]
        style(ax, f"Dev bpc against {key if key == 'tokens' else 'characters read'}", xlabel,
              "bits per character" if ax is axes[0] else "")
        label_ends(ax, ends)
    legend_below(fig, axes[0])
    save(fig, out / "learning_curves.png")


def domain_curves(curves: dict, out: Path):
    """Each held-out domain's bpc at the saved checkpoints: small multiples, one per domain."""
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(14, 3.2))
    for ax, (d, title) in zip(axes, DOMAINS.items(), strict=True):
        ends = {}
        for c, cur in curves.items():
            snaps = sorted(cur["snapshots"].values(), key=lambda p: p["tokens"])
            pts = [(p["tokens"] / 1e6, p[d]) for p in snaps]
            ax.plot(*zip(*pts, strict=True), color=COLORS[c], linestyle=STYLES[c], linewidth=2, marker="o",
                    markersize=3, label=LABELS[c])
            ends[c] = pts[-1]
        style(ax, title, "training tokens (millions)", "bpc" if ax is axes[0] else "")
        label_ends(ax, ends)
    legend_below(fig, axes[0])
    save(fig, out / "domain_curves.png")


def final_differences(primary: dict, out: Path):
    """Relative bpc against base at the end, per domain and the macro mean, with 95% paired-bootstrap intervals."""
    rows = [*(d for d in DOMAINS if d != "english"), "macro"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k, c in enumerate(("syllable-1000", "multisyllable-1000")):
        p = primary[f"{c} vs base"]
        for j, d in enumerate(rows):
            v = p["macro"] if d == "macro" else p["per_domain"][d]
            mid = 100 * (v["macro_rel_diff"] if d == "macro" else v["rel_diff"])
            lo, hi = (100 * x for x in (v["ci95"] if d == "macro" else v["rel_ci95"]))
            y = j + (k - 0.5) * 0.3
            ax.errorbar(mid, y, xerr=[[mid - lo], [hi - mid]], fmt="o", color=COLORS[c], markersize=6, capsize=3,
                        linewidth=1.5, label=LABELS[c] if j == 0 else None)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(rows)), [DOMAINS.get(d, "Macro mean (primary)") for d in rows])
    ax.invert_yaxis()
    style(ax, "Final bpc relative to base (negative = better than base)", "relative difference (%)", "")
    legend_below(fig, ax)
    save(fig, out / "final_differences.png")


def token_savings(cost: dict, out: Path):
    """Characters per token on each domain: grouped bars, base tokenizer and the two retrofits."""
    models = {"original": "base", "syllable-1000-final": "syllable-1000",
              "multisyllable-1000-final": "multisyllable-1000"}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    width = 0.26
    for k, (m, c) in enumerate(models.items()):
        xs = [j + (k - 1) * (width + 0.02) for j in range(len(DOMAINS))]
        vals = [cost[m][d] for d in DOMAINS]
        ax.bar(xs, vals, width=width, color=COLORS[c], label=LABELS[c], edgecolor=SURFACE, linewidth=1)
        for x, v in zip(xs, vals, strict=True):
            ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5,
                        color=INK2)
    ax.set_xticks(range(len(DOMAINS)), list(DOMAINS.values()))
    ax.set_ylim(0, 1.12 * max(cost[m][d] for m in models for d in DOMAINS))  # room for the value labels
    style(ax, "Characters per token (higher = fewer tokens for the same text)", "", "characters per token")
    legend_below(fig, ax)
    save(fig, out / "token_savings.png")


def training_curves(curves: dict, out: Path, window: int = 25):
    """Training bpc of every step, as a trailing mean over `window` steps (the raw per-step values are noisy)."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ends = {}
    for c, cur in curves.items():
        t = cur["train"]
        ys = [sum(r["train_bpc"] for r in t[max(0, i - window + 1):i + 1]) / len(t[max(0, i - window + 1):i + 1])
              for i in range(len(t))]
        xs = [r["tokens"] / 1e6 for r in t]
        ax.plot(xs, ys, color=COLORS[c], linestyle=STYLES[c], linewidth=1.5, label=LABELS[c])
        ends[c] = (xs[-1], ys[-1])
    style(ax, f"Training bpc (mean of the last {window} steps)", "training tokens (millions)", "bits per character")
    label_ends(ax, ends)
    legend_below(fig, ax)
    save(fig, out / "training_curves.png")


def pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def markdown(r: dict) -> str:
    """The tables behind every figure plus the benchmark and generation tables."""
    out = ["# Token-type experiment: results", "",
           "Generated by `vitok.cpt_figures` from `results.json`. Relative differences are bpc(A) / bpc(B) - 1;"
           " negative means A is better. Intervals: 95% paired bootstrap over documents (one training seed, so they"
           " do not cover run-to-run variation).", "", "## Primary comparison (final weights, macro mean over the"
           " four Vietnamese domains)", "", "| Comparison | Relative bpc difference | 95% interval | Decision |",
           "|---|---:|---|---|"]
    for name, p in r["primary"].items():
        m = p["macro"]
        out.append(f"| {name} | {pct(m['macro_rel_diff'])} | [{pct(m['ci95'][0])}, {pct(m['ci95'][1])}] | "
                   f"{p['decision']} |")
    out += ["", "## Per domain (final weights, relative to base)", "",
            "| Domain | syllable vs base | multisyllable vs base |", "|---|---|---|"]
    for d in (x for x in DOMAINS if x != "english"):
        cells = []
        for c in ("syllable-1000", "multisyllable-1000"):
            v = r["primary"][f"{c} vs base"]["per_domain"][d]
            cells.append(f"{pct(v['rel_diff'])} [{pct(v['rel_ci95'][0])}, {pct(v['rel_ci95'][1])}]")
        out.append(f"| {DOMAINS[d]} | {cells[0]} | {cells[1]} |")
    out += ["", "## English (final weights, relative to the original Qwen3)", "",
            "| Condition | Relative bpc difference | 95% interval |", "|---|---:|---|"]
    for name, m in r["english"].items():
        out.append(f"| {name} | {pct(m['macro_rel_diff'])} | [{pct(m['ci95'][0])}, {pct(m['ci95'][1])}] |")
    out += ["", "## bpc of every model", "", "| Model | " + " | ".join(DOMAINS.values()) + " |",
            "|---" * (len(DOMAINS) + 1) + "|"]
    for m, v in r["bpc"].items():
        out.append(f"| {m} | " + " | ".join(f"{v[d]['bpc']:.4f}" for d in DOMAINS) + " |")
    out += ["", "## Characters per token", "", "| Model | " + " | ".join(DOMAINS.values()) + " |",
            "|---" * (len(DOMAINS) + 1) + "|"]
    for m, v in r["cost"]["chars_per_token"].items():
        out.append(f"| {m} | " + " | ".join(f"{v[d]:.3f}" for d in DOMAINS) + " |")
    b = r["benchmarks"]
    sets = sorted(b["inclusion"])
    out += ["", "## Benchmarks (accuracy per answer character; inclusion decided on the original model before any"
            " trained model was scored)", "", "| Benchmark | Original | 95% lower bound | Chance | Included |",
            "|---|---:|---:|---:|---|"]
    for s in sets:
        v = b["inclusion"][s]
        out.append(f"| {s} | {v['original_acc']:.3f} | {v['lower95']:.3f} | {v['chance']:.3f} | "
                   f"{'yes' if v['included'] else 'no'} |")
    out += ["", "| Model | " + " | ".join(sets) + " |", "|---" * (len(sets) + 1) + "|"]
    for m, acc in b["accuracy"].items():
        out.append(f"| {m} | " + " | ".join(f"{acc[s]:.3f}" for s in sets) + " |")
    out += ["", "| Condition vs base (final) | " + " | ".join(f"{s} McNemar p" for s in sets) + " |",
            "|---" * (len(sets) + 1) + "|"]
    for c, v in b["mcnemar"].items():
        out.append(f"| {c} | " + " | ".join(f"{v[s]['p_value']:.3f}" for s in sets) + " |")
    out += ["", "## Generation (greedy, 256 tokens, 20 prompts, one GPU)", "",
            "| Model | Decode chars/s | Decode tokens/s | Prefill chars/s | Chars per generated token | "
            "4-syllable repetition | Peak GiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for m, g in r["generation"].items():
        out.append(f"| {m} | {g['decode_chars_per_second']:.1f} | {g['decode_tokens_per_second']:.1f} | "
                   f"{g['prefill_chars_per_second']:.0f} | {g['generated_chars_per_token']:.2f} | "
                   f"{100 * g['repetition_4_mean']:.1f}% | {g['max_memory_gib']:.2f} |")
    device = next(iter(r["generation"].values()))["device"]
    out += ["", f"Generation measured on {device}. Rates describe different generated text, not equally useful"
            " output.", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    r = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    learning_curves(r["curves"], args.out)
    domain_curves(r["curves"], args.out)
    final_differences(r["primary"], args.out)
    token_savings(r["cost"]["chars_per_token"], args.out)
    training_curves(r["curves"], args.out)
    (args.out / "report.md").write_text(markdown(r), encoding="utf-8")


if __name__ == "__main__":
    main()
