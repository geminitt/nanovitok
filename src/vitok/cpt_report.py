"""Results of the token-type experiment (PLAN.md, Evaluation), from the item-level files the runs wrote.

    python -m vitok.cpt_report --eval runs/.../eval/out --train runs/.../full/work --out results/token-type

Reads every evaluation folder (`vitok.evaluate` jobs: `<condition>-zero`, `<condition>-final`, `<condition>-step_N`,
`original`) and every training folder (train/dev logs, manifests), and writes `results.json`:
- primary: for each pair of trained conditions, the macro mean over the Vietnamese domains of the relative bpc
  difference, its stratified paired-bootstrap interval and the plan's decision (one seed: one interval);
- per-domain and English relative differences with paired-bootstrap intervals;
- benchmarks: accuracy per model, the inclusion rule decided on `original` (Clopper-Pearson lower bound above chance)
  and exact McNemar of each condition against base;
- cost: characters per token per domain, generation rates and repetition;
- curves: training bpc per step, dev bpc, and the snapshots' bpc per domain, against tokens and characters read.
Pairing is checked: two models are compared only on the same documents (digest and length) or items (gold).
"""

import argparse
import json
from pathlib import Path

from vitok import cpt_data, stats

VI = ["fineweb2", "wikipedia", "legal", "flores"]
CONDITIONS = ["base", "syllable-1000", "multisyllable-1000"]
PAIRS = [("syllable-1000", "base"), ("multisyllable-1000", "base"), ("multisyllable-1000", "syllable-1000")]


def records(folder: Path, part: str) -> list[dict]:
    rows = cpt_data.jsonl_rows(folder / f"{part}.jsonl")
    rows.sort(key=lambda r: r["i"])
    if [r["i"] for r in rows] != list(range(len(rows))):
        raise SystemExit(f"{folder / part}.jsonl is incomplete")
    return rows


def paired_nats(a: Path, b: Path, domain: str) -> tuple[list[float], list[float], list[int]]:
    ra, rb = records(a, f"lm_{domain}"), records(b, f"lm_{domain}")
    if [(r["digest"], r["chars"]) for r in ra] != [(r["digest"], r["chars"]) for r in rb]:
        raise SystemExit(f"{a.name} and {b.name} scored different {domain} documents")
    return [r["nats"] for r in ra], [r["nats"] for r in rb], [r["chars"] for r in ra]


def lm_summary(folder: Path, domain: str) -> dict:
    rows = records(folder, f"lm_{domain}")
    chars, tokens = sum(r["chars"] for r in rows), sum(r["tokens"] for r in rows)
    return {"bpc": stats.bpc([r["nats"] for r in rows], [r["chars"] for r in rows]), "chars_per_token": chars / tokens}


def compare(a: Path, b: Path, domains: list[str], seed: int = 0) -> dict:
    """Relative bpc of model a against model b: the macro mean over `domains` and each domain alone."""
    nats = {d: paired_nats(a, b, d) for d in domains}
    macro = stats.stratified_relative_bootstrap({d: (x, y) for d, (x, y, _) in nats.items()}, seed=seed)
    per = {d: stats.paired_bootstrap_bpc(x, y, c, seed=seed) for d, (x, y, c) in nats.items()}
    return {"macro": macro, "decision": stats.decide([tuple(macro["ci95"])]),
            "per_domain": {d: {"rel_diff": v["rel_diff"], "rel_ci95": v["rel_ci95"]} for d, v in per.items()}}


def benchmarks(eval_dir: Path, models: list[str], sets: list[str], chance: dict[str, float]) -> dict:
    out: dict = {"accuracy": {}, "inclusion": {}, "mcnemar": {}}
    for s in sets:
        orig = records(eval_dir / "original", f"mc_{s}")
        k = sum(r["correct_char"] for r in orig)
        lower = stats.binomial_lower(k, len(orig))
        out["inclusion"][s] = {"original_acc": k / len(orig), "lower95": lower, "chance": chance[s],
                               "included": lower > chance[s]}
        rows = {m: records(eval_dir / m, f"mc_{s}") for m in models}
        gold = [r["gold"] for r in orig]
        for m, rs in rows.items():
            if [r["gold"] for r in rs] != gold:
                raise SystemExit(f"{m} scored other {s} items than original")
            out["accuracy"].setdefault(m, {})[s] = sum(r["correct_char"] for r in rs) / len(rs)
        base = [r["correct_char"] for r in rows["base-final"]]
        for c in CONDITIONS[1:]:
            out["mcnemar"].setdefault(c, {})[s] = stats.mcnemar_exact([r["correct_char"] for r in rows[f"{c}-final"]],
                                                                      base)
    return out


def generation(eval_dir: Path, models: list[str]) -> dict:
    out = {}
    for m in models:
        s = json.loads((eval_dir / m / "summary.json").read_text())
        g = s["generation"]
        out[m] = {k: g[k] for k in ("decode_chars_per_second", "decode_tokens_per_second", "prefill_chars_per_second",
                                    "generated_chars_per_token", "repetition_4_mean", "max_memory_gib")}
        out[m]["device"] = s["device_name"]
    if len({v["device"] for v in out.values()}) != 1:
        raise SystemExit("generation was measured on different GPUs")
    return out


def curves(train_dir: Path, eval_dir: Path, steps_per_snapshot: list[int]) -> dict:
    out: dict = {}
    for c in CONDITIONS:
        t = cpt_data.jsonl_rows(train_dir / c / "train" / "train_log.jsonl")
        dev = cpt_data.jsonl_rows(train_dir / c / "train" / "dev_log.jsonl")
        by_step = {r["step"] + 1: r for r in t}  # a log record's step is zero-based: tokens after step + 1 steps
        snaps = {}
        start = eval_dir / ("original" if c == "base" else f"{c}-zero")
        points = [(0, start), *((s, eval_dir / f"{c}-step_{s:06d}") for s in steps_per_snapshot),
                  (max(by_step), eval_dir / f"{c}-final")]
        for step, folder in points:
            snaps[step] = {"tokens": by_step[step]["tokens"] if step else 0,
                           "chars": by_step[step]["chars"] if step else 0,
                           **{d: lm_summary(folder, d)["bpc"] for d in [*VI, "english"]}}
        out[c] = {"train": [{"step": r["step"] + 1, "tokens": r["tokens"], "chars": r["chars"],
                             "train_bpc": r["train_bpc"]} for r in t],
                  "dev": [{"step": r["step"], "bpc": r["bpc"],
                           "tokens": by_step[r["step"]]["tokens"] if r["step"] else 0,
                           "chars": by_step[r["step"]]["chars"] if r["step"] else 0} for r in dev],
                  "snapshots": snaps}
    return out


def report(eval_dir: Path, train_dir: Path, chance: dict[str, float], snapshot_steps: list[int]) -> dict:
    finals = {c: eval_dir / f"{c}-final" for c in CONDITIONS}
    models = ["original", *(f"{c}-zero" for c in CONDITIONS[1:]), *(f"{c}-final" for c in CONDITIONS)]
    out: dict = {"primary": {}, "english": {}, "bpc": {}, "cost": {}}
    for a, b in PAIRS:
        out["primary"][f"{a} vs {b}"] = compare(finals[a], finals[b], VI)
    for c in CONDITIONS:
        out["english"][f"{c}-final vs original"] = compare(finals[c], eval_dir / "original", ["english"])["macro"]
    for m in models:
        out["bpc"][m] = {d: lm_summary(eval_dir / m, d) for d in [*VI, "english"]}
    out["cost"]["chars_per_token"] = {m: {d: v["chars_per_token"] for d, v in out["bpc"][m].items()}
                                      for m in ("original", *(f"{c}-final" for c in CONDITIONS[1:]))}
    out["benchmarks"] = benchmarks(eval_dir, models, sorted(chance), chance)
    out["generation"] = generation(eval_dir, models)
    out["curves"] = curves(train_dir, eval_dir, snapshot_steps)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", type=Path, required=True, help="the evaluation jobs' output folders")
    ap.add_argument("--train", type=Path, required=True, help="the full runs' work folder")
    ap.add_argument("--evalsets", type=Path, required=True, help="vitok.evalsets folder (chance levels)")
    ap.add_argument("--snapshot-steps", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    man = json.loads((args.evalsets / "evalsets_manifest.json").read_text())
    chance = {n: m["chance"] for n, m in man.items() if m.get("kind") == "mc"}
    result = report(args.eval, args.train, chance, args.snapshot_steps)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["primary"], indent=2))


if __name__ == "__main__":
    main()
