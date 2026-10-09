"""Compare conditions scored by vitok.score against a reference condition, pairing on the same items.

    python -m vitok.compare --reference base=runs/.../score-base-final \
        --runs multisyllable=runs/.../score-multisyllable-final syllable=runs/.../score-syllable-final \
        --out runs/.../comparison.json

For each run against the reference: Vietnamese and English bpc with a paired bootstrap 95% interval of the
difference and of the relative difference (documents resampled; both sides scored on the same documents), Belebele
accuracy with an exact McNemar test on the paired answers, and the generation measurements as ratios
(run / reference). It refuses to compare scores of different documents or items.
"""

import argparse
import json
import math
from pathlib import Path

from vitok import stats


def _records(folder: Path, part: str) -> list[dict] | None:
    path = folder / f"{part}.jsonl"
    if not path.exists():
        return None
    recs = sorted((json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()), key=lambda r: r["i"])
    if [r["i"] for r in recs] != list(range(len(recs))):
        raise SystemExit(f"{path} is incomplete or has repeated items; finish the scoring run first")
    return recs


def compare_docs(run: list[dict], ref: list[dict], seed: int = 0) -> dict:
    if [(r["digest"], r["chars"]) for r in run] != [(r["digest"], r["chars"]) for r in ref]:
        raise SystemExit("the two scores are on different documents")
    nats_a, nats_b, chars = [r["nats"] for r in run], [r["nats"] for r in ref], [r["chars"] for r in ref]
    boot = stats.paired_bootstrap_bpc(nats_a, nats_b, chars, seed=seed)
    return {"bpc": stats.bpc(nats_a, chars), "bpc_reference": stats.bpc(nats_b, chars),
            "chars_per_token": sum(chars) / sum(r["tokens"] for r in run),
            "chars_per_token_reference": sum(chars) / sum(r["tokens"] for r in ref), **boot}


def compare_belebele(run: list[dict], ref: list[dict]) -> dict:
    if [r["gold"] for r in run] != [r["gold"] for r in ref]:
        raise SystemExit("the two Belebele scores are on different items")
    a, b = [r["correct_char"] for r in run], [r["correct_char"] for r in ref]
    return {"items": len(a), "acc_char": sum(a) / len(a), "acc_char_reference": sum(b) / len(b),
            "acc_sum": sum(r["correct_sum"] for r in run) / len(a),
            "acc_sum_reference": sum(r["correct_sum"] for r in ref) / len(b), "mcnemar_char": stats.mcnemar_exact(a, b)}


def compare_generation(run: dict, ref: dict) -> dict:
    keys = ["decode_chars_per_second", "decode_tokens_per_second", "prefill_chars_per_second",
            "generated_chars_per_token", "repetition_4_mean", "max_memory_gib"]
    out = {}
    for k in keys:
        a, b = run.get(k), ref.get(k)
        out[k] = a
        out[k + "_reference"] = b
        out[k + "_ratio"] = a / b if a is not None and b else None
    return out


def compare(run_dir: Path, ref_dir: Path) -> dict:
    run_sum = json.loads((run_dir / "summary.json").read_text())
    ref_sum = json.loads((ref_dir / "summary.json").read_text())
    if "generation" in run_sum and run_sum.get("device_name") != ref_sum.get("device_name"):
        raise SystemExit(f"generation speed measured on {run_sum.get('device_name')} and "
                         f"{ref_sum.get('device_name')} cannot be compared")
    out = {"snapshot": run_sum.get("snapshot"), "snapshot_reference": ref_sum.get("snapshot")}
    for part, key in (("vi_docs", "vi"), ("en_docs", "en")):
        a, b = _records(run_dir, part), _records(ref_dir, part)
        if a is not None and b is not None:
            out[key] = compare_docs(a, b)
    a, b = _records(run_dir, "belebele"), _records(ref_dir, "belebele")
    if a is not None and b is not None:
        out["belebele"] = compare_belebele(a, b)
    if "generation" in run_sum and "generation" in ref_sum:
        out["generation"] = compare_generation(run_sum["generation"], ref_sum["generation"])
    for v in out.get("vi", {}), out.get("en", {}):
        assert all(math.isfinite(x) for x in v.values() if isinstance(x, float))
    return out


def _named(spec: str) -> tuple[str, Path]:
    name, sep, path = spec.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected name=folder, got {spec!r}")
    return name, Path(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=_named, required=True, help="name=score folder of the reference condition")
    ap.add_argument("--runs", type=_named, nargs="+", required=True, help="name=score folder of each compared run")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ref_name, ref_dir = args.reference
    result = {"reference": ref_name, "runs": {name: compare(d, ref_dir) for name, d in args.runs}}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
