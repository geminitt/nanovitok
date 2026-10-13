"""Token-type report contracts on synthetic outputs with known answers: the primary comparison and its decision,
pairing refused across different documents or items, the benchmark inclusion rule, generation on one GPU, and
curves aligned to tokens and characters read."""

import json

import numpy as np
import pytest

from vitok import cpt_report

STEPS, SNAPS = 20, [10]
FACTOR = {"original": 1.10, "syllable-1000-zero": 1.40, "multisyllable-1000-zero": 1.50, "base-final": 1.00,
          "syllable-1000-final": 0.99, "multisyllable-1000-final": 0.98, "base-step_000010": 1.05,
          "syllable-1000-step_000010": 1.06, "multisyllable-1000-step_000010": 1.07}
CORRECT = {"easy": 0.6, "hard": 0.25}  # share right per item set; chance 0.25 for both


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build(tmp_path, digest_shift=None):
    rng = np.random.default_rng(0)
    ev, tr = tmp_path / "eval", tmp_path / "train"
    docs = {d: rng.integers(300, 2500, 60) for d in [*cpt_report.VI, "english"]}
    base_nats = {d: c * rng.uniform(0.6, 0.9, len(c)) for d, c in docs.items()}
    gold = rng.integers(0, 4, 400)
    for m, f in FACTOR.items():
        (ev / m).mkdir(parents=True)
        for d, c in docs.items():
            noise = rng.normal(1, 0.002, len(c)) if "step" in m else np.ones(len(c))
            jsonl(ev / m / f"lm_{d}.jsonl",
                  [{"i": i, "digest": f"{d}{i}" + ("x" if digest_shift == (m, d) else ""), "chars": int(c[i]),
                    "tokens": int(c[i] / (3.4 if "multi" not in m else 4.0)), "nats": float(base_nats[d][i] * f * n)}
                   for i, n in enumerate(noise)])
        if "step" not in m:
            for s, share in CORRECT.items():
                right = rng.random(400) < (share if m == "original" else 0.5)
                jsonl(ev / m / f"mc_{s}.jsonl", [{"i": i, "gold": int(gold[i]), "correct_char": bool(right[i])}
                                                 for i in range(400)])
            (ev / m / "summary.json").write_text(json.dumps({"device_name": "A100", "generation": {
                "decode_chars_per_second": 80.0 * (1.2 if "multi" in m else 1), "decode_tokens_per_second": 20.0,
                "prefill_chars_per_second": 8000.0, "generated_chars_per_token": 4.0, "repetition_4_mean": 0.5,
                "max_memory_gib": 3.0}}))
    for c in cpt_report.CONDITIONS:
        (tr / c / "train").mkdir(parents=True)
        cpt = 3.4 if "multi" not in c else 4.0
        jsonl(tr / c / "train" / "train_log.jsonl",
              [{"step": s, "tokens": 100 * (s + 1), "chars": int(100 * (s + 1) * cpt), "train_bpc": 1.5 - s / 100}
               for s in range(STEPS)])
        jsonl(tr / c / "train" / "dev_log.jsonl", [{"step": s, "bpc": 1.4 - s / 100} for s in (0, 10, 20)])
    return ev, tr


CHANCE = {"easy": 0.25, "hard": 0.25}


def test_the_primary_comparison_finds_the_known_differences(tmp_path):
    ev, tr = build(tmp_path)
    r = cpt_report.report(ev, tr, CHANCE, SNAPS)
    p = r["primary"]
    assert p["multisyllable-1000 vs base"]["macro"]["macro_rel_diff"] == pytest.approx(-0.02)
    assert p["multisyllable-1000 vs base"]["decision"] == "A better"
    assert p["multisyllable-1000 vs syllable-1000"]["macro"]["macro_rel_diff"] == pytest.approx(0.98 / 0.99 - 1)
    for d in cpt_report.VI:
        assert p["syllable-1000 vs base"]["per_domain"][d]["rel_diff"] == pytest.approx(-0.01)
    assert r["english"]["base-final vs original"]["macro_rel_diff"] == pytest.approx(1 / 1.1 - 1)
    cpt = r["cost"]["chars_per_token"]
    assert cpt["multisyllable-1000-final"]["fineweb2"] > cpt["original"]["fineweb2"]


def test_benchmarks_follow_the_inclusion_rule_on_original_and_compare_with_base(tmp_path):
    ev, tr = build(tmp_path)
    b = cpt_report.report(ev, tr, CHANCE, SNAPS)["benchmarks"]
    assert b["inclusion"]["easy"]["included"] and not b["inclusion"]["hard"]["included"]
    assert b["inclusion"]["easy"]["lower95"] > 0.25 >= b["inclusion"]["hard"]["lower95"]
    assert set(b["mcnemar"]) == {"syllable-1000", "multisyllable-1000"}
    assert "p_value" in b["mcnemar"]["syllable-1000"]["easy"]


def test_curves_put_each_point_at_the_tokens_and_characters_read(tmp_path):
    ev, tr = build(tmp_path)
    cur = cpt_report.report(ev, tr, CHANCE, SNAPS)["curves"]["multisyllable-1000"]
    assert sorted(cur["snapshots"]) == [0, 10, STEPS]
    assert cur["snapshots"][10]["tokens"] == 1000 and cur["snapshots"][10]["chars"] == 4000
    assert cur["snapshots"][0]["tokens"] == 0
    assert [d["tokens"] for d in cur["dev"]] == [0, 1000, 2000] and cur["train"][0]["step"] == 1


def test_comparisons_refuse_different_documents_and_mixed_gpus(tmp_path):
    ev, tr = build(tmp_path, digest_shift=("syllable-1000-final", "legal"))
    with pytest.raises(SystemExit, match="different legal documents"):
        cpt_report.report(ev, tr, CHANCE, SNAPS)
    ev2, tr2 = build(tmp_path / "b")
    s = json.loads((ev2 / "original" / "summary.json").read_text())
    (ev2 / "original" / "summary.json").write_text(json.dumps(s | {"device_name": "T4"}))
    with pytest.raises(SystemExit, match="different GPUs"):
        cpt_report.report(ev2, tr2, CHANCE, SNAPS)


def test_figures_and_tables_render_from_the_results(tmp_path):
    from vitok import cpt_figures
    ev, tr = build(tmp_path)
    (tmp_path / "results.json").write_text(json.dumps(cpt_report.report(ev, tr, CHANCE, SNAPS)))
    cpt_figures.main(["--results", str(tmp_path / "results.json"), "--out", str(tmp_path / "out")])
    for name in ("learning_curves", "domain_curves", "final_differences", "token_savings", "training_curves"):
        assert (tmp_path / "out" / f"{name}.png").stat().st_size > 10_000, name
    md = (tmp_path / "out" / "report.md").read_text()
    assert "| multisyllable-1000 vs base | -2.00% |" in md and "| easy | " in md and "A better" in md


def test_the_token_savings_chart_names_every_domain(tmp_path, monkeypatch):
    from vitok import cpt_figures
    ev, tr = build(tmp_path)
    cost = cpt_report.report(ev, tr, CHANCE, SNAPS)["cost"]["chars_per_token"]
    figs = []
    monkeypatch.setattr(cpt_figures, "save", lambda fig, path: figs.append(fig))
    cpt_figures.token_savings(cost, tmp_path)
    labels = [t.get_text() for t in figs[0].axes[0].get_xticklabels()]
    assert labels == list(cpt_figures.DOMAINS.values())
