import json
import math

import pytest

from vitok.analysis import (
    comparisons,
    load,
    noise_threshold,
    nonembedding_params,
    scaling,
    val_sensitivity,
    verdicts,
)

VARIANTS = ("clean",)
# (depth, transformer_matrices) as printed by nanochat in runs/*/train.log
LOGGED_PARAMS = [(6, 10_616_940), (8, 25_166_016)]


def fake_run(path, cond, depth, seed, bpc, n_docs=20):
    """A result file whose clean bpc is exactly `bpc` (1 nat-per-char unit = ln2 bits)."""
    chars = [100 + i for i in range(n_docs)]
    docs = {v: {"chars": chars, "nats": [bpc * math.log(2) * c for c in chars]} for v in VARIANTS}
    path.write_text(json.dumps({
        "step": 1, "max_seq_len": 1024,
        "user_config": {"depth": depth, "aspect_ratio": 64, "head_dim": 128},
        "doc_ids": list(range(n_docs)), "docs": docs,
        "pairs": {"ids": [0], "good_nats": [1.0], "bad_nats": [2.0]},
    }), encoding="utf-8")


@pytest.mark.parametrize("depth,logged", LOGGED_PARAMS)
def test_nonembedding_params_matches_nanochat(depth, logged):
    got = nonembedding_params({"depth": depth, "aspect_ratio": 64, "head_dim": 128})
    assert abs(got - logged) / logged < 1e-4


def test_scaling_reports_effect_and_seed_bar(tmp_path):
    results, figures = tmp_path / "results", tmp_path / "figures"
    results.mkdir()
    #            d6: super better by 0.002        d8: super worse by 0.002
    for cond, depth, seed, value in [("bpe-nfc", 6, 0, 1.0000), ("super-nfc", 6, 0, 0.9980),
                                     ("bpe-nfc", 6, 1, 1.0005), ("super-nfc", 6, 1, 0.9985),
                                     ("bpe-nfc", 8, 0, 0.9000), ("super-nfc", 8, 0, 0.9020)]:
        fake_run(results / f"{cond}_d{depth}_s{seed}.json", cond, depth, seed, value)

    text = scaling(load(results), figures)
    assert "| super-nfc − bpe-nfc | -0.0020 (s1: -0.0020) | +0.0020 |" in text
    assert "| d6 | 10,616,832 |" in text and "| d8 | 25,165,824 |" in text
    assert (figures / "h3_scaling.png").exists()


@pytest.mark.parametrize("worse,verdict", [(1.005, "met"), (1.02, "not met")])
def test_h1_is_a_non_inferiority_test(tmp_path, worse, verdict):
    # SuperBPE 0.5% worse passes the pre-registered 1% margin; 2% worse fails it
    for cond, value in [("bpe-nfc", 1.0), ("super-nfc", worse)]:
        fake_run(tmp_path / f"{cond}_d6_s0.json", cond, 6, 0, value)
    runs = load(tmp_path)
    text = verdicts(comparisons(runs), {"nfc": 0.18}, runs)
    assert f"- NFC pilot criterion: **{verdict}**." in text


def test_val_sensitivity_flags_a_sign_change(tmp_path):
    test_dir, val_dir = tmp_path / "test", tmp_path / "val"
    test_dir.mkdir()
    val_dir.mkdir()
    for cond, test_bpc, val_bpc in [("bpe-nfc", 1.000, 1.000), ("super-nfc", 1.002, 0.998)]:
        fake_run(test_dir / f"{cond}_d8_s0.json", cond, 8, 0, test_bpc)
        fake_run(val_dir / f"{cond}_d8_s0.json", cond, 8, 0, val_bpc)
    text = val_sensitivity(load(val_dir), comparisons(load(test_dir)))
    assert "| super-nfc − bpe-nfc | d8 | 20 | -0.0020 |" in text and text.rstrip().endswith("| no |")


def test_val_docs_follow_the_test_set_rule(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from vitok.val_docs import val_docs

    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"text": ["ngắn", "chữ " * 1000]}), shard)
    docs = val_docs(shard)
    assert len(docs) == 1 and len(docs[0]["text"]) <= 2500 and not docs[0]["text"].endswith(" ")


def _break(kind, a: dict, b: dict):
    """Make result b inconsistent with result a in one way the analysis must refuse."""
    if kind == "doc ids":
        b["doc_ids"] = b["doc_ids"][::-1]
    elif kind == "chars":
        b["docs"]["clean"]["chars"][0] += 1
    elif kind == "nothing scored by both":
        a["docs"]["clean"]["nats"] = [None] * len(a["docs"]["clean"]["nats"])


def test_result_files_must_be_one_per_run(tmp_path):
    for sub in ("test", "val"):
        (tmp_path / sub).mkdir()
        fake_run(tmp_path / sub / "bpe-nfc_d6_s0.json", "bpe-nfc", 6, 0, 1.0)
    with pytest.raises(ValueError, match="two result files"):
        load(tmp_path)


@pytest.mark.parametrize("kind", ["doc ids", "chars", "nothing scored by both"])
def test_paired_comparison_refuses_runs_that_cannot_be_paired(tmp_path, kind):
    # pairing indexes every run by position: same documents, same order, same lengths, at least one in common
    for cond in ("bpe-nfc", "super-nfc"):
        fake_run(tmp_path / f"{cond}_d6_s0.json", cond, 6, 0, 1.0)
    runs = load(tmp_path)
    _break(kind, runs[("bpe-nfc", 6, 0)], runs[("super-nfc", 6, 0)])
    message = "no document was scored" if kind == "nothing scored by both" else "cannot be paired"
    with pytest.raises(ValueError, match=message):
        comparisons(runs)


def test_noise_threshold_is_a_t_test_on_the_seed_spreads():
    # RMS of the spreads estimates the noise of a difference; two spreads -> t with 2 degrees of freedom
    assert noise_threshold([0.003, -0.004]) == pytest.approx(4.303 * math.sqrt((0.003 ** 2 + 0.004 ** 2) / 2), rel=1e-3)
    assert noise_threshold([]) is None
    assert noise_threshold([0.0, 0.0]) == 0.0


def test_published_summary_regenerates_from_the_committed_results(tmp_path, monkeypatch, capsys):
    """results/summary.md is a pure function of the committed result files (CI checks the same from the shell);
    in the suite it also lets mutation testing see every number of the published analysis."""
    import sys

    from conftest import REPO

    from vitok import analysis

    out = tmp_path / "summary.md"
    monkeypatch.setattr(sys, "argv", [
        "analysis", "--results", str(REPO / "kaggle" / "outputs"),
        "--compression", str(REPO / "kaggle" / "outputs" / "vitok-data" / "compression-16k.json"),
        "--val-results", str(REPO / "results" / "val"),
        "--out", str(out), "--figures", str(tmp_path / "figures")])
    analysis.main()
    capsys.readouterr()
    lines = lambda text: [line for line in text.splitlines() if not line.startswith("Figure")]
    committed = (REPO / "results" / "summary.md").read_text(encoding="utf-8")
    assert lines(out.read_text(encoding="utf-8")) == lines(committed)
    assert (tmp_path / "figures" / "h3_scaling.png").stat().st_size > 0
    import numpy as np
    from matplotlib.image import imread

    assert np.array_equal(imread(tmp_path / "figures/h3_scaling.png"), imread(REPO / "figures/h3_scaling.png"))


def _row(depth, a, b, variant, hyp, diff, rel_hi, significant=True, threshold=0.001):
    return {"depth": depth, "a": a, "b": b, "variant": variant, "hyp": hyp, "diff": diff, "ci95": [diff, diff],
            "significant": significant, "rel_diff": diff, "rel_ci95": [diff, rel_hi], "seed_spreads": [],
            "noise_threshold": threshold}


@pytest.mark.parametrize("reduction,upper,expected", [
    (0.18, 0.005, "met"), (0.10, 0.005, "not met"), (0.18, 0.012, "not met"),
    (0.18, 0.01, "not met"), (0.15, 0.005, "met"), (None, 0.005, "not met"),
])
def test_verdict_rules(reduction, upper, expected):
    rows = [_row(6, "super-nfc", "bpe-nfc", "clean", "H1", 0.001, upper)]
    text = verdicts(rows, {"nfc": reduction} if reduction is not None else {}, {})
    assert f"- NFC pilot criterion: **{expected}**." in text
    assert f"→ not worse by more than 1%: {'yes' if upper < 0.01 else 'no'}" in text


@pytest.mark.parametrize("uppers", [[], [0.005, 0.02], [0.02, 0.005]])
def test_pilot_margin_requires_every_measured_depth(uppers):
    rows = [_row(depth, "super-nfc", "bpe-nfc", "clean", "H1", 0.001, upper)
            for depth, upper in enumerate(uppers, start=6)]
    assert "- NFC pilot criterion: **not met**." in verdicts(rows, {"nfc": 0.18}, {})


@pytest.mark.parametrize("diff,threshold,significant,expected", [
    (-0.01, 0.001, True, "A lower"), (0.01, 0.001, True, "A higher"), (0.01, 0.02, True, "not resolved"),
    (0.01, 0.001, False, "not resolved"), (0.01, None, True, "A higher"),
    (0.001, 0.001, True, "not resolved")])
def test_direction_needs_both_the_interval_and_the_noise_test(diff, threshold, significant, expected):
    from vitok.analysis import beyond_noise, direction
    row = _row(6, "a", "b", "clean", "H2", diff, diff, significant=significant, threshold=threshold)
    assert direction(row) == expected
    assert beyond_noise(row) == ("no second seed" if threshold is None else ("yes" if abs(diff) > threshold else "no"))


def test_comparisons_and_val_sensitivity_skip_missing_runs_and_flag_sign_changes(tmp_path):
    from vitok.analysis import delta
    test_dir, val_dir = tmp_path / "test", tmp_path / "val"
    test_dir.mkdir()
    val_dir.mkdir()
    # d6 has only BPE runs (every comparison skipped), d8 has the H1 pair; the val shard agrees at d8
    for cond, depth, t, v in [("bpe-nfc", 6, 1.0, 1.0),
                              ("bpe-nfc", 8, 1.0, 1.0), ("super-nfc", 8, 1.002, 1.003)]:
        fake_run(test_dir / f"{cond}_d{depth}_s0.json", cond, depth, 0, t)
        fake_run(val_dir / f"{cond}_d{depth}_s0.json", cond, depth, 0, v)
    rows = comparisons(load(test_dir))
    assert {(r["depth"], r["a"], r["b"], r["variant"]) for r in rows} == {(8, "super-nfc", "bpe-nfc", "clean")}
    text = val_sensitivity(load(val_dir), rows)
    assert "| super-nfc − bpe-nfc | d8 | 20 | +0.0030 |" in text and text.rstrip().endswith("| yes |")
    no_test = val_sensitivity(load(val_dir), [])
    assert no_test.rstrip().endswith("| — | — |")
    runs = load(test_dir)
    assert delta(runs, "super-nfc", "bpe-nfc", 8, 0) == pytest.approx(0.002)
    assert delta(runs, "super-nfc", "bpe-nfc", 6, 0) is None


def test_verdicts_list_every_seed_of_each_depth(tmp_path):
    # d6 has seeds 0 and 1, d8 only seed 0: each depth lists its own seeds.
    for cond, depth, seed, value in [("bpe-nfc", 6, 0, 1.0), ("super-nfc", 6, 0, 0.998), ("bpe-nfc", 6, 1, 1.0),
                                     ("super-nfc", 6, 1, 0.999), ("bpe-nfc", 8, 0, 0.9), ("super-nfc", 8, 0, 0.902)]:
        fake_run(tmp_path / f"{cond}_d{depth}_s{seed}.json", cond, depth, seed, value)
    runs = load(tmp_path)
    text = verdicts(comparisons(runs), {"nfc": 0.18}, runs)
    assert "- d6: s0 -0.0020, s1 -0.0010\n" in text and "- d8: s0 +0.0020\n" in text


def test_scaling_uses_each_depths_common_documents_and_handles_missing_conditions(tmp_path):
    for cond, depth in (("bpe-nfc", 6), ("super-nfc", 6), ("bpe-nfc", 8)):
        fake_run(tmp_path / f"{cond}_d{depth}_s0.json", cond, depth, 0, 1.0)
    path = tmp_path / "super-nfc_d6_s0.json"
    row = json.loads(path.read_text())
    row["docs"]["clean"]["nats"][-1] = None
    path.write_text(json.dumps(row))
    text = scaling(load(tmp_path))
    assert "| super-nfc − bpe-nfc | +0.0000 | — |" in text
    assert "| d8 | 25,165,824 | 1.0000 | — |" in text
