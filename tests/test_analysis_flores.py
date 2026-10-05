import json
import math

import pytest

from vitok.analysis import (comparisons, h2_decision, load, noise_threshold, nonembedding_params, scaling,
                            val_sensitivity, verdicts)
from vitok.flores_ctc import calibrate, ctc

VARIANTS = ("clean", "strip50", "strip100")
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


def test_flores_ctc_counts_lines(tokenizers_dir):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizers_dir / "bpe-nfc" / "tokenizer.json"))
    lines = ["Hà Nội là thủ đô.", "Học sinh đi học."]
    assert ctc(tok, lines) == sum(len(tok.encode(l, add_special_tokens=False).ids) for l in lines)


def test_flores_calibrate_picks_closest_split(tokenizers_dir):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizers_dir / "bpe-nfc" / "tokenizer.json"))
    short, long = ["Hà Nội."], ["Hà Nội."] * 50
    cal = calibrate({"dev": short, "devtest": long}, tok, published=ctc(tok, long))
    assert cal["split"] == "devtest" and cal["gap"] == 0


def test_syllable_runs_and_superword_spans(tokenizers_dir):
    from tokenizers import Tokenizer

    from vitok.wordhood import superword_spans, syllable_runs

    text = "Hà Nội, học sinh"
    # "Hà Nội" and "học sinh" are runs of 2; the comma breaks the run between them
    assert [text[a:b] for a, b in syllable_runs(text, 2)] == ["Hà Nội", "học sinh"]
    assert syllable_runs(text, 3) == []

    tok = Tokenizer.from_file(str(tokenizers_dir / "super-nfc" / "tokenizer.json"))
    for start, end in superword_spans(tok, text):
        piece = text[start:end]
        assert piece == piece.strip() and " " in piece  # trimmed, and really more than one syllable


@pytest.mark.parametrize("worse,verdict", [(1.005, "supported"), (1.02, "not supported")])
def test_h1_is_a_non_inferiority_test(tmp_path, worse, verdict):
    # SuperBPE 0.5% worse passes the pre-registered 1% margin; 2% worse fails it
    for cond, value in [("bpe-nfc", 1.0), ("super-nfc", worse), ("bpe-nfd", 1.0), ("super-nfd", worse)]:
        fake_run(tmp_path / f"{cond}_d6_s0.json", cond, 6, 0, value)
    runs = load(tmp_path)
    text = verdicts(comparisons(runs), {"nfc": 0.18, "nfd": 0.18}, runs)
    assert f"- Verdict: **{verdict}**." in text.split("**H2**")[0]


def test_frequency_matched_baseline_uses_the_most_frequent_runs():
    import collections

    from vitok.wordhood import frequency_matched_rate

    count = collections.Counter({"học sinh": 10, "của các": 8, "rất xa": 1})
    hit = collections.Counter({"học sinh": 10, "của các": 0, "rất xa": 1})
    assert frequency_matched_rate(count, hit, 2) == pytest.approx(10 / 18)  # the rare run is left out
    assert frequency_matched_rate(count, hit, 3) == pytest.approx(11 / 19)


def test_val_sensitivity_flags_a_sign_change(tmp_path):
    test_dir, val_dir = tmp_path / "test", tmp_path / "val"
    test_dir.mkdir(); val_dir.mkdir()
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
    with pytest.raises(ValueError, match="no document was scored" if kind == "nothing scored by both" else "cannot be paired"):
        comparisons(runs)


def test_noise_threshold_is_a_t_test_on_the_seed_spreads():
    # RMS of the spreads estimates the noise of a difference; two spreads -> t with 2 degrees of freedom
    assert noise_threshold([0.003, -0.004]) == pytest.approx(4.303 * math.sqrt((0.003 ** 2 + 0.004 ** 2) / 2), rel=1e-3)
    assert noise_threshold([]) is None


def test_h2_needs_the_effect_at_every_size():
    row = lambda depth: {"depth": depth, "a": "bpe-nfd", "variant": "strip50", "rel_ci95": [0.0, 0.001]}
    cost = [row(6), row(8)]
    one_size = h2_decision([row(6), row(8)], ["A lower", "not resolved"], cost)
    assert one_size.startswith("not supported (1 of 2")
    assert h2_decision([row(6), row(8)], ["A lower", "A lower"], cost) == "supported"


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
        "--wordhood", str(REPO / "results" / "wordhood_16k.json"), "--val-results", str(REPO / "results" / "val"),
        "--out", str(out), "--figures", str(tmp_path / "figures")])
    analysis.main()
    capsys.readouterr()
    lines = lambda text: [line for line in text.splitlines() if not line.startswith("Figure")]
    assert lines(out.read_text(encoding="utf-8")) == lines((REPO / "results" / "summary.md").read_text(encoding="utf-8"))
    assert (tmp_path / "figures" / "h3_scaling.png").stat().st_size > 0


def _row(depth, a, b, variant, hyp, diff, rel_hi, significant=True, threshold=0.001):
    return {"depth": depth, "a": a, "b": b, "variant": variant, "hyp": hyp, "diff": diff, "ci95": [diff, diff],
            "significant": significant, "rel_diff": diff, "rel_ci95": [diff, rel_hi], "seed_spreads": [],
            "noise_threshold": threshold}


@pytest.mark.parametrize("rows,reductions,expect", [
    # H1 needs the token reduction on both normalizations AND every relative upper bound under the 1% margin
    ([_row(6, "super-nfc", "bpe-nfc", "clean", "H1", 0.001, 0.005)], {"nfc": 0.18, "nfd": 0.18}, ["H1 supported"]),
    ([_row(6, "super-nfc", "bpe-nfc", "clean", "H1", 0.001, 0.005)], {"nfc": 0.18, "nfd": 0.10}, ["H1 not supported", "(threshold 15%): not met."]),
    ([_row(6, "super-nfc", "bpe-nfc", "clean", "H1", 0.008, 0.012)], {"nfc": 0.18, "nfd": 0.18},
     ["H1 not supported", "not worse by more than 1%: no "]),
    ([_row(6, "super-nfc", "bpe-nfc", "clean", "H1", 0.001, 0.005)], {}, ["H1 not supported", "NFC nan%"]),
    # H2: NFD lower at every depth with the cost inside the margin -> supported; higher everywhere -> contradicted
    ([_row(6, "bpe-nfd", "bpe-nfc", "strip50", "H2", -0.01, -0.005), _row(6, "bpe-nfd", "bpe-nfc", "clean", "H2 cost", 0.001, 0.002),
      _row(8, "bpe-nfd", "bpe-nfc", "strip50", "H2", -0.01, -0.005), _row(8, "bpe-nfd", "bpe-nfc", "clean", "H2 cost", 0.001, 0.002)],
     {"nfc": 0.18, "nfd": 0.18}, ["H2 supported", "NFD better"]),
    ([_row(6, "bpe-nfd", "bpe-nfc", "strip50", "H2", -0.01, -0.005), _row(6, "bpe-nfd", "bpe-nfc", "clean", "H2 cost", 0.01, 0.02)],
     {"nfc": 0.18, "nfd": 0.18}, ["H2 not supported (1 of 1", "within 1%: no"]),
    ([_row(6, "bpe-nfd", "bpe-nfc", "strip50", "H2", 0.01, 0.02), _row(8, "bpe-nfd", "bpe-nfc", "strip50", "H2", 0.01, 0.02)],
     {"nfc": 0.18, "nfd": 0.18}, ["H2 contradicted (NFD worse at every size)", "strip50: +0.0100 → NFD worse"]),
    # one depth with NFD both better and worse is neither "lower everywhere" nor "higher everywhere"
    ([_row(6, "bpe-nfd", "bpe-nfc", "strip50", "H2", -0.01, -0.005), _row(6, "bpe-nfd", "bpe-nfc", "strip100", "H2", 0.01, 0.02)],
     {"nfc": 0.18, "nfd": 0.18},
     ["H2 not supported (2 of 2 stripped-text comparisons resolved: d6 bpe strip50 (NFD better); d6 bpe strip100 (NFD worse), not at every size)"]),
    ([_row(6, "bpe-nfd", "bpe-nfc", "strip100", "H2", 0.01, 0.02), _row(6, "bpe-nfd", "bpe-nfc", "strip50", "H2", -0.01, -0.005)],
     {"nfc": 0.18, "nfd": 0.18}, ["H2 not supported (2 of 2"]),
    ([_row(6, "bpe-nfd", "bpe-nfc", "strip50", "H2", 0.0001, 0.02, significant=True, threshold=0.01)],
     {"nfc": 0.18, "nfd": 0.18}, ["H2 not supported (0 of 1 stripped-text comparisons resolved, not at every size)"]),
])
def test_verdict_rules(rows, reductions, expect):
    text = verdicts(rows, reductions, {}).replace("**", "")
    flat = " ".join(text.split()) + " "
    for needle in expect:
        hyp, _, rest = needle.partition(" ")
        if hyp in ("H1", "H2") and rest.split(" ")[0] in ("supported", "not", "contradicted"):
            section = flat.split(f"{hyp} —")[1].split("H2 —")[0] if hyp == "H1" else flat.split("H2 —")[1].split("H3 —")[0]
            assert f"Verdict: {rest}" in section, (needle, section)
        else:
            assert needle in flat, (needle, flat)


@pytest.mark.parametrize("diff,threshold,significant,expected", [
    (-0.01, 0.001, True, "A lower"), (0.01, 0.001, True, "A higher"), (0.01, 0.02, True, "not resolved"),
    (0.01, 0.001, False, "not resolved"), (0.01, None, True, "A higher")])
def test_direction_needs_both_the_interval_and_the_noise_test(diff, threshold, significant, expected):
    from vitok.analysis import beyond_noise, direction
    row = _row(6, "a", "b", "clean", "H2", diff, diff, significant=significant, threshold=threshold)
    assert direction(row) == expected
    assert beyond_noise(row) == ("no second seed" if threshold is None else ("yes" if abs(diff) > threshold else "no"))


def test_comparisons_and_val_sensitivity_skip_missing_runs_and_flag_sign_changes(tmp_path):
    from vitok.analysis import delta
    test_dir, val_dir = tmp_path / "test", tmp_path / "val"
    test_dir.mkdir(); val_dir.mkdir()
    # d6 has only BPE runs (every comparison skipped), d8 has the H1 pair; the val shard agrees at d8
    for cond, depth, t, v in [("bpe-nfc", 6, 1.0, 1.0), ("bpe-nfd", 6, 1.0, 1.0),
                              ("bpe-nfc", 8, 1.0, 1.0), ("super-nfc", 8, 1.002, 1.003)]:
        fake_run(test_dir / f"{cond}_d{depth}_s0.json", cond, depth, 0, t)
        fake_run(val_dir / f"{cond}_d{depth}_s0.json", cond, depth, 0, v)
    rows = comparisons(load(test_dir))
    assert {(r["depth"], r["a"], r["b"], r["variant"]) for r in rows} >= {(8, "super-nfc", "bpe-nfc", "clean"),
                                                                           (6, "bpe-nfd", "bpe-nfc", "clean")}
    assert all(r["depth"] == 8 or r["a"].startswith("bpe") for r in rows)
    text = val_sensitivity(load(val_dir), rows)
    assert "| super-nfc − bpe-nfc | d8 | 20 | +0.0030 |" in text and text.rstrip().endswith("| yes |")
    no_test = val_sensitivity(load(val_dir), [])
    assert no_test.rstrip().endswith("| — | — |")
    runs = load(test_dir)
    assert delta(runs, "super-nfc", "bpe-nfc", 8, 0) == pytest.approx(0.002)
    assert delta(runs, "super-nfc", "bpe-nfc", 6, 0) is None and delta(runs, "bpe-nfd", "super-nfd", 8, 0) is None


def test_verdicts_list_every_seed_of_each_depth_and_skip_missing_wordhood(tmp_path):
    # d6 has seeds 0 and 1, d8 only seed 0: each depth lists its own seeds; H4 lists only the conditions measured
    for cond, depth, seed, value in [("bpe-nfc", 6, 0, 1.0), ("super-nfc", 6, 0, 0.998), ("bpe-nfc", 6, 1, 1.0),
                                     ("super-nfc", 6, 1, 0.999), ("bpe-nfc", 8, 0, 0.9), ("super-nfc", 8, 0, 0.902)]:
        fake_run(tmp_path / f"{cond}_d{depth}_s{seed}.json", cond, depth, seed, value)
    runs = load(tmp_path)
    wordhood = {"super-nfc": {"match_rate": 0.6, "frequency_matched_baseline": 0.5, "baseline_rate": 0.2,
                              "share_outside_2_4": 0.1}}
    text = verdicts(comparisons(runs), {"nfc": 0.18, "nfd": 0.18}, runs, wordhood)
    assert "- d6: s0 -0.0020, s1 -0.0010\n" in text and "- d8: s0 +0.0020\n" in text
    assert "- super-nfc: 60.0%; frequency-matched baseline 50.0%" in text and "- super-nfd:" not in text
