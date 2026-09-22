import json
import math

import pytest

from vitok.analysis import load, nonembedding_params, scaling
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
    assert "| super-nfc − bpe-nfc | -0.0020 ± 0.0005 | +0.0020 |" in text
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
