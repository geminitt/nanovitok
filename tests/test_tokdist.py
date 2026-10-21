"""Token Distillation contracts, on a tiny Qwen3 (CPU): snippets hold their token's pieces where the new tokenizer uses
it, the student merges exactly that occurrence, only rows with snippets move, every other weight stays, and a run
stopped after every batch resumes to the same rows."""

import json
import random

import pytest
import torch
from conftest import SENTENCES, qwen_like_base
from test_extend import tiny_qwen3
from tokenizers import Tokenizer

from vitok import extend, mixed_stream, retrofit, tokdist


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.set_num_threads(1)
    d = tmp_path_factory.mktemp("tokdist")
    base = qwen_like_base()
    corpus = d / "vi.txt"
    corpus.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    multi = retrofit.retrofit(base, retrofit.learn_merges(base, [corpus], "multi", 60), "multi")
    (d / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (d / "multi.json").write_text(json.dumps(multi), encoding="utf-8")
    base_tok, multi_tok = Tokenizer.from_str(json.dumps(base)), Tokenizer.from_str(json.dumps(multi))
    model = tiny_qwen3(base_tok.get_vocab_size(with_added_tokens=True), layers=2)
    extend.extend(model, extend.new_token_pieces(base_tok, multi_tok)).save_pretrained(d / "mean-model")
    rng = random.Random(0)
    docs = [("web", " ".join(rng.choices(SENTENCES, k=rng.randint(2, 6)))) for _ in range(80)]
    mixed_stream.write_mixed(d / "stream", iter(docs), {"test": 1})
    return d, base_tok, multi_tok


def argv(d, out, *extra):
    return ["--model", str(d / "mean-model"), "--base-tokenizer", str(d / "base.json"), "--tokenizer",
            str(d / "multi.json"), "--stream", str(d / "stream"), "--out", str(out), "--snippets", "6",
            "--snippet-len", "16", "--per-doc", "2", "--batch", "4", "--lr", "1e-2", "--log-every", "1",
            "--save-every", "3", "--device", "cpu", *extra]


def run(*a):
    return tokdist.run(tokdist.parse_args(argv(*a)))


def test_snippets_hold_the_pieces_where_the_new_tokenizer_uses_the_token(setup):
    d, base, new = setup
    pieces = extend.new_token_pieces(base, new)
    texts = [r["text"] for r in mixed_stream.mixed_records(d / "stream")]
    snips, stats = tokdist.find_snippets(texts, base, new, pieces, 6, 16, 2)
    assert stats["documents"] <= len(texts) and sum(map(len, snips.values())) > 0
    for tid, rows in snips.items():
        assert len(rows) <= 6 and all(sum(r["doc"] == doc for r in rows) <= 2 for doc in {r["doc"] for r in rows})
        for r in rows:
            k = len(pieces[tid])
            assert len(r["ids"]) <= 16 and r["ids"][r["at"]:r["at"] + k] == pieces[tid]
            base_ids = base.encode(texts[r["doc"]], add_special_tokens=False).ids
            assert any(base_ids[s:s + len(r["ids"])] == r["ids"] for s in range(len(base_ids)))
            assert tid in new.encode(texts[r["doc"]], add_special_tokens=False).ids
        assert [r["doc"] for r in rows] == sorted(r["doc"] for r in rows)  # stream order
    used = {t for text in texts for t in new.encode(text, add_special_tokens=False).ids if t in pieces}
    assert {t for t, rows in snips.items() if rows} <= used


def test_the_student_merges_one_occurrence_and_aligns_it_to_its_last_piece():
    student, align = tokdist.merge([5, 6, 7, 8, 9], 1, 2, 100)
    assert student == [5, 100, 8, 9] and align == [0, 2, 3, 4]
    assert tokdist.merge([7, 8], 0, 2, 100) == ([100], [1])


def test_only_rows_with_snippets_move_and_every_other_weight_stays(setup, tmp_path):
    from transformers import AutoModelForCausalLM

    d, base, new = setup
    assert run(d, tmp_path) == 0
    start = AutoModelForCausalLM.from_pretrained(d / "mean-model")
    out = AutoModelForCausalLM.from_pretrained(tmp_path / "model")
    man = json.loads((tmp_path / "model" / "tokdist_manifest.json").read_text())
    snips = {int(k): v for k, v in json.loads((tmp_path / "snippets.json").read_text())["snippets"].items()}
    a, b = dict(start.named_parameters()), dict(out.named_parameters())
    for name in a:
        if name != "model.embed_tokens.weight":
            assert torch.equal(a[name], b[name]), name
    ea, eb = a["model.embed_tokens.weight"], b["model.embed_tokens.weight"]
    first = man["first_new_id"]
    assert torch.equal(ea[:first], eb[:first]) and torch.equal(ea[first + len(snips):], eb[first + len(snips):])
    for tid, rows in snips.items():
        assert not torch.equal(ea[tid], eb[tid]) or not rows, tid  # with CE a row without snippets may move too
    assert man["tokens_without_snippets_moved_by_ce"] == sum(
        not rows and not torch.equal(ea[tid], eb[tid]) for tid, rows in snips.items())
    assert man["tokens_with_snippets"] == sum(bool(r) for r in snips.values()) and man["snippets"] == sum(
        map(len, snips.values()))
    assert out.get_output_embeddings().weight.data_ptr() == out.get_input_embeddings().weight.data_ptr()
    assert Tokenizer.from_file(str(tmp_path / "model" / "tokenizer.json")).get_vocab() == new.get_vocab()
    log = [json.loads(x) for x in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert [r["batch"] for r in log] == list(range(man["batches"])) and all("ce" in r for r in log)
    assert log[-1]["mse"] < log[0]["mse"]
    assert not (tmp_path / "checkpoint.pt").exists() and run(d, tmp_path) == 0  # finished: kept


def test_a_run_stopped_after_every_batch_resumes_to_the_same_rows(setup, tmp_path):
    d, _, _ = setup
    run(d, tmp_path / "whole")
    codes = []
    while (code := run(d, tmp_path / "pieces", "--deadline-minutes", "0")) != 0:
        codes.append(code)
    assert codes and set(codes) == {tokdist.EXIT_INCOMPLETE}
    from safetensors.torch import load_file
    a = load_file(tmp_path / "whole" / "model" / "model.safetensors")
    b = load_file(tmp_path / "pieces" / "model" / "model.safetensors")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    strip = lambda p: [{k: v for k, v in json.loads(x).items() if k != "elapsed_seconds"}  # noqa: E731
                       for x in (p / "train_log.jsonl").read_text().splitlines()]
    assert strip(tmp_path / "whole") == strip(tmp_path / "pieces")


def test_without_ce_the_loss_is_the_mse_and_a_changed_setting_is_refused(setup, tmp_path):
    d, _, _ = setup
    assert run(d, tmp_path / "mse", "--no-ce") == 0
    log = [json.loads(x) for x in (tmp_path / "mse" / "train_log.jsonl").read_text().splitlines()]
    assert all(r["loss"] == pytest.approx(r["mse"]) and "ce" not in r for r in log)
    from safetensors.torch import load_file
    snips = json.loads((tmp_path / "mse" / "snippets.json").read_text())["snippets"]
    a = load_file(d / "mean-model" / "model.safetensors")["model.embed_tokens.weight"]
    b = load_file(tmp_path / "mse" / "model" / "model.safetensors")["model.embed_tokens.weight"]
    assert {int(k) for k, rows in snips.items() if rows} == {i for i in map(int, snips) if not torch.equal(a[i], b[i])}
    assert run(d, tmp_path / "o", "--deadline-minutes", "0") == tokdist.EXIT_INCOMPLETE
    with pytest.raises(SystemExit, match="another run"):
        run(d, tmp_path / "o", "--lr", "1e-3")


def test_the_auto_weighted_loss_doubles_the_mse_in_value():
    """mse + ce * (mse / ce) equals 2 * mse; the weight only changes the gradient's mix."""
    torch.manual_seed(0)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = lambda inputs_embeds, attention_mask: type("o", (), {"last_hidden_state": inputs_embeds})()

    w_old = torch.randn(10, 4)
    new = torch.nn.Parameter(torch.randn(2, 4))
    batch = [{"student": [1, 10, 3], "align": [0, 2, 3], "teacher": [1, 2, 4, 3]}] * 2
    amp = dict(device_type="cpu", enabled=False)
    loss, parts = tokdist.loss_fn(Tiny(), w_old, new, 10, batch, "cpu", True, amp)
    assert float(loss.detach()) == pytest.approx(2 * parts["mse"]) and parts["ce"] > 0
