"""Demo contracts, on a tiny Qwen3 (CPU): streaming equals batch decoding, a race changes nothing, numbers are exact."""

import json
import threading

import pytest
import torch
from conftest import SENTENCES, qwen_like_base
from test_extend import tiny_qwen3
from tokenizers import Tokenizer

from vitok import demo, extend, retrofit, score


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.set_num_threads(1)
    d = tmp_path_factory.mktemp("demo")
    base = qwen_like_base()
    corpus = d / "vi.txt"
    corpus.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    multi = retrofit.retrofit(base, retrofit.learn_merges(base, [corpus], "multi", 120), "multi")
    (d / "multi.json").write_text(json.dumps(multi), encoding="utf-8")
    base_tok, tok = Tokenizer.from_str(json.dumps(base)), Tokenizer.from_str(json.dumps(multi))
    model = extend.extend(tiny_qwen3(base_tok.get_vocab_size(with_added_tokens=True), layers=2),
                          extend.new_token_pieces(base_tok, tok))
    model.save_pretrained(d / "extended")
    return d, model.eval(), tok


def banned_ids(tok):
    return sorted(i for i, t in tok.get_added_tokens_decoder().items() if t.special)


def test_streaming_yields_the_ids_that_batch_decoding_chooses(setup):
    _, model, tok = setup
    banned = banned_ids(tok)
    want = score.generate(model, tok, SENTENCES[0], 12, banned)
    ids = list(demo.stream_ids(model, tok, SENTENCES[0], 12, banned))
    assert len(ids) == 12 and not set(ids) & set(banned)
    assert tok.decode(ids, skip_special_tokens=False) == want["text"]


def test_a_stream_needs_a_prompt_and_a_positive_length(setup):
    _, model, tok = setup
    with pytest.raises(ValueError, match="new_tokens"):
        list(demo.stream_ids(model, tok, SENTENCES[0], 0, []))
    with pytest.raises(ValueError, match="prompt"):
        list(demo.stream_ids(model, tok, "", 4, []))


def test_numbers_are_counts_and_ratios_of_what_was_generated():
    s = demo.summarize(new_tokens=10, new_chars=36, decode_seconds=0.5)
    assert s == {"new_tokens": 10, "new_chars": 36, "chars_per_token": 3.6, "chars_per_second": 72.0,
                 "tokens_per_second": 20.0}
    with pytest.raises(ValueError, match="decode_seconds"):
        demo.summarize(new_tokens=10, new_chars=36, decode_seconds=0)


def test_a_race_gives_each_model_its_own_complete_text(setup):
    d, model, tok = setup
    banned = banned_ids(tok)
    seen: dict[str, list[str]] = {"a": [], "b": []}
    done = demo.race({"a": (model, tok), "b": (model, tok)}, SENTENCES[1], 9, on_text=lambda n, t: seen[n].append(t))
    want = score.generate(model, tok, SENTENCES[1], 9, banned)["text"]
    assert {n: r["text"] for n, r in done.items()} == {"a": want, "b": want}
    assert all(r["new_tokens"] == 9 for r in done.values())
    # every update shows the text so far, and the last one is the whole text
    assert all(texts and texts[-1] == want and len(texts) == 9 for texts in seen.values())


def test_a_race_runs_the_models_at_the_same_time(setup):
    _, model, tok = setup
    started = {"a": threading.Event(), "b": threading.Event()}
    met = {}

    def on_text(name, text):
        started[name].set()
        other = "b" if name == "a" else "a"
        # run one after the other, the first model would wait here for a second that never starts
        met.setdefault(name, started[other].wait(timeout=30))

    demo.race({"a": (model, tok), "b": (model, tok)}, SENTENCES[2], 4, on_text=on_text)
    assert met == {"a": True, "b": True}


def test_a_benchmark_runs_the_models_in_turn_and_reports_exact_counts(setup):
    _, model, tok = setup
    out = demo.benchmark({"a": (model, tok), "b": (model, tok)}, SENTENCES[4], 7)
    want = score.generate(model, tok, SENTENCES[4], 7, banned_ids(tok))["text"]
    for r in out.values():
        assert r["text"] == want and r["new_tokens"] == 7 and r["new_chars"] == len(want)
        assert r["chars_per_token"] == len(want) / 7 and r["chars_per_second"] > 0


def test_a_failing_model_fails_the_race(setup):
    _, model, tok = setup

    def boom(name, text):
        if name == "b":
            raise RuntimeError("update failed")

    with pytest.raises(RuntimeError, match="update failed"):
        demo.race({"a": (model, tok), "b": (model, tok)}, SENTENCES[3], 4, on_text=boom)


def test_loading_a_condition_applies_the_snapshot_and_checks_the_tokenizer(setup, tmp_path):
    d, model, tok = setup
    snap = {"params": {"model.norm.weight": torch.full_like(model.model.norm.weight, 2.0)}, "new_ids": [0, 0],
            "step": 7, "tokens": 1, "chars": 1}
    torch.save(snap, tmp_path / "final.pt")
    plain, _ = demo.load_condition(d / "extended", d / "multi.json", None, "cpu", torch.float32)
    loaded, loaded_tok = demo.load_condition(d / "extended", d / "multi.json", tmp_path / "final.pt", "cpu",
                                             torch.float32)
    assert float(plain.model.norm.weight.detach().mean()) != 2.0
    assert float(loaded.model.norm.weight.detach().mean()) == 2.0
    assert not loaded.training and loaded_tok.get_vocab_size() == tok.get_vocab_size()
    with pytest.raises(SystemExit, match="embedding rows"):
        demo.load_condition(d / "extended", _big_tokenizer(tmp_path), None, "cpu", torch.float32)


def _big_tokenizer(tmp_path):
    tok = Tokenizer.from_str(json.dumps(qwen_like_base()))
    tok.add_tokens([f"extra{i}" for i in range(5000)])
    path = tmp_path / "big.json"
    tok.save(str(path))
    return path


def test_race_events_stream_every_update_then_the_results(setup):
    _, model, tok = setup
    events = list(demo.race_events({"a": (model, tok), "b": (model, tok)}, SENTENCES[5], 6))
    kinds = [k for k, _ in events]
    assert kinds == ["update"] * 12 + ["done"]
    want = score.generate(model, tok, SENTENCES[5], 6, banned_ids(tok))["text"]
    for name in ("a", "b"):
        texts = [p[1] for k, p in events if k == "update" and p[0] == name]
        assert len(texts) == 6 and texts[-1] == want
    assert {n: r["text"] for n, r in events[-1][1].items()} == {"a": want, "b": want}


def test_race_events_raise_a_failure_after_the_updates(setup):
    _, model, tok = setup
    with pytest.raises(ValueError, match="prompt"):
        list(demo.race_events({"a": (model, tok)}, "", 4))
