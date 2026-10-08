"""Evaluation contracts, on a tiny Qwen3 (CPU): scores do not depend on batching or caching, runs resume exactly."""

import json
import math
import unicodedata

import pytest
import torch
from conftest import SENTENCES, qwen_like_base
from hypothesis import given
from hypothesis import strategies as st
from test_extend import tiny_qwen3
from tokenizers import Tokenizer

from vitok import extend, retrofit, score

ITEMS = [{"flores_passage": SENTENCES[i], "question": "Ai?", "mc_answer1": SENTENCES[(i + 1) % 8][:20],
          "mc_answer2": "Hà Nội", "mc_answer3": "không", "mc_answer4": SENTENCES[(i + 3) % 8][:30],
          "correct_answer_num": str(1 + i % 4)} for i in range(6)]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.set_num_threads(1)
    d = tmp_path_factory.mktemp("score")
    base = qwen_like_base()
    corpus = d / "vi.txt"
    corpus.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    multi = retrofit.retrofit(base, retrofit.learn_merges(base, [corpus], "multi", 120), "multi")
    (d / "multi.json").write_text(json.dumps(multi), encoding="utf-8")
    base_tok, tok = Tokenizer.from_str(json.dumps(base)), Tokenizer.from_str(json.dumps(multi))
    model = extend.extend(tiny_qwen3(base_tok.get_vocab_size(with_added_tokens=True), layers=2),
                          extend.new_token_pieces(base_tok, tok))
    model.save_pretrained(d / "model")
    docs = [" ".join(SENTENCES[i:] + SENTENCES[:i]) for i in range(8)] + ["The quick brown fox, don't stop!"]
    (d / "vi.jsonl").write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in docs), "utf-8")
    (d / "en.jsonl").write_text(json.dumps({"text": "The quick brown fox, don't stop! " * 3}) + "\n", "utf-8")
    (d / "belebele.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in ITEMS), "utf-8")
    return d, model.eval(), tok


def reference_logprob(model, seq, start):
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=torch.tensor([seq])).logits[0].double(), -1)
    return sum(float(lp[t - 1, seq[t]]) for t in range(start, len(seq)))


def test_scores_do_not_depend_on_batching_or_padding(setup):
    _, model, tok = setup
    seqs = [[0] + tok.encode(s, add_special_tokens=False).ids[: 3 + 4 * i] for i, s in enumerate(SENTENCES)]
    starts = [1 + i % 3 for i in range(len(seqs))]
    want = [reference_logprob(model, q, s) for q, s in zip(seqs, starts, strict=True)]
    for bs in (1, 3, 8):
        assert score.sum_logprobs(model, seqs, starts, bs) == pytest.approx(want, rel=1e-5, abs=1e-5)


def test_cached_greedy_decoding_matches_decoding_without_a_cache(setup):
    _, model, tok = setup
    banned = sorted(i for i, t in tok.get_added_tokens_decoder().items() if t.special)
    rec = score.generate(model, tok, SENTENCES[0], 12, banned)
    ids = tok.encode(SENTENCES[0], add_special_tokens=False).ids
    out = []
    with torch.no_grad():
        for _ in range(12):
            logits = model(input_ids=torch.tensor([ids + out])).logits[0, -1]
            logits[banned] = -math.inf
            out.append(int(logits.argmax()))
    assert rec["text"] == tok.decode(out, skip_special_tokens=False)
    assert rec["new_tokens"] == 12 and rec["new_chars"] == len(rec["text"]) and rec["prompt_chars"] == len(SENTENCES[0])
    assert rec["prefill_seconds"] > 0 and rec["decode_seconds"] > 0


@pytest.mark.parametrize("text,want", [("", 0.0), ("một hai ba", 0.0), ("a b c d a b c d", 1 / 5),
                                       ("A b c d a B c D", 1 / 5), ("x " * 10, 6 / 7)])
def test_repetition_counts_windows_seen_before(text, want):
    assert score.repetition(text) == pytest.approx(want)


@given(st.lists(st.sampled_from(["một", "hai", "ba", "bốn", "năm"]), max_size=40))
def test_repetition_is_a_share(words):
    r = score.repetition(" ".join(words))
    windows = list(zip(*(words[i:] for i in range(4)), strict=False))  # 4-word windows; shorter tails drop out
    assert 0 <= r < 1 and (r == 0) == (len(set(windows)) == len(windows))


def args(d, out, *extra):
    return score.parse_args(["--model", str(d / "model"), "--tokenizer", str(d / "multi.json"),
                             "--vi-docs", str(d / "vi.jsonl"), "--en-docs", str(d / "en.jsonl"),
                             "--belebele", str(d / "belebele.jsonl"), "--speed-prompts", "2", "--new-tokens", "6",
                             "--batch-size", "2", "--out", str(out), *extra])


TIMING = {"prefill_seconds", "decode_seconds"}


def records(out, part):
    return [{k: v for k, v in json.loads(line).items() if k not in TIMING}
            for line in (out / f"{part}.jsonl").read_text().splitlines()]


def test_a_full_run_scores_every_part_consistently(setup, tmp_path):
    d, model, tok = setup
    assert score.run(args(d, tmp_path)) == 0
    s = json.loads((tmp_path / "summary.json").read_text())
    vi = records(tmp_path, "vi_docs")
    texts = [json.loads(line)["text"] for line in (d / "vi.jsonl").read_text().splitlines()]
    assert [r["i"] for r in vi] == list(range(len(texts))) and [r["chars"] for r in vi] == list(map(len, texts))
    assert s["vi"]["bpc"] == pytest.approx(sum(r["nats"] for r in vi) / (math.log(2) * sum(map(len, texts))))
    eos = tok.token_to_id("<|endoftext|>")
    seq = [eos] + tok.encode(texts[3], add_special_tokens=False).ids
    assert vi[3]["nats"] == pytest.approx(-reference_logprob(model, seq, 1), rel=1e-5)
    assert s["en"]["docs"] == 1 and s["vi"]["tokens"] == sum(r["tokens"] for r in vi)
    bb = records(tmp_path, "belebele")
    it = ITEMS[2]
    ctx = tok.encode(score.PROMPT.format(passage=it["flores_passage"], question=it["question"]),
                     add_special_tokens=False).ids
    ans = tok.encode(" " + it["mc_answer4"], add_special_tokens=False).ids
    assert bb[2]["logprobs"][3] == pytest.approx(reference_logprob(model, ctx + ans, len(ctx)), rel=1e-5)
    for r in bb:
        per_char = [v / c for v, c in zip(r["logprobs"], r["chars"], strict=True)]
        assert r["correct_char"] == (per_char.index(max(per_char)) == r["gold"])
        assert r["correct_sum"] == (r["logprobs"].index(max(r["logprobs"])) == r["gold"])
    assert s["belebele"]["acc_char"] == sum(r["correct_char"] for r in bb) / len(ITEMS)
    gen = records(tmp_path, "generation")
    assert len(gen) == 2 and all(r["prompt_chars"] <= 500 and r["new_tokens"] == 6 for r in gen)
    assert s["generation"]["generated_chars_per_token"] == sum(r["new_chars"] for r in gen) / 12


def test_a_stopped_run_resumes_to_the_same_records(setup, tmp_path):
    d, _, _ = setup
    score.run(args(d, tmp_path / "whole"))
    out, codes = tmp_path / "pieces", []
    while not (out / "summary.json").exists():
        codes.append(score.run(args(d, out, "--deadline-minutes", "0")))
        for part in ("vi_docs", "belebele"):  # a crash can tear the last line
            p = out / f"{part}.jsonl"
            if p.exists() and len(codes) == 2:
                p.write_text(p.read_text() + '{"i": 99, "na', encoding="utf-8")
    assert codes[-1] == 0 and set(codes[:-1]) == {score.EXIT_INCOMPLETE} and len(codes) > 3
    for part in ("vi_docs", "en_docs", "belebele", "generation"):
        a, b = records(tmp_path / "whole", part), records(out, part)
        assert [r["i"] for r in b] == sorted(r["i"] for r in b)
        assert a == b, part


def test_snapshot_weights_are_used(setup, tmp_path):
    d, model, _ = setup
    snap = {"params": {"model.norm.weight": torch.full_like(model.model.norm.weight, 2.0)}, "new_ids": [0, 0],
            "step": 1, "tokens": 1, "chars": 1}
    torch.save(snap, tmp_path / "snap.pt")
    score.run(args(d, tmp_path / "plain"))
    score.run(args(d, tmp_path / "snap", "--snapshot", str(tmp_path / "snap.pt")))
    a, b = (json.loads((tmp_path / p / "summary.json").read_text()) for p in ("plain", "snap"))
    assert a["vi"]["bpc"] != b["vi"]["bpc"] and b["snapshot"]["step"] == 1


def test_bad_inputs_are_refused(setup, tmp_path):
    d, _, _ = setup
    score.run(args(d, tmp_path / "x", "--speed-prompts", "0"))
    with pytest.raises(SystemExit, match=r"another evaluation \(differs in \['new_tokens'\]\)"):
        score.run(args(d, tmp_path / "x", "--speed-prompts", "0", "--new-tokens", "7"))
    nfd = tmp_path / "nfd.jsonl"
    nfd.write_text(json.dumps({"text": unicodedata.normalize("NFD", SENTENCES[0])}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="must hold NFC"):
        score.run(args(d, tmp_path / "y", "--vi-docs", str(nfd), "--speed-prompts", "0"))
    for extra, message in [(["--dtype", "fp16", "--device", "cpu"], "needs a CUDA"),
                           (["--new-tokens", "1"], "at least 2"),
                           (["--eos-token", "<none>"], "has no")]:
        with pytest.raises(SystemExit, match=message):
            score.run(args(d, tmp_path / message.replace(" ", "-"), *extra))
    with pytest.raises(SystemExit, match="from --vi-docs"):
        score.parse_args(["--model", "m", "--tokenizer", "t", "--speed-prompts", "1", "--out", "o"])


def test_comparison_pairs_on_the_same_items_and_refuses_others(setup, tmp_path):
    from vitok import compare, stats
    d, model, _ = setup
    snap = {"params": {"model.norm.weight": torch.full_like(model.model.norm.weight, 1.5)}, "new_ids": [0, 0],
            "step": 1, "tokens": 1, "chars": 1}
    torch.save(snap, tmp_path / "snap.pt")
    score.run(args(d, tmp_path / "ref"))
    score.run(args(d, tmp_path / "run", "--snapshot", str(tmp_path / "snap.pt")))
    same = compare.compare(tmp_path / "ref", tmp_path / "ref")
    assert same["vi"]["diff"] == 0 and same["vi"]["ci95"] == [0, 0] and same["belebele"]["mcnemar_char"]["p_value"] == 1
    assert all(same["generation"][k + "_ratio"] == 1 for k in ("decode_chars_per_second", "generated_chars_per_token"))
    c = compare.compare(tmp_path / "run", tmp_path / "ref")
    a, b = records(tmp_path / "run", "vi_docs"), records(tmp_path / "ref", "vi_docs")
    want = stats.paired_bootstrap_bpc([r["nats"] for r in a], [r["nats"] for r in b], [r["chars"] for r in b])
    assert c["vi"]["ci95"] == want["ci95"] and c["vi"]["bpc"] - c["vi"]["bpc_reference"] == pytest.approx(want["diff"])
    out = tmp_path / "cmp.json"
    compare.main(["--reference", f"base={tmp_path / 'ref'}", "--runs", f"multi={tmp_path / 'run'}", "--out", str(out)])
    assert json.loads(out.read_text())["runs"]["multi"]["vi"]["ci95"] == want["ci95"]
    p = tmp_path / "run" / "vi_docs.jsonl"
    lines = p.read_text().splitlines()
    rec = json.loads(lines[0])
    p.write_text("\n".join([json.dumps({**rec, "digest": "00"}), *lines[1:]]) + "\n")
    with pytest.raises(SystemExit, match="different documents"):
        compare.compare(tmp_path / "run", tmp_path / "ref")
    p.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(SystemExit, match="incomplete"):
        compare.compare(tmp_path / "run", tmp_path / "ref")
    with pytest.raises(SystemExit):
        compare.main(["--reference", "noequals", "--runs", "a=b", "--out", str(out)])


def test_full_run_records_the_gold_answers_token_counts_and_generation_totals(setup, tmp_path):
    d, model, tok = setup
    score.run(args(d, tmp_path))
    bb = records(tmp_path, "belebele")
    assert [r["gold"] for r in bb] == [int(it["correct_answer_num"]) - 1 for it in ITEMS]
    texts = [json.loads(line)["text"] for line in (d / "vi.jsonl").read_text().splitlines()]
    assert [r["tokens"] for r in records(tmp_path, "vi_docs")] == \
        [len(tok.encode(t, add_special_tokens=False).ids) for t in texts]
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["vi"]["chars_per_token"] == s["vi"]["chars"] / s["vi"]["tokens"]
    assert s["vi"]["chars"] == sum(map(len, texts)) and s["vi"]["docs"] == len(texts)
    gen = [json.loads(line) for line in (tmp_path / "generation.jsonl").read_text().splitlines()]
    decode, prefill = sum(r["decode_seconds"] for r in gen), sum(r["prefill_seconds"] for r in gen)
    g = s["generation"]
    assert g["decode_chars_per_second"] == pytest.approx(sum(r["new_chars"] for r in gen) / decode)
    assert g["decode_tokens_per_second"] == pytest.approx(sum(r["new_tokens"] - 1 for r in gen) / decode)
    assert g["prefill_chars_per_second"] == pytest.approx(sum(r["prompt_chars"] for r in gen) / prefill)
    assert g["repetition_4_mean"] == pytest.approx(sum(r["repetition_4"] for r in gen) / len(gen))
    assert g["max_memory_gib"] is None and all(r["max_memory_gib"] is None for r in gen)
    assert g["prompts"] == 2 and g["new_tokens"] == 6 and s["device_name"] == "cpu"
    assert set(s) == {"signature", "snapshot", "vi", "en", "belebele", "generation", "device_name"}


def test_belebele_picks_by_log_likelihood_per_character_and_in_total(setup, tmp_path, monkeypatch):
    _, model, tok = setup
    items = [ITEMS[0] | {"correct_answer_num": "2"}, ITEMS[1] | {"correct_answer_num": "4"}]
    # item 0: answer 2 has the best total and per-character score; item 1: answer 1 has the best total, answer 4
    # (much longer) the best per character
    per_item = {0: [-9.0, -1.0, -8.0, -7.0], 1: [-2.0, -5.0, -6.0, -3.0]}

    def fake(model, seqs, starts, batch_size, amp_dtype=None):
        return [per_item[j // 4][j % 4] for j in range(len(seqs))]

    monkeypatch.setattr(score, "sum_logprobs", fake)
    items[1] = items[1] | {"mc_answer1": "ab", "mc_answer4": "x" * 60}
    s = score.score_belebele(model, tok, items, tmp_path / "b.jsonl", 8, None, lambda: False)
    recs = [json.loads(line) for line in (tmp_path / "b.jsonl").read_text().splitlines()]
    assert [(r["correct_sum"], r["correct_char"]) for r in recs] == [(True, True), (False, True)]
    assert recs[1]["chars"][0] == 3 and recs[1]["chars"][3] == 61
    assert s == {"items": 2, "acc_char": 1.0, "acc_sum": 0.5}


@pytest.mark.parametrize("change", ["vi_docs", "snapshot", "tokenizer", "en_docs", "belebele"])
def test_an_evaluation_of_other_inputs_is_refused(setup, tmp_path, change):
    d, model, _ = setup
    snap = {"params": {"model.norm.weight": torch.full_like(model.model.norm.weight, 2.0)}, "new_ids": [0, 0],
            "step": 1, "tokens": 1, "chars": 1}
    torch.save(snap, tmp_path / "s.pt")
    score.run(args(d, tmp_path / "o", "--speed-prompts", "0", "--snapshot", str(tmp_path / "s.pt")))
    if change == "snapshot":
        torch.save({**snap, "step": 2}, tmp_path / "s2.pt")
        extra = ["--snapshot", str(tmp_path / "s2.pt")]
    else:  # the same content written with an extra line or other whitespace: other bytes
        src = {"vi_docs": d / "vi.jsonl", "en_docs": d / "en.jsonl", "belebele": d / "belebele.jsonl",
               "tokenizer": d / "multi.json"}[change]
        copy = tmp_path / src.name
        text = src.read_text()
        other = json.dumps(json.loads(text), indent=1) if change == "tokenizer" else text + text.splitlines()[0] + "\n"
        copy.write_text(other)
        extra = [f"--{change.replace('_', '-')}", str(copy), "--snapshot", str(tmp_path / "s.pt")]
    with pytest.raises(SystemExit, match=rf"differs in \['{change}_sha256'\]"):
        score.run(args(d, tmp_path / "o", "--speed-prompts", "0", *extra))
    sig = json.loads((tmp_path / "o" / "signature.json").read_text())
    assert sig["model"] == str(d / "model") and sig["batch_size"] == 2 and sig["dtype"] == "fp32"


def test_a_tokenizer_as_large_as_the_matrix_fits_and_a_generous_deadline_finishes(setup, tmp_path):
    d, model, tok = setup
    from test_extend import tiny_qwen3
    tiny_qwen3(tok.get_vocab_size(with_added_tokens=True), layers=2).save_pretrained(tmp_path / "exact")
    a = args(d, tmp_path / "o", "--speed-prompts", "0", "--deadline-minutes", "100",
             "--model", str(tmp_path / "exact"))
    assert score.run(a) == 0


def test_scoring_loads_the_pinned_revision_in_fp32_and_main_exits_with_the_status(setup, tmp_path, monkeypatch):
    import transformers
    d, _, _ = setup
    seen, real = [], transformers.AutoModelForCausalLM.from_pretrained

    def spy(name, **kw):
        seen.append(dict(kw))
        kw.pop("revision")
        return real(name, **kw)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", spy)
    score.run(args(d, tmp_path / "o", "--speed-prompts", "0", "--revision", "abc"))
    assert seen == [{"revision": "abc", "torch_dtype": torch.float32}]
    with pytest.raises(SystemExit) as e:
        score.main(["--model", str(d / "model"), "--tokenizer", str(d / "multi.json"),
                    "--vi-docs", str(d / "vi.jsonl"), "--out", str(tmp_path / "m"), "--deadline-minutes", "0"])
    assert e.value.code == score.EXIT_INCOMPLETE


def test_score_defaults_and_required_arguments():
    a = score.parse_args(["--model", "m", "--tokenizer", "t", "--out", "o"])
    assert (a.speed_prompts, a.speed_prompt_chars, a.new_tokens, a.batch_size, a.dtype, a.revision, a.snapshot,
            a.deadline_minutes, a.eos_token) == (0, 500, 256, 8, "fp32", None, None, None, "<|endoftext|>")
    assert isinstance(a.tokenizer, score.Path) and isinstance(a.out, score.Path)
    assert a.device == ("cuda" if torch.cuda.is_available() else "cpu")
    for drop in ("--model", "--tokenizer", "--out"):
        full = ["--model", "m", "--tokenizer", "t", "--out", "o"]
        i = full.index(drop)
        with pytest.raises(SystemExit):
            score.parse_args(full[:i] + full[i + 2:])
    b = score.parse_args(["--model", "m", "--tokenizer", "t", "--out", "o", "--vi-docs", "v", "--speed-prompts",
                          "1", "--new-tokens", "2", "--speed-prompt-chars", "300", "--deadline-minutes", "1.5"])
    assert (b.new_tokens, b.speed_prompt_chars, b.deadline_minutes, b.speed_prompts) == (2, 300, 1.5, 1)


def test_comparison_values_and_one_sided_parts(setup, tmp_path):
    from vitok import compare
    d, model, _ = setup
    score.run(args(d, tmp_path / "ref"))
    score.run(args(d, tmp_path / "noen", "--speed-prompts", "0", "--en-docs", str(d / "vi.jsonl")))
    full = compare.compare(tmp_path / "ref", tmp_path / "ref")
    vi = records(tmp_path / "ref", "vi_docs")
    cpt = sum(r["chars"] for r in vi) / sum(r["tokens"] for r in vi)
    assert full["vi"]["chars_per_token"] == full["vi"]["chars_per_token_reference"] == pytest.approx(cpt)
    assert full["en"]["bpc"] == full["en"]["bpc_reference"]
    bb = records(tmp_path / "ref", "belebele")
    acc_c, acc_s = (sum(r[k] for r in bb) / len(bb) for k in ("correct_char", "correct_sum"))
    keys = ("items", "acc_char", "acc_char_reference", "acc_sum", "acc_sum_reference")
    want = {"items": len(ITEMS), "acc_char": acc_c, "acc_char_reference": acc_c, "acc_sum": acc_s,
            "acc_sum_reference": acc_s}
    assert {k: full["belebele"][k] for k in keys} == want
    assert full["snapshot"] is None and full["snapshot_reference"] is None
    g = full["generation"]
    assert g["decode_chars_per_second"] == g["decode_chars_per_second_reference"] and g["max_memory_gib_ratio"] is None
    # en_docs of different documents are refused; without generation on one side there is no generation entry
    with pytest.raises(SystemExit, match="different documents"):
        compare.compare(tmp_path / "noen", tmp_path / "ref")
    (tmp_path / "noen" / "en_docs.jsonl").unlink()
    part = compare.compare(tmp_path / "noen", tmp_path / "ref")
    assert "en" not in part and "generation" not in part and "vi" in part
    s = json.loads((tmp_path / "ref" / "summary.json").read_text())
    s["device_name"] = "Tesla T4"
    (tmp_path / "ref" / "summary.json").write_text(json.dumps(s))
    score.run(args(d, tmp_path / "cpu"))
    with pytest.raises(SystemExit, match="cannot be compared"):  # speed measured on two devices
        compare.compare(tmp_path / "ref", tmp_path / "cpu")
    for missing in (["--runs", "a=b", "--out", "o"], ["--reference", "a=b", "--out", "o"],
                    ["--reference", "a=b", "--runs", "a=b"]):
        with pytest.raises(SystemExit):
            compare.main(missing)
    assert compare._named("a=b=c") == ("a", score.Path("b=c"))



def test_generation_ratios_need_both_values_and_snapshots_are_reported():
    from vitok import compare
    g = compare.compare_generation({"decode_chars_per_second": 10.0, "max_memory_gib": None},
                                   {"decode_chars_per_second": 5.0, "max_memory_gib": 1.0})
    assert g["decode_chars_per_second_ratio"] == 2.0 and g["max_memory_gib_ratio"] is None
    assert g["max_memory_gib"] is None and g["max_memory_gib_reference"] == 1.0 and g["repetition_4_mean"] is None
    assert compare.compare_generation({"max_memory_gib": 2.0}, {"max_memory_gib": None})["max_memory_gib_ratio"] is None


def test_comparison_reports_snapshots_and_skips_parts_one_side_lacks(setup, tmp_path):
    from vitok import compare
    d, model, _ = setup
    snap = {"params": {"model.norm.weight": torch.full_like(model.model.norm.weight, 1.5)}, "new_ids": [0, 0],
            "step": 7, "tokens": 70, "chars": 300}
    torch.save(snap, tmp_path / "s.pt")
    score.run(args(d, tmp_path / "ref", "--speed-prompts", "0"))
    score.run(args(d, tmp_path / "run", "--speed-prompts", "0", "--snapshot", str(tmp_path / "s.pt")))
    (tmp_path / "run" / "belebele.jsonl").unlink()
    c = compare.compare(tmp_path / "run", tmp_path / "ref")
    assert c["snapshot"]["step"] == 7 and c["snapshot_reference"] is None and "belebele" not in c
    c2 = compare.compare(tmp_path / "ref", tmp_path / "run")
    assert c2["snapshot"] is None and c2["snapshot_reference"]["tokens"] == 70 and "belebele" not in c2
