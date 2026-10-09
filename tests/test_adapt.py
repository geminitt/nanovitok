"""Adaptation training contracts, on a tiny Qwen3 (CPU): what trains and what stays, equal compute, exact resume."""

import itertools
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from conftest import ROOT, SENTENCES, qwen_like_base
from hypothesis import given
from hypothesis import strategies as st
from test_extend import tiny_qwen3
from tokenizers import Tokenizer

from vitok import adapt, extend, retrofit

STEPS, BATCH, SEQ = 8, 4, 32


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.set_num_threads(1)  # bit-identical comparisons across runs and processes
    d = tmp_path_factory.mktemp("adapt")
    base = qwen_like_base()
    corpus = d / "vi.txt"
    corpus.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    multi = retrofit.retrofit(base, retrofit.learn_merges(base, [corpus], "multi", 120), "multi")
    (d / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (d / "multi.json").write_text(json.dumps(multi), encoding="utf-8")
    base_tok, multi_tok = Tokenizer.from_str(json.dumps(base)), Tokenizer.from_str(json.dumps(multi))
    rows = base_tok.get_vocab_size(with_added_tokens=True)
    model = tiny_qwen3(rows, layers=4)
    model.save_pretrained(d / "base-model")
    extend.extend(model, extend.new_token_pieces(base_tok, multi_tok)).save_pretrained(d / "multi-model")
    rng = random.Random(0)
    # "!" is id 0 of the byte-level alphabet, so the lowest id occurs in training (held-out docs are i % 3 == 0)
    docs = [" ".join(rng.choices(SENTENCES, k=rng.randint(1, 4))) + (" 7!" if i % 3 == 1 else "")
            for i in range(400)]
    for i in range(2):
        pq.write_table(pa.table({"text": docs[i * 200:(i + 1) * 200]}), d / f"shard_{i:05d}.parquet")
    (d / "held.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in docs[:30:3]), encoding="utf-8")
    return d


def argv(d, cond, out, *extra, steps=STEPS):
    return ["--model", str(d / f"{cond}-model"), "--base-tokenizer", str(d / "base.json"),
            "--tokenizer", str(d / f"{cond}.json"), "--shards", str(d / "shard_00000.parquet"),
            str(d / "shard_00001.parquet"), "--held-out", str(d / "held.jsonl"), "--out", str(out),
            "--steps", str(steps), "--batch", str(BATCH), "--seq-len", str(SEQ), "--lr", "3e-3", "--warmup", "2",
            "--layers", "0,-1", "--log-every", "1", "--save-every", "2", "--encode-batch", "8", *extra]


def run(*a, **k):
    return adapt.train(adapt.parse_args(argv(*a, **k)))


def final(out):
    return torch.load(out / "final.pt", map_location="cpu", weights_only=False)


def log(out):
    return [json.loads(line) for line in (out / "train_log.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("cond", ["multi", "base"])
def test_only_new_rows_and_chosen_layers_train(setup, tmp_path, cond):
    from transformers import AutoModelForCausalLM
    assert run(setup, cond, tmp_path, "--snapshot-steps", "4") == 0
    start = AutoModelForCausalLM.from_pretrained(setup / f"{cond}-model")
    trained = AutoModelForCausalLM.from_pretrained(setup / f"{cond}-model")
    meta = adapt.load_snapshot(trained, tmp_path / "final.pt")
    a, b = dict(start.named_parameters()), dict(trained.named_parameters())
    new = range(*meta["new_ids"])
    assert (len(new) > 0) == (cond == "multi")
    for name in a:
        layer = name.split(".")[2] if name.startswith("model.layers.") else None
        if layer in ("0", "3"):
            assert not torch.equal(a[name], b[name]), name
        elif name == adapt.EMBED:
            old = [i for i in range(a[name].shape[0]) if i not in new]
            assert torch.equal(a[name][old], b[name][old])
            assert (a[name][new.start:new.stop] != b[name][new.start:new.stop]).any(dim=1).all() or not len(new)
        else:
            assert torch.equal(a[name], b[name]), name
    rec = log(tmp_path)
    assert [r["step"] for r in rec] == list(range(STEPS)) and all(r["finite"] for r in rec)
    assert rec[-1]["loss"] < rec[0]["loss"] and rec[-1]["tokens"] == STEPS * BATCH * SEQ
    assert not (tmp_path / "checkpoint.pt").exists() and (tmp_path / "snapshots" / "step_000004.pt").exists()
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["steps"] == STEPS and man["new_tokens"] == len(new) and man["chars"] == rec[-1]["chars"]


def test_conditions_get_equal_compute_and_the_retrofit_reads_more_text(setup, tmp_path):
    run(setup, "base", tmp_path / "base")
    run(setup, "multi", tmp_path / "multi")
    b, m = log(tmp_path / "base")[-1], log(tmp_path / "multi")[-1]
    assert b["tokens"] == m["tokens"] and m["chars"] > b["chars"] > 0


def _same(a, b):
    assert a["params"].keys() == b["params"].keys()
    for k in a["params"]:
        assert torch.equal(a["params"][k], b["params"][k]), k
    assert (a["step"], a["tokens"], a["chars"]) == (b["step"], b["tokens"], b["chars"])


def test_runs_stopped_by_the_deadline_resume_to_the_same_weights(setup, tmp_path):
    run(setup, "multi", tmp_path / "whole")
    out = tmp_path / "pieces"
    codes = []
    while not (out / "final.pt").exists():
        codes.append(run(setup, "multi", out, "--deadline-minutes", "0"))  # stops after every step
    assert codes == [adapt.EXIT_INCOMPLETE] * (STEPS - 1) + [0]
    _same(final(tmp_path / "whole"), final(out))
    assert log(out) == [{**r, **{k: log(out)[i][k] for k in ("step_seconds", "elapsed_seconds")}}
                        for i, r in enumerate(log(tmp_path / "whole"))]


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_a_killed_process_resumes_to_the_same_weights(setup, tmp_path, sig):
    run(setup, "multi", tmp_path / "whole", steps=40)
    out = tmp_path / "killed"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src")}
    code = ("import sys, torch; torch.set_num_threads(1); from vitok import adapt; "
            "sys.exit(adapt.train(adapt.parse_args(sys.argv[1:])))")
    # cwd at the code root: under mutmut that is the mutants/ copy, whose config the mutated code looks up
    proc = subprocess.Popen([sys.executable, "-c", code, *argv(setup, "multi", out, steps=40)], env=env, cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and proc.poll() is None:
        if (out / "train_log.jsonl").exists() and len((out / "train_log.jsonl").read_text().splitlines()) >= 5:
            break
        time.sleep(0.05)
    assert proc.poll() is None, proc.stderr.read().decode()
    proc.send_signal(sig)
    rc = proc.wait(60)
    assert rc == (adapt.EXIT_INCOMPLETE if sig == signal.SIGTERM else -signal.SIGKILL)
    if sig == signal.SIGKILL:
        with open(out / "train_log.jsonl", "a") as f:
            f.write('{"step": 99, "lo')  # a line torn by the kill
    assert run(setup, "multi", out, steps=40) == 0
    _same(final(tmp_path / "whole"), final(out))
    assert [r["step"] for r in log(out)] == list(range(40))


def test_micro_batches_and_gradient_checkpointing_do_not_change_the_result(setup, tmp_path):
    run(setup, "multi", tmp_path / "whole")
    run(setup, "multi", tmp_path / "micro", "--micro-batch", "2", "--grad-checkpointing")
    a, b = final(tmp_path / "whole"), final(tmp_path / "micro")
    for k in a["params"]:
        torch.testing.assert_close(a["params"][k], b["params"][k], rtol=1e-4, atol=1e-5)


def test_a_checkpoint_of_another_run_is_refused_and_a_finished_run_is_kept(setup, tmp_path):
    assert run(setup, "multi", tmp_path, "--deadline-minutes", "0") == adapt.EXIT_INCOMPLETE
    with pytest.raises(SystemExit, match=r"another run \(differs in \['embed_lr', 'lr'\]\)"):
        adapt.train(adapt.parse_args([*argv(setup, "multi", tmp_path), "--lr", "1e-3"]))
    while run(setup, "multi", tmp_path) != 0:
        pass
    before = (tmp_path / "final.pt").stat().st_mtime_ns
    assert run(setup, "multi", tmp_path) == 0 and (tmp_path / "final.pt").stat().st_mtime_ns == before


def test_running_out_of_text_stops_with_a_message(setup, tmp_path):
    with pytest.raises(SystemExit, match="training text ran out at step"):
        run(setup, "multi", tmp_path, steps=10_000)


@pytest.mark.parametrize("extra,message", [
    (["--micro-batch", "3"], "multiple of"),
    (["--dtype", "fp16", "--device", "cpu"], "needs a CUDA device"),
    (["--snapshot-steps", "0"], "must lie in"),
    (["--layers", "0,4"], "out of range"),
    (["--layers", "0,-4"], "repeated"),
    (["--eos-token", "<none>"], "has no"),
])
def test_bad_arguments_are_refused(setup, tmp_path, extra, message):
    with pytest.raises((SystemExit, ValueError), match=message):
        adapt.train(adapt.parse_args([*argv(setup, "multi", tmp_path), *extra]))


def test_model_without_room_or_with_an_untied_head_is_refused(setup, tmp_path):
    with pytest.raises(SystemExit, match="extend it first"):
        adapt.train(adapt.parse_args([*argv(setup, "multi", tmp_path), "--model", str(setup / "base-model")]))
    base_tok = Tokenizer.from_file(str(setup / "base.json"))
    multi_tok = Tokenizer.from_file(str(setup / "multi.json"))
    untied = extend.extend(tiny_qwen3(base_tok.get_vocab_size(with_added_tokens=True), tied=False, layers=4),
                           extend.new_token_pieces(base_tok, multi_tok))
    untied.save_pretrained(tmp_path / "untied")
    with pytest.raises(SystemExit, match="untied output head"):
        adapt.train(adapt.parse_args([*argv(setup, "multi", tmp_path / "o"), "--model", str(tmp_path / "untied")]))


def test_new_ids_must_be_one_block_after_the_old_ones(setup):
    base = Tokenizer.from_file(str(setup / "base.json"))
    multi = Tokenizer.from_file(str(setup / "multi.json"))
    ids = adapt.new_token_ids(base, multi)
    assert len(ids) and ids.start == 1 + max(base.get_vocab(with_added_tokens=True).values())
    assert adapt.new_token_ids(base, base) == range(0)
    gap = json.loads(multi.to_str())
    last = max(gap["model"]["vocab"], key=gap["model"]["vocab"].get)
    gap["model"]["vocab"][last] += 5
    with pytest.raises(ValueError, match="one block"):
        adapt.new_token_ids(base, Tokenizer.from_str(json.dumps(gap)))


def test_snapshots_of_the_wrong_shape_or_names_are_refused(setup, tmp_path):
    from transformers import AutoModelForCausalLM
    run(setup, "multi", tmp_path)
    snap = final(tmp_path)
    for change, message in [(lambda s: s["params"].update(bogus=torch.zeros(1)), "not in the model"),
                            (lambda s: s.update(new_ids=[s["new_ids"][0], s["new_ids"][1] - 1]), "has shape")]:
        bad = {**snap, "params": dict(snap["params"])}
        change(bad)
        torch.save(bad, tmp_path / "bad.pt")
        with pytest.raises(ValueError, match=message):
            adapt.load_snapshot(AutoModelForCausalLM.from_pretrained(setup / "multi-model"), tmp_path / "bad.pt")


@given(st.integers(1, 300), st.integers(0, 50), st.floats(0, 0.5))
def test_learning_rate_warms_up_then_decays_to_the_floor(steps, warmup, floor):
    lrs = [adapt.lr_at(s, steps, 1.0, warmup, floor) for s in range(steps)]
    w = min(warmup, steps)
    assert lrs[:w] == pytest.approx([(s + 1) / warmup for s in range(w)])
    after = lrs[w:]
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(after)) and all(floor - 1e-12 <= x <= 1 for x in after)
    if steps > warmup + 1:
        assert after[0] == pytest.approx(1.0) and after[-1] == pytest.approx(floor)


def test_log_keeps_only_steps_the_checkpoint_covers(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text('{"step": 0}\n{"step": 1}\n{"step": 2}\n{"step": 3, "lo', encoding="utf-8")
    adapt._truncate_log(p, 2)
    assert p.read_text() == '{"step": 0}\n{"step": 1}\n'


def expected_batches(d, cond, n):
    """The first n batches the training run must see, built independently from the data module."""
    from vitok import cpt_data
    tok = Tokenizer.from_file(str(d / f"{cond}.json"))
    held = cpt_data.held_out_digests(cpt_data.jsonl_texts(d / "held.jsonl"))
    docs = cpt_data.training_docs(cpt_data.parquet_texts([d / "shard_00000.parquet", d / "shard_00001.parquet"]), held)
    return list(itertools.islice(cpt_data.packed_batches(tok, docs, tok.token_to_id("<|endoftext|>"), SEQ, BATCH), n))


def test_the_log_reports_the_true_loss_gradient_norm_and_text_read(setup, tmp_path):
    from transformers import AutoModelForCausalLM
    run(setup, "multi", tmp_path)
    rec = log(tmp_path)
    batches = expected_batches(setup, "multi", STEPS)
    assert Tokenizer.from_file(str(setup / "base.json")).token_to_id("!") == 0
    assert any((b.input_ids == 0).any() for b in batches)  # the lowest id is trained on
    assert [r["chars"] for r in rec] == list(itertools.accumulate(b.chars for b in batches))
    assert [r["tokens"] for r in rec] == [BATCH * SEQ * (i + 1) for i in range(STEPS)]
    model = AutoModelForCausalLM.from_pretrained(setup / "multi-model")
    base_tok, multi_tok = (Tokenizer.from_file(str(setup / f"{c}.json")) for c in ("base", "multi"))
    new = adapt.new_token_ids(base_tok, multi_tok)
    trainable = adapt.set_trainable(model, new, [0, 3])
    x = torch.from_numpy(batches[0].input_ids)
    loss = model(input_ids=x, labels=x).loss
    loss.backward()
    norm = float(torch.nn.utils.get_total_norm([p.grad for p in trainable.values()]))
    assert rec[0]["loss"] == pytest.approx(float(loss.detach()), rel=1e-5)
    assert rec[0]["grad_norm"] == pytest.approx(norm, rel=1e-4)
    assert all(0 < r["step_seconds"] <= r["elapsed_seconds"] < 600 for r in rec)
    assert [r["lr"] for r in rec] == pytest.approx([adapt.lr_at(i, STEPS, 3e-3, 2, 0.1) for i in range(STEPS)])


def test_log_every_keeps_the_cadence_and_the_last_step(setup, tmp_path):
    run(setup, "base", tmp_path, "--log-every", "3")
    assert [r["step"] for r in log(tmp_path)] == [2, 5, 7]


def test_a_loss_that_stays_non_finite_aborts_the_run(setup, tmp_path):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(setup / "base-model")
    with torch.no_grad():
        model.model.norm.weight.fill_(float("nan"))  # a frozen weight, so the loss is NaN at every step
    model.save_pretrained(tmp_path / "nan-model")
    a = argv(setup, "base", tmp_path / "out", "--max-bad-steps", "2")
    a[a.index("--model") + 1] = str(tmp_path / "nan-model")
    with pytest.raises(SystemExit, match="3 consecutive steps with a non-finite loss or gradient at step 2"):
        adapt.train(adapt.parse_args(a))
    rec = log(tmp_path / "out")
    assert [r["step"] for r in rec] == [0, 1, 2] and not any(r["finite"] for r in rec)


def test_running_out_of_text_leaves_the_last_regular_checkpoint(setup, tmp_path):
    with pytest.raises(SystemExit, match=r"training text ran out at step (\d+)") as e:
        run(setup, "multi", tmp_path, steps=10_000)
    done = int(str(e.value).split("at step ")[1].split()[0])
    ckpt = torch.load(tmp_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert ckpt["kind"] == "checkpoint" and ckpt["step"] == done - done % 2 and done > 2


def test_short_runs_and_generous_deadlines_finish_in_one_call(setup, tmp_path):
    assert run(setup, "base", tmp_path / "short", "--save-every", "50", steps=3) == 0
    assert not (tmp_path / "short" / "checkpoint.pt").exists() and final(tmp_path / "short")["step"] == 3
    assert run(setup, "base", tmp_path / "deadline", "--deadline-minutes", "100", "--snapshot-steps", "2") == 0
    assert final(tmp_path / "deadline")["kind"] == "final"
    snap = torch.load(tmp_path / "deadline" / "snapshots" / "step_000002.pt", map_location="cpu", weights_only=False)
    assert snap["kind"] == "snapshot" and snap["step"] == 2 and snap["tokens"] == 2 * BATCH * SEQ


def test_the_manifest_records_the_run(setup, tmp_path):
    run(setup, "multi", tmp_path)
    man = json.loads((tmp_path / "manifest.json").read_text())
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(adapt.__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unknown"
    from importlib.metadata import version
    assert set(man) == {"signature", "new_tokens", "steps", "tokens", "chars", "git_commit", "packages", "device",
                        "python", "created"}
    assert man["git_commit"] == head and man["device"] == "cpu" and man["python"] == sys.version.split()[0]
    assert man["packages"] == {p: version(p) for p in ("torch", "transformers", "tokenizers")}
    assert man["tokens"] == STEPS * BATCH * SEQ and man["signature"] == final(tmp_path)["signature"]
    assert time.strptime(man["created"], "%Y-%m-%dT%H:%M:%S%z")


def test_the_signature_holds_every_setting_that_changes_the_result(setup, tmp_path):
    args = adapt.parse_args(argv(setup, "multi", tmp_path, "--seed", "3", "--embed-lr", "5e-3"))
    sig = adapt.signature(args, "tok", ["held"])
    assert sig == {"model": str(setup / "multi-model"), "revision": None, "tokenizer_sha256": "tok",
                   "shards": [["shard_00000.parquet", (setup / "shard_00000.parquet").stat().st_size],
                              ["shard_00001.parquet", (setup / "shard_00001.parquet").stat().st_size]],
                   "held_out_sha256": ["held"], "steps": STEPS, "batch": BATCH, "micro_batch": BATCH, "seq_len": SEQ,
                   "lr": 3e-3, "embed_lr": 5e-3, "warmup": 2, "min_lr_ratio": 0.1, "weight_decay": 0.0,
                   "layers": "0,-1", "seed": 3, "dtype": "fp32", "grad_clip": 1.0, "grad_checkpointing": False}


@pytest.mark.parametrize("change", ["tokenizer", "held-out"])
def test_a_checkpoint_made_with_other_input_files_is_refused(setup, tmp_path, change):
    assert run(setup, "multi", tmp_path / "o", "--deadline-minutes", "0") == adapt.EXIT_INCOMPLETE
    a = argv(setup, "multi", tmp_path / "o")
    if change == "tokenizer":  # the same tokenizer, written with other whitespace: other bytes
        (tmp_path / "t.json").write_text(json.dumps(json.loads((setup / "multi.json").read_text()), indent=1))
        a[a.index("--tokenizer") + 1] = str(tmp_path / "t.json")
    else:
        (tmp_path / "h.jsonl").write_text((setup / "held.jsonl").read_text() + json.dumps({"text": "x"}) + "\n")
        a[a.index("--held-out") + 1] = str(tmp_path / "h.jsonl")
    key = {"tokenizer": "tokenizer_sha256", "held-out": "held_out_sha256"}[change]
    with pytest.raises(SystemExit, match=rf"differs in \['{key}'\]"):
        adapt.train(adapt.parse_args(a))


def test_defaults_are_the_approved_design():
    a = adapt.parse_args(["--model", "m", "--base-tokenizer", "b", "--tokenizer", "t", "--shards", "s", "--held-out",
                          "h", "--out", "o", "--steps", "10", "--batch", "32"])
    assert (a.seq_len, a.layers, a.lr, a.embed_lr, a.warmup, a.min_lr_ratio, a.weight_decay, a.grad_clip, a.seed,
            a.dtype, a.micro_batch, a.revision, a.encode_batch, a.save_every, a.log_every, a.max_bad_steps,
            a.snapshot_steps, a.deadline_minutes, a.grad_checkpointing, a.eos_token) == \
        (512, "0,1,-2,-1", 1e-4, 1e-4, 50, 0.1, 0.0, 1.0, 0, "fp32", 32, None, 256, 100, 10, 20, [], None, False,
         "<|endoftext|>")
    assert a.device == ("cuda" if torch.cuda.is_available() else "cpu")
    assert isinstance(a.out, Path) and isinstance(a.tokenizer, Path) and isinstance(a.shards[0], Path)
    required = ["--model", "--base-tokenizer", "--tokenizer", "--shards", "--held-out", "--out", "--steps", "--batch"]
    full = ["--model", "m", "--base-tokenizer", "b", "--tokenizer", "t", "--shards", "s", "--held-out", "h",
            "--out", "o", "--steps", "10", "--batch", "32"]
    for flag in required:
        i = full.index(flag)
        with pytest.raises(SystemExit):
            adapt.parse_args(full[:i] + full[i + 2:])
    with pytest.raises(SystemExit):
        adapt.parse_args([*full, "--dtype", "fp8"])


def test_the_optimizer_decays_only_layer_matrices_and_precision_follows_the_dtype():
    w, n = torch.nn.Parameter(torch.ones(2, 2)), torch.nn.Parameter(torch.ones(2))
    e = torch.nn.Parameter(torch.ones(3, 2))
    opt = adapt.make_optimizer({"model.layers.0.w": w, "model.layers.0.norm": n, adapt.EMBED: e}, 1e-3, 5e-3, 0.1)
    got = [(g["params"], g["weight_decay"], g["peak"], g["lr"], g["betas"], g["eps"]) for g in opt.param_groups]
    assert got == [([w], 0.1, 1e-3, 1e-3, (0.9, 0.95), 1e-8), ([n], 0.0, 1e-3, 1e-3, (0.9, 0.95), 1e-8),
                   ([e], 0.0, 5e-3, 1e-3, (0.9, 0.95), 1e-8)]
    assert [len(g["params"]) for g in adapt.make_optimizer({"model.layers.0.w": w}, 1, 1, 0).param_groups] == [1]
    assert adapt.precision("fp32") == (None, False) and adapt.precision("fp16") == (torch.float16, True)
    assert adapt.precision("bf16") == (torch.bfloat16, False)


def test_only_the_new_rows_receive_their_full_gradient(setup):
    from transformers import AutoModelForCausalLM
    base, multi = (Tokenizer.from_file(str(setup / f"{c}.json")) for c in ("base", "multi"))
    new = adapt.new_token_ids(base, multi)
    masked, plain = (AutoModelForCausalLM.from_pretrained(setup / "multi-model") for _ in range(2))
    adapt.set_trainable(masked, new, [0])
    plain.get_input_embeddings().weight.requires_grad_(True)
    x = torch.tensor([multi.encode(" ".join(SENTENCES), add_special_tokens=False).ids])
    for m in (masked, plain):
        m(input_ids=x, labels=x).loss.backward()
    g, ref = adapt.input_weight(masked).grad, adapt.input_weight(plain).grad
    torch.testing.assert_close(g[new.start:new.stop], ref[new.start:new.stop])
    assert g[:new.start].abs().max() == 0 and g[new.stop:].abs().max() == 0 and ref[new.start:new.stop].abs().max() > 0
    exact = tiny_qwen3(new.stop, layers=4)  # a matrix that ends exactly at the last new id is enough
    assert adapt.EMBED in adapt.set_trainable(exact, new, [0])
    with pytest.raises(ValueError, match="embedding rows"):
        adapt.set_trainable(tiny_qwen3(new.stop - 1, layers=4), new, [0])


def test_new_ids_are_exactly_the_added_tokens(setup):
    base, multi = (Tokenizer.from_file(str(setup / f"{c}.json")) for c in ("base", "multi"))
    old = base.get_vocab(with_added_tokens=True)
    want = sorted(i for t, i in multi.get_vocab(with_added_tokens=True).items() if t not in old)
    assert list(adapt.new_token_ids(base, multi)) == want


@pytest.mark.parametrize("cond", ["multi", "base"])
def test_gradient_checkpointing_trains_the_bottom_layers_in_every_condition(setup, tmp_path, cond):
    run(setup, cond, tmp_path / "plain", steps=3)
    run(setup, cond, tmp_path / "ckpt", "--grad-checkpointing", steps=3)
    a, b = final(tmp_path / "plain")["params"], final(tmp_path / "ckpt")["params"]
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=1e-5)
    start = torch.load(tmp_path / "plain" / "final.pt", map_location="cpu", weights_only=False)
    assert start["params"]["model.layers.0.mlp.down_proj.weight"].abs().sum() > 0


def test_the_model_is_loaded_at_the_pinned_revision_in_fp32(setup, tmp_path, monkeypatch):
    import transformers
    seen = []
    real = transformers.AutoModelForCausalLM.from_pretrained

    def spy(name, **kw):
        seen.append(dict(kw))
        kw.pop("revision")
        return real(name, **kw)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", spy)
    run(setup, "base", tmp_path, "--revision", "abc123", steps=1)
    assert seen == [{"revision": "abc123", "torch_dtype": torch.float32}]


def test_main_exits_with_the_run_status(setup, tmp_path):
    with pytest.raises(SystemExit) as e:
        adapt.main(argv(setup, "base", tmp_path, "--deadline-minutes", "0"))
    assert e.value.code == adapt.EXIT_INCOMPLETE


def test_the_cosine_is_halfway_at_the_middle_of_the_decay():
    # steps 10, warmup 1: decay runs over steps 1..9, so step 5 is its middle
    assert adapt.lr_at(5, 10, 1.0, 1, 0.2) == pytest.approx(0.2 + 0.8 * 0.5)
    assert adapt.lr_at(9, 10, 1.0, 1, 0.2) == pytest.approx(0.2) and adapt.lr_at(1, 10, 1.0, 1, 0.2) == 1.0


@pytest.mark.parametrize("extra,message", [(["--steps", "0"], "must be positive"),
                                           (["--batch", "0"], "must be positive")])
def test_empty_runs_are_refused(setup, tmp_path, extra, message):
    with pytest.raises(SystemExit, match=message):
        adapt.parse_args([*argv(setup, "base", tmp_path), *extra, "--micro-batch", "1"])


def test_numeric_options_are_parsed_as_numbers():
    a = adapt.parse_args(["--model", "m", "--base-tokenizer", "b", "--tokenizer", "t", "--shards", "s", "--held-out",
                          "h", "--out", "o", "--steps", "10", "--batch", "32", "--min-lr-ratio", "0.2",
                          "--grad-clip", "0.5", "--weight-decay", "0.01", "--embed-lr", "1e-3", "--seed", "7"])
    assert (a.min_lr_ratio, a.grad_clip, a.weight_decay, a.embed_lr, a.seed) == (0.2, 0.5, 0.01, 1e-3, 7)
    assert isinstance(a.held_out[0], Path) and isinstance(a.base_tokenizer, Path)


def test_without_git_the_manifest_takes_the_bundle_commit(setup, tmp_path, monkeypatch):
    def no_git(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", no_git)
    monkeypatch.setenv("VITOK_GIT_COMMIT", "b756b73")
    run(setup, "base", tmp_path, steps=1)
    assert json.loads((tmp_path / "manifest.json").read_text())["git_commit"] == "b756b73"
