"""The Kaggle notebook 03 run end to end on CPU with a tiny Qwen3: every code cell executes once, in order, on a fake
/kaggle/input tree built like the real datasets; only the GPU check and the model download are replaced."""

import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import REPO, SENTENCES, qwen_like_base
from test_extend import tiny_qwen3
from tokenizers import Tokenizer

from vitok import kaggle_session, retrofit

NOTEBOOK = REPO / "kaggle" / "notebooks" / "03_adaptation.ipynb"
NAMES = ("base", "syllable", "multisyllable", "syllable-1000", "multisyllable-1000")


def write_jsonl(path, texts):
    path.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")


@pytest.fixture(scope="module")
def kaggle_input(tmp_path_factory):
    root = tmp_path_factory.mktemp("kaggle")
    code, data = root / "input" / "vitok-code", root / "input" / "vitok-data"
    (code / "src" / "vitok").mkdir(parents=True)
    (code / "pyproject.toml").write_text("")
    (data / "shards").mkdir(parents=True)
    rng = random.Random(0)
    docs = [" ".join(rng.choices(SENTENCES, k=rng.randint(1, 4))) for _ in range(300)]
    docs += [" ".join(SENTENCES[i:] + SENTENCES[:i]) + f" Số {i}." for i in range(6)]  # long enough for the dev set
    pq.write_table(pa.table({"text": docs}), data / "shards" / "shard_00000.parquet")
    pq.write_table(pa.table({"text": docs[:5]}), data / "shards" / "shard_99999.parquet")  # the val shard, unused
    test = [" ".join(SENTENCES[i:] + SENTENCES[:i]) for i in range(6)]
    write_jsonl(data / "test.jsonl", test)
    # like the real vitok-data on Kaggle: no val_docs.jsonl there (the bundle ships it, in retrofit/)
    base = qwen_like_base()
    corpus = root / "vi.txt"
    corpus.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    r = code / "retrofit"
    learned = {"multi": retrofit.learn_merges(base, [corpus], "multi", 120),
               "single": retrofit.learn_merges(base, [corpus], "single", 120)}
    for name in NAMES:
        (r / name).mkdir(parents=True)
        tok = base if name == "base" else retrofit.retrofit(
            base, learned["multi" if name.startswith("multi") else "single"][: 60 if "1000" in name else 120],
            "multi" if name.startswith("multi") else "single")
        (r / name / "tokenizer.json").write_text(json.dumps(tok), encoding="utf-8")
    base_tok = Tokenizer.from_file(str(r / "base" / "tokenizer.json"))
    chars = sum(map(len, test))
    cpt = lambda t: chars / sum(len(t.encode(d, add_special_tokens=False).ids) for d in test)
    for name in NAMES[1:]:
        (r / name / "report.json").write_text(json.dumps(
            {"chars_per_token_base": cpt(base_tok),
             "chars_per_token_new": cpt(Tokenizer.from_file(str(r / name / "tokenizer.json")))}))
    write_jsonl(r / "check_texts.jsonl", ["The quick brown fox, don't stop!", "2026: 1.250.000", *test[:2]])
    write_jsonl(r / "en_docs.jsonl", ["The quick brown fox, don't stop! " * 10])
    items = [{"flores_passage": SENTENCES[i], "question": "Ai?", "mc_answer1": "a", "mc_answer2": "Hà Nội",
              "mc_answer3": "không", "mc_answer4": SENTENCES[i][:10], "correct_answer_num": str(1 + i % 4)}
             for i in range(3)]
    (r / "belebele_vie_Latn.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    write_jsonl(r / "val_docs.jsonl", docs[:10])
    (r / "bundle_manifest.json").write_text(json.dumps({"commit": "test"}))
    rows = base_tok.get_vocab_size(with_added_tokens=True)
    tiny_qwen3(rows, layers=4).save_pretrained(root / "qwen")
    return root


def code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.mark.parametrize("stage", ["probe", "full"])
def test_every_cell_runs_on_a_tiny_model(kaggle_input, tmp_path, monkeypatch, stage):
    root = kaggle_input
    work = tmp_path / "work"
    monkeypatch.setenv("VITOK_KAGGLE_INPUT", str(root / "input"))
    monkeypatch.setenv("VITOK_KAGGLE_WORK", str(work))
    monkeypatch.setattr(kaggle_session, "preflight", lambda need_gpus: [{"name": "cpu"}] * need_gpus)
    monkeypatch.setattr(kaggle_session, "prefetch", lambda model, revision: str(model))
    cells = code_cells()
    ns: dict = {}
    exec(cells[0], ns)  # parameters
    assert (ns["STAGE"], ns["DTYPE"], ns["SEQ_LEN"], ns["DEV_SHARD"], ns["DEV_DOCS"]) == \
        ("probe", "fp16", 512, "shard_00019.parquet", 500)
    ns.update(STAGE=stage, MODEL=str(root / "qwen"), DTYPE="fp32", STEPS=4, SEQ_LEN=32, BATCH=4, MICRO_BATCH=2,
              SNAPSHOT_STEPS=[] if stage == "probe" else [2], SPEED_PROMPTS=0 if stage == "probe" else 1,
              QUEUES={0: ["multisyllable-1000"], 1: ["multisyllable"]} if stage == "probe" else
              {0: ["base"], 1: ["multisyllable-1000"]}, DEV_SHARD="shard_00000.parquet", DEV_DOCS=4)
    for cell in cells[1:]:
        exec(cell, ns)
    assert ns["codes"] == {0: 0, 1: 0}, [p.read_text()[-3000:] for p in work.rglob("*.log")]
    names = ["multisyllable-1000", "multisyllable"] if stage == "probe" else ["base", "multisyllable-1000"]
    report = kaggle_session.report(work, names)
    assert all(report[n]["train"]["steps"] == 4 and report[n]["train"]["non_finite_steps"] == 0 for n in report)
    cfg = json.loads((work / f"config-{stage}.json").read_text())
    assert [p.split("/")[-1] for p in cfg["shards"]] == ["shard_00000.parquet"]
    if stage == "probe":
        assert set(report["multisyllable"]) == {"train", "score-final"} and cfg["score"]["speed_prompts"] == 0
        assert cfg["score"]["vi_docs"] == str(work / "dev_docs.jsonl") and not cfg["score"]["before_training"]
        cmp = json.loads((work / "probe-comparison.json").read_text())
        assert kaggle_session.probe_choice(cmp) in ("multisyllable-1000", "multisyllable")
        assert cmp["vi"]["chars_per_token"] < cmp["vi"]["chars_per_token_reference"]  # +1,000 compresses less
    else:
        assert {"score-before", "score-step_000002", "score-final"} <= set(report["multisyllable-1000"])
        assert "en" in report["base"]["score-final"] and "decode_chars_per_second" in report["base"]["score-final"]
        assert cfg["score"]["vi_docs"].endswith("test.jsonl")
    dev = [json.loads(line)["text"] for line in (work / "dev_docs.jsonl").read_text().splitlines()]
    assert len(dev) == 4 and cfg["held_out"][-1] == str(work / "dev_docs.jsonl")


def test_the_configuration_cell_refuses_a_full_run_without_its_step_count(kaggle_input, tmp_path, monkeypatch):
    monkeypatch.setenv("VITOK_KAGGLE_INPUT", str(kaggle_input / "input"))
    monkeypatch.setenv("VITOK_KAGGLE_WORK", str(tmp_path))
    monkeypatch.setattr(kaggle_session, "preflight", lambda need_gpus: [])
    monkeypatch.setattr(kaggle_session, "prefetch", lambda model, revision: str(model))
    cells = code_cells()
    ns: dict = {}
    exec(cells[0], ns)
    for cell in cells[1:3]:
        exec(cell, ns)
    ns.update(STAGE="full", STEPS=None)
    with pytest.raises(AssertionError, match="set STEPS"):
        exec(cells[3], ns)
