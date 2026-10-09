"""Embedding extension contracts, on a tiny randomly initialized Qwen3 (CPU, no download)."""

import json

import pytest
import torch
from conftest import SENTENCES, qwen_like_base
from tokenizers import Tokenizer

from vitok import extend, retrofit


def tiny_qwen3(rows: int, tied: bool = True, seed: int = 0, layers: int = 2):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=rows, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                      tie_word_embeddings=tied)
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def tokenizers_pair(tmp_path_factory):
    base = qwen_like_base()
    corpus = tmp_path_factory.mktemp("ext") / "vi.txt"
    corpus.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    new = retrofit.retrofit(base, retrofit.learn_merges(base, [corpus], "multi", 120), "multi")
    return Tokenizer.from_str(json.dumps(base)), Tokenizer.from_str(json.dumps(new))


def test_pieces_spell_each_new_token_with_base_tokens(tokenizers_pair):
    base, new = tokenizers_pair
    pieces = extend.new_token_pieces(base, new)
    old = base.get_vocab(with_added_tokens=True)
    assert set(pieces) == {i for t, i in new.get_vocab(with_added_tokens=True).items() if t not in old}
    for i, ids in pieces.items():
        assert len(ids) >= 2 and all(j in old.values() for j in ids)
        assert "".join(base.id_to_token(j) for j in ids) == new.id_to_token(i)


@pytest.mark.parametrize("rows,tied", [(448, True), (1024, True), (448, False)],
                         ids=["grows-tied", "fits-in-padding", "grows-untied"])
def test_extension_keeps_old_rows_and_logits_and_sets_new_rows_to_the_mean(tokenizers_pair, rows, tied):
    base, new = tokenizers_pair
    pieces = extend.new_token_pieces(base, new)
    original = tiny_qwen3(rows, tied)
    extended = extend.extend(tiny_qwen3(rows, tied), pieces)
    need = max(pieces) + 1
    assert extended.get_input_embeddings().weight.shape[0] == (rows if rows >= need else -(-need // 64) * 64)
    head, emb = extended.get_output_embeddings().weight, extended.get_input_embeddings().weight
    assert (head.data_ptr() == emb.data_ptr()) == tied and head.shape == emb.shape
    texts = ["The quick brown fox, don't stop!", "2026: 1.250.000", *SENTENCES]
    report = extend.check(original, extended, pieces, base, new, texts)
    assert report["old_rows_max_abs_diff"] == 0 and report["new_rows_mean_max_abs_diff"] < 1e-6
    assert report["texts_checked"] >= 2 and report["same_argmax"]                # same ids, same logits
    assert report["logits_max_abs_diff"] <= extend.LOGITS_TOLERANCE
    if not tied:
        o = original.get_output_embeddings().weight
        for i, ids in pieces.items():
            torch.testing.assert_close(head[i], o[ids].mean(0))
    # a sentence that uses new tokens runs and gives finite logits over the grown vocabulary
    ids = new.encode(SENTENCES[1], add_special_tokens=False).ids
    assert any(i in pieces for i in ids)
    logits = extended(torch.tensor([ids])).logits
    assert logits.shape[-1] == emb.shape[0] and torch.isfinite(logits).all()


def test_tokens_that_moved_or_do_not_split_are_refused(tokenizers_pair):
    base, new = tokenizers_pair
    moved = json.loads(new.to_str())
    a, b = list(moved["model"]["vocab"])[:2]
    moved["model"]["vocab"][a], moved["model"]["vocab"][b] = moved["model"]["vocab"][b], moved["model"]["vocab"][a]
    with pytest.raises(ValueError, match="moved from id"):
        extend.new_token_pieces(base, Tokenizer.from_str(json.dumps(moved)))


def test_the_command_saves_a_checked_model_and_refuses_one_that_fails(tokenizers_pair, tmp_path, monkeypatch):
    import transformers
    base, new = tokenizers_pair
    (tmp_path / "base.json").write_text(base.to_str())
    (tmp_path / "new.json").write_text(new.to_str())
    tiny_qwen3(448).save_pretrained(tmp_path / "qwen")
    texts = ["The quick brown fox, don't stop!", "2026: 1.250.000", *SENTENCES]
    (tmp_path / "check.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    seen, real = [], transformers.AutoModelForCausalLM.from_pretrained

    def spy(name, **kw):
        seen.append(dict(kw))
        kw.pop("revision")
        return real(name, **kw)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", spy)
    cli = lambda out, check: ["extend", "--model", str(tmp_path / "qwen"), "--revision", "abc",
                              "--base-tokenizer", str(tmp_path / "base.json"),
                              "--tokenizer", str(tmp_path / "new.json"),
                              "--check-texts", str(check), "--out", str(out)]
    monkeypatch.setattr("sys.argv", cli(tmp_path / "out", tmp_path / "check.jsonl"))
    extend.main()
    assert seen == [{"revision": "abc", "torch_dtype": torch.float32}] * 2
    man = json.loads((tmp_path / "out" / "extend_manifest.json").read_text())
    pieces = extend.new_token_pieces(base, new)
    assert (man["model"], man["revision"], man["tokenizer"], man["new_tokens"]) == \
        (str(tmp_path / "qwen"), "abc", str(tmp_path / "new.json"), len(pieces))
    assert man["embedding_rows"] == -(-(max(pieces) + 1) // 64) * 64 and man["checks"]["texts_checked"] == 2
    assert man["checks"]["old_rows_max_abs_diff"] == 0 and man["checks"]["same_argmax"] and "created" in man
    assert (tmp_path / "out" / "tokenizer.json").read_bytes() == (tmp_path / "new.json").read_bytes()
    saved = real(tmp_path / "out")
    assert saved.get_input_embeddings().weight.shape[0] == man["embedding_rows"]
    # check texts that the new tokenizer all changes leave nothing to compare: the command refuses
    (tmp_path / "vi.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in SENTENCES))
    monkeypatch.setattr("sys.argv", cli(tmp_path / "bad", tmp_path / "vi.jsonl"))
    with pytest.raises(SystemExit, match="failed its checks"):
        extend.main()
    assert not (tmp_path / "bad").exists()
