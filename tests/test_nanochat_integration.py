"""CPU-only integration with patched nanochat: dataloader, checkpoint loading, and evaluation math.
No training happens here; the model is randomly initialised."""

import json
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from conftest import BASE_DIR, SENTENCES, requires_nanochat

pytestmark = requires_nanochat


@pytest.fixture(scope="module")
def nanochat_base(tokenizers_dir):
    """Populate NANOCHAT_BASE_DIR with a tokenizer and tiny data shards, like a Kaggle run dir."""
    from vitok.hf_tokenizer import write_token_bytes

    tok_dir = BASE_DIR / "tokenizer"
    shutil.rmtree(tok_dir, ignore_errors=True)
    shutil.copytree(tokenizers_dir / "super-nfd", tok_dir)
    write_token_bytes(tok_dir)
    data_dir = BASE_DIR / "base_data_climbmix"
    data_dir.mkdir(exist_ok=True)
    for name in ("shard_00000.parquet", "shard_99999.parquet"):
        pq.write_table(pa.table({"text": SENTENCES * 20}), data_dir / name, row_group_size=32)
    return BASE_DIR


@pytest.fixture(scope="module")
def tiny_model(nanochat_base):
    from nanochat.checkpoint_manager import build_model, save_checkpoint
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer

    tok = get_tokenizer()
    cfg = GPTConfig(sequence_len=64, vocab_size=tok.get_vocab_size(), n_layer=2, n_head=2,
                    n_kv_head=2, n_embd=64, window_pattern="L")
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    from dataclasses import asdict
    ckpt = nanochat_base / "base_checkpoints" / "dtest"
    save_checkpoint(str(ckpt), 0, model.state_dict(), None, {"model_config": asdict(cfg), "user_config": {}})
    model, tok, meta = build_model(str(ckpt), 0, torch.device("cpu"), "eval")
    return model, tok, meta


def test_get_tokenizer_returns_hf(nanochat_base):
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from vitok.hf_tokenizer import HFTokenizer

    tok = get_tokenizer()
    assert isinstance(tok, HFTokenizer)
    tb = get_token_bytes()
    assert tb.shape[0] == tok.get_vocab_size()
    assert tb[tok.get_bos_token_id()] == 0


def test_dataloader_rows_start_with_bos(nanochat_base):
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    from nanochat.tokenizer import get_tokenizer

    tok = get_tokenizer()
    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, 2, 32, "train", device="cpu", buffer_size=16)
    x, y = next(loader)
    assert x.shape == (2, 32)
    assert (x[:, 0] == tok.get_bos_token_id()).all()
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_padding_does_not_change_nats(tiny_model):
    from vitok.eval import sequence_nats

    model, tok, meta = tiny_model
    short, long = SENTENCES[0][:20], " ".join(SENTENCES[:2])
    alone = sequence_nats(model, tok, [short], max_len=64, batch_size=1)[0]
    batched = sequence_nats(model, tok, [long, short], max_len=64, batch_size=2)
    assert batched[1] == pytest.approx(alone, rel=1e-5)
    assert batched[0] is not None


def test_too_long_documents_are_skipped(tiny_model):
    from vitok.eval import sequence_nats

    model, tok, _ = tiny_model
    out = sequence_nats(model, tok, [" ".join(SENTENCES * 5), "ngắn"], max_len=64)
    assert out[0] is None and out[1] is not None


def test_random_model_bpc_near_uniform(tiny_model):
    import math
    from vitok.eval import sequence_nats
    from vitok.stats import bpc

    model, tok, _ = tiny_model
    texts = [s[:60] for s in SENTENCES]
    nats = sequence_nats(model, tok, texts, max_len=64)
    n_tokens = sum(len(tok.encode(t)) for t in texts)
    per_token_bits = bpc(nats, [1]) * 1 / n_tokens  # bits per token
    uniform = math.log2(tok.get_vocab_size())
    assert abs(per_token_bits - uniform) / uniform < 0.15
