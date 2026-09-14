import json

from tokenizers import Tokenizer

from conftest import SENTENCES
from vitok.compression import stats_for
from vitok.hf_tokenizer import HFTokenizer
from vitok.text import nfc
from vitok.tokenizer_spec import CONDITIONS, SPECIAL_TOKENS

UNSEEN = "Ωμέγα ☃ 中文 emoji 🎉 và tiếng Việt"


def test_roundtrip_all_conditions(tokenizers_dir):
    for cond in CONDITIONS:
        hf = HFTokenizer.from_directory(tokenizers_dir / cond)
        for text in SENTENCES + [UNSEEN]:
            assert hf.decode(hf.encode(text)) == nfc(text), (cond, text)


def test_unseen_bytes_are_not_dropped(tokenizers_dir):
    # characters absent from the training corpus must still be encoded (full byte alphabet)
    for cond in CONDITIONS:
        hf = HFTokenizer.from_directory(tokenizers_dir / cond)
        assert hf.decode(hf.encode("🎉")) == "🎉"


def test_special_tokens_and_bos(tokenizers_dir):
    hf = HFTokenizer.from_directory(tokenizers_dir / "bpe-nfc")
    ids = hf.encode("xin chào", prepend="<|bos|>")
    assert ids[0] == hf.get_bos_token_id()
    assert hf.get_special_tokens() == set(SPECIAL_TOKENS)
    batch = hf.encode(["a", "b"], prepend=hf.get_bos_token_id())
    assert all(row[0] == hf.get_bos_token_id() for row in batch)


def test_nfd_tokenizers_see_combining_marks(tokenizers_dir):
    nfd_tok = Tokenizer.from_file(str(tokenizers_dir / "bpe-nfd" / "tokenizer.json"))
    nfc_tok = Tokenizer.from_file(str(tokenizers_dir / "bpe-nfc" / "tokenizer.json"))
    word = "Việt"
    # same text, NFD tokenizer works on decomposed bytes (dot below U+0323 = 0xCC 0xA3)
    assert nfd_tok.normalizer is not None
    assert len(nfd_tok.encode(word).ids) >= 1 and len(nfc_tok.encode(word).ids) >= 1
    nfd_bytes = "".join(nfd_tok.encode(word).tokens)
    assert "Ì£" in nfd_bytes  # byte-level rendering of U+0323


def test_superbpe_inherits_merges(tokenizers_dir):
    meta = json.loads((tokenizers_dir / "train_meta.json").read_text())
    for norm in ("nfc", "nfd"):
        assert meta[norm]["n_alphabet"] == 256
        assert meta[norm]["n_inherited_merges"] == round(0.9 * 600) - 256


def test_compression_stats(tokenizers_dir):
    tok = Tokenizer.from_file(str(tokenizers_dir / "super-nfc" / "tokenizer.json"))
    s = stats_for(tok, SENTENCES)
    assert s["chars_per_token"] > 1
    assert 0 <= s["superword_token_share"] <= 1
