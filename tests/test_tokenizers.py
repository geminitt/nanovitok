import json

import pytest
from conftest import SENTENCES
from tokenizers import Tokenizer

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


def test_token_bytes_are_utf8_lengths(tokenizers_dir, tmp_path):
    import shutil

    import torch

    from vitok.hf_tokenizer import write_token_bytes
    d = tmp_path / "tok"
    shutil.copytree(tokenizers_dir / "bpe-nfd", d)
    write_token_bytes(d)
    tb = torch.load(d / "token_bytes.pt")
    hf = HFTokenizer.from_directory(d)
    assert tb.dtype == torch.int32 and tb.shape[0] == hf.get_vocab_size()
    for token, i in hf.tok.get_vocab(with_added_tokens=True).items():
        if token in hf.get_special_tokens():
            assert tb[i] == 0
        else:  # byte-level tokens spell each byte as one character
            assert tb[i] == len(token)


def test_special_tokens_and_bos(tokenizers_dir):
    hf = HFTokenizer.from_directory(tokenizers_dir / "bpe-nfc")
    ids = hf.encode("xin chào", prepend="<|bos|>")
    assert ids[0] == hf.get_bos_token_id()
    assert hf.get_special_tokens() == set(SPECIAL_TOKENS)
    batch = hf.encode(["a", "b"], prepend=hf.get_bos_token_id())
    assert all(row[0] == hf.get_bos_token_id() for row in batch)
    # prepend/append accept an id or a special-token string, for one text or a batch, and add nothing else
    plain = hf.encode("xin chào")
    end = hf.encode_special("<|assistant_end|>")
    for app in (end, "<|assistant_end|>"):
        assert hf.encode("xin chào", append=app) == plain + [end]
        assert hf.encode(["xin chào"], prepend="<|bos|>", append=app) == [[hf.get_bos_token_id()] + plain + [end]]
    assert hf.encode(["a", "b"]) == [hf.encode("a"), hf.encode("b")]
    with pytest.raises(ValueError, match="Invalid input type"):
        hf.encode(42)
    # decoding keeps special tokens (nanochat reads them back) and returns NFC
    ids = hf.encode("xin chào", prepend="<|bos|>", append="<|assistant_end|>")
    assert hf.decode(ids) == "<|bos|>xin chào<|assistant_end|>"


def test_nfd_tokenizers_see_combining_marks(tokenizers_dir):
    nfd_tok = Tokenizer.from_file(str(tokenizers_dir / "bpe-nfd" / "tokenizer.json"))
    nfc_tok = Tokenizer.from_file(str(tokenizers_dir / "bpe-nfc" / "tokenizer.json"))
    word = "Việt"
    # same text, NFD tokenizer works on decomposed bytes (dot below U+0323 = 0xCC 0xA3)
    assert nfd_tok.normalizer is not None
    assert len(nfd_tok.encode(word).ids) >= 1 and len(nfc_tok.encode(word).ids) >= 1
    nfd_bytes = "".join(nfd_tok.encode(word).tokens)
    assert "Ì£" in nfd_bytes  # byte-level rendering of U+0323


def test_pretokenizers_keep_nfd_marks_on_letters():
    import unicodedata

    from tokenizers import Regex, pre_tokenizers

    from vitok.tokenizer_spec import STAGE1_REGEX, STAGE2_REGEX

    text = unicodedata.normalize("NFD", "Một người Việt đến.. tốt")
    for regex in (STAGE1_REGEX, STAGE2_REGEX):
        split = pre_tokenizers.Split(pattern=Regex(regex), behavior="isolated", invert=False)
        pieces = [p for p, _ in split.pre_tokenize_str(text)]
        # no piece may start with a combining mark (a mark cut off from its letter)
        assert not any(unicodedata.category(p[0]) == "Mn" for p in pieces), (regex, pieces)


def test_superbpe_inherits_merges(tokenizers_dir):
    meta = json.loads((tokenizers_dir / "train_meta.json").read_text())
    for norm in ("nfc", "nfd"):
        assert meta[norm]["n_alphabet"] == 256
        assert meta[norm]["n_inherited_merges"] == round(0.9 * 600) - 256


@pytest.mark.parametrize("cond", ["bpe-nfc", "super-nfc"])
def test_compression_stats_are_the_counts(tokenizers_dir, cond):
    import collections
    import re
    tok = Tokenizer.from_file(str(tokenizers_dir / cond / "tokenizer.json"))
    got = stats_for(tok, SENTENCES, top_k=5)
    ids = [i for d in SENTENCES for i in tok.encode(d, add_special_tokens=False).ids]
    vocab = tok.get_vocab(with_added_tokens=False)
    superwords = {i for t, i in vocab.items() if "Ġ" in t.strip("Ġ")}  # a space inside the token, not at its ends
    used = collections.Counter(ids)
    syllables = sum(len(re.findall(r"[^\W\d_]+", d)) for d in SENTENCES)
    chars = sum(map(len, SENTENCES))
    assert got["tokens"] == len(ids) and got["chars_nfc"] == chars and got["vocab_size"] == len(vocab)
    assert got["chars_per_token"] == pytest.approx(chars / len(ids))
    assert got["tokens_per_syllable"] == pytest.approx(len(ids) / syllables)
    assert got["superword_vocab"] == len(superwords)
    assert got["superword_token_share"] == pytest.approx(sum(used[i] for i in superwords) / len(ids))
    want_top = [(tok.decode([i]), c) for i, c in used.most_common() if i in superwords][:5]
    assert got["top_superwords"] == want_top
    is_super = cond == "super-nfc"
    assert (len(superwords) > 0) == is_super and (got["superword_token_share"] > 0) == is_super
