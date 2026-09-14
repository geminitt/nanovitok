"""nanochat-compatible wrapper around a Hugging Face `tokenizer.json`.

Implements the subset of nanochat's RustBPETokenizer interface used by pretraining and
evaluation. Decoded text is returned in NFC regardless of the tokenizer's normalizer.
"""

import os
import unicodedata

from tokenizers import Tokenizer


class HFTokenizer:
    def __init__(self, tok: Tokenizer, bos_token: str = "<|bos|>"):
        self.tok = tok
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        return cls(Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json")))

    def get_vocab_size(self):
        return self.tok.get_vocab_size(with_added_tokens=True)

    def get_special_tokens(self):
        return {t.content for t in self.tok.get_added_tokens_decoder().values() if t.special}

    def encode_special(self, text):
        token_id = self.tok.token_to_id(text)
        assert token_id is not None, f"unknown special token {text}"
        return token_id

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        pre = None if prepend is None else (prepend if isinstance(prepend, int) else self.encode_special(prepend))
        app = None if append is None else (append if isinstance(append, int) else self.encode_special(append))

        def wrap(ids):
            return ([pre] if pre is not None else []) + ids + ([app] if app is not None else [])

        if isinstance(text, str):
            return wrap(self.tok.encode(text, add_special_tokens=False).ids)
        if isinstance(text, list):
            return [wrap(e.ids) for e in self.tok.encode_batch(text, add_special_tokens=False)]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return unicodedata.normalize("NFC", self.tok.decode(ids, skip_special_tokens=False))

    def id_to_token(self, token_id):
        return self.decode([token_id])


def write_token_bytes(tokenizer_dir):
    """Write nanochat's token_bytes.pt (UTF-8 bytes per token, 0 for special tokens).

    Only used for nanochat's bpb monitoring during training; conclusions use bits per NFC
    character from vitok.eval (bpb favours NFD tokenizers, whose text has more bytes).
    """
    import torch

    hf = HFTokenizer.from_directory(tokenizer_dir)
    special = hf.get_special_tokens()
    # Byte-level tokens are strings over GPT-2's byte->unicode alphabet: one char per byte.
    # Counting chars avoids decoding partial UTF-8 sequences into replacement characters.
    counts = [0] * hf.get_vocab_size()
    for token, token_id in hf.tok.get_vocab(with_added_tokens=True).items():
        counts[token_id] = 0 if token in special else len(token)
    torch.save(torch.tensor(counts, dtype=torch.int32), os.path.join(tokenizer_dir, "token_bytes.pt"))
