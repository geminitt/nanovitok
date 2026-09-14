"""Shared tokenizer definitions. Must import nothing beyond the stdlib, so it also works in the
SuperBPE training environment (which has a forked `tokenizers` and no torch)."""

CONDITIONS = ["bpe-nfc", "bpe-nfd", "super-nfc", "super-nfd"]

# nanochat's special tokens, in nanochat's order.
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

# Stage 1 (subwords): SuperBPE's whitespace pretokenization. Letter classes include \p{M},
# so NFD combining marks stay attached to their base letter. nanochat's own pattern lacks
# \p{M} and would split NFD diacritics off as punctuation.
STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Stage 2 (superwords): SuperBPE's extension regex, which no longer splits on single spaces.
STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"


def parse(condition: str) -> tuple[str, str]:
    """'super-nfd' -> ('super', 'nfd')."""
    algo, norm = condition.split("-")
    assert algo in ("bpe", "super") and norm in ("nfc", "nfd"), condition
    return algo, norm
