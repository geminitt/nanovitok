"""NFC normalization shared by the tokenizers and corpus pipeline."""

import unicodedata


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)
