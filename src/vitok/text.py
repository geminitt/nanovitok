"""Vietnamese text utilities: normalization, diacritic stripping, tone manipulation."""

import random
import unicodedata

# Combining tone marks (NFD). Vowel-quality marks (circumflex U+0302, breve U+0306,
# horn U+031B) are not tones and are kept when changing tone.
TONE_MARKS = {
    "̀": "huyền",
    "́": "sắc",
    "̃": "ngã",
    "̉": "hỏi",
    "̣": "nặng",
}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def strip_diacritics(s: str) -> str:
    """Remove all combining marks and map đ/Đ -> d/D. Keeps the NFC character count unchanged."""
    d = "".join(ch for ch in nfd(s) if unicodedata.category(ch) != "Mn")
    return nfc(d.replace("đ", "d").replace("Đ", "D"))


def strip_diacritics_partial(s: str, frac: float, seed: int = 0) -> str:
    """Strip diacritics from a random `frac` of whitespace-separated syllables."""
    rng = random.Random(seed)
    parts = s.split(" ")
    return " ".join(strip_diacritics(p) if rng.random() < frac else p for p in parts)


def tone_of(syllable: str) -> str | None:
    """Return the tone mark (NFD combining char) of a syllable, or None for thanh ngang."""
    marks = [ch for ch in nfd(syllable) if ch in TONE_MARKS]
    return marks[0] if marks else None


def with_tone(syllable: str, tone: str | None) -> str | None:
    """Swap the tone of a syllable that already carries one, keeping it on the same vowel.

    `tone` is a key of TONE_MARKS, or None to remove it (thanh ngang). Returns None when the
    syllable has no tone to swap, since placing a new tone needs orthographic rules.
    """
    chars = list(nfd(syllable))
    pos = next((i for i, ch in enumerate(chars) if ch in TONE_MARKS), None)
    if pos is None:
        return None
    del chars[pos]
    if tone is not None:
        # Insert after the whole mark run of the tone's vowel, so NFC composes the
        # quality mark (â, ơ, ...) first and the tone on top of it.
        j = pos
        while j < len(chars) and unicodedata.category(chars[j]) == "Mn":
            j += 1
        chars.insert(j, tone)
    return nfc("".join(chars))
