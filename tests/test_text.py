import unicodedata

from vitok.text import nfc


def test_nfc_normalizes_combining_marks_and_is_idempotent():
    text = "Tiếng Việt có dấu"
    decomposed = unicodedata.normalize("NFD", text)
    assert nfc(decomposed) == text
    assert nfc(nfc(decomposed)) == text
