from vitok.text import nfc, nfd, strip_diacritics, strip_diacritics_partial, tone_of, with_tone


def test_strip_diacritics():
    assert strip_diacritics("Đường phố Hà Nội") == "Duong pho Ha Noi"
    assert strip_diacritics("đã đến Đà Nẵng") == "da den Da Nang"
    assert strip_diacritics("Nghiêng ngả ưu tư") == "Nghieng nga uu tu"


def test_strip_keeps_char_count():
    s = "Việt Nam đẹp lắm, ừ!"
    assert len(strip_diacritics(s)) == len(nfc(s))


def test_nfd_roundtrip():
    s = "Tiếng Việt có dấu"
    assert nfc(nfd(s)) == s
    assert len(nfd(s)) > len(s)


def test_partial_strip_is_deterministic():
    s = "học sinh giỏi nhất trường này"
    assert strip_diacritics_partial(s, 0.5, seed=1) == strip_diacritics_partial(s, 0.5, seed=1)
    assert strip_diacritics_partial(s, 0.0) == s
    assert strip_diacritics_partial(s, 1.0) == strip_diacritics(s)
    # only syllables change: spacing, line breaks and length are kept byte for byte
    spaced = "học  sinh\ngiỏi   nhất"
    for frac in (0.0, 0.5, 1.0):
        out = strip_diacritics_partial(spaced, frac, seed=2)
        assert len(out) == len(spaced) and [i for i, c in enumerate(out) if c in " \n"] == [i for i, c in enumerate(spaced) if c in " \n"]
    assert strip_diacritics_partial(spaced, 1.0) == strip_diacritics(spaced)


def test_tone_change():
    assert tone_of("má") == "́"
    assert tone_of("ma") is None
    assert with_tone("má", "̀") == "mà"
    assert with_tone("mà", None) == "ma"
    assert with_tone("ma", "́") is None  # no tone to swap
    # vowel-quality marks survive, tone stays on the same vowel
    assert with_tone("việt", "́") == "viết"
    assert with_tone("viết", "̣") == "việt"
    assert with_tone("người", "̃") == "ngưỡi"
    assert with_tone("Đặng", "̉") == "Đẳng"
    assert with_tone("hoà", "́") == "hoá"
