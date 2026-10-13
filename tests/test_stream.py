"""Training-stream contracts: a seed fixes one document order, held-out texts never enter, parts close at row-group
boundaries, an extended stream continues the same order, a crashed part is rewritten, other seeds are refused."""

import json

import pytest

from vitok import cpt_data, stream

UNITS = [(f"f{f}", rg) for f in range(3) for rg in range(4)]


def text(name, rg, i):
    return f"doc {name}-{rg}-{i} " + "chữ " * (20 + i)


def reader(log=None):
    def read(name, rg):
        if log is not None:
            log.append((name, rg))
        return [text(name, rg, i) for i in range(5)]
    return read


def build(tmp, seed=0, max_chars=2000, part_chars=500, held=frozenset(), read=None):
    return stream.build(tmp, seed, max_chars, part_chars, set(held), UNITS, read or reader(), {"src": 1})


def texts(folder):
    return list(stream.stream_texts(folder))


def test_a_seed_fixes_the_order_and_another_seed_changes_it(tmp_path):
    a, b, c = (tmp_path / x for x in "abc")
    build(a), build(b), build(c, seed=1)
    assert texts(a) == texts(b)
    assert texts(a) != texts(c)
    order = stream.unit_order(UNITS, 0)
    assert sorted(order) == sorted(UNITS) and order == stream.unit_order(UNITS, 0)


def test_held_out_texts_never_enter_and_are_counted(tmp_path):
    long = "văn bản dài " * 300  # 3,600 characters: held out by its test-style cut, not its full text
    def read(name, rg):
        return [text(name, rg, i) for i in range(4)] + [f"{name}-{rg} " + long]
    first = read(*stream.unit_order(UNITS, 0)[0])
    held = {cpt_data.digest(first[1]), cpt_data.digest(cpt_data.eval_text(first[4]))}
    man = build(tmp_path, held=held, read=read, max_chars=30000)
    got = texts(tmp_path)
    assert first[1] not in got and first[4] not in got and first[0] in got
    assert man["skipped_held_out"] == 2
    assert man["docs"] == len(got) and man["chars"] == sum(map(len, got))


def test_parts_close_at_unit_boundaries_and_reads_stop_after_the_budget(tmp_path):
    log = []
    man = build(tmp_path, max_chars=1200, part_chars=400, read=reader(log))
    assert man["chars"] >= 1200 and all(p["chars"] >= 400 or p is man["parts"][-1] for p in man["parts"])
    assert [u for p in man["parts"] for u in p["units"]] == man["units_read"]
    assert man["next_unit"] == len(man["units_read"])
    assert log == [tuple(u) for u in man["units_read"]] + [tuple(stream.unit_order(UNITS, 0)[man["next_unit"]])]
    for p in man["parts"]:  # each part holds whole row groups: 5 documents per unit, none held out here
        assert p["docs"] == 5 * len(p["units"])


def test_an_extended_stream_continues_the_same_order(tmp_path):
    short, long = tmp_path / "s", tmp_path / "l"
    build(short, max_chars=900)
    n = len(texts(short))
    build(short, max_chars=2500)
    build(long, max_chars=2500)
    assert texts(short) == texts(long)[:len(texts(short))] and len(texts(short)) > n
    before = json.loads((short / "stream_manifest.json").read_text())
    assert build(short, max_chars=100) == before  # already long enough: nothing changes


def test_a_part_left_open_by_a_crash_is_rewritten_and_a_changed_part_refused(tmp_path):
    build(tmp_path, max_chars=900)
    (tmp_path / "part-00099.jsonl").write_text("torn")
    build(tmp_path, max_chars=1800)
    assert not (tmp_path / "part-00099.jsonl").exists()
    reference = tmp_path / "ref"
    build(reference, max_chars=1800)
    assert texts(tmp_path) == texts(reference)
    first = tmp_path / "part-00000.jsonl"
    first.write_text(first.read_text() + "\n")
    with pytest.raises(SystemExit, match="changed since it was written"):
        build(tmp_path, max_chars=5000)
    with pytest.raises(SystemExit, match="does not match its manifest"):
        texts(tmp_path)


def test_another_seed_or_source_is_refused_and_too_little_text_is_reported(tmp_path):
    build(tmp_path)
    with pytest.raises(SystemExit, match="another seed or source"):
        build(tmp_path, seed=3)
    with pytest.raises(SystemExit, match="another seed or source"):
        stream.build(tmp_path, 0, 2000, 500, set(), UNITS, reader(), {"src": 2})
    with pytest.raises(SystemExit, match="hold only"):
        build(tmp_path / "x", max_chars=10**9)
