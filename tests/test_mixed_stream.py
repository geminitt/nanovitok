"""Mixed-stream contracts: training text is cut like the test sets (so a test document is caught by its window),
dev sets avoid test documents, test and dev documents never enter training, every prefix of the stream holds about
equal characters per domain, and a rerun keeps what is finished."""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from vitok import cpt_data, evalsets, mixed_stream, stream
from vitok.data import eval_text

HEAD = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n" * 30  # letterhead longer than the skip


def wiki_text(i):
    return f"Bài {i}. " + f"Nội dung số {i} của bài viết. " * (110 + i % 7)


def legal_html(i):
    body = "".join(f"<p>Điều {k} của văn bản {i}: quy định chi tiết.</p>" for k in range(90 + i % 5))
    return "<div>" + HEAD.replace("\n", "<br>") + "</div>" + body


@given(st.text(alphabet="ab \nĐđ", min_size=0, max_size=4000))
def test_training_text_is_cut_like_the_test_set(text):
    body = mixed_stream.wiki_body(text)
    assert evalsets.wiki_window(text) == (eval_text(body) if body is not None else None)
    html = text.replace("\n", "<br>")
    body = mixed_stream.legal_body(html)
    assert evalsets.legal_window(evalsets.html_text(html)) == (eval_text(body) if body is not None else None)


def rows(domain, n):
    if domain == "wikipedia":
        return [{"id": f"w{i}", "text": wiki_text(i)} for i in range(n)]
    return [{"id": i, "content_html": legal_html(i)} for i in range(n)]


def test_dev_sets_avoid_test_documents_by_id_and_by_text():
    src = rows("wikipedia", 40)
    test = [{"id": "w3", "text": evalsets.wiki_window(src[3]["text"])},
            {"id": "other", "text": evalsets.wiki_window(src[5]["text"])}]  # same text under another id
    dev = mixed_stream.dev_set("wikipedia", src, test, n=10)
    assert len(dev) == 10 and not {"w3", "w5"} & {r["id"] for r in dev}
    assert all(r["text"] == evalsets.wiki_window(src[int(r["id"][1:])]["text"]) for r in dev)
    assert dev == mixed_stream.dev_set("wikipedia", src, test, n=10)
    with pytest.raises(ValueError, match="usable documents"):
        mixed_stream.dev_set("wikipedia", src, test, n=39)


def gen(prefix, sizes):
    return iter([prefix * s for s in sizes])


@given(st.lists(st.integers(1, 30), min_size=1, max_size=60), st.lists(st.integers(1, 30), min_size=1, max_size=60),
       st.lists(st.integers(1, 30), min_size=1, max_size=60), st.integers(1, 200))
def test_every_prefix_holds_about_equal_characters_per_domain(a, b, c, target):
    streams = {"web": gen("w", a), "wikipedia": gen("k", b), "legal": gen("l", c)}
    if min(sum(a), sum(b), sum(c)) < target:
        with pytest.raises(SystemExit, match="stream ends"):
            list(mixed_stream.interleave(streams, target))
        return
    chars = dict.fromkeys(streams, 0)
    biggest = max(a + b + c)
    for d, text in mixed_stream.interleave(streams, target):
        assert chars[d] < target  # a full domain gives no more documents
        chars[d] += len(text)
        assert max(chars.values()) - min(chars.values()) <= biggest + target if min(chars.values()) >= target \
            else max(chars.values()) - min(chars.values()) <= biggest
    assert all(target <= v < target + 30 for v in chars.values())


def test_ties_follow_the_domain_order():
    out = list(mixed_stream.interleave({"legal": gen("l", [2] * 3), "web": gen("w", [2] * 3),
                                        "wikipedia": gen("k", [2] * 3)}, 4))
    assert [d for d, _ in out] == ["web", "wikipedia", "legal"] * 2


def fake_sources(n_wiki=60, n_legal=60, groups=6):
    data = {"wikipedia": rows("wikipedia", n_wiki), "legal": rows("legal", n_legal)}
    names = {"wikipedia": evalsets.WIKIPEDIA_FILES[:1], "legal": [evalsets.LEGAL_FILE]}

    def units(domain):
        return [(names[domain][0], g) for g in range(groups)]

    def read_rows(domain):
        def read(name, rg):
            return data[domain][rg::groups]
        return read

    return units, read_rows, lambda domain: iter(data[domain]), data


def write(path, recs):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


def setup(tmp_path, data):
    ev = tmp_path / "evalsets"
    ev.mkdir()
    write(ev / "wikipedia.jsonl", [{"id": f"w{i}", "text": evalsets.wiki_window(wiki_text(i))} for i in (1, 2)])
    write(ev / "legal.jsonl", [{"id": "4", "text": evalsets.legal_window(evalsets.html_text(legal_html(4)))}])
    web_held = "Trang web giữ lại. " * 30
    for name in stream.HELD_OUT_SETS:
        if name not in ("wikipedia", "legal"):
            write(ev / f"{name}.jsonl", [{"id": "h", "text": web_held}] if name == "fineweb2" else [])
    (ev / "evalsets_manifest.json").write_text("{}")
    web_texts = [f"Trang web {i}. " + "nội dung " * (40 + i % 9) for i in range(300)]
    web_texts.insert(5, web_held)
    web = tmp_path / "web-stream"
    mixed_stream.prefix_stream(web, web_texts, set(), sum(map(len, web_texts)), {"s": 1})  # kept in this order
    return ev, web, web_texts, web_held


def test_the_built_stream_keeps_test_and_dev_text_out_and_balances_domains(tmp_path):
    units, read_rows, all_rows, data = fake_sources()
    ev, web, web_texts, web_held = setup(tmp_path, data)
    out = tmp_path / "out"
    target = 20_000
    man = mixed_stream.build(ev, web, target, out, units, read_rows, all_rows, dev_docs=5)
    recs = list(mixed_stream.mixed_records(out / "stream"))
    texts = {d: [r["text"] for r in recs if r["domain"] == d] for d in mixed_stream.DOMAINS}
    dev = {d: cpt_data.jsonl_rows(out / f"dev-{d}.jsonl") for d in ("wikipedia", "legal")}
    assert all(len(v) == 5 for v in dev.values())
    assert not {r["id"] for r in dev["wikipedia"]} & {"w1", "w2"} and "4" not in {r["id"] for r in dev["legal"]}
    held = {r["text"] for rows_ in dev.values() for r in rows_} | {web_held} | {
        r["text"] for s in ("wikipedia", "legal") for r in cpt_data.jsonl_rows(ev / f"{s}.jsonl")}
    for r in recs:
        assert r["text"] not in held and eval_text(r["text"]) not in held
    assert list(stream.stream_texts(web)) == web_texts
    assert web_held not in texts["web"] and texts["web"] == [t for t in web_texts if t != web_held][:len(texts["web"])]
    assert all(target <= man["stream"]["domain_chars"][d] == sum(map(len, texts[d])) for d in texts)
    assert man["web"]["skipped_held_out"] == 1
    before = {p.name: p.stat().st_mtime_ns for p in out.rglob("*.jsonl")}
    again = mixed_stream.build(ev, web, target, out, units, read_rows, lambda d: pytest.fail("dev rebuilt"), 5)
    assert again == man and before == {p.name: p.stat().st_mtime_ns for p in out.rglob("*.jsonl")}
    (ev / "evalsets_manifest.json").write_text('{"changed": 1}')
    with pytest.raises(SystemExit, match="other evaluation sets"):
        mixed_stream.build(ev, web, target, out, units, read_rows, all_rows, 5)


def test_a_changed_mixed_part_is_refused_and_a_short_source_reported(tmp_path):
    units, read_rows, all_rows, data = fake_sources(n_legal=12)
    ev, web, _, _ = setup(tmp_path, data)
    with pytest.raises(SystemExit, match="characters"):
        mixed_stream.build(ev, web, 200_000, tmp_path / "short", units, read_rows, all_rows, 5)
    mixed_stream.build(ev, web, 5_000, tmp_path / "ok", units, read_rows, all_rows, 5)
    part = tmp_path / "ok" / "stream" / "part-00000.jsonl"
    part.write_text(part.read_text() + "{}\n")
    with pytest.raises(SystemExit, match="does not match"):
        list(mixed_stream.mixed_records(tmp_path / "ok" / "stream"))
