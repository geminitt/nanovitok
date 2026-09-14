import pyarrow.parquet as pq

from vitok.data import ShardWriter, truncate_chars, write_single_shard


def test_truncate_chars_cuts_at_whitespace():
    text = "học sinh giỏi nhất trường"
    out = truncate_chars(text, 12)
    assert len(out) <= 12 and text.startswith(out) and not out.endswith(" ")
    assert out == "học sinh"
    assert truncate_chars("ngắn", 100) == "ngắn"


def test_shard_writer_sorts_before_val(tmp_path):
    w = ShardWriter(tmp_path, shard_bytes=200, row_group_docs=3)
    docs = [f"văn bản số {i} " * 5 for i in range(40)]
    for d in docs:
        w.add(d)
    w.close()
    write_single_shard(tmp_path / "shard_99999.parquet", ["val"])
    files = sorted(p.name for p in tmp_path.glob("*.parquet"))
    assert files[-1] == "shard_99999.parquet" and len(files) > 2
    back = [t for f in files[:-1] for t in pq.read_table(tmp_path / f).column("text").to_pylist()]
    assert back == docs  # order preserved, nothing lost
