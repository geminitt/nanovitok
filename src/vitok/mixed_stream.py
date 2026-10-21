"""Training text of the Token Distillation study (PLAN.md, Stage 1): web, Wikipedia and legal documents with equal
characters per domain, interleaved into one stream, and the Wikipedia and legal dev sets.

    python -m vitok.mixed_stream --evalsets runs/evalsets-2026-10-13-final \
        --web-stream runs/stream-seed0-2026-10-13 --chars-per-domain 120_000_000 --out runs/mixed-data-2026-10-21

Steps, each skipped when its output is already complete (so a rerun resumes):
1. Dev sets `dev-wikipedia.jsonl` and `dev-legal.jsonl`: 500 windows each, cut like the domain's test documents
   (`evalsets.wiki_window`, `evalsets.legal_window`), from documents that are not test documents (by source id and
   by text), sampled with seed 0. The web dev set is the existing `dev.jsonl` of the evaluation sets.
2. One stream per domain (`web/`, `wikipedia/`, `legal/`), in the format of `vitok.stream`:
   - web: the seed-0 FineWeb-2 stream of the token-type study, in its order, read from its folder;
   - Wikipedia: whole articles of at least 3,000 characters; legal: the HTML text of documents of at least 3,500
     characters from the first line start after the 1,000-character letterhead (the rules of the test sets, so
     training and test text are alike); both in the seed-0 order of `vitok.stream` over their row groups.
   Documents whose source id is a test or dev document are dropped; any document whose text or test-style cut is a
   held-out text (every evaluation set, both new dev sets, the merge-learning text) is dropped and counted.
3. The mixed stream `stream/`: documents {"domain", "text"}, each next document taken from the domain with the
   fewest characters so far (ties: web, Wikipedia, legal), until every domain holds `--chars-per-domain`. Any prefix
   of the stream therefore holds about equal characters per domain.
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from vitok import cpt_data, evalsets, stream

DOMAINS = ["web", "wikipedia", "legal"]
TEST_SETS = {"wikipedia": "wikipedia", "legal": "legal"}
DEV_DOCS = 500
SEED = 0
PART_CHARS = 200_000_000
SOURCES = {"wikipedia": (evalsets.WIKIPEDIA, evalsets.WIKIPEDIA_FILES, ["id", "text"]),
           "legal": (evalsets.LEGAL, [evalsets.LEGAL_FILE], ["id", "content_html"])}


def legal_body(html: str) -> str | None:
    """A legal document's text after its letterhead (the start of `evalsets.legal_window`, not cut)."""
    text = evalsets.nfc(evalsets.html_text(html or ""))
    if len(text) < evalsets.LEGAL_MIN_SOURCE:
        return None
    start = text.find("\n", evalsets.LEGAL_SKIP)
    return None if start < 0 else text[start + 1:]


def wiki_body(text: str) -> str | None:
    text = evalsets.nfc(text)
    return text if len(text) >= evalsets.WIKI_MIN_SOURCE else None


def record(domain: str, row: dict) -> tuple[str, str | None]:
    """(source id, training text or None) of one source row."""
    if domain == "wikipedia":
        return str(row["id"]), wiki_body(row["text"])
    return str(row["id"]), legal_body(row["content_html"])


def window(domain: str, row: dict) -> tuple[str, str | None]:
    """(source id, dev/test-style window or None), exactly as the domain's test set cuts it."""
    if domain == "wikipedia":
        return str(row["id"]), evalsets.wiki_window(row["text"])
    return str(row["id"]), evalsets.legal_window(evalsets.html_text(row["content_html"] or ""))


def dev_set(domain: str, rows: Iterable[dict], test: list[dict], n: int = DEV_DOCS, seed: int = SEED) -> list[dict]:
    """n windows of documents that are not test documents, by source id or by window text."""
    ids, texts = {r["id"] for r in test}, {r["text"] for r in test}
    return evalsets.sample(((key, w) for key, w in (window(domain, r) for r in rows)
                            if w is not None and key not in ids and w not in texts), n, seed)


def read_unit_filtered(domain: str, read_rows: Callable[[str, int], list[dict]],
                       excluded_ids: set[str]) -> Callable[[str, int], list[str]]:
    """A `vitok.stream` reader: a row group's training texts, without the excluded source ids."""
    def read(name: str, rg: int) -> list[str]:
        out = []
        for row in read_rows(name, rg):
            key, text = record(domain, row)
            if key not in excluded_ids and text is not None:
                out.append(text)
        return out
    return read


def prefix_stream(out: Path, texts: Iterable[str], held: set[bytes], max_chars: int, source: dict,
                  part_chars: int = PART_CHARS) -> dict:
    """Write the first documents of an existing stream up to `max_chars` (held-out texts dropped and counted), in the
    format of `vitok.stream`; one part per `part_chars`. A finished folder of the same source is kept."""
    man_path = out / "stream_manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
        if man["source"] == source and man["chars"] >= max_chars:
            return man
        raise SystemExit(f"{out} holds another or a shorter stream; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    man = {"seed": None, "source": source, "parts": [], "docs": 0, "skipped_held_out": 0, "chars": 0}
    buf: list[str] = []
    buf_chars = buf_docs = skipped = 0
    for doc in texts:
        if man["chars"] + buf_chars >= max_chars:
            break
        cut = cpt_data.eval_text(doc)
        if cpt_data.digest(doc) in held or (cut is not None and cpt_data.digest(cut) in held):
            skipped += 1
            continue
        buf.append(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        buf_chars += len(doc)
        buf_docs += 1
        if buf_chars >= part_chars:
            _close_part(out, man, buf, buf_chars, buf_docs, skipped)
            buf, buf_chars, buf_docs, skipped = [], 0, 0, 0
    if buf:
        _close_part(out, man, buf, buf_chars, buf_docs, skipped)
    if man["chars"] < max_chars:
        raise SystemExit(f"the source stream holds only {man['chars']:,} usable characters, {max_chars:,} requested")
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def _close_part(out: Path, man: dict, buf: list[str], chars: int, docs: int, skipped: int) -> None:
    name = f"part-{len(man['parts']):05d}.jsonl"
    data = "".join(buf).encode("utf-8")
    (out / name).write_bytes(data)
    man["parts"].append({"file": name, "docs": docs, "chars": chars, "skipped_held_out": skipped,
                         "sha256": hashlib.sha256(data).hexdigest()})
    man["docs"] += docs
    man["chars"] += chars
    man["skipped_held_out"] += skipped


def interleave(streams: dict[str, Iterator[str]], chars_per_domain: int) -> Iterator[tuple[str, str]]:
    """(domain, text): the next document from the domain with the fewest characters so far, in `DOMAINS` order on
    ties, until every domain holds `chars_per_domain`; a domain that has reached it gives no more documents."""
    chars = dict.fromkeys(streams, 0)
    order = [d for d in DOMAINS if d in streams] + [d for d in streams if d not in DOMAINS]
    while True:
        open_ = [d for d in order if chars[d] < chars_per_domain]
        if not open_:
            return
        d = min(open_, key=lambda x: (chars[x], order.index(x)))
        text = next(streams[d], None)
        if text is None:
            raise SystemExit(f"the {d} stream ends at {chars[d]:,} characters, {chars_per_domain:,} needed")
        chars[d] += len(text)
        yield d, text


def write_mixed(out: Path, docs: Iterable[tuple[str, str]], source: dict, part_chars: int = PART_CHARS) -> dict:
    """The mixed stream in the format of `vitok.stream` with a "domain" field; written once (a short job)."""
    man_path = out / "stream_manifest.json"
    if man_path.exists():
        old = json.loads(man_path.read_text())
        if old["source"] != source:
            raise SystemExit(f"{out} holds another mixed stream; use a new --out")
        return old
    out.mkdir(parents=True, exist_ok=True)
    for stray in out.glob("part-*.jsonl"):  # left by a crash before the manifest was written
        stray.unlink()
    man: dict = {"seed": None, "source": source, "parts": [], "docs": 0, "chars": 0, "skipped_held_out": 0,
                 "domain_chars": dict.fromkeys(DOMAINS, 0), "domain_docs": dict.fromkeys(DOMAINS, 0)}
    buf: list[str] = []
    buf_chars = buf_docs = 0
    for domain, text in docs:
        buf.append(json.dumps({"domain": domain, "text": text}, ensure_ascii=False) + "\n")
        buf_chars += len(text)
        buf_docs += 1
        man["domain_chars"][domain] += len(text)
        man["domain_docs"][domain] += 1
        if buf_chars >= part_chars:
            _close_part(out, man, buf, buf_chars, buf_docs, 0)
            buf, buf_chars, buf_docs = [], 0, 0
    if buf:
        _close_part(out, man, buf, buf_chars, buf_docs, 0)
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def mixed_records(folder: Path) -> Iterator[dict]:
    """{"domain", "text"} of a finished mixed stream, in order, each part checked against its manifest hash."""
    man = json.loads((folder / "stream_manifest.json").read_text())
    for p in man["parts"]:
        path = folder / p["file"]
        if stream._sha256(path) != p["sha256"]:
            raise SystemExit(f"{path} does not match its manifest")
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


# ---- sources (local parquet files from the pinned revisions; replaced by fakes in the tests) ----

def _local_files(domain: str) -> list[Path]:
    from huggingface_hub import hf_hub_download

    (repo, rev), files, _ = SOURCES[domain]
    return [Path(hf_hub_download(repo, f, repo_type="dataset", revision=rev)) for f in files]


def _units(domain: str) -> list[tuple[str, int]]:
    import pyarrow.parquet as pq

    _, names, _ = SOURCES[domain]
    return [(name, rg) for name, path in zip(names, _local_files(domain), strict=True)
            for rg in range(pq.ParquetFile(path).num_row_groups)]


def _read_rows(domain: str) -> Callable[[str, int], list[dict]]:
    import pyarrow.parquet as pq

    _, names, columns = SOURCES[domain]
    paths = dict(zip(names, _local_files(domain), strict=True))

    def read(name: str, rg: int) -> list[dict]:
        return pq.ParquetFile(paths[name]).read_row_group(rg, columns=columns).to_pylist()
    return read


def _all_rows(domain: str) -> Iterator[dict]:
    read = _read_rows(domain)
    for name, rg in _units(domain):
        yield from read(name, rg)


def build(evalsets_dir: Path, web_stream: Path, chars_per_domain: int, out: Path,
          units: Callable[[str], list[tuple[str, int]]] = _units,
          read_rows: Callable[[str], Callable[[str, int], list[dict]]] = _read_rows,
          all_rows: Callable[[str], Iterable[dict]] = _all_rows, dev_docs: int = DEV_DOCS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "mixed_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    evalsets_sha = stream._sha256(evalsets_dir / "evalsets_manifest.json")
    if man and man.get("evalsets_manifest_sha256") != evalsets_sha:
        raise SystemExit(f"{out} was built from other evaluation sets; use a new --out")
    man["evalsets_manifest_sha256"] = evalsets_sha
    dev = {}
    for domain in TEST_SETS:
        path = out / f"dev-{domain}.jsonl"
        key = f"dev-{domain}"
        if key in man and path.exists() and stream._sha256(path) == man[key]["sha256"]:
            dev[domain] = cpt_data.jsonl_rows(path)
            continue
        test = cpt_data.jsonl_rows(evalsets_dir / f"{TEST_SETS[domain]}.jsonl")
        dev[domain] = dev_set(domain, all_rows(domain), test, dev_docs)
        man[key] = {"count": len(dev[domain]), "seed": SEED, "sha256": evalsets.write_jsonl(path, dev[domain]),
                    "rule": "windows cut like the test set, from non-test documents (id and text), seeded sample"}
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    held = stream.held_out_digests(evalsets_dir) | cpt_data.held_out_digests(
        r["text"] for rows in dev.values() for r in rows)
    domain_dirs = {}
    for domain in DOMAINS:
        folder = out / domain
        source: dict
        if domain == "web":
            source = {"stream": str(web_stream), "stream_manifest_sha256": stream._sha256(
                web_stream / "stream_manifest.json"), "held_out": "every evaluation set, dev-wikipedia, dev-legal"}
            dman = prefix_stream(folder, stream.stream_texts(web_stream), held, chars_per_domain, source)
        else:
            (repo, rev), files, _ = SOURCES[domain]
            test = cpt_data.jsonl_rows(evalsets_dir / f"{TEST_SETS[domain]}.jsonl")
            excluded = {r["id"] for r in test} | {r["id"] for r in dev[domain]}
            source = {"repo": repo, "revision": rev, "files": files, "excluded_ids": len(excluded),
                      "rule": ("whole articles >= 3,000 chars" if domain == "wikipedia" else
                               "HTML text >= 3,500 chars from the line start after 1,000 chars")}
            dman = stream.build(folder, SEED, chars_per_domain, PART_CHARS, held, units(domain),
                                read_unit_filtered(domain, read_rows(domain), excluded), source)
        man[domain] = {k: v for k, v in dman.items() if k in ("docs", "chars", "skipped_held_out")}
        domain_dirs[domain] = folder
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    mix_source = {"chars_per_domain": chars_per_domain, "domains": {
        d: stream._sha256(f / "stream_manifest.json") for d, f in domain_dirs.items()}}
    mixed = write_mixed(out / "stream", interleave(
        {d: stream.stream_texts(f) for d, f in domain_dirs.items()}, chars_per_domain), mix_source)
    man["stream"] = {k: mixed[k] for k in ("docs", "chars", "domain_chars", "domain_docs")}
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evalsets", type=Path, required=True, help="the folder vitok.evalsets wrote")
    ap.add_argument("--web-stream", type=Path, required=True, help="the seed-0 FineWeb-2 stream (vitok.stream)")
    ap.add_argument("--chars-per-domain", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    man = build(args.evalsets, args.web_stream, args.chars_per_domain, args.out)
    print(json.dumps({k: v for k, v in man.items() if not k.startswith("dev-")}, indent=2))


if __name__ == "__main__":
    main()
