"""The training text of one seed: FineWeb-2 documents in a seeded order, held-out texts removed, written once and read
by every condition of that seed (PLAN.md: the seed sets the document order).

    python -m vitok.stream --evalsets runs/evalsets-2026-10-13-final --seed 0 --max-chars 2_000_000_000 \
        --out runs/stream-seed0

Order: the row groups of the training files (`evalsets.TRAIN_FILES`) are permuted by the seed, and the documents of
each row group are permuted by a generator derived from the seed and the row group, so a different seed reads a
different mixture in a different order while each seed's order is fixed. Only the row groups read are downloaded.

Output: `part-00000.jsonl`, ... ({"text"} per line, NFC), each closed at `--part-chars` characters, and
`stream_manifest.json` (seed, sources, the row groups read in order, documents written and skipped, characters, the
sha256 of every part). A rerun with a larger `--max-chars` keeps the finished parts and continues the same order, so
a longer stream starts with the shorter one; a changed seed or source refuses to reuse the folder.
"""

import argparse
import hashlib
import json
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from vitok import cpt_data, evalsets

HELD_OUT_SETS = ["fineweb2", "wikipedia", "legal", "flores", "english", "dev", "merge"]


def unit_order(units: list[tuple[str, int]], seed: int) -> list[tuple[str, int]]:
    """The (file, row group) units in the seed's order."""
    return [units[i] for i in np.random.default_rng([seed, 0]).permutation(len(units))]


def doc_order(n: int, seed: int, unit_index: int) -> np.ndarray:
    """The order of a row group's n documents, fixed by the seed and the unit's place in the stream."""
    return np.random.default_rng([seed, 1, unit_index]).permutation(n)


def held_out_digests(evalsets_dir: Path) -> set[bytes]:
    """Digests of every held-out language-modeling text (as stored: NFC and cut)."""
    texts = (t for name in HELD_OUT_SETS for t in cpt_data.jsonl_texts(evalsets_dir / f"{name}.jsonl"))
    return cpt_data.held_out_digests(texts)


def documents(units: list[tuple[str, int]], read_unit: Callable[[str, int], list[str]], seed: int,
              start_unit: int = 0) -> Iterator[tuple[int, str]]:
    """(unit index, document) in stream order from `start_unit` on."""
    for k in range(start_unit, len(units)):
        texts = read_unit(*units[k])
        for i in doc_order(len(texts), seed, k):
            yield k, texts[i]


def _hf_units(revision: str) -> list[tuple[str, int]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs, out = HfFileSystem(), []
    for name in evalsets.TRAIN_FILES:
        with fs.open(f"datasets/{evalsets.FINEWEB2[0]}@{revision}/{name}", "rb") as f:
            out += [(name, rg) for rg in range(pq.ParquetFile(f).num_row_groups)]
    return out


def _hf_read_unit(revision: str) -> Callable[[str, int], list[str]]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def read(name: str, rg: int) -> list[str]:
        for attempt in range(5):  # transient network errors: retry with backoff
            try:
                with fs.open(f"datasets/{evalsets.FINEWEB2[0]}@{revision}/{name}", "rb") as f:
                    return pq.ParquetFile(f).read_row_group(rg, columns=["text"]).column("text").to_pylist()
            except OSError:
                if attempt == 4:
                    raise
                import time
                time.sleep(2 ** attempt * 5)
        raise AssertionError("unreachable")
    return read


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(out: Path, seed: int, max_chars: int, part_chars: int, held: set[bytes], units: list[tuple[str, int]],
          read_unit: Callable[[str, int], list[str]], source: dict) -> dict:
    """Write (or extend) the stream up to `max_chars`; every part is closed whole, so the last may run over."""
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "stream_manifest.json"
    order = unit_order(units, seed)
    man: dict[str, Any] = {"seed": seed, "source": source, "units_total": len(order), "parts": [], "units_read": [],
           "next_unit": 0, "docs": 0, "skipped_held_out": 0, "chars": 0}
    if man_path.exists():
        old = json.loads(man_path.read_text())
        if old["seed"] != seed or old["source"] != source:
            raise SystemExit(f"{out} holds the stream of another seed or source; use a new --out")
        man = old
        for p in man["parts"]:  # a finished part must be what the manifest says
            if _sha256(out / p["file"]) != p["sha256"]:
                raise SystemExit(f"{out / p['file']} changed since it was written")
    for stray in sorted(out.glob("part-*.jsonl")):  # a part left open by a crash is rewritten
        if stray.name not in {p["file"] for p in man["parts"]}:
            stray.unlink()
    if man["chars"] >= max_chars:
        return man
    # a part always ends at a unit boundary, so the stream resumes at the next unit
    buf: list[str] = []
    buf_chars = buf_docs = buf_skipped = 0
    first_unit = man["next_unit"]
    last_unit = first_unit - 1
    for k, text in documents(order, read_unit, seed, first_unit):
        if k != last_unit:
            if buf_chars >= part_chars or man["chars"] + buf_chars >= max_chars:
                _close(out, man, buf, buf_chars, buf_docs, buf_skipped, order, first_unit, k)
                man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
                buf, buf_chars, buf_docs, buf_skipped, first_unit = [], 0, 0, 0, k
                if man["chars"] >= max_chars:
                    return man
            last_unit = k
        doc = unicodedata.normalize("NFC", text)
        cut = cpt_data.eval_text(doc)
        if cpt_data.digest(doc) in held or (cut is not None and cpt_data.digest(cut) in held):
            buf_skipped += 1
            continue
        buf.append(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        buf_chars += len(doc)
        buf_docs += 1
    if buf:
        _close(out, man, buf, buf_chars, buf_docs, buf_skipped, order, first_unit, len(order))
        man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    if man["chars"] < max_chars:
        raise SystemExit(f"the training files hold only {man['chars']:,} characters, {max_chars:,} requested")
    return man


def _close(out, man, buf, chars, docs, skipped, order, first_unit, next_unit):
    name = f"part-{len(man['parts']):05d}.jsonl"
    data = "".join(buf).encode("utf-8")
    tmp = out / (name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out / name)
    man["parts"].append({"file": name, "docs": docs, "chars": chars, "skipped_held_out": skipped,
                         "units": [list(u) for u in order[first_unit:next_unit]],
                         "sha256": hashlib.sha256(data).hexdigest()})
    man["units_read"] += [list(u) for u in order[first_unit:next_unit]]
    man["next_unit"] = next_unit
    man["docs"] += docs
    man["skipped_held_out"] += skipped
    man["chars"] += chars


def stream_texts(folder: Path) -> Iterator[str]:
    """The documents of a finished stream, in order, each part checked against its manifest hash."""
    man = json.loads((folder / "stream_manifest.json").read_text())
    for p in man["parts"]:
        path = folder / p["file"]
        if _sha256(path) != p["sha256"]:
            raise SystemExit(f"{path} does not match its manifest")
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evalsets", type=Path, required=True, help="the folder vitok.evalsets wrote")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-chars", type=int, required=True)
    ap.add_argument("--part-chars", type=int, default=200_000_000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    revision = evalsets.FINEWEB2[1]
    source = {"repo": evalsets.FINEWEB2[0], "revision": revision, "files": evalsets.TRAIN_FILES,
              "held_out_sets": HELD_OUT_SETS,
              "evalsets_manifest_sha256": _sha256(args.evalsets / "evalsets_manifest.json")}
    man = build(args.out, args.seed, args.max_chars, args.part_chars, held_out_digests(args.evalsets),
                _hf_units(revision), _hf_read_unit(revision), source)
    print(json.dumps({k: v for k, v in man.items() if k not in ("parts", "units_read", "source")}, indent=2))


if __name__ == "__main__":
    main()
