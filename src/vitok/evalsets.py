"""Every held-out text of the token-type experiment, built from pinned revisions (PLAN.md, Evaluation).

    python -m vitok.evalsets --out runs/evalsets-2026-10-13 [--sets fineweb2 wikipedia ...]

Language-modeling sets (jsonl, one {"id", "text"} per line, NFC):
- `fineweb2`: FineWeb-2 vie_Latn test split, documents cut like the original test set (`eval_text`).
- `wikipedia`: Vietnamese Wikipedia articles of at least 3,000 characters (most bot-made stubs are shorter), cut
  from the start.
- `legal`: Vietnamese legal documents (vbpl.vn crawl), text extracted from HTML; the first 1,000 characters (the
  identical letterhead of every document) are skipped to the next line break, then cut.
- `flores`: FLORES+ vie_Latn devtest sentences, whole (a gated repository: accept its terms on the Hub first).
- `english`: English FineWeb sample-10BT documents, as `vitok.heldout` cuts them.
A set is a seeded sample of its eligible, exact-deduplicated windows, kept in source order.

Multiple-choice sets (jsonl, one {"id", "context", "choices", "gold"} per line, NFC): the prompt is fixed here, so
scoring is generic: each choice is a continuation of the context, scored by log-likelihood per character.

FineWeb-2 train text: the last train file is reserved. Its first documents up to `MERGE_CHARS` characters are the
merge-learning text (`merge.jsonl`); the next 500 usable documents are the dev set (`dev.jsonl`). Training reads
the other files (`TRAIN_FILES`) and drops any document whose text or test-style cut is a held-out text.

`evalsets_manifest.json` records each set's source, revision, selection rule, count and sha256. The same revisions
and arguments give byte-identical files.
"""

import argparse
import ast
import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from vitok import heldout
from vitok.data import eval_text

FINEWEB2 = ("HuggingFaceFW/fineweb-2", "af9c13333eb981300149d5ca60a8e9d659b276b9")
FINEWEB2_TRAIN = [f"data/vie_Latn/train/{a:03d}_{b:05d}.parquet" for a in range(5) for b in range(6)]
RESERVED_FILE = FINEWEB2_TRAIN[-1]
TRAIN_FILES = FINEWEB2_TRAIN[:-1]
FINEWEB2_TEST = "data/vie_Latn/test/000_00000.parquet"
WIKIPEDIA = ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa")
WIKIPEDIA_FILES = [f"20231101.vi/train-{i:05d}-of-00004.parquet" for i in range(4)]
LEGAL = ("th1nhng0/vietnamese-legal-documents", "8977887f17be2defae4c5171d55562e1cde7d695")
LEGAL_FILE = "data/content.parquet"
FLORES = ("openlanguagedata/flores_plus", "e707e62e762a9e2faca3933aa315f4665cc3cd8f")
FLORES_FILE = "devtest/vie_Latn.jsonl"
GLOBAL_MMLU = ("CohereLabs/Global-MMLU", "0e619dbeb34206cd48705a1a0ea7fb21cae09993")
XNLI = ("facebook/xnli", "b8dd5d7af51114dbda02c0e3f6133f332186418e")
OKAPI_ARC = ("alexandrainst/m_arc", "69b0991ee606079105c4f9e271bcc1fef0014e03")
OKAPI_HELLASWAG = ("alexandrainst/m_hellaswag", "9d31dc982bd6285e081e3e3136332a38b9c1d7b7")
HELLASWAG = ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76")

DOCS = 1000
DEV_DOCS = 500
MERGE_CHARS = 20_000_000
WIKI_MIN_SOURCE = 3000
LEGAL_MIN_SOURCE, LEGAL_SKIP = 3500, 1000
SEED = 0
# chance accuracy of each multiple-choice set (the benchmark inclusion rule compares against it)
CHANCE = {"belebele": 1 / 4, "global_mmlu": 1 / 4, "xnli": 1 / 3, "okapi_arc": 1 / 4, "okapi_hellaswag": 1 / 4,
          "hellaswag_en": 1 / 4}
# XNLI as continuation choice: "premise, đúng không? {word}, hypothesis" (Vietnamese "right?"; entailment
# "Đúng" = yes, neutral "Ngoài ra" = also, contradiction "Không" = no), experimental input
XNLI_WORDS = ("Đúng", "Ngoài ra", "Không")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def sample(windows: Iterable[tuple[str, str]], n: int, seed: int = SEED) -> list[dict]:
    """n of the (id, text) windows, exact duplicates dropped first, chosen by seed, kept in source order."""
    seen, pool = set(), []
    for key, text in windows:
        if text not in seen:
            seen.add(text)
            pool.append({"id": key, "text": text})
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} usable documents, {n} needed")
    keep = np.sort(np.random.default_rng(seed).choice(len(pool), size=n, replace=False))
    return [pool[i] for i in keep]


def wiki_window(text: str, min_source: int = WIKI_MIN_SOURCE) -> str | None:
    text = nfc(text)
    return eval_text(text) if len(text) >= min_source else None


def legal_window(text: str, min_source: int = LEGAL_MIN_SOURCE, skip: int = LEGAL_SKIP) -> str | None:
    """The document after its letterhead: from the first line start at or after `skip` characters, then cut."""
    text = nfc(text)
    if len(text) < min_source:
        return None
    start = text.find("\n", skip)
    return None if start < 0 else eval_text(text[start + 1:])


class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "title"}
    SKIP = {"script", "style", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skipping += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skipping = max(0, self.skipping - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(re.sub(r"\s+", " ", data))  # a line break inside text is a space in HTML


def html_text(html: str) -> str:
    """Visible text of an HTML document: one line per block element, inner whitespace collapsed, blank lines dropped."""
    p = _Text()
    p.feed(html)
    p.close()
    lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(p.parts).split("\n"))
    return "\n".join(line for line in lines if line)


def _mc(key: str, context: str, choices: list[str], gold: int) -> dict:
    """One item; every choice is a continuation starting with exactly one space (some sources pad answers)."""
    if not 0 <= gold < len(choices) or len(choices) < 2 or not all(c.strip() for c in choices):
        raise ValueError(f"item {key}: gold {gold} or choices {choices!r} malformed")
    return {"id": key, "context": nfc(context), "choices": [" " + nfc(c).strip() for c in choices], "gold": gold}


def belebele_mc(recs: Iterable[dict]) -> list[dict]:
    """Belebele with the prompt the original experiment used (`vitok.score.PROMPT`)."""
    out = []
    for i, r in enumerate(heldout.belebele_items(recs)):
        ctx = f"{r['flores_passage']}\nCâu hỏi: {r['question']}\nTrả lời:"  # "Question:" / "Answer:"
        out.append(_mc(f"belebele-{i}", ctx, [" " + r[f"mc_answer{k}"] for k in range(1, 5)],
                       int(r["correct_answer_num"]) - 1))
    return out


def letter_mc(recs: Iterable[dict], prefix: str, question: str, template: str) -> list[dict]:
    """Four-option items with option_a..d and a letter answer, as cloze continuations of the question."""
    out = []
    for i, r in enumerate(recs):
        opts = [o for c in "abcd" if isinstance(o := r.get(f"option_{c}"), str) and o.strip()]
        if r.get("answer") not in ("A", "B", "C", "D") or len(opts) != 4:
            continue  # untranslated or incomplete item
        out.append(_mc(f"{prefix}-{r.get('id', r.get('sample_id', i))}", template.format(q=r[question]),
                       [" " + o.strip() for o in opts], "ABCD".index(r["answer"])))
    return out


def xnli_mc(recs: Iterable[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(recs):
        premise = r["premise"].strip().rstrip(".")
        out.append(_mc(f"xnli-{i}", f"{premise}, đúng không?",
                       [f" {w}, {r['hypothesis'].strip()}" for w in XNLI_WORDS], int(r["label"])))
    return out


def _hellaswag_clean(text: str) -> str:
    """lm-evaluation-harness's HellaSwag preprocessing (WikiHow section markers out, double spaces collapsed), then
    stripped: the harness leaves a leading space where a text starts with a marker, so a choice would begin with two."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ").strip()


def hellaswag_mc(recs: Iterable[dict], prefix: str) -> list[dict]:
    out = []
    for i, r in enumerate(recs):
        endings = r["endings"]
        endings = ast.literal_eval(endings) if isinstance(endings, str) else list(endings)
        if len(endings) != 4 or str(r.get("label", "")) not in ("0", "1", "2", "3"):
            continue
        ctx = r["ctx_a"] + " " + (r.get("ctx_b") or "").capitalize()
        context = _hellaswag_clean(f"{r['activity_label']}: {ctx}")
        choices = [" " + _hellaswag_clean(e) for e in endings]
        if not all(c.strip() for c in choices):
            continue
        out.append(_mc(f"{prefix}-{i}-{r.get('ind', '')}", context, choices, int(r["label"])))  # `ind` repeats
    return out


# ---- sources (network; replaced by fakes in the tests) ----

def _parquet_rows(repo_rev: tuple[str, str], files: list[str], columns: list[str]) -> Iterator[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for name in files:
        path = hf_hub_download(repo_rev[0], name, repo_type="dataset", revision=repo_rev[1])
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            yield from pf.read_row_group(rg, columns=columns).to_pylist()


def _streamed_texts(repo_rev: tuple[str, str], name: str) -> Iterator[str]:
    """Texts of one parquet file read row group by row group (only the byte ranges read are fetched)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{repo_rev[0]}@{repo_rev[1]}/{name}", "rb") as f:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            yield from pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()


def _jsonl_rows(repo_rev: tuple[str, str], name: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_rev[0], name, repo_type="dataset", revision=repo_rev[1])
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# ---- the sets ----

def build_fineweb2() -> tuple[list[dict], dict]:
    rows = _parquet_rows(FINEWEB2, [FINEWEB2_TEST], ["id", "text"])
    docs = sample(((r["id"], w) for r in rows if (w := eval_text(nfc(r["text"]))) is not None), DOCS)
    return docs, {"repo": FINEWEB2[0], "revision": FINEWEB2[1], "files": [FINEWEB2_TEST],
                  "rule": "eval_text (300-2,500 chars), exact dedup, seeded sample", "seed": SEED}


def build_wikipedia() -> tuple[list[dict], dict]:
    rows = _parquet_rows(WIKIPEDIA, WIKIPEDIA_FILES, ["id", "text"])
    docs = sample(((r["id"], w) for r in rows if (w := wiki_window(r["text"])) is not None), DOCS)
    return docs, {"repo": WIKIPEDIA[0], "revision": WIKIPEDIA[1], "files": WIKIPEDIA_FILES,
                  "rule": f"articles >= {WIKI_MIN_SOURCE} chars, eval_text from the start, exact dedup, seeded sample",
                  "seed": SEED}


def build_legal() -> tuple[list[dict], dict]:
    rows = _parquet_rows(LEGAL, [LEGAL_FILE], ["id", "content_html"])
    windows = ((str(r["id"]), legal_window(html_text(r["content_html"] or ""))) for r in rows)
    docs = sample(((key, w) for key, w in windows if w is not None), DOCS)
    return docs, {"repo": LEGAL[0], "revision": LEGAL[1], "files": [LEGAL_FILE],
                  "rule": f"HTML text; documents >= {LEGAL_MIN_SOURCE} chars; skip {LEGAL_SKIP} chars to a line start; "
                          "eval_text; exact dedup; seeded sample", "seed": SEED}


def build_flores() -> tuple[list[dict], dict]:
    try:
        rows = _jsonl_rows(FLORES, FLORES_FILE)
    except Exception as e:  # GatedRepoError: the owner has not accepted the terms yet
        raise SystemExit(f"FLORES+ is gated: accept its terms at https://huggingface.co/datasets/{FLORES[0]} ({e})") \
            from e
    docs = [{"id": str(r.get("id", i)), "text": nfc(r["text"])} for i, r in enumerate(rows)]
    return docs, {"repo": FLORES[0], "revision": FLORES[1], "files": [FLORES_FILE], "rule": "every sentence, whole"}


def build_english() -> tuple[list[dict], dict]:
    texts = heldout.english_docs(heldout._fineweb_texts(heldout.FINEWEB_REVISION), DOCS)
    docs = [{"id": f"fineweb-en-{i}", "text": t} for i, t in enumerate(texts)]
    return docs, {"repo": heldout.FINEWEB_REPO, "revision": heldout.FINEWEB_REVISION, "files": [heldout.FINEWEB_FILE],
                  "rule": "first usable documents in file order, eval_text, exact dedup"}


def build_merge_and_dev() -> tuple[tuple[list[dict], list[dict]], dict]:
    merge, dev, chars, seen = [], [], 0, set()
    for i, text in enumerate(_streamed_texts(FINEWEB2, RESERVED_FILE)):
        text = nfc(text)
        if chars < MERGE_CHARS:
            merge.append({"id": f"reserved-{i}", "text": text})
            chars += len(text)
            continue
        w = eval_text(text)
        if w is not None and w not in seen:
            seen.add(w)
            dev.append({"id": f"reserved-{i}", "text": w})
            if len(dev) == DEV_DOCS:
                break
    if len(dev) < DEV_DOCS:
        raise ValueError(f"the reserved file gave only {len(dev)} dev documents")
    return (merge, dev), {"repo": FINEWEB2[0], "revision": FINEWEB2[1], "file": RESERVED_FILE,
                          "merge_rule": f"first documents in file order up to {MERGE_CHARS:,} chars, whole",
                          "dev_rule": "the next usable documents, eval_text, exact dedup", "merge_chars": chars,
                          "train_files": TRAIN_FILES}


MC_SETS: dict[str, Callable[[], tuple[list[dict], dict]]] = {
    "belebele": lambda: (belebele_mc(heldout._belebele_records(heldout.BELEBELE_REVISION)),
                         {"repo": heldout.BELEBELE_REPO, "revision": heldout.BELEBELE_REVISION,
                          "files": [heldout.BELEBELE_FILE]}),
    "global_mmlu": lambda: (letter_mc(_parquet_rows(GLOBAL_MMLU, ["vi/test-00000-of-00001.parquet"],
                                                    ["sample_id", "question", "option_a", "option_b", "option_c",
                                                     "option_d", "answer"]),
                                      "global_mmlu", "question", "{q}\nTrả lời:"),
                            {"repo": GLOBAL_MMLU[0], "revision": GLOBAL_MMLU[1],
                             "files": ["vi/test-00000-of-00001.parquet"]}),
    "xnli": lambda: (xnli_mc(_parquet_rows(XNLI, ["vi/test-00000-of-00001.parquet"],
                                           ["premise", "hypothesis", "label"])),
                     {"repo": XNLI[0], "revision": XNLI[1], "files": ["vi/test-00000-of-00001.parquet"]}),
    "okapi_arc": lambda: (letter_mc(_jsonl_rows(OKAPI_ARC, "data/vi/test.jsonl"), "okapi_arc", "instruction",
                                    "Câu hỏi: {q}\nTrả lời:"),
                          {"repo": OKAPI_ARC[0], "revision": OKAPI_ARC[1], "files": ["data/vi/test.jsonl"]}),
    "okapi_hellaswag": lambda: (hellaswag_mc(_jsonl_rows(OKAPI_HELLASWAG, "data/vi/val.jsonl"), "okapi_hellaswag"),
                                {"repo": OKAPI_HELLASWAG[0], "revision": OKAPI_HELLASWAG[1],
                                 "files": ["data/vi/val.jsonl"]}),
    "hellaswag_en": lambda: (hellaswag_mc(_parquet_rows(HELLASWAG, ["data/validation-00000-of-00001.parquet"],
                                                        ["ind", "activity_label", "ctx_a", "ctx_b", "endings",
                                                         "label"]), "hellaswag_en"),
                             {"repo": HELLASWAG[0], "revision": HELLASWAG[1],
                              "files": ["data/validation-00000-of-00001.parquet"]}),
}
LM_SETS: dict[str, Callable[[], tuple[list[dict], dict]]] = {
    "fineweb2": build_fineweb2, "wikipedia": build_wikipedia, "legal": build_legal, "flores": build_flores,
    "english": build_english}
ALL = [*LM_SETS, *MC_SETS, "merge_dev"]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", nargs="+", choices=ALL, default=ALL)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    man_path = args.out / "evalsets_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for name in args.sets:
        if name == "merge_dev":
            (merge, dev), meta = build_merge_and_dev()
            meta |= {"merge_docs": len(merge), "dev_docs": len(dev),
                     "merge_sha256": write_jsonl(args.out / "merge.jsonl", merge),
                     "dev_sha256": write_jsonl(args.out / "dev.jsonl", dev)}
        else:
            kind = "lm" if name in LM_SETS else "mc"
            rows, meta = (LM_SETS | MC_SETS)[name]()
            assert all(r["text"] == nfc(r["text"]) for r in rows) if kind == "lm" else True
            meta |= {"kind": kind, "count": len(rows), "sha256": write_jsonl(args.out / f"{name}.jsonl", rows)}
            if kind == "mc":
                meta["chance"] = CHANCE[name]
        manifest[name] = meta
        print(name, json.dumps({k: v for k, v in meta.items() if k != "train_files"}), flush=True)
    man_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
