"""Remeasure the retained tokenizer survey from checksum-pinned tokenizers and documents."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from tokenizers import Tokenizer


def measure(tokenizer: Tokenizer, docs: list[str]) -> dict:
    if not docs or not all(docs):
        raise ValueError("survey requires nonempty documents")
    tokens = [len(row.ids) for row in tokenizer.encode_batch(docs, add_special_tokens=False)]
    chars = [len(text) for text in docs]
    if not all(tokens):
        raise ValueError("survey tokenizer produced an empty encoding")
    return {"chars": chars, "tokens": tokens, "chars_per_token": sum(chars) / sum(tokens)}


def regenerate(recorded: list[dict], docs_root: Path, offline: bool = False) -> list[dict]:
    from huggingface_hub import hf_hub_download

    docs = {}
    if not recorded or len({row["name"] for row in recorded}) != len(recorded):
        raise ValueError("survey sources must be nonempty and unique")
    for part, filename in (("test", "test.jsonl"), ("val", "val_docs.jsonl")):
        blob = (docs_root / filename).read_bytes()
        docs[part] = [json.loads(line)["text"] for line in blob.decode("utf-8").splitlines()]
        expected = {"sha256": hashlib.sha256(blob).hexdigest(), "documents": len(docs[part])}
        if any(row["inputs"][part] != expected for row in recorded):
            raise ValueError("survey documents differ from the pinned input")
    result = []
    for row in recorded:
        name = row["name"]
        if name.startswith("nanovitok/"):
            condition = name.split("/")[1].removesuffix("-16k")
            path = docs_root / "tokenizers-16k" / condition / "tokenizer.json"
        else:
            for attempt in range(3):
                try:
                    path = Path(hf_hub_download(name, "tokenizer.json", revision=row["revision"],
                                                local_files_only=offline))
                    break
                except Exception:
                    if offline or attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != row["tokenizer_sha256"] or len(blob) != row["tokenizer_bytes"]:
            raise ValueError(f"survey tokenizer checksum differs: {name}")
        tokenizer = Tokenizer.from_file(str(path))
        measured = {part: measure(tokenizer, texts) for part, texts in docs.items()}
        roundtrips = sum(tokenizer.decode(tokenizer.encode(text, add_special_tokens=False).ids) == text
                         for text in docs["test"][:200])
        result.append({**row, "results": measured, "vocab": tokenizer.get_vocab_size(), "roundtrips": roundtrips})
        print(f"Measured {name}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    recorded = json.loads(args.recorded.read_text(encoding="utf-8"))
    result = regenerate(recorded, args.docs_root, args.offline)
    if result != recorded:
        raise ValueError("survey remeasurement differs from the recorded results")
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
