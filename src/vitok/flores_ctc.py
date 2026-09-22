"""Compare our tokenizers with the released MonTok tokenizers (arXiv 2510.21909) on FLORES-200.

    python -m vitok.flores_ctc --tokenizers vitok-data/tokenizers-16k --out results/flores_ctc.json

Their metric is the corpus token count (CTC) of the Vietnamese side of FLORES-200, counted line by
line with special tokens excluded (scripts/calculate_ctc.py of their repo). Their paper does not say
which split, so `calibrate` reports the CTC of every split for their own BPE tokenizer: the split
that reproduces PUBLISHED["bpe_16384"] is the one to compare on.

Published Vietnamese numbers, from data/ctc_all.csv and data/ctc_superbpe.csv of
https://github.com/catherinearnett/explaining_tokenizer_inequities (300MB training data each).
"""

import argparse
import json
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

from vitok.tokenizer_spec import CONDITIONS

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
MONTOK_REPO = "catherinearnett/montok"
MONTOK = {
    "bpe_16384": "bpe_unscaled_tokenizers/bpe_vie_latn_16384_300mb_unscaled.json",
    "bpe_32768": "bpe_unscaled_tokenizers/bpe_vie_latn_32768_300mb_unscaled.json",
    "superbpe_32768_t0.9": "superbpe_tokenizers/superbpe_vie_latn_32768_300mb_unscaled_0_9/tokenizer.json",
}
PUBLISHED = {"bpe_16384": 68619, "bpe_32768": 67011, "superbpe_32768_t0.9": 52831}


def flores_lines(cache_dir: Path) -> dict[str, list[str]]:
    """Vietnamese FLORES-200 sentences per split, NFC normalized like the rest of the project."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "flores200_dataset.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(FLORES_URL, archive)
    splits = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name.startswith("vie_Latn."):
                text = tar.extractfile(member).read().decode("utf-8")
                splits[name.split(".")[-1]] = [unicodedata.normalize("NFC", l) for l in text.splitlines()]
    splits["dev+devtest"] = splits.get("dev", []) + splits.get("devtest", [])
    return splits


def ctc(tok: Tokenizer, lines: list[str]) -> int:
    """Corpus token count: tokens for every line, special tokens excluded."""
    return sum(len(e.ids) for e in tok.encode_batch(lines, add_special_tokens=False))


def calibrate(splits: dict[str, list[str]], reference: Tokenizer, published: int) -> dict:
    """Which FLORES split comes closest to the published CTC of their own tokenizer.

    The tarball here is FLORES-200 from the NLLB release; their `flores_texts/` file is not published,
    so an exact match is not guaranteed. `gap` says how far off we are.
    """
    counts = {name: ctc(reference, lines) for name, lines in splits.items()}
    best = min(counts, key=lambda k: abs(counts[k] - published))
    return {"split": best, "counts": counts, "published": published,
            "gap": counts[best] - published, "gap_frac": (counts[best] - published) / published}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizers", type=Path, default=None, help="directory holding bpe-nfc/, super-nfc/, ...")
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/flores"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    splits = flores_lines(args.cache_dir)
    montok = {name: Tokenizer.from_file(hf_hub_download(MONTOK_REPO, path, repo_type="dataset"))
              for name, path in MONTOK.items()}
    cal = calibrate(splits, montok["bpe_16384"], PUBLISHED["bpe_16384"])
    lines = splits[cal["split"]]
    chars = sum(len(l) for l in lines)
    print(f"FLORES-200 vie_Latn, closest split: {cal['split']} ({len(lines)} lines, {chars:,} NFC chars). "
          f"Their bpe_16384 counts {cal['counts'][cal['split']]:,} here vs {cal['published']:,} published "
          f"({cal['gap_frac']:+.1%}); compare ratios, not absolute counts.\n")

    result = {"calibration": cal, "lines": len(lines), "chars_nfc": chars,
              "published": PUBLISHED, "montok": {}, "ours": {}}
    print(f"{'tokenizer':28s} {'vocab':>7s} {'CTC':>9s} {'published':>10s} {'chars/token':>12s}")
    for name, tok in montok.items():
        n = ctc(tok, lines)
        result["montok"][name] = {"ctc": n, "vocab_size": tok.get_vocab_size(), "chars_per_token": chars / n}
        print(f"{name:28s} {tok.get_vocab_size():7d} {n:9,d} {PUBLISHED[name]:10,d} {chars / n:12.3f}")
    if args.tokenizers:
        for cond in CONDITIONS:
            tok = Tokenizer.from_file(str(args.tokenizers / cond / "tokenizer.json"))
            n = ctc(tok, lines)
            result["ours"][cond] = {"ctc": n, "vocab_size": tok.get_vocab_size(), "chars_per_token": chars / n}
            print(f"{'ours: ' + cond:28s} {tok.get_vocab_size():7d} {n:9,d} {'':10s} {chars / n:12.3f}")
        for norm in ("nfc", "nfd"):
            ours = 1 - result["ours"][f"super-{norm}"]["ctc"] / result["ours"][f"bpe-{norm}"]["ctc"]
            result[f"token_reduction_{norm}"] = ours
            print(f"SuperBPE token reduction on FLORES ({norm}): {ours:.1%}")
        theirs = 1 - PUBLISHED["superbpe_32768_t0.9"] / PUBLISHED["bpe_32768"]
        result["token_reduction_published_32k"] = theirs
        print(f"Their published reduction at vocab 32,768, transition 0.9: {theirs:.1%}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
