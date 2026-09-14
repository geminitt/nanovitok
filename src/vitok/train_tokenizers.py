"""Train the four tokenizers (BPE / SuperBPE x NFC / NFD) on the same corpus.

Run inside the SuperBPE environment (forked `tokenizers`, see kaggle/01_data_tokenizers.ipynb):

    python -m vitok.train_tokenizers --corpus tok_train.txt --out tokenizers --vocab-size 16000

The fork resumes BPE training when `merges.txt` exists in the working directory; stage 2 of
SuperBPE relies on that. With the stock `tokenizers` library the script still runs, but stage 2
silently trains from scratch, so only use stock `tokenizers` for smoke tests.
"""

import argparse
import json
import os
import unicodedata
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from vitok.tokenizer_spec import SPECIAL_TOKENS, STAGE1_REGEX, STAGE2_REGEX


def _train(workdir: Path, files: list[str], vocab_size: int, regex: str) -> Tokenizer:
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(regex), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])
    # Full 256-byte alphabet: otherwise bytes unseen in the corpus cannot be encoded at all.
    trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True,
                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    cwd = os.getcwd()
    os.chdir(workdir)  # the fork looks for merges.txt in the cwd
    try:
        tok.train(files, trainer)
        tok.model.save(".")  # merges.txt, vocab.json
    finally:
        os.chdir(cwd)
    return tok


def _finalize(tok: Tokenizer, norm: str, out_dir: Path) -> None:
    tok.normalizer = normalizers.NFD() if norm == "nfd" else normalizers.NFC()
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIAL_TOKENS)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))


def _write_corpus(src: Path, dst: Path, norm: str) -> None:
    form = norm.upper()
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(unicodedata.normalize(form, line))


def train_all(corpus: Path, out: Path, vocab_size: int, transition: float, workroot: Path) -> dict:
    meta = {"vocab_size": vocab_size, "transition": transition, "corpus_bytes": corpus.stat().st_size}
    for norm in ("nfc", "nfd"):
        corpus_norm = workroot / f"corpus_{norm}.txt"
        _write_corpus(corpus, corpus_norm, norm)
        files = [str(corpus_norm.resolve())]

        # Stage 1 = the plain BPE condition.
        bpe_dir = workroot / f"bpe-{norm}"
        bpe_dir.mkdir(parents=True, exist_ok=True)
        (bpe_dir / "merges.txt").unlink(missing_ok=True)
        bpe = _train(bpe_dir, files, vocab_size, STAGE1_REGEX)
        _finalize(bpe, norm, out / f"bpe-{norm}")

        # Stage 2: inherit the first merges of the BPE run, then extend without whitespace splits.
        merges = (bpe_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
        header, merges = merges[0], merges[1:]
        alphabet = json.loads((bpe_dir / "vocab.json").read_text(encoding="utf-8"))
        n_alphabet = len(alphabet) - len(merges)
        n_inherit = round(transition * vocab_size) - n_alphabet
        super_dir = workroot / f"super-{norm}"
        super_dir.mkdir(parents=True, exist_ok=True)
        (super_dir / "merges.txt").write_text("\n".join([header, *merges[:n_inherit]]) + "\n", encoding="utf-8")
        sup = _train(super_dir, files, vocab_size, STAGE2_REGEX)
        _finalize(sup, norm, out / f"super-{norm}")

        meta[norm] = {"n_alphabet": n_alphabet, "n_inherited_merges": n_inherit}
        corpus_norm.unlink()
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True, help="NFC text file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--transition", type=float, default=0.9, help="SuperBPE transition point as a fraction of vocab")
    ap.add_argument("--workdir", type=Path, default=Path("tok_work"))
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(train_all(args.corpus, args.out, args.vocab_size, args.transition, args.workdir), indent=2))


if __name__ == "__main__":
    main()
