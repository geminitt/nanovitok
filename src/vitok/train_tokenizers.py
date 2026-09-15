"""Train the four tokenizers (BPE / SuperBPE x NFC / NFD) on the same corpus.

    python -m vitok.train_tokenizers --corpus tok_train.txt --out tokenizers --vocab-size 16000

Stage 2 of SuperBPE runs in `vitok.superbpe` by default (`--stage2 fast`, stock `tokenizers` is enough).
`--stage2 fork` uses the SuperBPE fork instead, which resumes BPE training when `merges.txt` exists in
the working directory; it only works inside the fork's environment (with stock `tokenizers` it silently
trains from scratch) and does not finish on the full corpus. Notebook 01 uses it to check `fast` on a
small corpus.
"""

import argparse
import json
import os
import time
import unicodedata
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from vitok import superbpe
from vitok.tokenizer_spec import SPECIAL_TOKENS, STAGE1_REGEX, STAGE2_REGEX


def _pre_tokenizer(regex: str):
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(regex), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
    ])


def _train(workdir: Path, files: list[str], vocab_size: int, regex: str) -> Tokenizer:
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = _pre_tokenizer(regex)
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


def _stage2_fast(files: list[str], vocab: dict[str, int], merges: list[tuple[str, str]], vocab_size: int) -> tuple[Tokenizer, dict]:
    start = time.time()
    ids = superbpe.encode_corpus(files, vocab, merges, _pre_tokenizer(STAGE2_REGEX))
    encoded = time.time()
    print(f"stage 2: encoded {len(ids):,} ids with {len(merges)} inherited merges in {encoded - start:.0f}s", flush=True)
    full_vocab, new_merges = superbpe.train_stage2(ids, vocab, vocab_size)
    tok = Tokenizer(BPE(vocab=full_vocab, merges=merges + new_merges))
    tok.pre_tokenizer = _pre_tokenizer(STAGE2_REGEX)
    stats = {"stage2_ids": len(ids), "stage2_new_merges": len(new_merges),
             "encode_seconds": round(encoded - start), "merge_seconds": round(time.time() - encoded)}
    return tok, stats


def train_all(corpus: Path, out: Path, vocab_size: int, transition: float, workroot: Path, stage2: str = "fast") -> dict:
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
        vocab = json.loads((bpe_dir / "vocab.json").read_text(encoding="utf-8"))
        n_alphabet = len(vocab) - len(merges)
        n_inherit = round(transition * vocab_size) - n_alphabet
        meta[norm] = {"n_alphabet": n_alphabet, "n_inherited_merges": n_inherit}
        if stage2 == "fork":
            super_dir = workroot / f"super-{norm}"
            super_dir.mkdir(parents=True, exist_ok=True)
            (super_dir / "merges.txt").write_text("\n".join([header, *merges[:n_inherit]]) + "\n", encoding="utf-8")
            sup = _train(super_dir, files, vocab_size, STAGE2_REGEX)
        else:
            # ids follow merge order (alphabet first), so the inherited vocab is a prefix of the ids
            inherited = [tuple(m.split(" ")) for m in merges[:n_inherit]]
            inherited_vocab = {t: i for t, i in vocab.items() if i < n_alphabet + len(inherited)}
            assert len(inherited_vocab) == n_alphabet + len(inherited), "stage 1 produced a duplicate token"
            sup, stats = _stage2_fast(files, inherited_vocab, inherited, vocab_size)
            meta[norm].update(stats)
        _finalize(sup, norm, out / f"super-{norm}")

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
    ap.add_argument("--stage2", choices=["fast", "fork"], default="fast")
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    meta = train_all(args.corpus, args.out, args.vocab_size, args.transition, args.workdir, args.stage2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
