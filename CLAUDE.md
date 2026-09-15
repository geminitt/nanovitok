# vitok — SuperBPE & NFD tokenizers for small Vietnamese LMs

Deep-learning class project (solo, 3 weeks). Question: does merging across whitespace (SuperBPE) and
decomposing diacritics (NFD) help small Vietnamese language models, and does the effect change with
model size (nanochat d6/d8/d10)? Full research plan: `plan-tokenizer-vietnamese-steps.md` (Vietnamese).
Pre-registered analysis: `docs/analysis_plan.md`. The user communicates in Vietnamese.

## Environment rules (strict)

- **Never install or use `uv`.** Python envs: **pixi** (`pixi.toml`). Rust/CLI tools: **mise**.
  System packages: apt, following `~/wsl/apt-packages.txt`; ask before anything needing sudo.
- **No training on the local machine.** Tokenizer training, pretraining and evaluation run on **Kaggle**
  (`kaggle/01_data_tokenizers.ipynb` CPU, `kaggle/02_train_eval.ipynb` T4×2). Locally: code + CPU unit tests only.

## Commands

```bash
pixi run test          # CPU unit + nanochat integration tests (needs third_party/nanochat, see below)
pixi run python -m vitok.analysis --results results --compression compression-16k.json --out results/summary.md
```

`third_party/` is gitignored. Recreate it with:

```bash
git clone https://github.com/karpathy/nanochat third_party/nanochat
git -C third_party/nanochat checkout $(cat patches/NANOCHAT_COMMIT)
git -C third_party/nanochat apply ../../patches/nanochat.patch
```

After editing nanochat, regenerate the patch: `git -C third_party/nanochat diff > patches/nanochat.patch`.

## Layout

- `src/vitok/` — `data` (FineWeb-2 vie_Latn splits), `train_tokenizers` (stage 1 = HF `BpeTrainer`), `superbpe` (stage 2 in numpy;
  the SuperBPE fork can't finish it on 500MB and is only used to check it on 3MB in notebook 01),
  `tokenizer_spec` (stdlib-only constants), `hf_tokenizer` (nanochat wrapper), `compression`, `conditions`
  (equal-text config), `eval` (per-doc nats), `minimal_pairs`, `stats`, `analysis`, `kaggle_run` (per-GPU queue).
- `patches/nanochat.patch` — 3 small changes: load `tokenizer.json` if present, `--scaling-batch-size`, `NANOCHAT_SEED`.
- `kaggle/` — the two notebooks. `tests/` — pytest.

## Invariants that silently break the science if violated

- Compare models with **bits per NFC character** (`vitok.stats.bpc`), never loss/token and never nanochat's bpb
  (NFD text has more bytes, which flatters NFD tokenizers).
- Equal-text design: same steps and 64 sequences/step for all conditions; `max_seq_len` scaled by chars/token;
  LR/WD pinned via `--scaling-batch-size` (`vitok.conditions`).
- Pretokenizer letter classes must include `\p{M}`, or NFD combining marks get split from their letters.
- BPE trainers must use the full 256-byte initial alphabet, or unseen bytes are dropped and bpc looks better than it is.
- Eval batches are right-padded; correctness relies on causal attention (tested in `test_padding_does_not_change_nats`).
- nanochat on T4: `NANOCHAT_DTYPE=float16` (auto-detect would pick fp32), `--window-pattern L` (SDPA has no sliding window).
