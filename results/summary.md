# From-scratch NFC pilot

Pilot evidence under the original equal-text design; see README for the Qwen3 retrofit study.

## d6

| run | bpc | chars/token | tokens/syllable | superword share |
|---|---:|---:|---:|---:|
| bpe-nfc s0 | 1.0914 | 3.861 | 1.191 | 0.0% |
| bpe-nfc s1 | 1.0913 | 3.861 | 1.191 | 0.0% |
| super-nfc s0 | 1.0896 | 4.694 | 0.979 | 19.6% |
| super-nfc s1 | 1.0901 | 4.694 | 0.979 | 19.6% |

Documents scored by every run: 1996.

Seed spreads (s1 − s0): -0.0001, +0.0005; approximate t-based noise threshold 0.0014.

| A − B | Δbpc | 95% interval | Relative difference [95% interval] | Interval excludes 0 | Beyond seed noise |
|---|---:|---|---|---|---|
| super-nfc − bpe-nfc | -0.0018 | [-0.0024, -0.0012] | -0.16% [-0.22%, -0.11%] | yes | yes |

## d8

| run | bpc | chars/token | tokens/syllable | superword share |
|---|---:|---:|---:|---:|
| bpe-nfc s0 | 1.0019 | 3.861 | 1.191 | 0.0% |
| bpe-nfc s1 | 1.0023 | 3.861 | 1.191 | 0.0% |
| super-nfc s0 | 1.0037 | 4.694 | 0.979 | 19.6% |
| super-nfc s1 | 1.0037 | 4.694 | 0.979 | 19.6% |

Documents scored by every run: 1996.

Seed spreads (s1 − s0): +0.0005, -0.0000; approximate t-based noise threshold 0.0014.

| A − B | Δbpc | 95% interval | Relative difference [95% interval] | Interval excludes 0 | Beyond seed noise |
|---|---:|---|---|---|---|
| super-nfc − bpe-nfc | +0.0018 | [+0.0012, +0.0024] | +0.18% [+0.12%, +0.24%] | yes | yes |

## d10

| run | bpc | chars/token | tokens/syllable | superword share |
|---|---:|---:|---:|---:|
| bpe-nfc s0 | 0.9370 | 3.861 | 1.191 | 0.0% |
| super-nfc s0 | 0.9399 | 4.694 | 0.979 | 19.6% |

Documents scored by every run: 1996.

| A − B | Δbpc | 95% interval | Relative difference [95% interval] | Interval excludes 0 | Beyond seed noise |
|---|---:|---|---|---|---|
| super-nfc − bpe-nfc | +0.0028 | [+0.0023, +0.0034] | +0.30% [+0.24%, +0.36%] | yes | no second seed |

## NFC pilot quality margin

Token reduction: NFC 17.8% (threshold 15%): met.
- d6 super-nfc − bpe-nfc: -0.16% [-0.22%, -0.11%] → not worse by more than 1%: yes
- d8 super-nfc − bpe-nfc: +0.18% [+0.12%, +0.24%] → not worse by more than 1%: yes
- d10 super-nfc − bpe-nfc: +0.30% [+0.24%, +0.36%] → not worse by more than 1%: yes
- NFC pilot criterion: **met**.

## NFC pilot size trend

- d6: s0 -0.0018, s1 -0.0012
- d8: s0 +0.0018, s1 +0.0013
- d10: s0 +0.0028
- Three sizes and at most two seeds: a trend, not a law.

## Sensitivity of H1 (bpc clean, seed 0)

| A − B | depth | all docs | without the 5% longest | short half | long half | docs favouring A |
|---|---|---|---|---|---|---|
| super-nfc − bpe-nfc | d6 | -0.0018 | -0.0019 | -0.0025 | -0.0013 | 1105/1996 |
| super-nfc − bpe-nfc | d8 | +0.0018 | +0.0018 | +0.0015 | +0.0020 | 882/1996 |
| super-nfc − bpe-nfc | d10 | +0.0028 | +0.0028 | +0.0017 | +0.0036 | 806/1996 |

## nanochat validation bpb vs test bpc (super-nfc − bpe-nfc, seed 0)

| depth | val bpb bpe-nfc | val bpb super-nfc | val relative | test bpc relative |
|---|---|---|---|---|
| d6 | 0.7862 | 0.7795 | -0.86% | -0.16% |
| d8 | 0.7200 | 0.7167 | -0.46% | +0.18% |
| d10 | 0.6756 | 0.6732 | -0.35% | +0.30% |

nanochat evaluates a fixed number of tokens of the val shard, so the two tokenizers are scored on
different documents (SuperBPE covers 22% more text), unpaired. Only the paired test bpc is used for
conclusions; the table shows how far the two disagree.

## Sensitivity of H1 on the val shard (seed 0)

| A − B | depth | val docs | val Δbpc | val relative [95% CI] | test relative | same sign |
|---|---|---|---|---|---|---|
| super-nfc − bpe-nfc | d6 | 4963 | -0.0019 | -0.18% [-0.21%, -0.14%] | -0.16% | yes |
| super-nfc − bpe-nfc | d8 | 4963 | +0.0020 | +0.20% [+0.17%, +0.24%] | +0.18% | yes |
| super-nfc − bpe-nfc | d10 | 4963 | +0.0026 | +0.28% [+0.25%, +0.32%] | +0.30% | yes |

## NFC pilot effect vs model size (bpc clean)

| A − B | d6 | d8 | d10 |
|---|---|---|---|
| super-nfc − bpe-nfc | -0.0018 (s1: -0.0012) | +0.0018 (s1: +0.0013) | +0.0028 |

Seed 0, with the same difference for seed 1 in brackets where both conditions have a second seed.
The seed only changes weight init (the data order is identical), so the gap between the two is a
lower bound on run-to-run noise.

| depth | non-embedding params | bpe-nfc | super-nfc |
|---|---|---|---|
| d6 | 10,616,832 | 1.0914 | 1.0896 |
| d8 | 25,165,824 | 1.0019 | 1.0037 |
| d10 | 49,152,000 | 0.9370 | 0.9399 |

Figure: figures/h3_scaling.png
