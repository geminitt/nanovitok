# Results summary

## d6

| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token | tokens/syllable | superword share |
|---|---|---|---|---|---|---|---|
| bpe-nfc s0 | 1.0914 | 2.2392 | 2.1982 | 0.993 | 3.861 | 1.191 | 0.0% |
| bpe-nfc s1 | 1.0913 | 2.2368 | 2.1936 | 0.993 | 3.861 | 1.191 | 0.0% |
| bpe-nfd s0 | 1.0926 | 2.2276 | 2.2152 | 0.996 | 3.860 | 1.191 | 0.0% |
| super-nfc s0 | 1.0896 | 2.2481 | 2.1910 | 0.995 | 4.694 | 0.979 | 19.6% |
| super-nfc s1 | 1.0901 | 2.2478 | 2.1842 | 0.992 | 4.694 | 0.979 | 19.6% |
| super-nfd s0 | 1.0891 | 2.2495 | 2.1921 | 0.993 | 4.692 | 0.980 | 19.6% |

Documents scored by every run: clean 1996, strip50 1993, strip100 1987

Seed noise (largest |s1 − s0| among conditions with two seeds; a lower bound): clean 0.0005, strip50 0.0023, strip100 0.0068

| hypothesis | A − B | variant | Δbpc | 95% CI | Δ relative [95% CI] | CI excludes 0 | |Δ| > seed noise |
|---|---|---|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | clean | -0.0018 | [-0.0024, -0.0012] | -0.16% [-0.22%, -0.11%] | yes | yes |
| H1 | super-nfd − bpe-nfd | clean | -0.0035 | [-0.0041, -0.0028] | -0.32% [-0.38%, -0.25%] | yes | yes |
| H2 | bpe-nfd − bpe-nfc | strip100 | +0.0171 | [+0.0160, +0.0181] | +0.78% [+0.73%, +0.82%] | yes | yes |
| H2 | bpe-nfd − bpe-nfc | strip50 | -0.0115 | [-0.0125, -0.0106] | -0.52% [-0.56%, -0.47%] | yes | yes |
| H2 cost | bpe-nfd − bpe-nfc | clean | +0.0012 | [+0.0007, +0.0018] | +0.11% [+0.06%, +0.16%] | yes | yes |
| H2 | super-nfd − super-nfc | strip100 | +0.0011 | [-0.0000, +0.0022] | +0.05% [-0.00%, +0.10%] | no | no |
| H2 | super-nfd − super-nfc | strip50 | +0.0013 | [+0.0004, +0.0023] | +0.06% [+0.02%, +0.10%] | yes | no |
| H2 cost | super-nfd − super-nfc | clean | -0.0005 | [-0.0010, +0.0001] | -0.04% [-0.10%, +0.01%] | no | yes |

| pairs | A − B | A only right | B only right | p |
|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | 9 | 4 | 0.267 |
| H1 | super-nfd − bpe-nfd | 3 | 11 | 0.0574 |
| H2 cost | bpe-nfd − bpe-nfc | 11 | 3 | 0.0574 |
| H2 cost | super-nfd − super-nfc | 6 | 11 | 0.332 |

## d8

| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token | tokens/syllable | superword share |
|---|---|---|---|---|---|---|---|
| bpe-nfc s0 | 1.0019 | 2.1036 | 2.0177 | 0.994 | 3.861 | 1.191 | 0.0% |
| bpe-nfc s1 | 1.0023 | 2.0941 | 2.0105 | 0.993 | 3.861 | 1.191 | 0.0% |
| bpe-nfd s0 | 1.0026 | 2.0985 | 2.0183 | 0.993 | 3.860 | 1.191 | 0.0% |
| super-nfc s0 | 1.0037 | 2.1228 | 2.0281 | 0.995 | 4.694 | 0.979 | 19.6% |
| super-nfc s1 | 1.0037 | 2.1243 | 2.0241 | 0.992 | 4.694 | 0.979 | 19.6% |
| super-nfd s0 | 1.0039 | 2.1230 | 2.0235 | 0.993 | 4.692 | 0.980 | 19.6% |

Documents scored by every run: clean 1996, strip50 1993, strip100 1987

Seed noise (largest |s1 − s0| among conditions with two seeds; a lower bound): clean 0.0005, strip50 0.0095, strip100 0.0072

| hypothesis | A − B | variant | Δbpc | 95% CI | Δ relative [95% CI] | CI excludes 0 | |Δ| > seed noise |
|---|---|---|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | clean | +0.0018 | [+0.0012, +0.0024] | +0.18% [+0.12%, +0.24%] | yes | yes |
| H1 | super-nfd − bpe-nfd | clean | +0.0013 | [+0.0007, +0.0019] | +0.13% [+0.07%, +0.19%] | yes | yes |
| H2 | bpe-nfd − bpe-nfc | strip100 | +0.0007 | [-0.0002, +0.0016] | +0.03% [-0.01%, +0.08%] | no | no |
| H2 | bpe-nfd − bpe-nfc | strip50 | -0.0051 | [-0.0060, -0.0042] | -0.24% [-0.28%, -0.20%] | yes | no |
| H2 cost | bpe-nfd − bpe-nfc | clean | +0.0008 | [+0.0002, +0.0013] | +0.08% [+0.02%, +0.13%] | yes | yes |
| H2 | super-nfd − super-nfc | strip100 | -0.0047 | [-0.0056, -0.0038] | -0.23% [-0.27%, -0.19%] | yes | no |
| H2 | super-nfd − super-nfc | strip50 | +0.0001 | [-0.0008, +0.0010] | +0.01% [-0.04%, +0.05%] | no | no |
| H2 cost | super-nfd − super-nfc | clean | +0.0003 | [-0.0002, +0.0008] | +0.03% [-0.02%, +0.08%] | no | no |

| pairs | A − B | A only right | B only right | p |
|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | 7 | 4 | 0.549 |
| H1 | super-nfd − bpe-nfd | 10 | 10 | 1 |
| H2 cost | bpe-nfd − bpe-nfc | 5 | 9 | 0.424 |
| H2 cost | super-nfd − super-nfc | 4 | 11 | 0.118 |

## d10

| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token | tokens/syllable | superword share |
|---|---|---|---|---|---|---|---|
| bpe-nfc s0 | 0.9370 | 2.0003 | 1.9160 | 0.995 | 3.861 | 1.191 | 0.0% |
| super-nfc s0 | 0.9399 | 2.0121 | 1.9185 | 0.994 | 4.694 | 0.979 | 19.6% |

Documents scored by every run: clean 1996, strip50 1993, strip100 1987

| hypothesis | A − B | variant | Δbpc | 95% CI | Δ relative [95% CI] | CI excludes 0 | |Δ| > seed noise |
|---|---|---|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | clean | +0.0028 | [+0.0023, +0.0034] | +0.30% [+0.24%, +0.36%] | yes | no second seed |

| pairs | A − B | A only right | B only right | p |
|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | 3 | 6 | 0.508 |

## Pre-registered verdicts

A comparison counts as resolved only when its 95% CI excludes 0 **and** |Δ| exceeds the seed noise (where a second seed exists).

**H1** — token reduction on the val shard: NFC 17.8%, NFD 17.7% (threshold 15%): met.
- d6 super-nfc − bpe-nfc: -0.16% [-0.22%, -0.11%] → not worse by more than 1%: yes
- d6 super-nfd − bpe-nfd: -0.32% [-0.38%, -0.25%] → not worse by more than 1%: yes
- d8 super-nfc − bpe-nfc: +0.18% [+0.12%, +0.24%] → not worse by more than 1%: yes
- d8 super-nfd − bpe-nfd: +0.13% [+0.07%, +0.19%] → not worse by more than 1%: yes
- d10 super-nfc − bpe-nfc: +0.30% [+0.24%, +0.36%] → not worse by more than 1%: yes
- Verdict: **supported**.

**H2** — NFD tokenizers give lower bpc on diacritic-stripped text, at a clean-text cost within 1%:
- d6 bpe-nfd − bpe-nfc, strip100: +0.0171 → NFD worse
- d6 bpe-nfd − bpe-nfc, strip50: -0.0115 → NFD better
- d6 super-nfd − super-nfc, strip100: +0.0011 → not resolved
- d6 super-nfd − super-nfc, strip50: +0.0013 → not resolved
- d8 bpe-nfd − bpe-nfc, strip100: +0.0007 → not resolved
- d8 bpe-nfd − bpe-nfc, strip50: -0.0051 → not resolved
- d8 super-nfd − super-nfc, strip100: -0.0047 → not resolved
- d8 super-nfd − super-nfc, strip50: +0.0001 → not resolved
- d6 bpe-nfd − bpe-nfc, clean (cost): +0.11% [+0.06%, +0.16%] → within 1%: yes
- d6 super-nfd − super-nfc, clean (cost): -0.04% [-0.10%, +0.01%] → within 1%: yes
- d8 bpe-nfd − bpe-nfc, clean (cost): +0.08% [+0.02%, +0.13%] → within 1%: yes
- d8 super-nfd − super-nfc, clean (cost): +0.03% [-0.02%, +0.08%] → within 1%: yes
- Verdict: **not supported (resolved comparisons point in opposite directions)**.

**H3** — Δbpc clean, super-nfc − bpe-nfc, by depth and seed:
- d6: s0 -0.0018, s1 -0.0012
- d8: s0 +0.0018, s1 +0.0013
- d10: s0 +0.0028
- Three sizes and at most two seeds: a trend, not a law.

**H4** (exploratory, no test) — superwords of 2–4 syllables that are exactly one underthesea word:
- super-nfc: 65.6%; frequency-matched baseline 57.0%; all adjacent syllables 25.2%; 9.8% of superword occurrences span another number of syllables
- super-nfd: 65.6%; frequency-matched baseline 57.0%; all adjacent syllables 25.2%; 9.8% of superword occurrences span another number of syllables


## Sensitivity of H1 (bpc clean, seed 0)

| A − B | depth | all docs | without the 5% longest | short half | long half | docs favouring A |
|---|---|---|---|---|---|---|
| super-nfc − bpe-nfc | d6 | -0.0018 | -0.0018 | -0.0025 | -0.0013 | 1105/1996 |
| super-nfd − bpe-nfd | d6 | -0.0035 | -0.0036 | -0.0040 | -0.0031 | 1189/1996 |
| super-nfc − bpe-nfc | d8 | +0.0018 | +0.0018 | +0.0015 | +0.0020 | 882/1996 |
| super-nfd − bpe-nfd | d8 | +0.0013 | +0.0013 | +0.0004 | +0.0019 | 917/1996 |
| super-nfc − bpe-nfc | d10 | +0.0028 | +0.0027 | +0.0017 | +0.0036 | 806/1996 |

## nanochat validation bpb vs test bpc (super-nfc − bpe-nfc, seed 0)

| depth | val bpb bpe-nfc | val bpb super-nfc | val relative | test bpc relative |
|---|---|---|---|---|
| d6 | 0.7862 | 0.7795 | -0.86% | -0.16% |
| d8 | 0.7200 | 0.7167 | -0.46% | +0.18% |
| d10 | 0.6756 | 0.6732 | -0.35% | +0.30% |

nanochat evaluates a fixed number of tokens of the val shard, so the two tokenizers are scored on
different documents (SuperBPE covers 22% more text), unpaired. Only the paired test bpc is used for
conclusions; the table shows how far the two disagree. NFD runs are left out: their bpb counts NFD bytes.

## H3: effect vs model size (bpc clean)

| A − B | d6 | d8 | d10 |
|---|---|---|---|
| super-nfc − bpe-nfc | -0.0018 (s1: -0.0012) | +0.0018 (s1: +0.0013) | +0.0028 |
| super-nfd − bpe-nfd | -0.0035 | +0.0013 | — |

Seed 0, with the same difference for seed 1 in brackets where both conditions have a second seed.
The seed only changes weight init (the data order is identical), so the gap between the two is a
lower bound on run-to-run noise.

| depth | non-embedding params | bpe-nfc | bpe-nfd | super-nfc | super-nfd |
|---|---|---|---|---|---|
| d6 | 10,616,832 | 1.0914 | 1.0926 | 1.0896 | 1.0891 |
| d8 | 25,165,824 | 1.0019 | 1.0026 | 1.0037 | 1.0039 |
| d10 | 49,152,000 | 0.9370 | — | 0.9399 | — |

Figure: figures/h3_scaling.png
