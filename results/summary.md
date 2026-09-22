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

Documents scored by every run: clean 1996, strip100 1987

| hypothesis | A − B | variant | Δbpc | 95% CI | significant |
|---|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | clean | -0.0018 | [-0.0024, -0.0012] | yes |
| H1 | super-nfd − bpe-nfd | clean | -0.0035 | [-0.0041, -0.0028] | yes |
| H2 | bpe-nfd − bpe-nfc | strip100 | +0.0171 | [+0.0160, +0.0181] | yes |
| H2 | bpe-nfd − bpe-nfc | strip50 | -0.0115 | [-0.0125, -0.0106] | yes |
| H2 cost | bpe-nfd − bpe-nfc | clean | +0.0012 | [+0.0007, +0.0018] | yes |
| H2 | super-nfd − super-nfc | strip100 | +0.0011 | [-0.0000, +0.0022] | no |
| H2 cost | super-nfd − super-nfc | clean | -0.0005 | [-0.0010, +0.0001] | no |

| pairs | A − B | A only right | B only right | p |
|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | 9 | 4 | 0.267 |
| H1 | super-nfd − bpe-nfd | 3 | 11 | 0.0574 |
| H2 cost | bpe-nfd − bpe-nfc | 11 | 3 | 0.0574 |
| H2 cost | super-nfd − super-nfc | 6 | 11 | 0.332 |

Seed noise bpe-nfc: s1 − s0 = -0.0001 bpc

Seed noise super-nfc: s1 − s0 = +0.0005 bpc

## d8

| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token | tokens/syllable | superword share |
|---|---|---|---|---|---|---|---|
| bpe-nfc s0 | 1.0019 | 2.1036 | 2.0177 | 0.994 | 3.861 | 1.191 | 0.0% |
| bpe-nfc s1 | 1.0023 | 2.0941 | 2.0105 | 0.993 | 3.861 | 1.191 | 0.0% |
| bpe-nfd s0 | 1.0026 | 2.0985 | 2.0183 | 0.993 | 3.860 | 1.191 | 0.0% |
| super-nfc s0 | 1.0037 | 2.1228 | 2.0281 | 0.995 | 4.694 | 0.979 | 19.6% |
| super-nfc s1 | 1.0037 | 2.1243 | 2.0241 | 0.992 | 4.694 | 0.979 | 19.6% |
| super-nfd s0 | 1.0039 | 2.1230 | 2.0235 | 0.993 | 4.692 | 0.980 | 19.6% |

Documents scored by every run: clean 1996, strip100 1987

| hypothesis | A − B | variant | Δbpc | 95% CI | significant |
|---|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | clean | +0.0018 | [+0.0012, +0.0024] | yes |
| H1 | super-nfd − bpe-nfd | clean | +0.0013 | [+0.0007, +0.0019] | yes |
| H2 | bpe-nfd − bpe-nfc | strip100 | +0.0007 | [-0.0002, +0.0016] | no |
| H2 | bpe-nfd − bpe-nfc | strip50 | -0.0051 | [-0.0060, -0.0042] | yes |
| H2 cost | bpe-nfd − bpe-nfc | clean | +0.0008 | [+0.0002, +0.0013] | yes |
| H2 | super-nfd − super-nfc | strip100 | -0.0047 | [-0.0056, -0.0038] | yes |
| H2 cost | super-nfd − super-nfc | clean | +0.0003 | [-0.0002, +0.0008] | no |

| pairs | A − B | A only right | B only right | p |
|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | 7 | 4 | 0.549 |
| H1 | super-nfd − bpe-nfd | 10 | 10 | 1 |
| H2 cost | bpe-nfd − bpe-nfc | 5 | 9 | 0.424 |
| H2 cost | super-nfd − super-nfc | 4 | 11 | 0.118 |

Seed noise bpe-nfc: s1 − s0 = +0.0005 bpc

Seed noise super-nfc: s1 − s0 = -0.0000 bpc

## d10

| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token | tokens/syllable | superword share |
|---|---|---|---|---|---|---|---|
| bpe-nfc s0 | 0.9370 | 2.0003 | 1.9160 | 0.995 | 3.861 | 1.191 | 0.0% |
| super-nfc s0 | 0.9399 | 2.0121 | 1.9185 | 0.994 | 4.694 | 0.979 | 19.6% |

Documents scored by every run: clean 1996, strip100 1987

| hypothesis | A − B | variant | Δbpc | 95% CI | significant |
|---|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | clean | +0.0028 | [+0.0023, +0.0034] | yes |

| pairs | A − B | A only right | B only right | p |
|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | 3 | 6 | 0.508 |

## Sensitivity of H1 (bpc clean, seed 0)

| A − B | depth | all docs | without the 5% longest | short half | long half | docs favouring A |
|---|---|---|---|---|---|---|
| super-nfc − bpe-nfc | d6 | -0.0018 | -0.0018 | -0.0025 | -0.0013 | 1105/1996 |
| super-nfd − bpe-nfd | d6 | -0.0035 | -0.0036 | -0.0040 | -0.0031 | 1189/1996 |
| super-nfc − bpe-nfc | d8 | +0.0018 | +0.0018 | +0.0015 | +0.0020 | 882/1996 |
| super-nfd − bpe-nfd | d8 | +0.0013 | +0.0013 | +0.0004 | +0.0019 | 917/1996 |
| super-nfc − bpe-nfc | d10 | +0.0028 | +0.0027 | +0.0017 | +0.0036 | 806/1996 |

## H3: effect vs model size (bpc clean, seed 0)

| A − B | d6 | d8 | d10 |
|---|---|---|---|
| super-nfc − bpe-nfc | -0.0018 ± 0.0005 | +0.0018 ± 0.0005 | +0.0028 |
| super-nfd − bpe-nfd | -0.0035 | +0.0013 | — |

± is the spread between the two seeds of the same condition (a lower bound on run-to-run
noise: the seed only changes weight init, the data order is identical).

| depth | non-embedding params | bpe-nfc | bpe-nfd | super-nfc | super-nfd |
|---|---|---|---|---|---|
| d6 | 10,616,832 | 1.0914 | 1.0926 | 1.0896 | 1.0891 |
| d8 | 25,165,824 | 1.0019 | 1.0026 | 1.0037 | 1.0039 |
| d10 | 49,152,000 | 0.9370 | — | 0.9399 | — |

Figure: figures/h3_scaling.png
