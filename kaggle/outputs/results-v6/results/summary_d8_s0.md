# Results summary

## d8

| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token |
|---|---|---|---|---|---|
| bpe-nfc s0 | 1.0019 | 2.1036 | 2.0177 | 0.994 | 3.861 |
| bpe-nfd s0 | 1.0026 | 2.0985 | 2.0183 | 0.993 | 3.860 |
| super-nfc s0 | 1.0037 | 2.1228 | 2.0281 | 0.995 | 4.694 |
| super-nfd s0 | 1.0039 | 2.1230 | 2.0235 | 0.993 | 4.692 |

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
