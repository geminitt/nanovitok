# Results summary

## d6

| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token |
|---|---|---|---|---|---|
| bpe-nfc s0 | 1.0914 | 2.2392 | 2.1982 | 0.993 | 3.861 |
| bpe-nfd s0 | 1.0926 | 2.2276 | 2.2152 | 0.996 | 3.860 |
| super-nfc s0 | 1.0896 | 2.2481 | 2.1910 | 0.995 | 4.694 |
| super-nfd s0 | 1.0891 | 2.2495 | 2.1921 | 0.993 | 4.692 |

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
