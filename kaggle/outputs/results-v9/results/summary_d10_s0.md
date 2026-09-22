# Results summary

## d10

| run | bpc clean | bpc strip50 | bpc strip100 | pair acc | chars/token |
|---|---|---|---|---|---|
| bpe-nfc s0 | 0.9370 | 2.0003 | 1.9160 | 0.995 | 3.861 |
| super-nfc s0 | 0.9399 | 2.0121 | 1.9185 | 0.994 | 4.694 |

Documents scored by every run: clean 1996, strip100 1987

| hypothesis | A − B | variant | Δbpc | 95% CI | significant |
|---|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | clean | +0.0028 | [+0.0023, +0.0034] | yes |

| pairs | A − B | A only right | B only right | p |
|---|---|---|---|---|
| H1 | super-nfc − bpe-nfc | 3 | 6 | 0.508 |
