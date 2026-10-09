# Retrofit results

Canonical-tokenization bpc; lower is better. Paired document bootstrap: 10,000 resamples, seed 0. The fixed criterion requires the upper relative 95% bound below +1%.

## Final quality

| Condition | VI bpc | Difference vs base [95% interval] | Within 1% | EN bpc | Belebele | Token saving |
|---|---:|---|---|---:|---:|---:|
| base | 1.092801 | reference | reference | 1.061271 | 35.00% | 0.00% |
| syllable-1000 | 1.102770 | +0.912% [+0.869%, +0.955%] | yes | 1.063913 | 35.22% | 7.50% |
| multisyllable-1000 | 1.109404 | +1.519% [+1.471%, +1.571%] | no | 1.065679 | 35.33% | 15.82% |
| multisyllable-1000-s1 | 1.108361 | +1.424% [+1.377%, +1.473%] | no | 1.065461 | 35.33% | 15.82% |

## Recovery

| Condition | Step | Processed tokens | Characters read | VI bpc | EN bpc | Belebele |
|---|---:|---:|---:|---:|---:|---:|
| base | 0 | 0 | 0 | 1.258153 | 1.106317 | 34.78% |
| base | 300 | 4,915,200 | 16,805,500 | 1.147809 | 1.066919 | 34.89% |
| base | 600 | 9,830,400 | 33,575,833 | 1.116182 | 1.062539 | 34.44% |
| base | 1200 | 19,660,800 | 67,094,352 | 1.092801 | 1.061271 | 35.00% |
| syllable-1000 | 0 | 0 | 0 | 1.500032 | 1.106680 | 32.00% |
| syllable-1000 | 300 | 4,915,200 | 18,154,360 | 1.169721 | 1.072155 | 35.44% |
| syllable-1000 | 600 | 9,830,400 | 36,256,275 | 1.136757 | 1.071499 | 34.22% |
| syllable-1000 | 1200 | 19,660,800 | 72,456,286 | 1.102770 | 1.063913 | 35.22% |
| multisyllable-1000 | 0 | 0 | 0 | 1.594315 | 1.106479 | 34.33% |
| multisyllable-1000 | 300 | 4,915,200 | 19,886,409 | 1.180214 | 1.072604 | 33.67% |
| multisyllable-1000 | 600 | 9,830,400 | 39,657,905 | 1.144851 | 1.074204 | 34.11% |
| multisyllable-1000 | 1200 | 19,660,800 | 79,230,093 | 1.109404 | 1.065679 | 35.33% |
| multisyllable-1000-s1 | 0 | 0 | 0 | 1.594312 | 1.106480 | 34.33% |
| multisyllable-1000-s1 | 300 | 4,915,200 | 19,886,409 | 1.169043 | 1.063662 | 35.33% |
| multisyllable-1000-s1 | 600 | 9,830,400 | 39,657,905 | 1.134850 | 1.066329 | 34.22% |
| multisyllable-1000-s1 | 1200 | 19,660,800 | 79,230,093 | 1.108361 | 1.065461 | 35.33% |

## Common-session generation

20 paired prompts; 256 greedy new tokens; fp16; two Tesla T4 GPUs.

| Model | Decode chars/s | Ratio vs base | Prefill chars/s | Repetition | Peak GiB |
|---|---:|---:|---:|---:|---:|
| base | 71.159 | 1.0000 | 7918.752 | 66.76% | 3.449 |
| multisyllable-1000 | 79.136 | 1.1121 | 8107.866 | 79.08% | 3.449 |
| multisyllable-1000-before | 72.285 | 1.0158 | 7843.946 | 71.30% | 3.449 |
| multisyllable-1000-s1 | 79.080 | 1.1113 | 8029.422 | 72.77% | 3.449 |
| original | 75.932 | 1.0671 | 7972.040 | 60.44% | 3.449 |
| syllable-1000 | 77.311 | 1.0865 | 7774.054 | 68.50% | 3.450 |
| syllable-1000-before | 78.088 | 1.0974 | 7887.778 | 68.68% | 3.450 |

Repetition is the mean share of repeated 4-syllable windows in each generated continuation. Text differs across models; these descriptive rates do not measure the throughput of equally useful text.
Base/syllable use GPU 0 and multisyllable uses GPU 1; physical-device variation is not separated.
Notebook wall time: 1127.766 seconds, excluding platform startup/teardown.

## Paired downstream check

| Condition vs base | A only correct | Base only correct | Exact McNemar p |
|---|---:|---:|---:|
| syllable-1000 | 34 | 32 | 0.902159 |
| multisyllable-1000 | 37 | 34 | 0.812589 |
| multisyllable-1000-s1 | 40 | 37 | 0.819893 |

Intervals resample documents and do not cover full training-seed uncertainty. Two multisyllable seeds and one base seed were run. The canonical-versus-marginal text likelihood gap is unmeasured.
All conditions processed 19,660,800 tokens. fp16 scaling skipped updates on non-finite gradients; losses and saved tensors are finite. No clear Belebele improvement was established.

| Condition | Processed steps | Successful updates | Characters read |
|---|---:|---:|---:|
| base | 1,200 | 1,200 | 67,094,352 |
| syllable-1000 | 1,200 | 1,200 | 72,456,286 |
| multisyllable-1000 | 1,200 | 1,199 | 79,230,093 |
| multisyllable-1000-s1 | 1,200 | 1,198 | 79,230,093 |

## Retained tokenizer survey

Tokenizer-only measurements on 2,000 test and 5,000 validation documents; no LLM quality comparison. Recovered from the original command with tokenizer revisions resolved at its timestamp. File sizes, vocabulary sizes, rounded results and 200 round trips match the historical transcript.

| Tokenizer | Vocabulary | Test chars/token | Validation chars/token |
|---|---:|---:|---:|
| HuggingFaceTB/SmolLM2-360M | 49,152 | 1.303 | 1.302 |
| mistralai/Mistral-7B-v0.1 | 32,000 | 1.573 | 1.572 |
| Qwen/Qwen3-0.6B | 151,669 | 3.420 | 3.434 |
| Qwen/Qwen2.5-0.5B | 151,665 | 3.420 | 3.434 |
| sail/Sailor2-1B | 151,665 | 3.420 | 3.434 |
| SeaLLMs/SeaLLMs-v3-7B-Chat | 151,646 | 3.420 | 3.434 |
| unsloth/Llama-3.2-1B | 128,256 | 3.568 | 3.587 |
| unsloth/gemma-3-1b-pt | 262,145 | 3.577 | 3.590 |
| SeaLLMs/SeaLLM-7B-v2.5 | 256,000 | 3.578 | 3.593 |
| vilm/vinallama-7b | 46,303 | 3.829 | 3.847 |
| vinai/PhoGPT-4B | 20,480 | 3.889 | 3.906 |
| nanovitok/bpe-nfc-16k | 16,009 | 3.846 | 3.869 |
| nanovitok/super-nfc-16k | 16,009 | 4.686 | 4.743 |
