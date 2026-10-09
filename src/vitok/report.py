"""Regenerate the retrofit report from committed, audited item-level measurements.

    python -m vitok.report --data results/adaptation --out results/adaptation.md --figures figures
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from vitok import compare, stats

CONDITIONS = ("base", "syllable-1000", "multisyllable-1000", "multisyllable-1000-s1")
STAGES = ("score-before", "score-step_000300", "score-step_000600", "score-final")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(folder: Path, part: str, count: int) -> list[dict]:
    rows = [json.loads(line) for line in (folder / f"{part}.jsonl").read_text().splitlines() if line]
    if [r["i"] for r in rows] != list(range(count)):
        raise ValueError(f"{folder}/{part}: incomplete or repeated items")
    return rows


def _generation(rows: list[dict]) -> dict:
    for row in rows:
        if (row["new_tokens"] != 256 or row["new_chars"] != len(row["text"]) or
                not 0 <= row["repetition_4"] <= 1 or
                any(not math.isfinite(row[k]) or row[k] <= 0 for k in
                    ("decode_seconds", "prefill_seconds", "max_memory_gib"))):
            raise ValueError("invalid generation record")
    return {
        "decode_chars_per_second": sum(r["new_chars"] for r in rows) / sum(r["decode_seconds"] for r in rows),
        "decode_tokens_per_second": sum(r["new_tokens"] - 1 for r in rows) / sum(r["decode_seconds"] for r in rows),
        "prefill_chars_per_second": sum(r["prompt_chars"] for r in rows) / sum(r["prefill_seconds"] for r in rows),
        "generated_chars_per_token": sum(r["new_chars"] for r in rows) / sum(r["new_tokens"] for r in rows),
        "repetition_4_mean": sum(r["repetition_4"] for r in rows) / len(rows),
        "max_memory_gib": max(r["max_memory_gib"] for r in rows),
    }


def _training(folder: Path) -> list[dict]:
    rows = [json.loads(line) for line in (folder / "train_log.jsonl").read_text().splitlines() if line]
    steps = [r["step"] for r in rows]
    if steps != sorted(set(steps)) or not set(range(4, 1200, 5)).issubset(steps) or steps[-1] != 1199:
        raise ValueError("incomplete training")
    if any(r["tokens"] != (r["step"] + 1) * 32 * 512 for r in rows):
        raise ValueError("training budget differs")
    if any(not math.isfinite(r["loss"]) for r in rows):
        raise ValueError("non-finite training loss")
    return rows


def analyze(data: Path) -> dict:
    manifest = _read(data / "manifest.json")
    if (manifest["quality_margin"], manifest["bootstrap_resamples"], manifest["bootstrap_seed"]) != (0.01, 10000, 0):
        raise ValueError("report protocol differs from the fixed experiment")
    speed_names = set(CONDITIONS) | {"original", "syllable-1000-before", "multisyllable-1000-before"}
    required = {f"quality/{name}/{stage}/{filename}" for name in CONDITIONS for stage in STAGES
                for filename in ("summary.json", "vi_docs.jsonl", "en_docs.jsonl", "belebele.jsonl")}
    required |= {f"training/{name}/train_log.jsonl" for name in CONDITIONS}
    required |= {f"speed/{name}/{filename}" for name in speed_names
                 for filename in ("summary.json", "generation.jsonl")}
    required |= {"speed/session-manifest.json", "speed/session-results.json", "tokenizers/survey.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("measurement manifest is incomplete")
    for relative, record in manifest["files"].items():
        blob = (data / relative).read_bytes()
        if hashlib.sha256(blob).hexdigest() != record["sha256"] or len(blob) != record["bytes"]:
            raise ValueError(f"measurement checksum differs: {relative}")
    quality: dict[str, dict] = {}
    raw: dict[str, dict] = {}
    training = {}
    for name in CONDITIONS:
        quality[name], raw[name] = {}, {}
        for stage in STAGES:
            folder = data / "quality" / name / stage
            summary = _read(folder / "summary.json")
            rows = {part: _rows(folder, part, count) for part, count in
                    (("vi_docs", 2000), ("en_docs", 2000), ("belebele", 900))}
            for part, key in (("vi_docs", "vi"), ("en_docs", "en")):
                docs = rows[part]
                if any(not math.isfinite(d["nats"]) or d["nats"] < 0 or d["chars"] <= 0 or d["tokens"] <= 0
                       for d in docs):
                    raise ValueError("invalid document measurement")
                measured = stats.bpc([d["nats"] for d in docs], [d["chars"] for d in docs])
                chars, tokens = sum(d["chars"] for d in docs), sum(d["tokens"] for d in docs)
                if (not math.isclose(summary[key]["bpc"], measured, rel_tol=1e-12) or
                        (summary[key]["docs"], summary[key]["chars"], summary[key]["tokens"]) !=
                        (len(docs), chars, tokens) or
                        not math.isclose(summary[key]["chars_per_token"], chars / tokens, rel_tol=1e-12)):
                    raise ValueError("document aggregate differs")
            for metric in ("acc_char", "acc_sum"):
                correct = [r["correct_" + metric.removeprefix("acc_")] for r in rows["belebele"]]
                if (summary["belebele"]["items"] != len(correct) or
                        not math.isclose(summary["belebele"][metric], sum(correct) / len(correct), rel_tol=1e-12)):
                    raise ValueError("downstream aggregate differs")
            quality[name][stage], raw[name][stage] = summary, rows
        training[name] = _training(data / "training" / name)
    reference = raw["base"]["score-before"]
    for stages in raw.values():
        for rows in stages.values():
            for part in ("vi_docs", "en_docs"):
                if [(r["digest"], r["chars"]) for r in rows[part]] != [
                        (r["digest"], r["chars"]) for r in reference[part]]:
                    raise ValueError("the scores are on different documents")
    comparisons = {}
    for name in CONDITIONS[1:]:
        a, b = raw[name]["score-final"], raw["base"]["score-final"]
        comparisons[name] = {
            "vi": compare.compare_docs(a["vi_docs"], b["vi_docs"]),
            "en": compare.compare_docs(a["en_docs"], b["en_docs"]),
            "belebele": compare.compare_belebele(a["belebele"], b["belebele"]),
        }
    speed_root = data / "speed"
    session, result = _read(speed_root / "session-manifest.json"), _read(speed_root / "session-results.json")
    expected = set(CONDITIONS) | {"original", "syllable-1000-before", "multisyllable-1000-before"}
    if (result["status"] != "done" or result["session_id"] != session["session_id"] or
            set(result["models"]) != expected or set(session["jobs"]) != expected or
            (session["dtype"], session["speed_prompts"], session["prompt_chars"], session["new_tokens"]) !=
            ("fp16", 20, 500, 256)):
        raise ValueError("incomplete or mixed speed session")
    speed, prompt_lengths = {}, None
    for name in sorted(expected):
        job = result["models"][name]
        summary = _read(speed_root / name / "summary.json")
        signature = summary["signature"]
        if (job["status"] != "done" or job["exit_code"] != 0 or job["session_id"] != session["session_id"] or
                summary["device_name"] != "Tesla T4" or signature["vi_docs_sha256"] != session["prompt_sha256"] or
                signature["snapshot_sha256"] != session["jobs"][name].get("snapshot_sha256")):
            raise ValueError("speed job provenance differs")
        generations = _rows(speed_root / name, "generation", 20)
        lengths = [r["prompt_chars"] for r in generations]
        if prompt_lengths is not None and lengths != prompt_lengths:
            raise ValueError("speed prompt lengths differ")
        prompt_lengths = lengths
        speed[name] = _generation(generations)
        for key, value in speed[name].items():
            if not math.isclose(summary["generation"][key], value, rel_tol=1e-12):
                raise ValueError("generation aggregate differs")
    survey = _read(data / "tokenizers" / "survey.json")
    if len(survey) != 13 or len({row["name"] for row in survey}) != 13:
        raise ValueError("incomplete tokenizer survey")
    for row in survey:
        for part in ("test", "val"):
            measured = row["results"][part]
            chars, tokens = measured["chars"], measured["tokens"]
            if (len(chars) != row["inputs"][part]["documents"] or len(tokens) != len(chars) or
                    any(not isinstance(n, int) or n <= 0 for n in chars + tokens) or
                    not math.isclose(measured["chars_per_token"], sum(chars) / sum(tokens), rel_tol=1e-12)):
                raise ValueError("invalid tokenizer survey aggregate")
            if part == "test" and chars != [r["chars"] for r in reference["vi_docs"]]:
                raise ValueError("survey and quality test documents differ")
    return {"manifest": manifest, "quality": quality, "comparisons": comparisons, "speed": speed, "survey": survey,
            "training": training, "session": session, "wall_seconds": result["wall_seconds"]}


def render(result: dict) -> str:
    lines = ["# Retrofit results", "", "Canonical-tokenization bpc; lower is better. Paired document bootstrap: "
             "10,000 resamples, seed 0. The fixed criterion requires the upper relative 95% bound below +1%.", "",
             "## Final quality", "", "| Condition | VI bpc | Difference vs base [95% interval] | Within 1% | "
             "EN bpc | Belebele | Token saving |",
             "|---|---:|---|---|---:|---:|---:|"]
    base = result["quality"]["base"]["score-final"]
    for name in CONDITIONS:
        summary = result["quality"][name]["score-final"]
        paired = result["comparisons"].get(name)
        difference, passed = "reference", "reference"
        if paired:
            vi = paired["vi"]
            difference = f"{vi['rel_diff']:+.3%} [{vi['rel_ci95'][0]:+.3%}, {vi['rel_ci95'][1]:+.3%}]"
            passed = "yes" if vi["rel_ci95"][1] < 0.01 else "no"
        saving = 1 - base["vi"]["chars_per_token"] / summary["vi"]["chars_per_token"]
        lines.append(f"| {name} | {summary['vi']['bpc']:.6f} | {difference} | {passed} | "
                     f"{summary['en']['bpc']:.6f} | {summary['belebele']['acc_char']:.2%} | {saving:.2%} |")
    lines += ["", "## Recovery", "", "| Condition | Step | Processed tokens | Characters read | "
              "VI bpc | EN bpc | Belebele |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in CONDITIONS:
        for stage in STAGES:
            summary = result["quality"][name][stage]
            snapshot = summary["snapshot"] or {"step": 0, "tokens": 0, "chars": 0}
            lines.append(f"| {name} | {snapshot['step']} | {snapshot['tokens']:,} | {snapshot['chars']:,} | "
                         f"{summary['vi']['bpc']:.6f} | {summary['en']['bpc']:.6f} | "
                         f"{summary['belebele']['acc_char']:.2%} |")
    lines += ["", "## Common-session generation", "",
              "20 paired prompts; 256 greedy new tokens; fp16; two Tesla T4 GPUs.", "",
              "| Model | Decode chars/s | Ratio vs base | Prefill chars/s | Repetition | Peak GiB |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, generation in result["speed"].items():
        ratio = generation["decode_chars_per_second"] / result["speed"]["base"]["decode_chars_per_second"]
        lines.append(f"| {name} | {generation['decode_chars_per_second']:.3f} | {ratio:.4f} | "
                     f"{generation['prefill_chars_per_second']:.3f} | {generation['repetition_4_mean']:.2%} | "
                     f"{generation['max_memory_gib']:.3f} |")
    lines += ["", "Repetition is the mean share of repeated 4-syllable windows in each generated continuation. "
              "Text differs across models; these descriptive rates do not measure the throughput "
              "of equally useful text.",
              "Base/syllable use GPU 0 and multisyllable uses GPU 1; physical-device variation is not separated.",
              f"Notebook wall time: {result['wall_seconds']:.3f} seconds, excluding platform startup/teardown.", "",
              "## Paired downstream check", "",
              "| Condition vs base | A only correct | Base only correct | Exact McNemar p |",
              "|---|---:|---:|---:|"]
    for name, paired in result["comparisons"].items():
        m = paired["belebele"]["mcnemar_char"]
        lines.append(f"| {name} | {m['a_only']} | {m['b_only']} | {m['p_value']:.6f} |")
    lines += ["", "Intervals resample documents and do not cover full training-seed uncertainty. Two multisyllable "
              "seeds and one base seed were run. The canonical-versus-marginal text likelihood gap is unmeasured.",
              "All conditions processed 19,660,800 tokens. fp16 scaling skipped updates on non-finite gradients; "
              "losses and saved tensors are finite. No clear Belebele improvement was established.", "",
              "| Condition | Processed steps | Successful updates | Characters read |",
              "|---|---:|---:|---:|"]
    for name, log in result["training"].items():
        steps = log[-1]["step"] + 1
        updates = steps - sum(not r["finite"] for r in log)
        lines.append(f"| {name} | {steps:,} | {updates:,} | {log[-1]['chars']:,} |")
    lines += ["", "## Retained tokenizer survey", "",
              "Tokenizer-only measurements on 2,000 test and 5,000 validation documents; no LLM quality comparison. "
              "Recovered from the original command with tokenizer revisions resolved at its timestamp. "
              "File sizes, vocabulary sizes, rounded results and 200 round trips match the historical transcript.", "",
              "| Tokenizer | Vocabulary | Test chars/token | Validation chars/token |",
              "|---|---:|---:|---:|"]
    for row in result["survey"]:
        lines.append(f"| {row['name']} | {row['vocab']:,} | {row['results']['test']['chars_per_token']:.3f} | "
                     f"{row['results']['val']['chars_per_token']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def plot_recovery(result: dict, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, stages in result["quality"].items():
        for ax, key in zip(axes, ("tokens", "chars"), strict=True):
            xs = [(stages[stage]["snapshot"] or {"tokens": 0, "chars": 0})[key] / 1e6 for stage in STAGES]
            ys = [stages[stage]["vi"]["bpc"] for stage in STAGES]
            ax.plot(xs, ys, marker="o", label=name)
    for ax, label in zip(axes, ("Training tokens (millions)", "Training characters (millions)"), strict=True):
        ax.set_xlabel(label)
        ax.set_ylabel("Vietnamese bpc (lower is better)")
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.savefig(folder / "recovery.png", dpi=180, metadata={"Software": "nanovitok"})
    fig.savefig(folder / "recovery.pdf")
    plt.close(fig)


def plot_measurements(result: dict, folder: Path) -> None:
    """Plot verified aggregates; quality intervals resample paired documents only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    names = list(CONDITIONS)
    colors = ["#6B7F4E", "#498AF2", "#A19654", "#9370A8"]
    final = [result["quality"][name]["score-final"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    saving = [100 * (1 - final[0]["vi"]["chars_per_token"] / row["vi"]["chars_per_token"]) for row in final]
    axes[0].barh(names, saving, color=colors)
    axes[0].set_xlabel("Test tokens saved vs base (%)")
    axes[0].set_xlim(0, 20)
    for index, value in enumerate(saving):
        axes[0].text(value + 0.3, index, f"{value:.2f}%", va="center")
    for index, name in enumerate(names[1:], start=1):
        paired = result["comparisons"][name]["vi"]
        value = 100 * paired["rel_diff"]
        lo, hi = [100 * bound for bound in paired["rel_ci95"]]
        axes[1].errorbar(value, index, xerr=[[value - lo], [hi - value]],
                         fmt="o", capsize=5, color=colors[index])
    axes[1].plot(0, 0, "o", color=colors[0])
    axes[1].set_yticks(range(len(names)), names)
    axes[1].axvline(1, color="#B54040", linestyle="--", label="Fixed +1% margin")
    axes[1].set_xlabel("VI bpc increase vs base (%) — paired 95% interval")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Token savings and quality cost after adaptation")
    fig.savefig(folder / "quality_cost.png", dpi=180, metadata={"Software": "nanovitok"})
    plt.close(fig)

    speed_names = ["original", "base", "syllable-1000-before", "syllable-1000",
                   "multisyllable-1000-before", "multisyllable-1000", "multisyllable-1000-s1"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    metrics = (("decode_chars_per_second", "Decode (characters/s)", 1),
               ("repetition_4_mean", "Repeated 4-syllable windows (%)", 100))
    for ax, (key, label, scale) in zip(axes.flat, metrics, strict=True):
        values = [scale * result["speed"][name][key] for name in speed_names]
        ax.barh(speed_names, values, color="#498AF2")
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_xlim(0, max(values) * 1.2)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        for index, value in enumerate(values):
            ax.text(value + max(values) * 0.015, index, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle("Common-session Tesla T4 generation — descriptive rates, different generated text")
    fig.savefig(folder / "generation.png", dpi=180, metadata={"Software": "nanovitok"})
    plt.close(fig)

def render_readme(result: dict) -> str:
    full = render(result)
    generation_notes = "Repetition is" + full.split("Repetition is", 1)[1].split("\n## Paired downstream check", 1)[0]
    transfer = ["| Condition | EN bpc | Belebele |", "|---|---:|---:|"]
    for name in CONDITIONS:
        final = result["quality"][name]["score-final"]
        transfer.append(f"| {name} | {final['en']['bpc']:.6f} | {final['belebele']['acc_char']:.2%} |")
    generation = ["| Model | Prefill chars/s | Peak GiB |", "|---|---:|---:|"]
    for name, measurement in result["speed"].items():
        generation.append(f"| {name} | {measurement['prefill_chars_per_second']:.3f} | "
                          f"{measurement['max_memory_gib']:.3f} |")
    transfer_table, generation_table = "\n".join(transfer), "\n".join(generation)
    return f'''<div align="center">

# nanovitok

[![CI](https://img.shields.io/github/actions/workflow/status/geminitt/nanovitok/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/geminitt/nanovitok/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/PYTHON-3.12-498AF2?style=for-the-badge)](./pixi.toml)
[![Kaggle](https://img.shields.io/badge/GPU-Kaggle_T4×2-A19654?style=for-the-badge)](https://www.kaggle.com/code)
[![License](https://img.shields.io/badge/LICENSE-MIT-6B7F4E?style=for-the-badge)](./LICENSE)

**Adapting an LLM with Vietnamese Multi-Syllable Tokens**

</div>

---

## Question

Does adding Vietnamese multi-syllable tokens to an existing LLM reduce processing cost while keeping quality,
under a small adaptation budget?

We extend Qwen3-0.6B's tokenizer by continuing its BPE merges. The main extension may join syllables across spaces
(for example, học sinh); a syllable-only extension is the control. Existing token IDs stay fixed. Each new token
embedding is initialized to the mean of the original embeddings of its pieces; input and output weights are tied.

**Result.** Multi-syllable tokens save more test tokens and show higher descriptive decode characters/s, but both
seeds fail the quality margin fixed before the runs. The syllable-only extension saves fewer tokens and meets that
margin. Pronounced repetition in greedy continuations limits what the measured generation rates establish.

---

## Design

| Item | Fixed protocol |
|---|---|
| Source model | [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), revision {result['manifest']['revision']} |
| Conditions | base: unchanged tokenizer; syllable: within syllables; multisyllable: may cross spaces |
| Vocabulary addition | 1,000 tokens, chosen by a development probe before the full runs |
| Trainable parameters | New embedding rows plus transformer layers 0, 1, 26 and 27; existing token rows frozen |
| Budget | 1,200 steps × 32 sequences × 512 tokens = 19,660,800 processed tokens per condition |
| Data order | Same ordered FineWeb-2 Vietnamese stream, deduplicated against test and merge-learning documents |
| Quality | Canonical-tokenization bits per NFC character on the same 2,000 Vietnamese test documents |
| Checks | 2,000 English documents; 900 Belebele vie_Latn items, zero-shot per-character answer scoring |
| Decision rule | Relative VI bpc vs base: upper 95% bound below +1%; paired bootstrap, 10,000 resamples, seed 0 |

The protocol matches processed tokens and steps. Multi-syllable tokens cover more text per step; exact training
FLOPs are unmeasured. One base seed and two multi-syllable seeds were run. Intervals resample documents and do not
cover full training-seed uncertainty.

---

## Results

![Token savings and paired quality intervals](./figures/quality_cost.png)

The dashed line is the fixed +1% quality margin. Error bars resample paired test documents, not training seeds.

Bits per character is the total negative log probability of the reference text's canonical token sequence,
divided by its NFC character count and ln(2). Documents are scored given an end-of-text context token; spaces and
punctuation count as characters. This evaluates fixed held-out text. Likelihood marginalized over every possible
tokenization of the same text is unmeasured.

![Recovery against training tokens and characters](./figures/recovery.png)

Adding tokens initially increases bpc; continued training recovers much of that cost. Continued training also
improves the unchanged-tokenizer baseline. The full [generated report](./results/adaptation.md) includes original,
zero-shot, snapshot and final scores, the English check and paired Belebele comparisons. Belebele does not show a
clear improvement over base under this protocol.

### Transfer checks

{transfer_table}

These are descriptive final scores; the report gives the paired comparisons. Small accuracy differences do not
establish a downstream improvement.

### Generation

20 paired prompts; 256 greedy new tokens; fp16; two Tesla T4 GPUs.

![Decode speed and repetition measurements](./figures/generation.png)

{generation_notes}

{generation_table}

Rates describe different generated text, not equally useful output. Memory is peak PyTorch allocated memory,
not total GPU usage; base/syllable and multisyllable were measured on different physical T4s in the same session.

### Training limitation

fp16 scaling skipped updates on non-finite gradients. All saved tensors and logged losses are finite. The
processed-token budgets match; successful optimizer-update counts differ, as recorded in the generated report.
No further training was used to change the fixed comparison after seeing the results.

---

## Reproduce

The [measurement bundle](./results/adaptation/manifest.json) records commits, model revision, source kernels and
sha256 hashes of the committed per-document/per-prompt measurements. Final weights remain in the Kaggle outputs.
The analysis is deterministic; CI regenerates both reports and this README and checks for any difference.

```bash
pixi install
pixi run python -m vitok.report --data results/adaptation --out results/adaptation.md \\
    --readme README.md --figures figures
```

The [retained tokenizer survey](./results/adaptation.md#retained-tokenizer-survey) is a tokenizer-only comparison.
Its original command was recovered; the pinned remeasurement matches the historical rounded values. To remeasure
it from tokenizer files and the committed test/validation documents (downloads tokenizer files, not model weights):

```bash
pixi run python -m vitok.survey --recorded results/adaptation/tokenizers/survey.json \\
    --docs-root kaggle/outputs/vitok-data --out /tmp/tokenizer-survey.json
```

To reproduce the GPU experiment, use [notebook 03](./kaggle/notebooks/03_adaptation.ipynb) for the development probe
and full adaptation queues, then [notebook 04](./kaggle/notebooks/04_generation_speed.ipynb) for every generation
measurement in one T4 session. Attach the vitok-code/vitok-data bundles and both adaptation outputs as described
in the notebooks. The input bundle is built by vitok.kaggle_session.build_bundle from a committed checkout,
retrofit tokenizer outputs and the pinned held-out files.

The main CLIs are vitok.retrofit, vitok.extend, vitok.cpt_data, vitok.adapt, vitok.heldout, vitok.score,
vitok.compare and vitok.report. Training resumes from checkpoints; scoring resumes item by item under a matching
signature. Default/dev environments run the CPU suite; the gpu environment provides CUDA PyTorch.

## NFC pilot evidence

The retained [from-scratch NFC pilot](./results/summary.md) compares BPE and SuperBPE under its earlier equal-text
design, including second seeds, model-size trends and the val-shard check. Its models and budget differ from the
Qwen3 retrofit; it serves as tokenization-only evidence. nanochat's fixed-token validation bpb scores different
documents between tokenizers, so the paired per-document bpc is used for conclusions.

![Retained NFC pilot quality and model-size effects](./figures/h3_scaling.png)

```bash
git clone https://github.com/karpathy/nanochat third_party/nanochat
git -C third_party/nanochat checkout $(cat patches/NANOCHAT_COMMIT)
git -C third_party/nanochat apply ../../patches/nanochat.patch
pixi run python -m vitok.analysis --results kaggle/outputs \\
    --compression kaggle/outputs/vitok-data/compression-16k.json \\
    --val-results results/val --out results/summary.md --figures figures
pixi run -e dev check
```

The dropped NFD, stripped-text, minimal-pairs, wordhood and FLORES branches remain available in Git history at
[f2bb633](https://github.com/geminitt/nanovitok/tree/f2bb633); they are outside the current research question.

---

## Sources and license

- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B): source model and tokenizer.
- [SuperBPE](https://arxiv.org/abs/2503.13423): cross-whitespace BPE merges; [reference implementation](https://github.com/PythonNut/superbpe).
- [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2): Vietnamese corpus, ODC-By 1.0.
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb): pinned sample-10BT English documents.
- [Belebele](https://huggingface.co/datasets/facebook/belebele): downstream multilingual reading-comprehension check.
- [nanochat](https://github.com/karpathy/nanochat): the from-scratch pilot, pinned and patched under patches/.

Code is [MIT licensed](./LICENSE). Corpus/model assets retain their respective licenses.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args()
    result = analyze(args.data)
    args.out.write_text(render(result), encoding="utf-8")
    if args.figures is not None:
        plot_recovery(result, args.figures)
        plot_measurements(result, args.figures)
    if args.readme is not None:
        args.readme.write_text(render_readme(result), encoding="utf-8")


if __name__ == "__main__":
    main()
