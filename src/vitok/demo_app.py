"""Streamlit demo: the same prompt continued by two trained models side by side.

    pixi run -e gpu streamlit run src/vitok/demo_app.py -- --models-dir runs/token-type-models-2026-10-13

Race streams both models at once; they share the GPU, so its speeds only illustrate. Benchmark runs them one after
the other and gives the comparable numbers. The logic is in `vitok.demo`.
"""

import argparse
from pathlib import Path

import streamlit as st
import torch

from vitok import demo

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = ("syllable-1000", "multisyllable-1000")
PROMPT = "Thủ đô của Việt Nam là"  # Vietnamese example prompt (experimental input)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path, default=REPO / "runs" / "token-type-models-2026-10-13")
    return p.parse_args()


@st.cache_resource(show_spinner="Loading both models onto the device")
def load(models_dir: Path) -> dict[str, tuple]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return {c: demo.load_condition(models_dir / c / "extended", models_dir / f"{c}.json", models_dir / c / "final.pt",
                                   device, dtype) for c in CONDITIONS}


def shown(text: str) -> str:
    """A token can end inside a UTF-8 character; hide the replacement mark until the next token completes it."""
    return text.rstrip("�")


def table(results: dict[str, dict]) -> str:
    """A Markdown table (st.dataframe segfaulted in pandas 3.0.6 with pyarrow 25.0.0 after a race, so no pandas)."""
    keys = ("new_tokens", "new_chars", "chars_per_token", "chars_per_second", "tokens_per_second")
    rows = ["| model | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    rows += [f"| {name} | " + " | ".join(f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |"
             for name, r in results.items()]
    return "\n".join(rows)


def main():
    args = arguments()
    st.set_page_config(page_title="nanovitok demo", layout="wide")
    st.title("Vietnamese tokens: two models, one prompt")
    models = load(args.models_dir)
    device = next(iter(models.values()))[0].device
    hardware = torch.cuda.get_device_name() if device.type == "cuda" else "CPU"
    st.caption(f"Qwen3-0.6B after continued pretraining on 100 M tokens; greedy decoding, batch 1, "
               f"{next(iter(models.values()))[0].dtype}, on {hardware}.")
    prompt = st.text_area("Prompt", PROMPT)
    new_tokens = st.slider("New tokens", 16, 512, 128, step=16)
    race_col, bench_col = st.columns(2)
    race_clicked, bench_clicked = race_col.button("Race"), bench_col.button("Benchmark")
    columns = dict(zip(CONDITIONS, st.columns(2), strict=True))
    boxes = {}
    for name, col in columns.items():
        col.subheader(name)
        boxes[name] = (col.empty(), col.empty())

    if race_clicked and prompt.strip():
        counts = dict.fromkeys(CONDITIONS, 0)
        for kind, payload in demo.race_events(models, prompt, new_tokens):
            if kind == "update":
                name, text = payload
                counts[name] += 1
                boxes[name][0].markdown(shown(text))
                boxes[name][1].caption(f"{counts[name]} tokens, {len(text)} characters, "
                                       f"{len(text) / counts[name]:.2f} characters per token")
            else:
                st.warning("Race: both models shared the GPU, so the speeds below are not comparable. "
                           "Characters per token is exact. Press Benchmark for comparable speeds.")
                st.markdown(table({n: demo.summarize(r["new_tokens"], r["new_chars"], r["decode_seconds"])
                                   for n, r in payload.items()}))
    if bench_clicked and prompt.strip():
        with st.spinner("Generating with each model in turn"):
            results = demo.benchmark(models, prompt, new_tokens)
        for name, r in results.items():
            boxes[name][0].markdown(shown(r["text"]))
        st.success(f"Benchmark on {hardware}: one model at a time, warmed up, same prompt and length.")
        st.markdown(table(results))
        a, b = (results[c]["chars_per_second"] for c in CONDITIONS)
        st.caption(f"{CONDITIONS[1]} / {CONDITIONS[0]} characters per second: {b / a:.3f}. One prompt is an "
                   "illustration, not a measurement of quality; greedy text repeats often.")


main()
