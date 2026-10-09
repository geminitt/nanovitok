import hashlib
import json
import math
import shutil
import sys

import pytest
from conftest import REPO

from vitok import report

DATA = REPO / "results" / "adaptation"


def test_report_checks_and_reproduces_the_measured_results(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    savefig = Figure.savefig

    def save_and_check_initial_coordinates(figure, path, **kwargs):
        if str(path).endswith("recovery.png"):
            assert len(figure.axes) == 2
            for axis in figure.axes:
                assert len(axis.lines) == 4
                assert all(line.get_xdata()[0] == 0 for line in axis.lines)
        return savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save_and_check_initial_coordinates)
    out = tmp_path / "adaptation.md"
    monkeypatch.setattr(sys, "argv", ["report", "--data", str(DATA), "--out", str(out),
                                    "--figures", str(tmp_path / "figures"), "--readme", str(tmp_path / "README.md")])
    report.main()
    assert out.read_text() == (REPO / "results" / "adaptation.md").read_text()
    assert (tmp_path / "figures" / "recovery.png").stat().st_size > 0
    assert (tmp_path / "figures" / "recovery.pdf").stat().st_size > 0
    for name in ("recovery", "quality_cost", "generation"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 0
        assert (tmp_path / "figures" / f"{name}.png").read_bytes() == (REPO / "figures" / f"{name}.png").read_bytes()
        assert f"./figures/{name}.png" in (tmp_path / "README.md").read_text()
    assert (tmp_path / "README.md").read_text() == (REPO / "README.md").read_text()
    readme = (tmp_path / "README.md").read_text()
    assert "| Condition | VI bpc |" not in readme
    assert "| Model | Decode chars/s |" not in readme
    assert "| Condition | EN bpc | Belebele |" in readme
    assert "| Model | Prefill chars/s | Peak GiB |" in readme
    assert "./figures/transfer.png" not in readme


def _rewrite(data, relative, value, jsonl=False):
    path = data / relative
    text = "".join(json.dumps(r) + "\n" for r in value) if jsonl else json.dumps(value)
    path.write_text(text)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["files"][relative] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size,
    }
    (data / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("kind,message", [
    ("checksum", "checksum differs"), ("partial documents", "incomplete or repeated"),
    ("pairing", "different documents"), ("summary", "document aggregate"),
    ("session", "mixed speed session"), ("generation", "invalid generation record"),
    ("partial training", "incomplete training"), ("budget", "training budget"),
    ("token saving", "document aggregate"), ("downstream summary", "downstream aggregate"),
    ("survey", "survey aggregate"),
    ("manifest coverage", "manifest is incomplete"), ("protocol", "protocol differs"),
    ("checksum size", "checksum differs"), ("provenance", "provenance differs"),
    ("prompt lengths", "prompt lengths differ"), ("generation summary", "generation aggregate differs"),
])
def test_report_refuses_incomplete_inconsistent_or_unpaired_measurements(tmp_path, kind, message):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    if kind == "checksum":
        (data / "quality/base/score-final/summary.json").write_text("{}")
    elif kind in {"manifest coverage", "protocol", "checksum size"}:
        path = data / "manifest.json"
        manifest = json.loads(path.read_text())
        if kind == "manifest coverage":
            del manifest["files"]["quality/base/score-final/summary.json"]
        elif kind == "protocol":
            manifest["quality_margin"] = 0.02
        else:
            manifest["files"]["quality/base/score-final/summary.json"]["bytes"] += 1
        path.write_text(json.dumps(manifest))
    elif kind in {"partial documents", "pairing"}:
        relative = "quality/syllable-1000/score-final/vi_docs.jsonl"
        rows = [json.loads(line) for line in (data / relative).read_text().splitlines()]
        if kind == "pairing":
            rows[0]["digest"] = "wrong document"
        else:
            rows.pop()
        _rewrite(data, relative, rows, jsonl=True)
    elif kind in {"summary", "token saving", "downstream summary"}:
        relative = "quality/base/score-final/summary.json"
        summary = json.loads((data / relative).read_text())
        if kind == "summary":
            summary["vi"]["bpc"] += 0.1
        elif kind == "token saving":
            summary["vi"]["chars_per_token"] += 0.1
        else:
            summary["belebele"]["acc_char"] += 0.1
        _rewrite(data, relative, summary)
    elif kind == "survey":
        relative = "tokenizers/survey.json"
        rows = json.loads((data / relative).read_text())
        rows[0]["results"]["test"]["tokens"][0] += 1
        _rewrite(data, relative, rows)
    elif kind == "session":
        relative = "speed/session-results.json"
        session = json.loads((data / relative).read_text())
        session["session_id"] = "another session"
        _rewrite(data, relative, session)
    elif kind in {"provenance", "generation summary"}:
        relative = "speed/base/summary.json"
        summary = json.loads((data / relative).read_text())
        if kind == "provenance":
            summary["device_name"] = "different hardware"
        else:
            summary["generation"]["decode_chars_per_second"] *= 1.1
        _rewrite(data, relative, summary)
    elif kind == "prompt lengths":
        relative = "speed/base/generation.jsonl"
        rows = [json.loads(line) for line in (data / relative).read_text().splitlines()]
        rows[0]["prompt_chars"] -= 1
        _rewrite(data, relative, rows, jsonl=True)
        relative = "speed/base/summary.json"
        summary = json.loads((data / relative).read_text())
        summary["generation"]["prefill_chars_per_second"] = (
            sum(r["prompt_chars"] for r in rows) / sum(r["prefill_seconds"] for r in rows))
        _rewrite(data, relative, summary)
    elif kind == "generation":
        relative = "speed/base/generation.jsonl"
        rows = [json.loads(line) for line in (data / relative).read_text().splitlines()]
        rows[0]["decode_seconds"] = 0
        _rewrite(data, relative, rows, jsonl=True)
    else:
        relative = "training/base/train_log.jsonl"
        rows = [json.loads(line) for line in (data / relative).read_text().splitlines()]
        if kind == "budget":
            rows[-1]["tokens"] -= 512
        else:
            rows.pop()
        _rewrite(data, relative, rows, jsonl=True)
    with pytest.raises((ValueError, SystemExit), match=message):
        report.analyze(data)


@pytest.mark.parametrize("repetition", [0, 1])
def test_generation_aggregates_require_valid_counts_ranges_and_finite_timings(repetition):
    row = {"new_tokens": 256, "text": "a" * 256, "new_chars": 256, "prompt_chars": 500,
           "decode_seconds": 2, "prefill_seconds": 4, "max_memory_gib": 3, "repetition_4": repetition}
    measured = report._generation([row])
    assert measured["decode_chars_per_second"] == 128 and measured["prefill_chars_per_second"] == 125
    assert measured["repetition_4_mean"] == repetition
    for key, value in (("new_tokens", 255), ("new_chars", 255), ("repetition_4", -0.1),
                       ("repetition_4", 1.1), ("decode_seconds", 0), ("prefill_seconds", math.nan),
                       ("max_memory_gib", math.inf)):
        with pytest.raises(ValueError, match="invalid generation record"):
            report._generation([{**row, key: value}])


@pytest.mark.parametrize("damage", ["middle gap", "duplicate step", "order", "nonfinite loss"])
def test_training_evidence_requires_complete_ordered_finite_logs(tmp_path, damage):
    rows = [json.loads(line) for line in (DATA / "training/base/train_log.jsonl").read_text().splitlines()]
    if damage == "middle gap":
        rows.pop(len(rows) // 2)
    elif damage == "duplicate step":
        rows.insert(0, rows[0])
    elif damage == "order":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[0]["loss"] = math.nan
    (tmp_path / "train_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="incomplete training|non-finite training loss"):
        report._training(tmp_path)


@pytest.mark.parametrize("path", ["quality/base/score-before/vi_docs.jsonl", "training/base/train_log.jsonl",
                                  "speed/session-manifest.json", "speed/base/generation.jsonl",
                                  "tokenizers/survey.json"])
def test_every_consumed_measurement_requires_a_manifest_checksum(tmp_path, path):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    manifest = json.loads((data / "manifest.json").read_text())
    del manifest["files"][path]
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest is incomplete"):
        report.analyze(data)


def test_report_quality_decision_uses_the_strict_prespecified_margin():
    result = report.analyze(DATA)
    vi = result["comparisons"]["syllable-1000"]["vi"]
    vi["rel_ci95"] = [vi["rel_ci95"][0], 0.01]
    line = next(line for line in report.render(result).splitlines() if line.startswith("| syllable-1000 |"))
    assert line.split("|")[4].strip() == "no"


@pytest.mark.parametrize("field,value", [("nats", -1), ("nats", math.nan), ("chars", 0), ("tokens", 0)])
def test_document_measurements_require_finite_nonnegative_loss_and_positive_counts(tmp_path, field, value):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    path = "quality/base/score-final/vi_docs.jsonl"
    rows = [json.loads(line) for line in (data / path).read_text().splitlines()]
    rows[0][field] = value
    _rewrite(data, path, rows, jsonl=True)
    with pytest.raises(ValueError, match="invalid document measurement"):
        report.analyze(data)


@pytest.mark.parametrize("field,value", [("nats", 0.0), ("tokens", 1)])
def test_valid_measurement_boundaries_are_not_rejected(tmp_path, field, value):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    path = "quality/base/score-final/vi_docs.jsonl"
    rows = [json.loads(line) for line in (data / path).read_text().splitlines()]
    rows[0][field] = value
    _rewrite(data, path, rows, jsonl=True)
    path = "quality/base/score-final/summary.json"
    summary = json.loads((data / path).read_text())
    summary["vi"].update(tokens=sum(row["tokens"] for row in rows),
                         bpc=report.stats.bpc([row["nats"] for row in rows], [row["chars"] for row in rows]),
                         chars_per_token=sum(row["chars"] for row in rows) / sum(row["tokens"] for row in rows))
    _rewrite(data, path, summary)
    result = report.analyze(data)
    assert result["quality"]["base"]["score-final"] == summary
    assert set(result["comparisons"]["syllable-1000"]) == {"vi", "en", "belebele"}
    assert "session" in result


@pytest.mark.parametrize("damage", ["status", "exit", "session", "prompt", "snapshot", "missing model", "settings"])
def test_speed_provenance_requires_complete_successful_jobs_and_one_protocol(tmp_path, damage):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    path = "speed/session-results.json"
    rows = json.loads((data / path).read_text())
    if damage in {"status", "exit", "session"}:
        key, value = {"status": ("status", "failed"), "exit": ("exit_code", 1),
                      "session": ("session_id", "wrong session")}[damage]
        rows["models"]["base"][key] = value
    elif damage == "missing model":
        del rows["models"]["base"]
    elif damage == "settings":
        path = "speed/session-manifest.json"
        rows = json.loads((data / path).read_text())
        rows["dtype"] = "fp32"
    else:
        path = "speed/base/summary.json"
        rows = json.loads((data / path).read_text())
        key = "vi_docs_sha256" if damage == "prompt" else "snapshot_sha256"
        rows["signature"][key] = "wrong source"
    _rewrite(data, path, rows)
    with pytest.raises(ValueError, match="provenance differs|mixed speed session"):
        report.analyze(data)


@pytest.mark.parametrize("damage", ["duplicate", "count", "zero tokens", "noninteger", "different test"])
def test_survey_evidence_requires_unique_sources_complete_counts_and_the_same_test(tmp_path, damage):
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    path = "tokenizers/survey.json"
    rows = json.loads((data / path).read_text())
    if damage == "duplicate":
        rows[0]["name"] = rows[1]["name"]
    elif damage == "count":
        rows[0]["inputs"]["val"]["documents"] += 1
    else:
        measured = rows[0]["results"]["test"]
        key, value = {"zero tokens": ("tokens", 0), "noninteger": ("tokens", 1.5),
                      "different test": ("chars", measured["chars"][0] + 1)}[damage]
        measured[key][0] = value
        measured["chars_per_token"] = sum(measured["chars"]) / sum(measured["tokens"])
    _rewrite(data, path, rows)
    with pytest.raises(ValueError, match="incomplete tokenizer survey|survey aggregate|test documents differ"):
        report.analyze(data)


@pytest.mark.parametrize("missing", ["data", "out"])
def test_report_cli_requires_both_measurements_and_an_output(tmp_path, monkeypatch, missing):
    argv = ["report", "--data", str(DATA)] if missing == "out" else ["report", "--out", str(tmp_path / "report.md")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        report.main()
    assert error.value.code == 2
