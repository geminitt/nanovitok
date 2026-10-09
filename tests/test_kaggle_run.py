import re

from conftest import NANOCHAT, requires_nanochat
from vitok.conditions import train_args

FIXED_FLAGS = ["run", "model-tag", "core-metric-every", "sample-every", "eval-every", "eval-tokens"]


@requires_nanochat
def test_all_train_flags_exist_in_patched_base_train():
    source = (NANOCHAT / "scripts" / "base_train.py").read_text()
    known = set(re.findall(r'add_argument\("--([a-z0-9-]+)"', source))
    cfg = train_args({"bpe-nfc": 4.0, "super-nfc": 5.0}, "super-nfc", depth=10)
    missing = [k for k in [*cfg, *FIXED_FLAGS] if k not in known]
    assert not missing, missing


@requires_nanochat
def test_seed_env_is_patched():
    assert "NANOCHAT_SEED" in (NANOCHAT / "nanochat" / "common.py").read_text()


def test_fetch_checkpoints_lays_out_runs_and_resumes(tmp_path, tokenizers_dir):
    from vitok.fetch_checkpoints import checkpoint_files, fetch

    assert checkpoint_files("super-nfc_d10_s0") == ["runs/super-nfc_d10_s0/base_checkpoints/d10/meta_015258.json",
                                                    "runs/super-nfc_d10_s0/base_checkpoints/d10/model_015258.pt"]
    calls, broken = [], {"runs/super-nfc_d6_s0/base_checkpoints/d6/meta_003814.json"}

    def fake_download(version, rel, stage):
        calls.append(rel)
        if rel in broken:
            raise ConnectionError("network down")
        path = stage / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel)
        return path

    only = ["bpe-nfc_d6_s0", "super-nfc_d6_s0"]
    assert fetch(tmp_path, tokenizers_dir, only, download=fake_download) == sorted(broken)
    run = tmp_path / "results-v4" / "runs" / "bpe-nfc_d6_s0"
    assert (run / "base_checkpoints" / "d6" / "model_003814.pt").read_text().endswith("model_003814.pt")
    assert (run / "tokenizer" / "tokenizer.json").exists() and (run / "tokenizer" / "token_bytes.pt").exists()
    assert len(calls) == 4 and (tmp_path / ".fetch-staging").exists()  # a failure does not stop the others

    calls.clear()
    broken.clear()
    assert fetch(tmp_path, tokenizers_dir, only, download=fake_download) == []
    assert calls == ["runs/super-nfc_d6_s0/base_checkpoints/d6/meta_003814.json"]  # only the missing file
    assert not (tmp_path / ".fetch-staging").exists()
