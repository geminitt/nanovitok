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
