import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# mutmut runs the tests from a copy in mutants/; the files to protect are the real repository's
REPO = ROOT.parent if ROOT.name == "mutants" else ROOT
NANOCHAT = REPO / "third_party" / "nanochat"

# Sandbox, set before anything else is imported: HOME, XDG folders, Kaggle credentials and nanochat's base dir
# all point at a temporary folder, so no test (or mutant) can write into the owner's data. Model caches stay real.
# conftest can be imported twice (as pytest's plugin and by `from conftest import …`, e.g. under mutmut); both copies
# must share one sandbox, or the second would re-point NANOCHAT_BASE_DIR and HOME away from the first's folders
if "VITOK_TEST_SANDBOX" not in os.environ:
    os.environ["VITOK_TEST_REAL_HOME"] = str(Path.home())
    os.environ["VITOK_TEST_SANDBOX"] = tempfile.mkdtemp(prefix="vitok_sandbox_")
REAL_HOME = Path(os.environ["VITOK_TEST_REAL_HOME"])
SANDBOX = Path(os.environ["VITOK_TEST_SANDBOX"])
os.environ.setdefault("HF_HOME", str(REAL_HOME / ".cache" / "huggingface"))
for var, sub in [("HOME", "home"), ("XDG_CACHE_HOME", "home/.cache"), ("XDG_CONFIG_HOME", "home/.config"),
                 ("XDG_DATA_HOME", "home/.local/share"), ("KAGGLE_CONFIG_DIR", "home/.kaggle"),
                 ("MPLCONFIGDIR", "home/.config/matplotlib")]:
    (SANDBOX / sub).mkdir(parents=True, exist_ok=True)
    os.environ[var] = str(SANDBOX / sub)
# nanochat reads NANOCHAT_BASE_DIR at import time
BASE_DIR = SANDBOX / "nanochat"
BASE_DIR.mkdir(exist_ok=True)
os.environ["NANOCHAT_BASE_DIR"] = str(BASE_DIR)
if NANOCHAT.exists():
    sys.path.insert(0, str(NANOCHAT))

# Guard: the session fails if any file of the real repository (or nanochat's real cache) appears, disappears or
# changes while the tests run. Byproducts of running Python and pytest are ignored.
SKIP_DIRS = {".git", ".pixi", "mutants", "__pycache__", ".pytest_cache", ".hypothesis"}


# mutation-testing logs are written by the runner while the tests run
SKIP_PATHS = {REPO / "runs" / "mutmut"}


def snapshot(roots) -> dict:
    state = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and Path(dirpath) / d not in SKIP_PATHS]
            for name in filenames:
                path = Path(dirpath) / name
                st = path.lstat()
                state[str(path)] = (st.st_size, st.st_mtime_ns)
    return state


GUARDED = [REPO, REAL_HOME / ".cache" / "nanochat"]
BEFORE = snapshot(GUARDED)


@pytest.fixture(scope="session", autouse=True)
def sandbox_cwd():
    """Run every test from inside the sandbox, so relative paths and default folders land there too."""
    old = os.getcwd()
    work = SANDBOX / "cwd"
    work.mkdir(exist_ok=True)
    os.chdir(work)
    yield
    os.chdir(old)


def pytest_sessionfinish(session, exitstatus):
    after = snapshot(GUARDED)
    changed = sorted(p for p in BEFORE.keys() | after.keys() if BEFORE.get(p) != after.get(p))
    if changed:
        print("\nTEST SANDBOX BREACH: these real files changed during the tests:", *changed, sep="\n  ")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED

SENTENCES = [
    "Học sinh giỏi nhất trường đã đạt giải nhất cuộc thi toán học quốc gia năm nay.",
    "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam.",
    "Người dân ở đồng bằng sông Cửu Long trồng lúa và nuôi cá tra.",
    "Máy tính của tôi bị hỏng nên tôi phải mang đi sửa ở cửa hàng gần nhà.",
    "Thời tiết hôm nay rất đẹp, trời trong xanh và có nắng nhẹ.",
    "Các nhà khoa học đang nghiên cứu cách giảm ô nhiễm không khí ở thành phố.",
    "Bà ngoại kể chuyện cổ tích cho các cháu nghe mỗi tối trước khi đi ngủ.",
    "Đội tuyển bóng đá Việt Nam đã chiến thắng với tỉ số 2-1 vào ngày 15/9.",
]


@pytest.fixture(scope="session")
def corpus_file(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("corpus") / "tok_train.txt"
    path.write_text("\n".join(SENTENCES * 40) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def tokenizers_dir(tmp_path_factory, corpus_file) -> Path:
    from vitok.train_tokenizers import train_all

    out = tmp_path_factory.mktemp("tokenizers")
    work = tmp_path_factory.mktemp("tok_work")
    train_all(corpus_file, out, vocab_size=600, transition=0.9, workroot=work)
    return out


requires_nanochat = pytest.mark.skipif(not NANOCHAT.exists(), reason="third_party/nanochat not cloned")


# Qwen2.5/Qwen3's pre-tokenizer regex, so a tiny base splits text the way the real base does
QWEN_RE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"
           r"|\s+(?!\S)|\s+")
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def qwen_like_base(vocab_size: int = 420) -> dict:
    """A small Qwen-like byte-level BPE tokenizer.json (NFC, regex Split + ByteLevel, specials after the vocabulary)."""
    import json

    from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(QWEN_RE), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["The quick brown fox, don't stop!"] * 5 + SENTENCES * 3,
                            BpeTrainer(vocab_size=vocab_size, show_progress=False,
                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.add_special_tokens(SPECIALS)
    return json.loads(tok.to_str())
