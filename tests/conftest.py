import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NANOCHAT = ROOT / "third_party" / "nanochat"

# nanochat reads NANOCHAT_BASE_DIR at import time, so set it before any test imports nanochat.
BASE_DIR = Path(tempfile.mkdtemp(prefix="vitok_nanochat_"))
os.environ["NANOCHAT_BASE_DIR"] = str(BASE_DIR)
if NANOCHAT.exists():
    sys.path.insert(0, str(NANOCHAT))

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
