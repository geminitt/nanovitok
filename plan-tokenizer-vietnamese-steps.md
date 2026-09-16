# Tokenizer cho Language Model tiếng Việt cỡ nhỏ: SuperBPE, NFD và quy mô
## Plan thực thi 3 tuần — bản theo thứ tự các bước

---

## 0. Cách dùng tài liệu này

- **Phần A** (mục 1–4): hiểu project. Đọc một lần trước khi bắt tay.
- **Phần B** (mục 5): 12 bước theo thứ tự, mỗi bước ghi rõ ngày dự kiến.
- **Phần C** (mục 6–10): tra cứu. Cấu hình, công thức, sự cố, nguồn, ngân sách.

Mỗi bước có bốn ô cố định:

| Ô | Nghĩa |
|---|---|
| **Làm gì** | Việc cụ thể |
| **Đầu ra** | File hoặc con số phải có sau khi xong |
| **Kiểm tra** | Dấu hiệu để biết bước đó đã đúng, trước khi đi tiếp |
| **Tra cứu** | Link cần dùng |

Nguyên tắc: **không đi tiếp khi ô "Kiểm tra" chưa đạt.** Trong đề tài này, lỗi nguy hiểm nhất là **so sánh không công bằng giữa các tokenizer**. Nó không làm code crash, chỉ âm thầm làm sai kết luận.

---

# PHẦN A — HIỂU PROJECT

## 1. Project này làm gì, nói trong 10 dòng

Tokenizer BPE chuẩn cắt văn bản ở khoảng trắng trước khi gộp token. Với tiếng Anh, khoảng trắng tách **từ**. Với tiếng Việt, khoảng trắng tách **âm tiết**: "học sinh" là một từ nhưng bị cắt thành hai mảnh, và BPE **không bao giờ** gộp được cả từ.

**SuperBPE** cho phép gộp qua khoảng trắng. Trên tiếng Anh nó giảm tới 33% số token và cải thiện model. Ngoài tiếng Anh, kết quả **trái chiều**: kém hơn BPE ở tiếng Hungary và tiếng Trung, nhưng nén tốt hơn trên 97 ngôn ngữ. **Chưa ai train language model tiếng Việt bằng SuperBPE.**

Yếu tố thứ hai là **dấu thanh**. "ệ" viết được dạng **NFC** (1 ký tự) hoặc **NFD** ("e" + dấu mũ + dấu nặng). Tokenizer train trên NFD có thể giúp model chia sẻ phần chữ gốc giữa các dạng có dấu, và **bền hơn khi người dùng gõ không dấu**. Chưa ai đo điều này cho language model tiếng Việt.

Ta train **4 tokenizer** (BPE / SuperBPE × NFC / NFD), pretrain model nhỏ kiểu nanochat ở **3 cỡ**, rồi so sánh công bằng.

## 2. Câu hỏi và giả thuyết — chốt trước khi train

**Câu hỏi chính**: với language model tiếng Việt cỡ nhỏ, gộp token qua khoảng trắng (SuperBPE) và tách dấu (NFD) ảnh hưởng thế nào tới **chất lượng mô hình hoá**, **độ nén**, và **độ bền khi văn bản mất dấu**? Ảnh hưởng đó có đổi theo cỡ model không?

| # | Giả thuyết | Cách kiểm định |
|---|---|---|
| **H1** | SuperBPE giảm ≥15% số token so với BPE cùng vocab, và với **cùng lượng văn bản train** thì bits per character (bpc) **không tệ hơn** BPE quá 1% | Bước 3 (độ nén), bước 8–10 (bpc), bootstrap ghép cặp theo văn bản |
| **H2** | Tokenizer NFD cho bpc **thấp hơn** NFC trên văn bản **bỏ dấu**, và chỉ thiệt ≤1% trên văn bản chuẩn | Bước 6 (bộ đánh giá bỏ dấu), bước 8–10 |
| **H3** | Lợi thế (hoặc bất lợi) của SuperBPE **thay đổi có hệ thống theo cỡ model** (d6 → d8 → d10) | Đồ thị bpc theo cỡ, bước 11 |
| **H4** *(khám phá)* | Các superword token của SuperBPE trùng với **từ nhiều âm tiết** trong từ điển nhiều hơn mức ngẫu nhiên | Đối chiếu với bộ tách từ tiếng Việt, bước 11 |

Nếu giả thuyết sai (SuperBPE kém hơn, NFD không giúp), **đó vẫn là kết quả hợp lệ** và phải báo cáo nguyên vẹn. Kết quả trái chiều ở Hungary/Trung cho thấy SuperBPE thất bại ở tiếng Việt là hoàn toàn có thể. Project chỉ thất bại nếu **so sánh không công bằng**.

**Phân cấp bằng chứng, chốt từ đầu:**
- **Chính**: bpc trên văn bản chuẩn và văn bản bỏ dấu, độ nén (H1, H2).
- **Phụ**: xu hướng theo cỡ (H3). Chỉ có 3 cỡ, nên chỉ nói về *xu hướng*, không nói về *quy luật*.
- **Khám phá**: độ trùng superword–từ (H4), độ chính xác cặp tối thiểu.

## 3. Ba thứ phải hiểu trước khi đọc các bước

### 3.1. Vì sao không so bằng loss trên mỗi token

Loss trên mỗi token **không so được** giữa các tokenizer. SuperBPE gộp nhiều chữ vào một token, nên mỗi token khó đoán hơn và loss/token cao hơn, dù model có thể tốt hơn.

Đơn vị so sánh đúng là **bits per character (bpc)** trên văn bản **NFC gốc**:

$$\text{bpc} = \frac{\sum_{\text{token}} -\ln p(\text{token})}{\ln 2 \times \text{số ký tự NFC của văn bản}}$$

⚠️ **Không dùng bits per byte của nanochat để kết luận.** Văn bản NFD có **nhiều byte hơn** NFC cho cùng nội dung, nên bpb sẽ thiên vị tokenizer NFD một cách giả tạo. bpb của nanochat chỉ dùng để theo dõi lúc train.

### 3.2. Công bằng nghĩa là cố định cái gì

Khác tokenizer thì cùng một đoạn văn cho ra số token khác nhau. Có hai cách cố định, dẫn tới hai kết luận khác nhau:

| Cố định | SuperBPE được lợi hay thiệt? | Trả lời câu hỏi |
|---|---|---|
| **Cùng số token train** | Lợi: đọc được **nhiều văn bản hơn** | ❌ Không công bằng, không dùng |
| **Cùng lượng văn bản** (thiết kế chính) | Hơi thiệt: ít token nên **ít FLOPs hơn** | "Cùng dữ liệu, tokenizer nào cho model tốt hơn?" |
| **Cùng FLOPs** (kiểm tra phụ, chỉ ở d8) | SuperBPE đọc thêm văn bản | "Cùng ngân sách tính toán, tokenizer nào tốt hơn?" |

**Thiết kế chính** (giống cách bài SuperBPE làm):
- Mọi điều kiện train trên **đúng cùng các văn bản, cùng thứ tự**.
- **Cùng số bước train**, **cùng số chuỗi mỗi bước**.
- **Độ dài context theo token được co giãn** để mỗi context chứa **cùng số ký tự trung bình**.

Kết quả: mọi điều kiện thấy cùng một lượng văn bản, cùng lịch learning rate; chỉ khác cách cắt văn bản thành token.

### 3.3. Tham số embedding

Ở model nhỏ, bảng embedding chiếm **phần lớn** tham số (nanochat dùng embedding và output head **không chia sẻ trọng số**). Luôn báo cáo **tham số không tính embedding** và **tổng tham số**. Mọi điều kiện cùng vocab size, nên phần embedding giống nhau, nhưng người đọc cần biết tỷ lệ này để hiểu kết quả.

## 4. Cổng quyết định

| Cổng | Ở đâu | Câu hỏi | Nếu ĐẠT | Nếu KHÔNG ĐẠT |
|---|---|---|---|---|
| **0** | Cuối bước 7 (ngày 6) | nanochat chạy fp16 trên Kaggle T4 ổn định, và ước tính thời gian d10 ≤ 6 giờ/run? | Giữ ngân sách ở mục 10 | Hạ cỡ lớn nhất xuống d9, hoặc giảm lượng văn bản d10 còn 60% (ghi rõ) |
| **1** | Cuối bước 3 (ngày 3) | SuperBPE giảm ≥15% số token so với BPE ở vocab đã chọn? | Giữ H1 là giả thuyết chính | Vẫn chạy đủ, nhưng **dời trọng tâm sang H2 (NFD)**; H1 thành phụ |
| **2** | Cuối bước 8 (ngày 9) | Có đủ bpc (chuẩn + bỏ dấu) cho 4 điều kiện ở d6, và bpc tính lại khớp bpb của nanochat (sau quy đổi) trong 1%? | **Có project nộp được.** Mọi bước sau là mở rộng | Sửa harness trước; không train d8 khi số đo còn nghi ngờ |
| **3** | Cuối bước 9 (ngày 14) | Có kết quả d8? Chọn 2 điều kiện cho d10 **theo luật đã chốt** | Train d10 | Nếu hết quota: bỏ d10, báo cáo 2 cỡ, H3 thành "hướng mở rộng" |

**Luật chọn điều kiện cho d10** (chốt trong `docs/analysis_plan.md` ở bước 4): luôn gồm **BPE-NFC** (baseline), cộng thêm **biến thể SuperBPE có bpc trên văn bản chuẩn thấp nhất ở d8**. Không chọn theo kết quả mình mong muốn.

---

# PHẦN B — CÁC BƯỚC

## 5. Sơ đồ phụ thuộc và lịch

```
1 ──► 2 ──► 3 [CỔNG 1] ──► 4 (chốt analysis plan)
                               │
              5 (nối tokenizer) ┼──► 6 (harness đánh giá)
                               │            │
                               └──► 7 [CỔNG 0] ─────┘
                                            │
                                            ▼
                                   8 d6 [CỔNG 2] ──► 9 d8 [CỔNG 3] ──► 10 d10
                                                                        │
                                                        11 phân tích ◄──┘
                                                                        │
                                                        12 báo cáo, demo
```

| Tuần | Ngày | Bước |
|---|---|---|
| **1** | 1–2 | 1, 2 |
| | 3 | 3 (Cổng 1), 4 |
| | 4–5 | 5, 6 |
| | 6 | 7 (Cổng 0) |
| | 7 | 8 bắt đầu |
| **2** | 8–9 | 8 xong (Cổng 2) |
| | 10–14 | 9 (Cổng 3) |
| **3** | 15–18 | 10 (chạy nền), song song bắt đầu 11 trên d6 và d8 |
| | 19–21 | 11 xong, 12 |

GPU chạy nền trong khi bạn làm việc khác: lúc d8/d10 đang train thì viết phân tích cho các cỡ đã xong.

---

## Bước 1 — Dựng repo và môi trường (ngày 1)

**Làm gì**
1. Tạo repo Git theo cấu trúc ở mục 9.
2. Clone nanochat, **ghi lại commit hash** vào `README.md`. nanochat thay đổi nhanh, không pin là không tái lập được.
3. Môi trường (**đã triển khai**, xem `README.md`):
   - **Máy local**: chỉ `pixi` để viết code và chạy unit test CPU. **Không train gì ở local.**
   - **Kaggle notebook 01 (CPU)**: cài Rust + **bản fork `tokenizers` của SuperBPE** ngay trong notebook để train tokenizer. Bản fork xung đột với `tokenizers` gốc, nên chỉ cài trong notebook này.
   - **Kaggle notebook 02 (GPU)**: nanochat + `tokenizers` **gốc**. Tokenizer SuperBPE sau khi train là file `tokenizer.json` chuẩn, nạp được bằng thư viện gốc.
4. Kaggle: tạo notebook, bật GPU T4 ×2, thêm secret `HF_TOKEN` (để đẩy checkpoint lên HF Hub private).

**Đầu ra**: repo có cấu trúc; hai `requirements-*.txt` đã pin; commit hash của nanochat.

**Kiểm tra**
- [ ] `pixi run test` qua hết ở local
- [ ] Notebook 01 build xong bản fork (cần Rust) trên Kaggle
- [ ] Notebook 02 in `torch.cuda.get_device_capability()` ra `(7, 5)` và 2 GPU

**Tra cứu**
- https://github.com/PythonNut/superbpe
- https://github.com/karpathy/nanochat

---

## Bước 2 — Chuẩn bị dữ liệu (ngày 1–2)

**Làm gì**
1. Tải một phần **FineWeb-2, subset `vie_Latn`**. Cần khoảng **6–7GB văn bản** (đủ cho d10 ~1 tỷ token BPE cộng phần dự phòng).
2. **Chuẩn hoá toàn bộ về NFC** trước tiên. Dữ liệu web lẫn cả NFC lẫn NFD; không chuẩn hoá thì hai điều kiện NFC/NFD xuất phát từ văn bản khác nhau.
3. Chia **theo văn bản** (không theo dòng) thành bốn phần **không giao nhau**:

| Phần | Kích thước | Dùng cho |
|---|---|---|
| `tok_train` | ~500MB | Train cả 4 tokenizer (cùng một tập) |
| `pretrain` | ~6GB | Pretrain, chia shard parquet theo format nanochat |
| `val` | ~2.000 văn bản | Theo dõi lúc train |
| `test` | ~2.000 văn bản (~5M ký tự) | **Chỉ dùng ở bước 8–10**, không đụng lúc chỉnh tham số |

4. Khử trùng lặp gần giữa `test` và `pretrain` (ví dụ MinHash hoặc so hash của các đoạn 50 ký tự).
5. Bản NFD của mọi phần được **sinh ra từ bản NFC** bằng `unicodedata.normalize("NFD", ...)`, không tải riêng.

**Đầu ra**: `data/{tok_train,pretrain,val,test}/` (NFC), `data/stats.json` (số văn bản, số ký tự, số âm tiết).

**Kiểm tra**
- [ ] `normalize("NFC", text) == text` với mọi văn bản đã lưu
- [ ] Không văn bản nào của `test` xuất hiện trong `pretrain` (theo hash)
- [ ] Đọc tay 20 văn bản ngẫu nhiên: đúng tiếng Việt, không phải rác

**Tra cứu**
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- https://docs.python.org/3/library/unicodedata.html

---

## Bước 3 — Train 4 tokenizer và đo độ nén (ngày 3) ⟶ **CỔNG 1**

**Làm gì**
1. Train trên **cùng `tok_train`**, cùng vocab, cùng quy tắc tách số và dấu câu:

| Tên | Chuẩn hoá | Giai đoạn 2 (gộp qua khoảng trắng) |
|---|---|---|
| `bpe-nfc` | NFC | Không |
| `bpe-nfd` | NFD | Không |
| `super-nfc` | NFC | Có |
| `super-nfd` | NFD | Có |

2. Vocab: thử **16k và 32k** (chỉ đo độ nén, không train model). Điểm chuyển giai đoạn của SuperBPE: mặc định **90% vocab** (giống bài gốc dùng t=180k/200k, và 2510.21909 dùng tỷ lệ 90/10). Thử thêm 80% chỉ để đo độ nén.
3. Đo trên `val`:
   - Số ký tự NFC trên mỗi token (**cpt**, càng cao càng nén tốt).
   - Số token trên mỗi âm tiết.
   - Tỷ lệ superword token trong vocab và tần suất dùng chúng.
4. Đọc tay: 30 superword token phổ biến nhất, là từ ("học sinh"), cụm ("của các"), hay rác?
5. **Chọn vocab** theo luật: lấy 16k nếu SuperBPE đạt ≥15% giảm token ở 16k; nếu không thì thử 32k; nếu cả hai không đạt thì giữ 16k và đi nhánh KHÔNG ĐẠT của Cổng 1.

**Vì sao chưa chọn 32k ngay**: với model d6 (~11M tham số không tính embedding), vocab 32k làm embedding chiếm quá nửa tổng tham số (xem mục 6.2).

**Đầu ra**: 4 file `tokenizers/<tên>/tokenizer.json`; `results/compression.json`; `results/superword_top30.md`.

**Kiểm tra — CỔNG 1**
- [ ] Encode rồi decode 1.000 văn bản `val`, chuẩn hoá NFC → **giống hệt** văn bản gốc, cho cả 4 tokenizer
- [ ] Hai tokenizer NFD thật sự thấy dấu tách rời (in token của "Việt" ra xem)
- [ ] `super-*` giảm ≥15% token so với `bpe-*` cùng chuẩn hoá (quyết định Cổng 1)
- [ ] Ghi lại cpt của 4 tokenizer: đây là con số dùng để co giãn context ở bước 5

**Tra cứu**
- https://arxiv.org/abs/2503.13423 (SuperBPE, mục phương pháp)
- https://arxiv.org/abs/2510.21909 (SuperBPE cho 97 ngôn ngữ, có tiếng Việt, chỉ đo độ nén: so số của bạn với họ)
- https://arxiv.org/abs/2604.05192 (bản train SuperBPE nhanh hơn, dùng nếu bản gốc quá chậm)

---

## Bước 4 — Chốt analysis plan (ngày 3)

**Làm gì**: viết `docs/analysis_plan.md` và **commit trước khi train model đầu tiên**. Nội dung:
- H1–H4 và ngưỡng ở mục 2.
- Định nghĩa bpc (mục 3.1), cách bỏ dấu (mục 6.4), cách tạo cặp tối thiểu (mục 6.5).
- Thiết kế công bằng: cùng văn bản, cùng số bước, context co giãn theo cpt (mục 3.2).
- Luật chọn 2 điều kiện cho d10 (mục 4).
- Kiểm định: bootstrap ghép cặp theo văn bản cho chênh lệch bpc; McNemar cho cặp tối thiểu.
- Seed: điều kiện nào có 2 seed (mục 10).

**Vì sao**: có 4 điều kiện × 3 cỡ × nhiều số đo, rất dễ vô thức chọn số đo nào đẹp để báo cáo. Chốt trước thì không ai (kể cả bạn) nghi ngờ được.

**Đầu ra**: `docs/analysis_plan.md` có commit hash **trước** mọi commit chứa checkpoint.

**Kiểm tra**: [ ] `git log` cho thấy commit analysis plan có thời điểm sớm hơn commit kết quả đầu tiên.

---

## Bước 5 — Nối tokenizer vào nanochat (ngày 4)

**Làm gì**
1. nanochat hiện chỉ có `RustBPETokenizer` (tiktoken). Viết lớp `HFTokenizer` trong `src/hf_tokenizer.py`, bọc `tokenizers.Tokenizer.from_file(...)`, **cung cấp đúng các hàm** mà nanochat gọi (encode, decode, id của BOS, vocab size, special token). Đọc `nanochat/tokenizer.py` để lấy danh sách hàm.
2. Thêm special token `<|bos|>` (và các token chat nếu code nanochat đòi) vào cả 4 tokenizer **cùng một cách**.
3. Bước chuẩn hoá: văn bản NFC → chuyển NFD nếu điều kiện là NFD → encode. Decode → chuyển NFC.
4. **Co giãn theo cpt** (mục 6.3):
   - `max_seq_len` của mỗi điều kiện = `round(1024 × cpt_bpe-nfc / cpt_điều_kiện)`.
   - `total_batch_size` = 64 chuỗi × `max_seq_len`.
   - `num_iterations` **giống nhau** cho mọi điều kiện cùng cỡ.
5. **Tắt việc tự co giãn learning rate theo batch size** của nanochat (nó nhân LR với √(B/B_ref) theo token). Giữ LR y hệt giữa các điều kiện, vì số chuỗi mỗi bước đã bằng nhau.

**Đầu ra**: `src/hf_tokenizer.py`, `configs/conditions.yaml` (4 điều kiện × 3 cỡ: seq len, batch, số bước).

**Kiểm tra**
- [ ] Dataloader của nanochat với `HFTokenizer` sinh batch; decode batch đầu ra văn bản đọc được, có BOS ở đầu văn bản
- [ ] Với 1.000 văn bản, **tổng số ký tự NFC mỗi bước** giữa 4 điều kiện chênh nhau <5%
- [ ] In LR thực tế ở bước 1 và bước 100 của cả 4 điều kiện: **bằng nhau**

**Tra cứu**: `nanochat/tokenizer.py`, `nanochat/dataloader.py`, `scripts/base_train.py` trong repo nanochat.

---

## Bước 6 — Harness đánh giá (ngày 4–5)

**Làm gì**: viết `src/eval.py`, nhận checkpoint + tokenizer, xuất JSON. Chạy trên `test`.

| Số đo | Cách tính | Giả thuyết |
|---|---|---|
| **bpc chuẩn** | Mục 3.1, tính **theo từng văn bản** trên văn bản test đã cắt ≤2.500 ký tự | H1, H3 |
| **bpc bỏ dấu 100%** | Bỏ dấu toàn bộ (mục 6.4), tính bpc trên số ký tự của văn bản đã bỏ dấu | H2 |
| **bpc bỏ dấu 50%** | Bỏ dấu ngẫu nhiên 50% âm tiết (seed cố định) | H2 |
| **Cặp tối thiểu** | Câu gốc so với câu đổi thanh một âm tiết; model đúng nếu log-prob câu gốc cao hơn (mục 6.5) | Khám phá |
| **Độ nén** | cpt, token/âm tiết trên `test` | H1 |
| **Tốc độ sinh** | Ký tự NFC sinh ra mỗi giây (cùng số token/giây, khác cpt) | Ứng dụng |
| **Superword–từ** | Tỷ lệ superword token trùng ranh giới từ theo bộ tách từ tiếng Việt | H4 |

**Thay cho cửa sổ trượt (đã triển khai)**: văn bản test được **cắt theo ký tự** (≤2.500 ký tự, tại khoảng trắng) ngay từ bước 2, nên gần như luôn vừa một context ở mọi điều kiện. Văn bản nào vẫn vượt context ở bất kỳ run nào bị loại khỏi **mọi** run cùng cỡ (`vitok.analysis` lấy giao). Cách này tránh sai số khi ranh giới cửa sổ rơi giữa superword token.

**Đầu ra**: `src/eval.py`; `data/test_stripped100.jsonl`, `data/test_stripped50.jsonl`, `data/minimal_pairs.jsonl`.

**Kiểm tra**
- [ ] Model **khởi tạo ngẫu nhiên**: bpc ≈ log2(vocab) / cpt (sai lệch <5%)
- [ ] Sau khi train thử 200 bước (bước 7): bpc bạn tính khớp bpb của nanochat **sau khi quy đổi** (nhân với số byte/ký tự của văn bản tương ứng) trong 1%, cho điều kiện `bpe-nfc`
- [ ] Hàm bỏ dấu biến "Đường phố Hà Nội" thành "Duong pho Ha Noi"
- [ ] Đọc tay 20 cặp tối thiểu: câu đổi thanh là âm tiết **có thật** trong tiếng Việt

**Tra cứu**
- https://github.com/undertheseanlp/underthesea (tách từ tiếng Việt)
- https://docs.python.org/3/library/unicodedata.html

---

## Bước 7 — Smoke test trên Kaggle T4 (ngày 6) ⟶ **CỔNG 0**

**Làm gì**
1. Chạy `bpe-nfc` ở **d6** 200 bước, rồi **d8** và **d10** mỗi cỡ 50 bước, trên **1×T4**.
2. Xác nhận log in `COMPUTE_DTYPE: torch.float16` và `GradScaler enabled`. **Lưu ý**: trên T4, nanochat tự chọn **fp32**, nên phải đặt `NANOCHAT_DTYPE=float16` (`vitok.kaggle_run` đã đặt sẵn). SDPA thay FlashAttention, `--window-pattern L`.
3. ~~Kiểm tra Muon~~: **đã xác nhận khi đọc code**: Newton–Schulz chỉ ép bf16 khi COMPUTE_DTYPE là bf16, nên fp16 không cần sửa.
4. Đo **token/giây** cho từng cỡ, ước tính thời gian mỗi run theo mục 10.
5. Thử `torch.compile`. Nếu lỗi trên T4, tắt đi và ghi lại (chậm hơn nhưng vẫn chạy được).
6. Chạy thử **hai tiến trình song song** (`CUDA_VISIBLE_DEVICES=0` và `=1`) để xác nhận T4 ×2 chạy được 2 run cùng lúc.

**Đầu ra**: `results/throughput.json`; bảng ngân sách ở mục 10 được cập nhật bằng số đo thật.

**Kiểm tra — CỔNG 0**
- [ ] Loss giảm, không NaN, loss scaler không bỏ quá 1% số bước
- [ ] Thời gian ước tính cho d10 ≤ 6 giờ/run
- [ ] Checkpoint đẩy lên HF Hub private được giữa chừng

**Mọi run (kể cả debug) đều chạy trên Kaggle T4 fp16.** Máy local chỉ chạy unit test CPU, không train.

---

## Bước 8 — Pretrain d6, 4 điều kiện (ngày 7–9) ⟶ **CỔNG 2**

**Làm gì**
1. Train 4 điều kiện ở d6, 1 seed, lượng văn bản theo mục 10. Chạy 2 run song song trên 2 GPU.
2. Chạy `src/eval.py` cho cả 4.
3. Vẽ bảng sơ bộ: bpc chuẩn, bpc bỏ dấu, cpt.

**Đầu ra**: `checkpoints/d6_<điều kiện>/` (trên HF Hub), `results/d6_*.json`.

**Kiểm tra — CỔNG 2**
- [ ] Cả 4 run xong, không NaN, cùng số bước
- [ ] bpc của `bpe-nfc` hợp lý (thấp hơn rõ so với model ngẫu nhiên, đường val loss đã chậm lại)
- [ ] Bộ số đo đầy đủ cho 4 điều kiện

**Qua Cổng 2 là có project nộp được**: một so sánh công bằng 2×2 ở một cỡ, có kiểm định. Mọi thứ sau đây là mở rộng.

**Không** coi "SuperBPE thắng" là điều kiện kiểm tra. Đó là kết quả cần đo, không phải dấu hiệu pipeline đúng.

---

## Bước 9 — Pretrain d8 (ngày 10–14) ⟶ **CỔNG 3**

**Làm gì**
1. Train 4 điều kiện ở d8.
2. **Seed thứ hai** cho `bpe-nfc` và `super-nfc`, để ước lượng độ nhiễu giữa các seed. Nếu chênh lệch bpc giữa hai seed **lớn hơn** chênh lệch giữa hai tokenizer, mọi kết luận H1 phải ghi rõ là không chắc chắn.
3. *(Tuỳ chọn, nếu còn quota)* **Kiểm tra cùng FLOPs**: một run `super-nfc` d8 train thêm bước tới khi bằng FLOPs của `bpe-nfc`.
4. Eval, rồi **áp luật chọn điều kiện cho d10** (mục 4) và ghi quyết định vào `results/gate3.json`, **commit trước khi train d10**.

**Đầu ra**: `results/d8_*.json`, `results/gate3.json`.

**Kiểm tra — CỔNG 3**
- [ ] 6 run chính xong
- [ ] Độ lệch giữa 2 seed đã được ghi lại
- [ ] Quyết định d10 đã commit

---

## Bước 10 — Pretrain d10 (ngày 15–18, chạy nền)

**Làm gì**
1. Train `bpe-nfc` và biến thể SuperBPE được chọn ở Cổng 3, mỗi cái 1 seed, song song trên 2 GPU.
2. Mỗi run có thể dài ~5 giờ: dùng notebook **commit** (Save & Run All), đẩy checkpoint lên HF Hub mỗi ~1.000 bước để chạy tiếp được nếu session bị ngắt.
3. Eval.

**Đầu ra**: `results/d10_*.json`.

**Kiểm tra**: [ ] 2 run xong cùng số bước, không NaN; nếu phải giảm văn bản ở Cổng 0, cả 2 run giảm như nhau.

---

## Bước 11 — Phân tích và kiểm định (ngày 15–20)

Làm song song với bước 10: phân tích d6 và d8 trước, thêm d10 khi xong.

**Làm gì**
1. **Bảng chính** (mỗi cỡ): bpc chuẩn, bpc bỏ dấu 50%/100%, cpt, token/âm tiết, tốc độ sinh, độ chính xác cặp tối thiểu.
2. **Kiểm định H1, H2**: với mỗi cặp điều kiện cần so, **bootstrap ghép cặp theo văn bản** (10.000 lần) cho chênh lệch bpc → khoảng tin cậy 95%. Ghép cặp vì hai model được đánh giá trên **cùng** văn bản, loại được nhiễu do độ khó của từng văn bản.
3. **Cặp tối thiểu**: McNemar giữa các điều kiện.
4. **H3**: đồ thị bpc theo tham số không tính embedding (log), mỗi đường là một tokenizer; đồ thị chênh lệch bpc (SuperBPE − BPE) theo cỡ. Ghi rõ chỉ có 3 điểm, chênh lệch giữa seed ở d8 làm thanh sai số tham chiếu.
5. **H4**: tách từ `test` bằng underthesea; tính tỷ lệ superword token khớp trọn một từ nhiều âm tiết, so với mốc ngẫu nhiên (các cặp âm tiết liền kề chọn ngẫu nhiên có cùng tần suất).
6. **Phân tích lỗi**: văn bản nào SuperBPE hơn/kém BPE nhiều nhất (tên riêng? số? từ mượn?); ví dụ tokenization cạnh nhau.
7. **Phân tích độ nhạy**: kết luận H1 có đổi khi dùng bpc trên `val` thay vì `test`, hoặc khi bỏ 5% văn bản dài nhất?

**Đầu ra**: `results/tests.json`, `figures/`.

**Kiểm tra**
- [ ] Mọi con số trong bảng đều có khoảng tin cậy hoặc p-value
- [ ] Mọi hình dùng bpc, không dùng loss/token

---

## Bước 12 — Báo cáo và demo (ngày 19–21)

**Cấu trúc báo cáo đề nghị**
1. **Vấn đề**: khoảng trắng tách âm tiết trong tiếng Việt; dấu thanh; ví dụ tokenization.
2. **Related work**: SuperBPE; 2510.21909 (SuperBPE 97 ngôn ngữ, chỉ đo độ nén); 2606.15044 (tokenizer Đông Nam Á, model 1,5B, không SuperBPE/NFD); MELLM 2026 (SuperBPE kém ở Hungary/Trung); TokSuite.
3. **Phương pháp**: 4 tokenizer, thiết kế công bằng (mục 3.2), bpc, bậc thang cỡ, analysis plan và thời điểm commit.
4. **Kết quả** theo phân cấp bằng chứng: chính → phụ → khám phá.
5. **Thảo luận**: tiếng Việt đứng ở đâu so với tiếng Anh, Hungary, Trung; ý nghĩa cho model tiếng Việt trên thiết bị.
6. **Hạn chế** (mục 8.2).

**Demo gợi ý** (chọn một):
- Trang so sánh tokenization: gõ câu tiếng Việt, xem 4 tokenizer cắt thế nào, số token, và model d10 tương ứng sinh tiếp nhanh bao nhiêu ký tự/giây.
- Gõ câu không dấu, xem model NFC và NFD đoán tiếp khác nhau ra sao.

---

# PHẦN C — TRA CỨU

## 6. Chi tiết kỹ thuật

### 6.1. Kích thước model (nanochat, `--aspect-ratio 64`, `--head-dim 128`)

| Depth | n_embd | Tham số không embedding (≈12·d²·L) | Embedding + head, vocab 16k | Tổng ≈ |
|---|---|---|---|---|
| d6 | 384 | ~11M | ~12M | ~23M |
| d8 | 512 | ~25M | ~16M | ~41M |
| d10 | 640 | ~49M | ~20M | ~70M |

Số chính xác: in ra từ model thật ở bước 7 (kiến trúc nanochat có thêm vài thành phần nhỏ).

### 6.2. Vì sao vocab 16k

Vocab 32k ở d6 làm embedding + head ≈ 25M, gấp hơn 2 lần phần không embedding. Model sẽ dành phần lớn tham số để nhớ token thay vì học ngôn ngữ, và kết quả nghiêng về đặc tính của embedding. Chỉ lên 32k nếu 16k không đủ chỗ cho superword (Cổng 1).

### 6.3. Co giãn context và batch

```python
cpt = {"bpe-nfc": ..., "bpe-nfd": ..., "super-nfc": ..., "super-nfd": ...}  # từ bước 3
BASE_SEQ = 1024          # token, cho bpe-nfc
SEQS_PER_STEP = 64       # giống nhau mọi điều kiện

def seq_len(cond):
    return round(BASE_SEQ * cpt["bpe-nfc"] / cpt[cond])

def total_batch_tokens(cond):
    return SEQS_PER_STEP * seq_len(cond)

# num_iterations: giống nhau cho mọi điều kiện cùng cỡ (mục 10)
```

`device_batch_size` chọn sao cho `total_batch_tokens` chia hết cho `device_batch_size × seq_len` (gradient accumulation). Kiểm tra VRAM ở bước 7.

### 6.4. Bỏ dấu tiếng Việt

```python
import unicodedata

def strip_diacritics(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    d = "".join(ch for ch in d if unicodedata.category(ch) != "Mn")
    d = d.replace("đ", "d").replace("Đ", "D")   # "đ" không tách được bằng NFD
    return unicodedata.normalize("NFC", d)
```

Bỏ dấu 50%: tách theo khoảng trắng, chọn ngẫu nhiên 50% âm tiết (seed cố định) để áp hàm trên.

### 6.5. Cặp tối thiểu

1. Lập danh sách âm tiết hợp lệ từ `pretrain` (âm tiết xuất hiện ≥50 lần).
2. Với mỗi câu trong `test`, chọn ngẫu nhiên một âm tiết có dấu thanh, đổi sang thanh khác sao cho âm tiết mới **nằm trong danh sách hợp lệ** ("má" → "mà").
3. Chấm: tổng log-prob của **cả câu** (bắt đầu từ BOS) dưới model. Đúng nếu câu gốc cao hơn. Vì so tổng xác suất của cả câu, cách chấm này không phụ thuộc tokenizer.
4. ~3.000 cặp.

### 6.6. Bootstrap ghép cặp

```python
import numpy as np
# nats[i], chars[i]: tổng loss (nats) và số ký tự NFC của văn bản i, cho model A và B
def paired_bootstrap(natsA, natsB, chars, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(chars))
    diffs = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        bpcA = natsA[s].sum() / (np.log(2) * chars[s].sum())
        bpcB = natsB[s].sum() / (np.log(2) * chars[s].sum())
        diffs.append(bpcA - bpcB)
    return np.percentile(diffs, [2.5, 50, 97.5])
```

Chênh lệch có ý nghĩa nếu khoảng 95% không chứa 0.

## 7. Checklist bàn giao

- [ ] 4 tokenizer (`tokenizer.json`) + script train
- [ ] Code: tách dữ liệu, `HFTokenizer`, eval, phân tích
- [ ] `docs/analysis_plan.md` kèm commit hash chứng minh chốt trước khi train
- [ ] Mọi file kết quả JSON, đủ để vẽ lại mọi hình
- [ ] Checkpoint trên HF Hub (private hoặc public)
- [ ] Commit hash nanochat, `requirements-*.txt`, danh sách seed, thay đổi đã patch vào nanochat (fp16/Muon/LR)
- [ ] Báo cáo + demo

## 8. Sự cố thường gặp và hạn chế

### 8.1. Sự cố

| Triệu chứng | Thử theo thứ tự |
|---|---|
| Không build được fork `tokenizers` của SuperBPE | Kiểm tra Rust và Python 3.12; dùng bản train nhanh của 2604.05192; cuối cùng mới tự cài giai đoạn 2 |
| Decode không ra văn bản gốc | Thiếu bước NFC sau decode; pre-tokenizer ByteLevel không khớp decoder; special token bị decode |
| Loss NaN trên T4 | Kiểm tra GradScaler có bật; giảm LR; Muon Newton–Schulz sang float32; giảm `device_batch_size` |
| `torch.compile` lỗi trên T4 | Tắt compile, ghi lại; chấp nhận chậm hơn |
| OOM | Giảm `device_batch_size`, tăng gradient accumulation (không đổi `total_batch_size`) |
| bpc tự tính lệch bpb nanochat >1% | Sai số ký tự (đếm NFD thay vì NFC); quên token BOS; nhớ rằng bpb tính trên val shard còn bpc trên test, nên chỉ so khi cùng tập |
| SuperBPE nén kém dự kiến | Superword có thể cần tok_train lớn hơn (2510.21909 cảnh báo điều này): tăng lên 1GB rồi đo lại, trước khi kết luận |
| Session Kaggle bị ngắt | Chạy tiếp từ checkpoint trên HF Hub; notebook commit thay vì phiên tương tác |
| Hết quota | Bỏ kiểm tra cùng FLOPs → bỏ seed thứ hai → bỏ d10. Ghi rõ trong báo cáo |

### 8.2. Hạn chế phải ghi trong báo cáo

- Model ≤~70M tham số; kết luận về SuperBPE có thể **đảo chiều ở model lớn** (chính là lý do có H3, nhưng 3 điểm không đủ để ngoại suy).
- Một vocab size, một điểm chuyển giai đoạn của SuperBPE.
- Chỉ đo mô hình hoá ngôn ngữ (bpc, cặp tối thiểu), **không đo benchmark tác vụ**: model quá nhỏ.
- Văn bản bỏ dấu là **mô phỏng**; người gõ thật bỏ dấu không đều và có lỗi gõ.
- Ít seed; độ nhiễu giữa seed chỉ đo ở d8 cho 2 điều kiện.
- Chạy fp16 trên T4, có patch nanochat; kết quả tuyệt đối có thể khác một chút so với bf16.

## 9. Cấu trúc thư mục đề nghị

```
tokenizer-vi/
├── README.md                     # commit nanochat, patch đã áp, cách chạy
├── requirements-superbpe.txt
├── requirements-train.txt
├── configs/
│   └── conditions.yaml           # 4 điều kiện × 3 cỡ: seq len, batch, số bước
├── docs/
│   └── analysis_plan.md          # CHỐT trước khi train
├── src/
│   ├── prepare_data.py           # bước 2
│   ├── train_tokenizers.sh       # bước 3
│   ├── compression.py            # bước 3
│   ├── hf_tokenizer.py           # bước 5
│   ├── diacritics.py             # mục 6.4
│   ├── minimal_pairs.py          # mục 6.5
│   ├── eval.py                   # bước 6
│   └── analysis.py               # bước 11
├── tokenizers/<tên>/tokenizer.json
├── data/                         # không commit dữ liệu lớn
├── results/
├── figures/
└── report/
```

## 10. Ngân sách GPU (ước tính thô, cập nhật ở bước 7)

Lượng văn bản quy về token của `bpe-nfc` (các điều kiện khác cùng lượng văn bản, cùng số bước):

**Đo thật ở bước 7** (smoke test 2026-09-16, `results/throughput_d*.json`; mỗi run một GPU, 2 run song song):

| Cỡ | Số bước | bpe-nfc (đo) | super-nfc (đo) | Cách xếp phiên | Thời gian phiên |
|---|---|---|---|---|---|
| d6 | 3.814 | 0,71 giờ (669 ms/bước) | 0,57 giờ (534 ms/bước) | 4 điều kiện, 2 mỗi GPU | ~1,4 giờ |
| d8 | 7.629 | 2,87 giờ (1.352 ms/bước) | 2,30 giờ (1.084 ms/bước) | 4 điều kiện, 2 mỗi GPU | ~5,7 giờ |
| d8 seed 1 | 7.629 | | | `bpe-nfc` và `super-nfc`, 1 mỗi GPU | ~2,9 giờ |
| d10 | 15.258 | 8,75 giờ (2.063 ms/bước) | 8,39 giờ (1.979 ms/bước) | **1 điều kiện mỗi GPU** (2 điều kiện/GPU vượt 12 giờ) | ~8,8 giờ |
| **Tổng** | | | | | **~19 giờ** |

Ước tính ban đầu (giữ lại để đối chiếu): d6 ~0,4 giờ, d8 ~1,4 giờ, d10 ~5 giờ mỗi run — tức thực tế chậm hơn 1,5–2 lần, nằm trong biên "sai số có thể gấp 2".

- Peak memory đo được: d6 6,4GB, d8 9,0GB (`device-batch-size` 32), d10 7,1GB (`device-batch-size` 16). Còn dư so với 16GB của T4.
- d10 vượt ngưỡng 6 giờ/run của Cổng 0 (8,75 giờ) nhưng vẫn gọn trong một phiên 12 giờ nếu mỗi GPU chỉ chạy một điều kiện.
- SuperBPE dùng ít token hơn cho cùng văn bản, nên run SuperBPE **nhanh hơn** con số trong bảng.
- Chạy 2 run song song trên T4 ×2 để giảm thời gian chờ; quota Kaggle tính theo thời gian session.
- ~23 giờ vừa quota ~30 giờ/tuần, nhưng nên **trải qua 2 tuần**, và giữ dự phòng cho run lỗi.

**Các bước không cần GPU**: 1, 2, 3, 4, 5 (phần lớn), 11, 12.

### Công sức làm tay (ước lượng)

| Việc | Thời gian |
|---|---|
| Học thêm: Unicode/chính tả tiếng Việt, SuperBPE, so sánh công bằng giữa tokenizer, fp16, bootstrap | ~3 ngày |
| Dữ liệu + tokenizer (bước 1–4) | ~8–10 giờ |
| Nối nanochat + harness (bước 5–7) | ~10–12 giờ |
| Theo dõi train, eval (bước 8–10) | ~4–6 giờ |
| Phân tích, hình, báo cáo, demo | ~15 giờ |

## 11. Nguồn tham khảo

### Bắt buộc trong Related work
| Bài | Liên quan | Link |
|---|---|---|
| SuperBPE: Space Travel for Language Models | Phương pháp chính | https://arxiv.org/abs/2503.13423 |
| Explaining and Mitigating Crosslingual Tokenizer Inequities | Train SuperBPE cho 97 ngôn ngữ **có tiếng Việt**, nhưng **không train language model**, không NFD. Khoảng trống trực tiếp của đề tài | https://arxiv.org/abs/2510.21909 |
| Tokenizer cho 11 ngôn ngữ Đông Nam Á (2606.15044) | Có tiếng Việt, model 1,5B; không SuperBPE, NFD, độ bền mất dấu, model nhỏ | https://arxiv.org/abs/2606.15044 |
| Benchmarking BPE Tokenizers with Bits per Byte (MELLM 2026) | SuperBPE kém BPE ở tiếng Hungary và tiếng Trung | https://aclanthology.org/2026.mellm-1.27/ |
| TokSuite | Đo tác động của lựa chọn tokenizer lên hành vi model | https://arxiv.org/abs/2512.20757 |
| Faster Superword Tokenization | Train SuperBPE nhanh hơn | https://arxiv.org/abs/2604.05192 |

### Công cụ và dữ liệu
| Việc | Link |
|---|---|
| nanochat | https://github.com/karpathy/nanochat |
| SuperBPE code | https://github.com/PythonNut/superbpe |
| FineWeb-2 (`vie_Latn`) | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| HF tokenizers | https://huggingface.co/docs/tokenizers |
| underthesea (tách từ tiếng Việt) | https://github.com/undertheseanlp/underthesea |
| unicodedata | https://docs.python.org/3/library/unicodedata.html |
| Kaggle Notebooks | https://www.kaggle.com/code |

### Hướng mở rộng về sau
- Pretrain model lớn hơn bằng tokenizer thắng, rồi SFT → RL theo pipeline nanochat.
- Mở rộng tokenizer của model có sẵn (Qwen) bằng superword tiếng Việt + continued pretraining.
- Nén KV cache cho văn bản dài tiếng Việt: token là từ thay vì âm tiết có giúp giữ nghĩa khi nén không.
- Thêm tiếng Thái, tiếng Khmer (ngôn ngữ không dùng khoảng trắng tách từ).
