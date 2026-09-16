# Cần học

Mọi link dưới đây đã kiểm tra mở được (16/09/2026). Ưu tiên đọc **phần được chỉ rõ**, không đọc cả bài.

**Thứ tự đề nghị**: 1 → 2 → 3 (cần cho báo cáo, ~2 ngày) → 6 (lúc bắt đầu viết) → 4, 5 (khi viết related work và làm H4) → 7–10 (sau project).

---

## 1. Thống kê thực nghiệm (~1 ngày) — ưu tiên cao nhất

Cần nắm:

* Khoảng tin cậy nghĩa là gì, và nó **không** nói gì về cái gì (CI của bạn không nói gì về nhiễu seed).
* Bootstrap, đặc biệt là **bootstrap ghép cặp**: vì sao ghép cặp loại được nhiễu do độ khó của từng văn bản.
* Kiểm định McNemar cho dữ liệu nhị phân ghép cặp.
* Phân biệt **effect size** và **statistical significance**: chênh 0,17% vẫn có thể "significant" mà chẳng quan trọng.
* Vì sao phải đăng ký phân tích trước (pre-registration), và vì sao plan bắt commit `gate3.json` trước khi train d10.

**Nguồn**

| Nguồn | Đọc phần nào | Vì sao |
|---|---|---|
| [The Hitchhiker's Guide to Testing Statistical Significance in NLP](https://aclanthology.org/P18-1128/) (ACL 2018) | Mục 2–4 | Bài tổng quan chuẩn: chọn kiểm định nào cho bài toán nào, có cả McNemar và bootstrap |
| [Statistical Significance Tests for MT Evaluation](https://aclanthology.org/W04-3250/) (Koehn 2004) | Cả bài, 6 trang | Chính là bootstrap ghép cặp mà `vitok.stats` đang dùng, giải thích bằng ví dụ |
| [With Little Power Comes Great Responsibility](https://arxiv.org/abs/2010.06595) (Card 2020) | Mục 1–3 | Power và effect size trong NLP; trả lời đúng câu "chênh 0,17% thì có ý nghĩa gì" |
| [Fine-Tuning Pretrained LMs: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/abs/2002.06305) (Dodge 2020) | Mục 3 | Đo nhiễu giữa các seed — chính xác việc bạn vừa làm với d6 seed 1 |
| [Reporting Score Distributions Makes a Difference](https://arxiv.org/abs/1707.09861) (Reimers 2017) | Cả bài | Vì sao báo cáo một con số từ một seed là sai lầm phổ biến |
| [Preregistering NLP research](https://aclanthology.org/2021.naacl-main.51/) (NAACL 2021) | Mục 2–3 | Nền tảng của `docs/analysis_plan.md` và các cổng trong plan |

## 2. Lý thuyết thông tin cho mô hình ngôn ngữ (~nửa ngày)

Cross-entropy ↔ bits per character ↔ perplexity là **cùng một đại lượng đổi đơn vị**. Hiểu chỗ này thì tự giải thích được vì sao bpb thiên vị NFD, vì sao loss/token không so được giữa hai tokenizer, và vì sao nén văn bản với dự đoán ngôn ngữ là một.

**Nguồn**

| Nguồn | Đọc phần nào | Vì sao |
|---|---|---|
| [Jurafsky & Martin, SLP3 chương 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf) | Mục "Perplexity" và "Entropy, Cross-Entropy" | Nền tảng, viết dễ hiểu, miễn phí |
| [Benchmarking BPE Tokenizers with Bits per Byte](https://aclanthology.org/2026.mellm-1.27/) (MELLM 2026) | Cả bài | Đúng cái bẫy bpb/bpc của project bạn, lại có kết quả SuperBPE kém ở tiếng Hungary và Trung — dùng luôn cho related work |
| Notebook của chính bạn: [notebooks/toy_bpe_superbpe.ipynb](notebooks/toy_bpe_superbpe.ipynb) | Bước 3 | Tỷ lệ nén 2,47× ở đó chính là cùng một khái niệm với bpc |

## 3. Scaling law và đếm tham số (~nửa ngày)

Chinchilla (tỷ lệ token/tham số), vì sao tách tham số embedding khỏi phần thân, và ba kiểu so sánh công bằng: **cùng lượng văn bản** (cái bạn đang dùng), cùng số token, cùng FLOPs. Đây là nền của H3.

**Nguồn**

| Nguồn | Đọc phần nào | Vì sao |
|---|---|---|
| [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) (Chinchilla 2022) | Mục 3 và Bảng 3 | Tỷ lệ ~20 token/tham số; nanochat đặt mặc định 12, ngân sách d8 của bạn là 500M/25M = 20 |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan 2020) | Mục 2.1 | Định nghĩa "tham số không tính embedding" và vì sao phải tách ra |
| [SuperBPE](https://arxiv.org/abs/2503.13423) | Mục thiết lập thí nghiệm | Cách họ so sánh công bằng giữa hai tokenizer — đối chiếu với thiết kế equal-text của bạn |

## 4. Tokenization ngoài BPE (~nửa ngày)

Bạn đã nắm BPE rất chắc. Bổ sung: Unigram LM / SentencePiece khác BPE ở đâu, các độ đo nén (fertility, CTC, chars/token) và điểm yếu của từng cái.

**Nguồn**

| Nguồn | Đọc phần nào | Vì sao |
|---|---|---|
| [Subword Regularization](https://arxiv.org/abs/1804.10959) (Kudo 2018) | Mục 3 | Thuật toán Unigram LM: chọn token theo xác suất chứ không theo tần suất cặp |
| [SentencePiece](https://arxiv.org/abs/1808.06226) | Mục 2 | Vì sao coi khoảng trắng là ký tự thường — liên quan trực tiếp tới ý tưởng SuperBPE |
| [How Good is Your Tokenizer?](https://arxiv.org/abs/2012.15613) (Rust 2021) | Mục 3 | Fertility và vì sao tokenizer kém làm hỏng model đa ngữ |
| [Tokenization Is More Than Compression](https://arxiv.org/abs/2402.18376) (Schmidt 2024) | Cả bài | Phản biện: nén tốt hơn chưa chắc model tốt hơn — đúng với kết quả 0,17% của bạn |
| [Explaining and Mitigating Crosslingual Tokenizer Inequities](https://arxiv.org/abs/2510.21909) | Mục về CTC và vocab tối ưu | Bài có số tiếng Việt để so; code và tokenizer công khai |
| [Karpathy, Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) | 2 tiếng | Ôn lại BPE và các lỗi thực tế của tokenizer |

## 5. Unicode và chính tả tiếng Việt (~nửa ngày)

NFC/NFD, combining marks, quy tắc đặt dấu thanh, khác biệt giữa âm tiết và từ, tách từ tiếng Việt. Cần cho mục "Vấn đề" và cho H4.

**Nguồn**

| Nguồn | Đọc phần nào | Vì sao |
|---|---|---|
| [UAX #15: Unicode Normalization Forms](https://www.unicode.org/reports/tr15/) | Mục 1–3 | Định nghĩa chuẩn của NFC/NFD, thứ tự canonical của combining marks |
| [underthesea](https://github.com/undertheseanlp/underthesea) | README + API tách từ | Công cụ tách từ cho H4: superword có khớp ranh giới từ không |
| Code của bạn: [text.py](src/vitok/text.py) | `with_tone`, `strip_diacritics` | Quy tắc đặt dấu thanh đã cài sẵn ở đây |

## 6. Cách viết bài thực nghiệm (~nửa ngày, đọc khi bắt đầu viết)

Phân cấp bằng chứng (chính → phụ → khám phá), cách trình bày kết quả âm tính, cách viết mục hạn chế. Project của bạn có "SuperBPE ngang bằng nhưng ít token hơn" và "NFD giúp ở strip50 nhưng hại ở strip100" — viết khéo thì hay, viết vụng thì thành mơ hồ.

**Nguồn**

| Nguồn | Đọc phần nào | Vì sao |
|---|---|---|
| [Simon Peyton Jones, How to Write a Great Research Paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/) | Slide + video 1 tiếng | Cách kể một ý chính, cách viết phần đóng góp |
| [Workshop on Insights from Negative Results in NLP](https://aclanthology.org/venues/insights/) | Đọc 2–3 bài ngắn | Mẫu để viết kết quả "không khác biệt" cho đàng hoàng |
| [Beyond Accuracy: Behavioral Testing with CheckList](https://aclanthology.org/2020.acl-main.442/) | Mục 2 | Ý tưởng minimal pair; giúp bạn giải thích vì sao bộ cặp hiện tại chạm trần và bộ khó nên làm thế nào |

## 7. Mixed precision (sau project)

fp16 vs bf16, GradScaler, vì sao T4 cần scaler mà A100 thì không. Bạn đang dùng hằng ngày mà chưa mở ra xem.

* [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (Micikevicius 2017) — mục 3, loss scaling.
* [NVIDIA: Train With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — phần fp16 vs bf16.
* [PyTorch AMP](https://pytorch.org/docs/stable/amp.html) — API thực tế, đúng thứ nanochat đang gọi.

## 8. SFT rồi RLHF/DPO (sau project)

* [InstructGPT](https://arxiv.org/abs/2203.02155) — mục 3, quy trình SFT → reward model → PPO.
* [DPO](https://arxiv.org/abs/2305.18290) — cách bỏ reward model, đang là mặc định thực tế.
* [TRL](https://huggingface.co/docs/trl) — thư viện để tự chạy thử.
* `third_party/nanochat/scripts/` — nanochat có sẵn phần SFT, đọc code quen thuộc trước khi đọc paper.

## 9. LoRA/PEFT (sau project)

* [LoRA](https://arxiv.org/abs/2106.09685) — mục 4.
* [QLoRA](https://arxiv.org/abs/2305.14314) — mục 3, hợp với máy 6GB.
* [PEFT](https://huggingface.co/docs/peft) — thực hành.

## 10. Đánh giá LLM (sau project)

* [HELM](https://arxiv.org/abs/2211.09110) — mục 1–2, vì sao đánh giá nhiều chiều.
* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — công cụ chuẩn.
* [NLP Evaluation in trouble: data contamination](https://arxiv.org/abs/2310.18018) — vì sao benchmark hay bị nhiễm dữ liệu train.

---

## Nếu chỉ chọn một thứ

Mục 1. Giá trị của project nằm ở chỗ so sánh **công bằng** và kết luận **đúng mức** — đó là việc của thống kê, không phải của model. Người chấm sẽ hỏi đúng chỗ đó.
