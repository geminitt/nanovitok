# CoT Distillation in Small Language Models: A Strategy-Diversity Study
## Plan thực thi — bản theo thứ tự các bước

---

## 0. Cách dùng tài liệu này

- **Phần A** (mục 1-4): hiểu project. Đọc một lần trước khi bắt tay.
- **Phần B** (mục 5): 16 bước theo thứ tự. Đây là phần bạn mở ra mỗi lần ngồi vào máy.
- **Phần C** (mục 6-10): tra cứu. Công thức, cấu hình, sự cố, nguồn.

Mỗi bước có bốn ô cố định:

| Ô | Nghĩa |
|---|---|
| **Làm gì** | Việc cụ thể |
| **Đầu ra** | File hoặc con số phải có sau khi xong |
| **Kiểm tra** | Dấu hiệu để biết bước đó đã đúng, trước khi đi tiếp |
| **Tra cứu** | Link cần dùng |

Nguyên tắc quan trọng nhất: **không đi tiếp khi ô "Kiểm tra" chưa đạt.** Lỗi ở bước sinh dữ liệu mà không phát hiện sẽ làm hỏng toàn bộ những gì đến sau.

---

# PHẦN A — HIỂU PROJECT

## 1. Project này làm gì, nói trong 10 dòng

Có một model to giải toán giỏi (**teacher**: Qwen2.5-14B). Có một model bé giải toán dở (**student**: Llama-3.2-1B). Ta bắt teacher giải 3.000 bài toán và ghi lại lời giải từng bước, rồi huấn luyện student bắt chước những lời giải đó. Student sẽ giỏi lên — điều này chắc chắn.

Câu hỏi là: **nó giỏi lên bằng cách học được cách suy nghĩ, hay bằng cách học thuộc một lối giải duy nhất?**

Ta đo bằng cách so **chính model đó trước và sau khi học**:

- **A** = Llama-3.2-1B nguyên bản, chưa động vào
- **B** = Llama-3.2-1B học (câu hỏi → đáp án), không có lời giải
- **C** = Llama-3.2-1B học (câu hỏi → lời giải đầy đủ → đáp án)

Giả thuyết **H1**: C chính xác hơn A, nhưng phân bố chiến lược giải của C hẹp hơn A. Tức là **giỏi hơn nhưng hẹp hơn**.

Nếu H1 sai — C vừa chính xác hơn vừa đa dạng ngang hoặc hơn A — thì đó **vẫn là kết quả hợp lệ** và phải báo cáo nguyên vẹn. Tương tự, nếu C **không** chính xác hơn A (negative transfer đã được ghi nhận trong literature, xem mục 10), đó cũng là kết quả chứ không phải lỗi. Project này không thất bại vì kết quả âm tính; nó chỉ thất bại nếu đo sai.

**Phân cấp bằng chứng — chốt ngay từ đầu, không đổi sau khi thấy kết quả:**

- **Bằng chứng định lượng chính**: pass@$k$ và entropy phân bố đáp án trên 200 câu GSM8K test (bước 7). Tự động hoàn toàn, đủ mẫu, không phụ thuộc codebook.
- **Phân tích khám phá**: phân bố chiến lược trên bộ bài tự soạn (bước S, 8, 12-14). Chỉ có 10 họ bài nên công suất thống kê thấp — chỉ phát hiện được hiệu ứng lớn. Kết quả không có ý nghĩa thống kê ở phần này **không** làm đổ báo cáo.

## 2. Ba thứ phải hiểu trước khi đọc các bước

### 2.1. Vì sao so với A chứ không so với teacher

Nếu so teacher với C rồi thấy C hẹp hơn, có hai cách giải thích: (i) distillation làm nó hẹp, hoặc (ii) model 1B vốn đã hẹp sẵn, chẳng liên quan gì tới distillation. Không phân biệt được.

So A với C thì loại được cách giải thích thứ hai, vì A và C là **cùng một model**, chỉ khác ở chỗ C đã qua huấn luyện. Teacher vẫn được đo, nhưng chỉ làm **mốc tham chiếu trần**, không phải nhóm đối chứng.

### 2.2. Vì sao cần điều kiện B

Khi C giỏi hơn A, có hai nguyên nhân khả dĩ: (i) nhờ được dạy suy luận, hoặc (ii) đơn giản vì C đã quen dạng đề GSM8K, quen cách trình bày, quen viết đáp án ở đâu — thứ mà **bất kỳ** lần fine-tune nào cũng cho.

B được huấn luyện trên **đúng những câu hỏi đó** nhưng chỉ thấy đáp án, không thấy lời giải. Nên B cũng quen dạng đề, mà không được dạy suy luận. Phần **C trừ B** mới là công của tín hiệu CoT.

Lưu ý: B chắc chắn thua C về accuracy, vì lúc infer B không sinh lời giải. Đó là **hệ quả hiển nhiên của thiết kế, không phải phát hiện**. Báo cáo không được trình bày nó như phát hiện.

B không tham gia phần phân tích chiến lược, vì output của nó chỉ là một con số, không có gì để gán nhãn.

### 2.2b. Điều kiện D (tuỳ chọn) — self-distillation

B không tách được hai khả năng: (i) **trace của teacher** làm student hẹp đi, hay (ii) **bất kỳ SFT nào trên lời giải đúng đã lọc** cũng làm hẹp đi, kể cả khi lời giải đến từ chính student. Và vì B không có lời giải, nó không giúp gì cho phần đa dạng.

- **D** = Llama-3.2-1B học trên **lời giải đúng do chính A sinh ra** (rejection sampling), cùng format, cùng cấu hình train với C.

Nếu D cũng hẹp như C → hiện tượng thuộc về SFT/lọc đáp án, không riêng gì distillation. Nếu D giữ được đa dạng còn C thì không → mới quy được cho tín hiệu của teacher. Đây là đối chứng mạnh nhất cho H1.

D **không bắt buộc**. Quyết định làm hay không đưa ra tại Cổng 2 (xem mục 3). Nếu làm, D lấy chỗ của B seed 1 và seed 2 để giữ ngân sách (chi tiết ở bước 9).

### 2.3. Hai chỉ số chính

**pass@k.** Sinh $n$ lời giải cho một bài, đếm $c$ lời giải đúng. Ước lượng không thiên lệch:

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

- **pass@1** = "cho một lần thử, model đúng bao nhiêu phần trăm" — chính là accuracy thông thường.
- **pass@20** = "cho thử 20 lần, có ít nhất một lần đúng bao nhiêu phần trăm" — tức **độ phủ**: model có *chạm được tới* lời giải đúng hay không, dù không chắc chọn đúng ngay.

Nếu C có pass@1 cao hơn A nhưng pass@20 **không** cao hơn: distillation không mở rộng vùng bài model giải được, nó chỉ làm model chắc tay hơn trong vùng cũ, đồng thời đánh mất những đường giải hiếm mà A thi thoảng mò ra. Đây là bằng chứng định lượng trực tiếp cho H1, và nó **không cần gán nhãn một chữ nào**.

Nguồn công thức: Chen et al., *Evaluating Large Language Models Trained on Code* (Codex), mục 2.1 — https://arxiv.org/abs/2107.03374

**Tỷ lệ tập trung chiến lược top-1.** Với mỗi bài, sinh 20 lời giải, gán nhãn chiến lược cho từng lời giải, rồi tính chiến lược phổ biến nhất chiếm bao nhiêu phần trăm. 20/20 nghĩa là model hoàn toàn cứng nhắc; 10/20 nghĩa là nó lưỡng lự giữa hai lối.

## 3. Năm cổng quyết định

Bốn trong năm cổng nằm ở cuối một bước cụ thể. Cổng là chỗ bạn dừng lại, nhìn bằng chứng, và quyết định có đi tiếp theo hướng cũ hay rẽ.

**Quyết định ở mỗi cổng phải làm ngay tại đó, không hoãn.** Các quyết định này chỉ có giá trị khi bạn còn thời gian. Hoãn tới lúc sắp hết hạn thì bạn sẽ chọn theo hoảng loạn.

| Cổng | Ở đâu | Câu hỏi | Nếu ĐẠT | Nếu KHÔNG ĐẠT |
|---|---|---|---|---|
| **0** | Trong lúc làm bước 2-4 | Được duyệt quyền truy cập Llama trên HF chưa? | Tiếp tục | Bước 2-4 vẫn làm được (teacher không bị gate). Nếu tới lúc cần train mà vẫn chưa duyệt → đổi student sang `Qwen2.5-1.5B-Instruct` (không gate) |
| **1** | Cuối bước 3 | Teacher 14B-AWQ serve được trên **2×T4 (tensor parallel)** với tốc độ chấp nhận được chưa? | Tiếp tục | Hạ ngay xuống `Qwen2.5-7B-Instruct-AWQ`. **Không cố sửa quá nửa ngày.** |
| **2** | Cuối bước 7 | Đã có accuracy A/C + đường cong pass@k chưa? | **Có project nộp được rồi.** Mọi bước sau là phần thêm. Quyết định luôn có làm điều kiện D (mục 2.2b) hay không | Dừng mọi kế hoạch mở rộng, dồn toàn bộ thời gian còn lại vào lõi |
| **3** | Cuối bước 8 | A có đủ đa dạng chiến lược để đo sự thu hẹp không? (ngưỡng ở bước 8) | Tiếp tục phân tích chiến lược như kế hoạch | **Không sửa bài để "tạo" đa dạng.** Ghi nhận đây là một phát hiện (1B vốn không có đa dạng để mất), hạ phần chiến lược xuống mô tả trên teacher và A, dồn thời gian vào bằng chứng chính (pass@k, entropy đáp án, transfer) |
| **4** | Cuối bước 12 | Cohen's kappa ≥ 0,6 chưa? | Giữ LLM-judge cho toàn bộ ~4.700 lời giải | **Không sửa codebook rồi chạy lại vòng hai.** Rút xuống 5 họ bài, gán nhãn tay **chỉ A và C seed 0**, $k = 10$ mỗi bài (~400 lời giải, ~3-4 giờ), bỏ judge, bỏ kappa |

Cổng 2 là cổng an toàn: qua được nó, bạn luôn có một project hoàn chỉnh trong tay bất kể chuyện gì xảy ra sau đó.

Cổng 3 tồn tại vì rủi ro lớn nhất về mặt khoa học của project: model 1B có thể gần như **luôn** tính tuần tự. Khi đó A đã tập trung ~100% vào một chiến lược, không còn chỗ để C hẹp hơn, và H1 không kiểm định được dù đúng hay sai. Phát hiện điều này ở bước 8 rẻ hơn rất nhiều so với phát hiện ở bước 13.

## 4. Bảng tra nhanh tài nguyên

### Model

| Vai trò | Tên trên HF | Link | Ghi chú |
|---|---|---|---|
| Teacher | `Qwen/Qwen2.5-14B-Instruct-AWQ` | https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ | ~10GB weights. Load được trên 1×T4 nhưng chỉ còn ~2-3GB KV cache → **chạy trên 2×T4, `tensor_parallel_size=2`** |
| Teacher dự phòng | `Qwen/Qwen2.5-7B-Instruct-AWQ` | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ | ~5GB |
| Student | `meta-llama/Llama-3.2-1B-Instruct` | https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct | **Bị gate — xin quyền ở bước 1** |
| Student dự phòng | `Qwen/Qwen2.5-1.5B-Instruct` | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct | Không gate |

### Dataset

| Tên | Link | Vai trò |
|---|---|---|
| GSM8K | https://huggingface.co/datasets/openai/gsm8k | Sinh data train + đo accuracy in-domain |
| SVAMP | https://huggingface.co/datasets/ChilleD/SVAMP | Chỉ đo transfer |
| MultiArith | https://huggingface.co/datasets/ChilleD/MultiArith | Chỉ đo transfer |

GSM8K dùng config `main`, có split `train` (7.473) và `test` (1.319). Đáp án chuẩn nằm sau dấu `####` trong trường `answer`.

MultiArith trên `ChilleD/MultiArith` chia 420/180. Plan này dùng **toàn bộ 600 câu** (gộp cả hai split) để có power thống kê tốt hơn.

### Thư viện và tài liệu

| Việc | Thư viện | Tài liệu |
|---|---|---|
| Sinh data, eval, sampling | vLLM | https://docs.vllm.ai/en/latest/ |
| Huấn luyện | TRL `SFTTrainer` (API 1.x — xem lưu ý ở bước 2 và mục 6.1) | https://huggingface.co/docs/trl/sft_trainer |
| LoRA | PEFT | https://huggingface.co/docs/peft/index |
| Nạp dataset | `datasets` | https://huggingface.co/docs/datasets/index |
| McNemar test | statsmodels | https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html |
| Wilson CI | statsmodels `proportion_confint` | https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html |
| Permutation test | scipy | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html |
| Bootstrap | scipy | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html |
| Cohen's kappa | scikit-learn | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html |

### Hạ tầng

| Việc | Link |
|---|---|
| Kaggle Notebooks (toàn bộ tác vụ GPU) | https://www.kaggle.com/code |
| Kaggle Secrets (cất HF token) | https://www.kaggle.com/docs/notebooks#adding-secrets |
| HF Hub, repo private (lưu checkpoint — dễ hơn Kaggle Dataset khi đang chạy dở) | https://huggingface.co/docs/huggingface_hub/guides/upload |
| Kaggle Datasets (lưu output cuối) | https://www.kaggle.com/docs/datasets |
| Tạo HF access token | https://huggingface.co/settings/tokens |

**Ràng buộc phần cứng**: T4 thuộc kiến trúc Turing (sm75), **không hỗ trợ bf16**. Mọi cấu hình train phải dùng `fp16` (AMP) kèm `gradient_checkpointing`. P100 (sm60) cũng vậy. T4 cũng không có kernel `awq_marlin` (cần sm80), vLLM sẽ dùng kernel AWQ thường — chậm hơn, đã tính vào ngân sách.

Kaggle cho ~30 giờ GPU mỗi tuần; mỗi session có giới hạn thời gian (kiểm tra con số hiện hành trong docs Kaggle). Tổng project ~22 giờ GPU (phụ lục), vừa quota nhưng **không dư nhiều nếu dồn vào một tuần** — nên trải qua ít nhất hai tuần.

**Chạy nền thay vì ngồi canh**: dùng **"Save Version → Save & Run All (Commit)"** cho mọi tác vụ dài (sinh dữ liệu, train, eval). Notebook commit chạy headless tới hết giới hạn session, không bị ngắt vì idle như session tương tác. Session tương tác chỉ dùng để debug.

**Máy local (RTX 1000 Ada 6GB, hỗ trợ bf16 + FlashAttention)**: dùng để **debug pipeline với model 1B** — eval harness, parser, format dữ liệu, train thử vài chục bước — trước khi đẩy lên Kaggle. Không chạy teacher 14B được. **Mọi lần train và eval chính thức chạy trên cùng một loại phần cứng (Kaggle T4)** để kết quả so sánh được.

---

# PHẦN B — CÁC BƯỚC

## 5. Sơ đồ phụ thuộc

```
1 ──► 2 ──► 3 [CỔNG 1] ──► 4 ──► 5 ──► 6 ──► 7 [CỔNG 2]
                                                  │
            └──► S (nhánh song song, không cần GPU)│
                          │                        │
                          └────────┬───────────────┘
                                   ▼
                                   8 [CỔNG 3] ──► 9 ──► 10 ──► 11
                                                              │
                                   ┌──────────────────────────┘
                                   ▼
                                  12 [CỔNG 4] ──► 13 ──► 14 ──► 15 ──► 16
```

**Nhánh S chạy song song.** Nó không cần GPU, nên làm được trong lúc bước 4, 6, 7 đang chạy máy. Điều kiện duy nhất: **S phải xong trước bước 8** — codebook, luật tính chỉ số và ngưỡng Cổng 3 phải được commit trước khi A sinh bất kỳ lời giải nào trên bộ bài tự soạn.

**Các bước bắt buộc theo thứ tự tuyệt đối**: 4 → 5 → 6 → 7 (mỗi bước ăn đầu ra của bước trước). Từ 8 trở đi cũng vậy.

---

## Bước 1 — Xin quyền truy cập và dựng repo

Làm đầu tiên, trước tất cả, vì phải chờ duyệt.

**Làm gì**

1. Vào https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct, đăng nhập, đọc và chấp nhận license, bấm xin quyền truy cập. Việc duyệt thường nhanh nhưng **không tức thì**.
2. Tạo access token tại https://huggingface.co/settings/tokens. Cần quyền **`write`** (fine-grained: đọc repo gated + ghi vào repo của bạn), vì checkpoint sẽ được push lên HF Hub.
3. Tạo sẵn một model repo **private** trên HF Hub để chứa checkpoint (vd `<user>/cot-distill-ckpt`). Derivative của Llama để private không vướng license; chỉ khi công khai mới cần tuân thủ mục 7.
4. Mở một Kaggle Notebook, vào Add-ons → Secrets, thêm secret tên `HF_TOKEN`.
5. Tạo repo Git, dựng sẵn cấu trúc thư mục (mục 9).

**Đầu ra**: repo có cấu trúc; secret đã cất trên Kaggle; repo checkpoint private trên HF.

**Kiểm tra**: chạy được đoạn này trong notebook mà không lỗi:

```python
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("HF_TOKEN")
print(tok[:6], "...")
```

**Tra cứu**: https://www.kaggle.com/docs/notebooks#adding-secrets

---

## Bước 2 — Dựng môi trường

**Làm gì**

1. Tạo notebook Kaggle, bật GPU (chọn T4 ×2).
2. Cài thư viện. Colab và Kaggle thay đổi image mặc định khá thường xuyên, nên sau khi tìm được tổ hợp chạy được, **pin version ngay** vào `requirements*.txt` và commit.

**Lưu ý về vLLM trên T4**: T4 không có FlashAttention-2, vLLM phải dùng backend attention khác. Các bản vLLM mới (0.29 tại thời điểm viết) đã bỏ engine V0 và hỗ trợ cho kiến trúc cũ ngày càng hẹp. **Đừng bắt đầu từ bản mới nhất rồi lùi dần** — mỗi vòng thử trên Kaggle tốn 10-20 phút. Thay vào đó:

1. Tìm trên Kaggle Code một notebook công khai gần đây chạy vLLM trên T4×2 (từ khoá `vllm T4`, `vllm AWQ T4`), lấy đúng tổ hợp `vllm` + `torch` + `transformers` của nó.
2. Cài đúng tổ hợp đó, chạy phần Kiểm tra dưới đây và bước 3.
3. Chạy được → pin ngay.

**Lưu ý về TRL**: TRL hiện ở bản 1.x. Nhiều hướng dẫn cũ trên mạng **không còn chạy**:

- `DataCollatorForCompletionOnlyLM` **đã bị xoá**.
- `max_seq_length` trong `SFTConfig` **đã đổi thành `max_length`**.
- `assistant_only_loss=True` đòi chat template có thẻ `{% generation %}`; template Llama 3.2 **không có** → báo lỗi.
- Cách đúng: dataset dạng **prompt–completion** (mục 6.1), loss chỉ tính trên completion.

Vì vLLM và TRL có thể kéo các bản `torch`/`transformers` khác nhau, nếu xung đột không giải được thì tách thành **hai notebook với hai môi trường**: một cho sinh/eval (vLLM), một cho train (TRL + PEFT). Pin riêng thành `requirements-infer.txt` và `requirements-train.txt`.

**Đầu ra**: `requirements*.txt` đã pin, commit vào repo.

**Kiểm tra**

```python
import torch, vllm, transformers, peft, trl
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))   # kỳ vọng (7, 5) cho T4
print(vllm.__version__, transformers.__version__, peft.__version__, trl.__version__)
print(torch.cuda.device_count())             # kỳ vọng 2 khi chọn T4 ×2
```

Nếu đã tách hai môi trường, chạy phần import `vllm` trong notebook sinh/eval và phần import `peft, trl` trong notebook train.

**Tra cứu**: https://docs.vllm.ai/en/latest/getting_started/installation.html

---

## Bước 3 — Smoke test teacher ⟶ **CỔNG 1**

**Làm gì**

1. Load `Qwen/Qwen2.5-14B-Instruct-AWQ` bằng vLLM trên **2×T4**:

```python
from vllm import LLM
llm = LLM(
    model="Qwen/Qwen2.5-14B-Instruct-AWQ",
    quantization="awq",
    dtype="float16",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    enable_prefix_caching=True,   # bắt buộc khi làm judge ở bước 12 — codebook là prefix chung
    seed=0,
)
```

2. Sinh thử **100** lời giải cho 100 bài GSM8K train bất kỳ, gửi **một lần dưới dạng batch** (không lặp từng câu). 10 bài quá ít để đo throughput khi batch.
3. Bấm giờ, ghi throughput (token output/giây) từ log vLLM. Nhân lên để ước tính 3.300 bài.
4. Đọc bằng mắt 10 lời giải: mạch lạc không, dài bao nhiêu token, có kết thúc bằng `Answer: <number>` như prompt yêu cầu không.

**Prompt teacher** (dùng nguyên ở bước 4, commit vào `configs/gen.yaml`):

```
Solve the following math problem. Show your reasoning step by step
in plain text (no LaTeX, no \boxed{}), then end your response with
"Answer: <number>" on its own line.

Problem: {question}
```

Qwen2.5 mặc định hay viết LaTeX và `\boxed{}`. Nếu teacher vẫn viết LaTeX dù đã dặn, chấp nhận — nhưng parser phải xử lý được `\boxed{}` và báo cáo phải ghi rõ student học văn phong đó.

**Cấu hình sinh cho teacher**:

```python
temperature = 0.7
top_p       = 0.95
max_tokens  = 1024
```

`max_tokens = 1024` chứ không phải 512: nếu trace bị cắt cụt ở 512 token, bài đó rơi vào nhóm "sai" và bị lọc bỏ, tập train sẽ **lệch về lời giải ngắn** một cách không chủ đích. Sinh rộng rồi lọc theo độ dài một cách có chủ đích ở bước 4.

(Qwen2.5 không có chế độ thinking — không cần tắt gì.)

**Đầu ra**: 100 lời giải mẫu; throughput; con số ước tính thời gian cho 3.300 bài.

**Kiểm tra — CỔNG 1**

- [ ] Model load được trên 2×T4, không OOM. Log vLLM báo đủ KV cache cho ít nhất ~30 chuỗi đồng thời ở độ dài ~1.500 token
- [ ] 10 lời giải đọc tay mạch lạc, có `Answer:` rõ
- [ ] Ước tính cho 3.300 bài **dưới 4 giờ**

Nếu chỉ có 1×T4 (tensor parallel lỗi): vẫn chạy được với `max_model_len=2048`, nhưng KV cache chỉ ~2-3GB (~15 chuỗi đồng thời), throughput khoảng 80-150 token/giây → 3.300 bài mất 2-4 giờ, sát ngưỡng.

Nếu một trong ba mục không đạt: **hạ xuống `Qwen/Qwen2.5-7B-Instruct-AWQ` ngay, không cố sửa quá nửa ngày.** Teacher 7B đạt ~91,6% GSM8K so với ~93-94% của 14B; khoảng cách với student (~44,4%) vẫn còn ~47 điểm, thừa rộng.

**Tra cứu**
- https://docs.vllm.ai/en/latest/getting_started/quickstart.html
- https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ

---

## Bước 4 — Sinh dữ liệu huấn luyện

**Làm gì**

1. Nạp GSM8K train (7.473 bài). Xáo với seed cố định.
2. **Tách dev split 200 câu trước tiên.** Cất riêng, không đụng tới.
3. Lấy 3.300 câu từ phần còn lại.
4. Cho teacher sinh **một** lời giải cho mỗi câu, ở `temperature = 0.7`, dùng prompt teacher ở bước 3. Chạy dưới dạng notebook **commit** (chạy nền), ghi `gen_raw.jsonl` theo từng lô 500 câu để nếu session chết thì không mất hết.
5. Trích đáp án, so với đáp án chuẩn (phần sau `####`). Giữ bài đúng, vứt bài sai. Kỳ vọng còn ~3.000.
6. **Làm sạch trace**: cắt bỏ dòng `Answer: ...` cuối cùng (và `\boxed{}` nếu có) khỏi output của teacher, lưu phần còn lại vào trường `cot`. Lúc train sẽ ghép lại `f"{cot}\nAnswer: {gold}"` (bước 6) — nếu không cắt, mỗi mẫu train có **hai** dòng đáp án.
7. **Lọc độ dài có chủ đích**: bỏ các mẫu mà prompt + `cot` + dòng đáp án vượt `max_length = 1024` token theo tokenizer **của student** (Llama). Ghi lại số mẫu bị bỏ.
8. Tạo bản answer-only bằng cách cắt phần lời giải khỏi **đúng tập đã lọc đó**.

**Vì sao `temperature = 0.7` chứ không phải 0** — điểm dễ làm sai nhất trong cả project. Sinh ở temperature 0 thì mỗi câu chỉ ra đúng một lời giải khả dĩ nhất, tức **mode collapse đã bị nhốt sẵn vào tập train ngay từ khâu tạo dữ liệu**. Khi đó dù có quan sát thấy C hẹp hơn A, cũng không quy kết được cho SFT, vì nguyên nhân có thể chỉ nằm ở cách bạn curate dữ liệu. Cả kết luận của project sẽ vô giá trị.

**Vì sao phải tách dev TRƯỚC** — nếu bạn thử cả 2 và 3 epoch rồi chọn cái nào cho điểm cao trên *test set*, bạn đã lén dùng test set để tối ưu, và mọi con số accuracy báo cáo bị thổi phồng.

**Đầu ra** (JSONL, một object mỗi dòng)

- `data/train_cot.jsonl` — `question`, `cot`, `gold_answer`, `extracted_answer`
- `data/train_answer_only.jsonl` — `question`, `gold_answer`
- `data/dev.jsonl` — 200 câu
- `data/gen_raw.jsonl` — **toàn bộ generation thô kể cả bài bị lọc bỏ**

**Kiểm tra**

- [ ] Tỷ lệ lọc ~90% trở lên (3.300 → ~3.000). Thấp hơn nhiều → teacher hoặc parser có vấn đề, dừng lại kiểm tra
- [ ] `train_cot.jsonl` và `train_answer_only.jsonl` cùng số dòng, cùng bộ `question`
- [ ] `set(dev.question) & set(train.question) == set()`
- [ ] Không mẫu nào trong `train_cot.jsonl` còn chứa `Answer:` hoặc `\boxed` trong trường `cot`
- [ ] Tỷ lệ bị cắt cụt (chạm `max_tokens = 1024`) dưới 1%
- [ ] Ghi lại phân bố độ dài trace (trung vị, p95). Qwen2.5-14B trên GSM8K thường ra khoảng 200-400 token. Đây là **số để báo cáo**, không phải điều kiện dừng — chỉ dừng lại nếu trung vị bất thường (dưới 80 hoặc trên 800 token)

**Tra cứu**: https://huggingface.co/datasets/openai/gsm8k

---

## Bước 5 — Dựng eval harness và chạy điều kiện A

**Làm gì**

1. Viết hàm trích đáp án theo **luật cứng** (mục 6.3).
2. Viết eval harness: nạp model bằng vLLM → sinh greedy → trích đáp án → so gold → trả mảng đúng/sai theo từng câu.
3. Chạy A trên GSM8K test full (1.319 câu).
4. Chạy thêm A với prompt 8-shot kinh điển làm **sanity check**.

**Prompt protocol** — phải thống nhất giữa các điều kiện, nếu không phép so sánh mất giá trị:

- **Kết quả chính**: chat template chuẩn của Llama 3.2, **zero-shot, hoàn toàn giống nhau cho A, B và C**.
- Prompt phải yêu cầu rõ định dạng đáp án, vì A chưa được train theo format này:

```
Solve the following math problem. Show your reasoning step by step,
then end your response with "Answer: <number>".

Problem: {question}
```

- **Sanity check**: A với prompt 8-shot GSM8K kinh điển, để xác nhận A không bị thiệt thòi khi phải theo format của model đã fine-tune. Nếu A (8-shot) cao hơn hẳn A (zero-shot), **cả hai con số đều phải báo cáo**.

**Cố định ngày trong chat template — bắt buộc.** Template Llama 3.2 tự chèn dòng `Today Date: ...` vào system header bằng `strftime_now`, tức là **ngày chạy máy**. Train hôm nay, eval tuần sau → prompt khác nhau, và không ai tái lập được đúng prompt. Luôn truyền ngày cố định ở **mọi** chỗ áp template (train, eval, sampling, cả HF lẫn vLLM):

```python
DATE_STRING = "26 Jul 2024"   # commit vào configs/gen.yaml

def render_prompt(tok, question):
    messages = [{"role": "user", "content": PROMPT.format(question=question)}]
    return tok.apply_chat_template(messages, add_generation_prompt=True,
                                   tokenize=False, date_string=DATE_STRING)

# vLLM: KHÔNG dùng llm.chat — tự render bằng hàm trên rồi đưa text vào llm.generate
outputs = llm.generate([render_prompt(tok, q) for q in questions], sampling_params)
```

Một hàm `render_prompt` duy nhất, dùng chung cho eval, sampling và (phần prompt của) dữ liệu train. Kiểm tra bằng cách in một prompt đã render và xác nhận có dòng `Today Date: 26 Jul 2024`.

**Decode cho eval accuracy**: `temperature = 0`, `max_tokens = 1024`. Ghi lại tỷ lệ output bị cắt cụt (chạm `max_tokens`) cho từng model.

**Đầu ra**
- `results/eval_A_gsm8k.jsonl` — `question_id`, `output`, `extracted`, `gold`, `correct`
- Một con số accuracy kèm Wilson CI

**Kiểm tra**

- [ ] Accuracy của A rơi vào **35-50%**. Model card Meta công bố ~44,4% (8-shot CoT); zero-shot thấp hơn là bình thường. Dưới 20% → gần như chắc chắn lỗi prompt hoặc lỗi parser, **không phải model dở**. Dừng lại sửa
- [ ] Tỷ lệ trích xuất thất bại dưới 5%
- [ ] Prompt đã render có đúng `Today Date: 26 Jul 2024`, không phải ngày chạy máy

**Mẹo**: viết và debug toàn bộ harness trên **máy local** (Llama 1B chạy thoải mái trong 6GB) với 50 câu, rồi mới đẩy lên Kaggle chạy full.

**Tra cứu**: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct (phần Evaluation results)

---

## Bước 6 — Train C, seed 0

**Làm gì**

1. Định dạng `train_cot.jsonl` thành dạng **prompt–completion** (bên dưới).
2. **Debug trên máy local trước**: train 50 bước với 200 mẫu, xác nhận loss giảm, mask đúng, không lỗi API. Rồi mới đẩy lên Kaggle.
3. Train LoRA với cấu hình ở mục 6.1, chạy dạng notebook **commit**.
4. **Push adapter lên repo HF private sau mỗi epoch** (`push_to_hub` / `hub_strategy="every_save"`), để không mất tiến độ khi session bị ngắt.
5. Đánh giá từng checkpoint trên dev 200 câu, chọn epoch tốt nhất. Dev 200 câu nhiễu (±7 điểm): nếu các epoch chênh nhau dưới 3 điểm, **chọn epoch 2** theo luật cố định này thay vì chạy theo nhiễu. Ghi luật vào `configs/train_lora.yaml` trước khi train.
6. Merge LoRA vào base weights.

**Định dạng dữ liệu — prompt–completion, prompt đã render sẵn**

```python
{
    "prompt":     render_prompt(tok, q),          # hàm ở bước 5, có DATE_STRING cố định
    "completion": f"{cot}\nAnswer: {gold}<|eot_id|>",
}
```

Vì sao render sẵn thay vì đưa `messages` cho TRL: nếu để TRL tự áp chat template, nó gọi `apply_chat_template` **không có** `date_string` → dòng `Today Date` lại là ngày chạy máy (bước 5). Render sẵn bằng cùng hàm `render_prompt` đảm bảo prompt lúc train giống hệt lúc eval.

Phải dùng **cùng một `PROMPT`** với lúc eval ở bước 5. Nếu lệch, B và C được train trên một format rồi lại bị đo trên format khác.

**Quan trọng — chỉ tính loss trên phần completion.** Nếu tính loss cả phần câu hỏi, model sẽ học cách *sinh ra đề bài* thay vì học cách giải. Với dataset prompt–completion, TRL 1.x chỉ tính loss trên completion (`completion_only_loss`, mặc định bật cho dạng này). **Không** dùng `assistant_only_loss` (template Llama 3.2 không hỗ trợ) và **không** tìm `DataCollatorForCompletionOnlyLM` (đã bị xoá).

**Hai lỗi token đặc biệt phải tự kiểm tra** (hành vi khác nhau giữa các bản TRL):

- **BOS lặp**: prompt đã render bắt đầu bằng `<|begin_of_text|>`; nếu tokenizer lại tự thêm BOS thì mẫu có hai BOS. Nếu thấy, cắt `<|begin_of_text|>` khỏi chuỗi prompt.
- **EOS**: completion phải kết thúc bằng **đúng một** `<|eot_id|>`. Thiếu → model không biết dừng, sinh lan man tới `max_tokens`. Thừa → cũng sai.

**Đầu ra**: adapter từng epoch trên HF Hub, `checkpoints/C_seed0/` (đã merge), log training.

**Kiểm tra**

- [ ] Decode `input_ids` của 2 mẫu đầu tiên sau khi collate: đúng một BOS, prompt có `Today Date: 26 Jul 2024`, completion kết thúc bằng đúng một `<|eot_id|>`
- [ ] In mask của 1 mẫu: nhãn phần prompt là `-100`, chỉ phần completion được tính loss
- [ ] Loss giảm đều, không NaN. NaN → xem mục 8, gần như luôn là vấn đề fp16
- [ ] Sinh thử 3 câu bằng C, đọc bằng mắt: có lời giải từng bước, kết thúc bằng `Answer: <number>` và **dừng lại** ngay sau đó
- [ ] Tỷ lệ trích xuất được đáp án trên dev của C ≥ 95%

**Không** đặt "C cao hơn A trên dev" làm điều kiện kiểm tra. Đó là **kết quả cần đo**, không phải dấu hiệu pipeline chạy đúng: Llama-3.2-1B-Instruct đã qua post-training nặng, SFT trên ~3.000 trace có thể chỉ tăng vài điểm hoặc không tăng (negative transfer đã được ghi nhận, mục 10). Hơn nữa dev 200 câu có CI khoảng ±7 điểm, không đủ để thấy "rõ rệt". Chỉ nghi ngờ lỗi khi C **tụt mạnh** so với A (vd hơn 15 điểm) — lúc đó đi theo bảng sự cố ở mục 8.

**Tra cứu**
- https://huggingface.co/docs/trl/sft_trainer
- https://huggingface.co/docs/peft/developer_guides/lora
- https://huggingface.co/docs/peft/developer_guides/lora#merge-adapters

---

## Bước 7 — Eval C, pass@k, entropy đáp án ⟶ **CỔNG 2**

**Làm gì**

1. Chạy C (seed 0) trên GSM8K test full. Accuracy + Wilson CI.
2. **McNemar exact test** giữa A và C trên cùng bộ câu.
3. Lấy 200 câu cố định từ GSM8K test. Với cả A và C, sinh **20 lời giải mỗi câu** ở `temperature = 0.7`, `top_p = 0.95`.
4. Tính pass@$k$ cho $k = 1, 2, 5, 10, 20$. Vẽ hai đường cong trên cùng một hình.
5. Tính entropy phân bố đáp án (mục 6.5).

**Về McNemar**: khi so hai model trên **cùng bộ đề**, bạn không chỉ có hai con số tổng, mà có từng câu. McNemar chỉ nhìn vào hai ô lệch — số câu A đúng C sai, và số câu A sai C đúng — rồi trả lời "chênh lệch này có khả năng là ngẫu nhiên không?". Kết quả là giá trị $p$; quy ước $p < 0{,}05$ thì coi là có ý nghĩa.

**Đầu ra**
- `results/eval_C_seed0_gsm8k.jsonl`
- `results/samples_A_k20.jsonl`, `results/samples_C_seed0_k20.jsonl`
- `figures/passk_A_vs_C.png` ← **hình chủ đạo của báo cáo**
- Bảng: accuracy A/C + CI + $p$-value

**Kiểm tra — CỔNG 2**

- [ ] Có accuracy của A và C kèm CI và $p$-value
- [ ] Có đường cong pass@$k$ của cả hai
- [ ] pass@1 tính từ sampling **xấp xỉ** accuracy greedy (không cần bằng hệt, nhưng lệch quá 10 điểm là dấu hiệu có lỗi)

**Qua được cổng 2, bạn đã có một project hoàn chỉnh và nộp được.** Toàn bộ phần còn lại là mở rộng. Nếu bất cứ lúc nào sau đây mọi thứ đổ vỡ, bạn vẫn viết được báo cáo từ những gì có tại điểm này.

**Quyết định tại cổng 2 — có làm điều kiện D (mục 2.2b) không?** Làm nếu cả hai điều sau đúng:

- Còn ít nhất ~3 giờ GPU dư so với ngân sách ở phụ lục
- Bước S đang đúng tiến độ (codebook sắp chốt)

Nếu làm, thêm vào bước 9 (xem ở đó). Nếu không, ghi D vào mục "hướng mở rộng" của báo cáo.

**Lưu ý khi đọc đường cong pass@$k$**: cùng `temperature = 0.7`, nhưng SFT làm phân bố xác suất của C nhọn hơn A, nên độ ngẫu nhiên *thực tế* khi sampling không bằng nhau. Một phần chênh lệch pass@20 có thể đến từ đây chứ không phải từ việc "mất đường giải". Nếu còn ngân sách, sample thêm A và C ở `temperature = 1.0` trên cùng 200 câu (~20 phút GPU) và báo cáo cả hai; nếu không, ghi vào mục hạn chế.

**Tra cứu**
- https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html
- https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
- https://arxiv.org/abs/2107.03374

---

## Bước S — Soạn bộ bài và codebook (nhánh song song, không cần GPU)

Bắt đầu được ngay sau bước 2. Làm trong lúc bước 4, 6, 7 đang chạy máy. **Phải xong trước bước 8.**

Chính vì nhánh này chạy song song được nên bản plan đầy đủ mới khả thi.

**Làm gì**

1. **Soạn 10 "họ bài toán", mỗi họ 3-4 biến thể số liệu, tổng ~40 bài.** Mỗi họ phải được thiết kế sao cho **tồn tại ít nhất hai chiến lược giải tự nhiên**. Ví dụ: lập phương trình so với tính tuần tự; dùng tỷ lệ và mẹo tắt so với làm thẳng theo trình tự. Bài chỉ có một cách giải thì không đo được đa dạng.

2. **Soạn 7 biến thể đảo chiến lược.** Mỗi biến thể lấy một họ bài và đổi số liệu sao cho **chiến lược tối ưu đổi hẳn**. Ví dụ: họ bài vốn nên lập phương trình, nay đổi số sao cho nhẩm ra ngay còn lập phương trình thành lòng vòng.

3. **Viết codebook.** Với mỗi họ bài:
   - 2-3 nhãn chiến lược cụ thể, kèm tiêu chí nhận diện và ví dụ mẫu
   - nhãn `other` cho lời giải hợp lệ nhưng không khớp nhãn nào
   - nhãn `unparseable` cho lời giải hỏng hoặc không có reasoning

4. **CHỐT codebook trước khi xem bất kỳ output nào của model.** Bắt buộc, không phải khuyến nghị. Nếu định nghĩa nhãn sau khi đã đọc output, bạn sẽ vô thức uốn nhãn theo kết quả mình mong đợi. Cách đảm bảo: commit codebook vào Git với timestamp, **trước** commit đầu tiên chứa output sampling.

   **Cùng commit đó, chốt luôn `docs/analysis_plan.md`** — mọi quyết định phân tích có thể bị uốn theo kết quả:
   - Luật xử lý `unparseable` và luật cân bằng cỡ mẫu (bước 13)
   - Có tính cả lời giải sai đáp án hay chỉ lời giải đúng (mặc định: **tính mọi lời giải hợp lệ**, vì ta đo *lựa chọn chiến lược* chứ không đo độ chính xác; phân tích chỉ-lời-giải-đúng là phân tích độ nhạy)
   - Thống kê kiểm định chính và cách hoán vị (bước 14)
   - Ngưỡng Cổng 3 (bước 8)
   - Danh sách 5 họ bài giữ lại nếu Cổng 4 không đạt (bước 12)
   - Phân cấp bằng chứng chính / khám phá (mục 1)

5. **Chạy decontamination check**: đo n-gram overlap giữa 40 bài tự soạn và GSM8K train. Nếu trùng, model đã thấy bài đó lúc train, kết quả vô nghĩa.

**Gợi ý mạnh — đừng định nghĩa nhãn từ con số không.** Có sẵn hai khung đã được hiệu chỉnh theo phán đoán của người:

- *Are We Measuring Strategy or Phrasing?* — định nghĩa **approach-level diversity** (khác biệt về chiến lược nền tảng) so với **surface-level diversity** (khác biệt về từ ngữ, ký hiệu, định dạng), và liệt kê các chiều để phân biệt: công cụ toán học được dùng, cách thiết lập cấu trúc bài toán, và các chiều khác. https://arxiv.org/abs/2606.29985
- *Beyond Accuracy: Evaluating Strategy Diversity in LLM Mathematical Reasoning* — có sẵn prompt phân loại chiến lược, dùng được gần như nguyên văn. https://arxiv.org/abs/2605.09292

Neo codebook vào định nghĩa có sẵn vừa tiết kiệm thời gian, vừa khiến kappa dễ đạt ≥ 0,6, vừa giúp kết quả so sánh được với họ.

**Đầu ra**
- `data/strategy_families.jsonl` — 40 bài: `family_id`, `variant_id`, `question`, `gold_answer`
- `data/strategy_flip.jsonl` — 7 biến thể đảo
- `docs/codebook.md` — đã commit và khoá
- `docs/analysis_plan.md` — đã commit và khoá, cùng commit với codebook
- `src/judge_label.py` + prompt judge — viết xong ở đây vì Cổng 3 (bước 8) cần dùng
- `results/decontamination.json`

**Kiểm tra**

- [ ] Mỗi bài đã tự giải tay và xác nhận đáp án
- [ ] Mỗi họ thật sự có ≥ 2 cách giải — tự viết ra cả hai cách cho ít nhất một biến thể mỗi họ
- [ ] Codebook và `analysis_plan.md` đã commit trước mọi output sampling trên bộ bài tự soạn
- [ ] Không bài nào trùng GSM8K train

---

## Bước 8 — Hiệu chỉnh độ khó bộ bài ⟶ **CỔNG 3**

Cần cả bước 5 (đã có A chạy được) và bước S (đã có bộ bài, codebook và `analysis_plan.md` đã khoá).

### 8a. Hiệu chỉnh độ khó

1. Chạy A trên 40 bài tự soạn, greedy, đo tỷ lệ giải đúng.
2. **Mục tiêu: A đạt khoảng 50-70%.**
3. Thấp hơn → đơn giản hoá số liệu. Cao hơn → tăng số bước.
4. Lặp cho tới khi vào khoảng.

**Vì sao bắt buộc**: student chỉ 1B. Bộ bài quá khó thì phần lớn lời giải rơi vào nhãn `unparseable` và phép so sánh phân bố mất ý nghĩa. Quá dễ thì bài tầm thường, mọi model đều làm giống nhau.

**Ràng buộc tuyệt đối**: việc hiệu chỉnh chỉ đụng tới **độ khó**, không bao giờ đụng tới **cấu trúc chiến lược** đã định nghĩa trong codebook. Nếu bạn thấy mình đang sửa bài để nó ra chiến lược mình muốn — dừng lại, đó là uốn dữ liệu.

**Phải xong trước bước 12**, để cả ba nhóm cùng sample trên đúng một bộ bài cuối cùng.

### 8b. Kiểm tra sàn đa dạng của A — CỔNG 3

Làm **ngay sau** khi bộ bài đã chốt ở 8a, **trước** khi train thêm bất kỳ model nào ở bước 9.

1. Sample A $k = 20$ mỗi bài trên toàn bộ 40 bài cuối cùng + 7 biến thể đảo, `temperature = 0.7`, `top_p = 0.95` — **đúng cấu hình bước 12**. Lưu thành `results/samples_strategy_A.jsonl`; đây chính là mẫu A của bước 12, không phải sinh lại.
2. Gán nhãn **tạm** bằng judge (`src/judge_label.py`, prompt đã khoá). Nhãn này chưa được kiểm chứng bằng kappa, chỉ dùng cho quyết định ở cổng này. Ở bước 12 judge chạy lại cùng cấu hình (greedy) nên nhãn cuối sẽ trùng.
3. Tính top-1 concentration theo đúng luật ở bước 13 (chỉ lời giải hợp lệ, $n_{\text{valid}} \geq 10$).
4. Đọc tay **10 lời giải của 2 họ bài** để xác nhận judge không gán nhãn vô lý. Ghi `sample_id` của 20 lời giải này; chúng **bị loại** khỏi mẫu gán tay có ẩn nguồn ở bước 12, vì bạn đã biết chúng là của A.

**Ngưỡng (chốt sẵn trong `analysis_plan.md`)**: ĐẠT nếu **ít nhất 5/10 họ bài** có top-1 concentration trung bình của A **≤ 85%**. Tức là ở ít nhất một nửa số họ, A thường xuyên dùng hơn một chiến lược — còn chỗ để đo sự thu hẹp.

**Nếu KHÔNG ĐẠT** (xem bảng cổng ở mục 3):

- **Không** sửa bài, không đổi codebook, không tăng temperature để "tạo ra" đa dạng. Mọi thay đổi như vậy sau khi đã thấy output của A là uốn dữ liệu.
- Ghi nhận như một phát hiện: model 1B gần như không có đa dạng chiến lược để mất, nên câu hỏi "distillation có làm hẹp không" không đặt ra được ở quy mô này.
- Phần chiến lược chỉ còn mô tả: teacher vs A (teacher có đa dạng không?) và C seed 0 để minh hoạ. Bỏ C seed 1-2 ở bước 12, bỏ bước 14 phần chiến lược.
- Dồn thời gian dư vào bằng chứng chính: pass@$k$ ở hai temperature, entropy đáp án, transfer, và điều kiện D nếu đã chọn.

**Đầu ra**: `data/strategy_families_final.jsonl`, kèm log các vòng hiệu chỉnh; `results/samples_strategy_A.jsonl`; `results/gate3.json` (top-1 từng họ + quyết định).

**Kiểm tra**

- [ ] A đạt 50-70% (greedy) trên bộ bài cuối cùng
- [ ] Cổng 3 đã quyết định và ghi vào `results/gate3.json`, **có commit** trước khi bắt đầu bước 9

---

## Bước 9 — Train 5 lần còn lại

**Làm gì**

Cùng bộ hyperparameter, khác seed:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **B** (answer-only) | train | train | train |
| **C** (CoT) | ✓ đã xong ở bước 6 | train | train |

Mỗi lần C ~1 giờ; B nhanh hơn nhiều vì completion chỉ vài token. Merge từng cái. Chọn epoch theo dev cho từng lần, cùng luật cố định ở bước 6.

**Chạy song song trên 2×T4**: model 1B vừa thoải mái một T4, nên mỗi GPU chạy một lần train riêng (hai tiến trình, `CUDA_VISIBLE_DEVICES=0` và `=1`), **không** dùng DDP. Ví dụ: GPU0 chạy C seed 1 → C seed 2, GPU1 chạy B seed 0 → 1 → 2. Thời gian chờ giảm khoảng một nửa. Mỗi tiến trình ghi log và push checkpoint vào thư mục riêng.

**Nếu đã chọn làm điều kiện D tại Cổng 2** (mục 2.2b):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **B** (answer-only) | train | — | — |
| **C** (CoT) | ✓ đã xong ở bước 6 | train | train |
| **D** (self-distill) | train | — | — |

B chỉ còn 1 seed — mất mean ± std của B, nhưng B không phải điều kiện của giả thuyết chính nên chấp nhận được. Ghi rõ trong báo cáo.

Chuẩn bị dữ liệu D (trước khi train, ~0,5 giờ GPU với vLLM):

1. Với **đúng 3.300 câu** đã dùng ở bước 4, cho A sinh tối đa **4 lời giải** mỗi câu (`temperature = 0.7`, `top_p = 0.95`, cùng prompt eval).
2. Với mỗi câu, giữ **lời giải đúng đầu tiên**; câu nào cả 4 lần đều sai thì bỏ.
3. Làm sạch trace và lọc độ dài giống bước 4. Lưu `data/train_selfdistill.jsonl`.
4. Ghi lại độ phủ (số câu còn lại / 3.300). A chỉ giải được ~40% ở lần đầu nên D sẽ có **ít câu hơn và câu dễ hơn** C — đây là confound phải ghi vào mục hạn chế. Không cố cân bằng bằng cách cắt tập của C, vì C seed 0 đã train xong.

**Vì sao 3 seed**: seed là con số khởi tạo bộ sinh ngẫu nhiên. Cùng cấu hình, khác seed, kết quả lệch nhau vài phần trăm. Nếu chỉ train một lần và thấy C đạt 41,2% còn B đạt 38,7%, bạn không biết chênh 2,5 điểm là thật hay may rủi. Chạy 3 seed rồi báo cáo $41{,}2 \pm 0{,}4$ so với $38{,}7 \pm 0{,}5$ thì mới nói được.

**Đầu ra**: 5 checkpoint đã merge (hoặc 4 + D nếu làm D; khi đó thêm `data/train_selfdistill.jsonl`).

**Kiểm tra**: cả 5 train xong không NaN; accuracy dev của mỗi cái nằm trong khoảng hợp lý so với cái cùng điều kiện; các kiểm tra token đặc biệt ở bước 6 đạt cho từng loại dữ liệu (B, D).

---

## Bước 10 — Eval toàn bộ

**Làm gì**

1. Chạy cả 7 model (A, B×3, C×3 — hoặc A, B, C×3, D) trên GSM8K test full (1.319), SVAMP (1.000), MultiArith (600). Hai GPU chạy hai model song song.
2. Tổng ~20.000 lượt sinh. Đây là lý do bắt buộc dùng vLLM — với `transformers.generate` trên T4 sẽ mất hàng chục giờ và buộc phải cắt test set.
3. Mean ± std qua 3 seed cho B và C (chỉ C nếu làm D), kèm Wilson CI.
4. McNemar cho cặp A–C và B–C (dùng seed 0); thêm A–D và C–D nếu có D.
5. **Lưu toàn bộ output thô** của mọi model — bước 11 sẽ trích xuất lại từ đây mà không cần sinh lại.

**Vì sao có SVAMP và MultiArith**: hai bộ này không tham gia train. Chúng đo xem model có **chuyển giao** được kỹ năng sang dạng bài lạ hay chỉ học vẹt GSM8K.

**Đầu ra**: `results/eval_all.jsonl`, bảng accuracy đầy đủ.

**Kiểm tra**

- [ ] B thua C rõ rệt (dự kiến — B không sinh CoT lúc infer). Nếu B ≈ C thì gần như chắc có lỗi (B đang sinh CoT, hoặc C không sinh)
- [ ] Accuracy của A trên GSM8K khớp với bước 5 (cùng harness, cùng output greedy — phải trùng hoàn toàn hoặc gần như vậy)
- [ ] Tỷ lệ output bị cắt cụt ở `max_tokens` được ghi riêng cho từng model
- [ ] Thời gian chạy quanh 1,5 giờ. Vượt xa → kiểm tra vLLM có batch đúng không

"C hơn A" **không** phải điều kiện kiểm tra — xem lý do ở bước 6.

---

## Bước 11 — Audit trích xuất

**Làm gì**

1. Lọc ra các ca trích xuất thất bại (không moi được số nào, hoặc parser trả giá trị lạ).
2. **Đọc tay 50 ca**, phân loại: model thật sự sai, hay parser sai.
3. Nếu parser sai ở tỷ lệ đáng kể → sửa luật trích xuất và **trích xuất lại** trên output thô đã lưu của toàn bộ 7 model (vài giây, không cần GPU). **Không sinh lại** — output greedy không đổi, chỉ luật đọc nó thay đổi. Ghi lại cả số liệu trước và sau khi sửa parser.
4. Báo cáo tỷ lệ `unparseable` **riêng cho từng nhóm**. Nếu tỷ lệ này lệch nhau giữa A và C thì bản thân điều đó đã là một phát hiện cần bàn.

**Vì sao đừng bỏ bước này**: nếu parser hỏng ở 3% số ca, accuracy lệch 3 điểm — đủ để đảo ngược kết luận. Bước này thường bị bỏ qua và là nguồn sai lệch âm thầm phổ biến.

**Đầu ra**: `results/extraction_audit.md` — 50 ca kèm phán quyết.

**Kiểm tra**: trên 50 ca, tỷ lệ do parser sai dưới 10%.

---

## Bước 12 — Sampling và gán nhãn ⟶ **CỔNG 4**

**Làm gì**

1. **Sampling**, `temperature = 0.7`, `top_p = 0.95`, $k = 20$ mỗi bài, trên **40 bài họ chiến lược + 7 biến thể đảo = 47 bài**:

| Nhóm | Số model | Họ chiến lược (40 bài) | Bộ đảo (7 bài) | Tổng |
|---|---|---|---|---|
| Teacher (Qwen2.5-14B) | 1 | 800 | 140 | 940 |
| A (Llama-3.2-1B gốc) | 1 | 800 | 140 | 940 — **đã có từ bước 8b** |
| C (đã CoT-distill) | 3 seed | 2.400 | 420 | 2.820 |
| D (tuỳ chọn) | 1 | 800 | 140 | 940 |

Tổng **~4.700 lời giải** (~5.640 nếu có D). Bộ đảo của teacher rẻ, sample luôn để làm mốc tham chiếu.

Nếu Cổng 3 KHÔNG ĐẠT: chỉ sample teacher và C seed 0 (xem bước 8b).

2. **Gán nhãn tự động** toàn bộ bằng LLM-judge: đưa từng lời giải cho Qwen2.5-14B-AWQ kèm codebook, bảo nó chọn nhãn. Judge chạy **greedy** (`temperature = 0`), cấu hình load như bước 3 với `enable_prefix_caching=True` và codebook đặt ở **đầu** prompt, lời giải ở cuối — nhờ vậy phần codebook (~1.500 token) chỉ phải prefill một lần cho mỗi họ bài thay vì ~4.700 lần.

3. **Người gán nhãn 150 lời giải**: mẫu phân tầng, cân bằng theo họ bài và theo nhóm, **trộn ngẫu nhiên và ẩn nguồn gốc** — giao diện gán nhãn chỉ hiện đề và lời giải, không hiện nhóm. **Loại trừ 20 lời giải đã đọc ở bước 8b.**

   Ẩn nguồn sẽ **không hoàn hảo**: teacher và C viết theo văn phong của Qwen (có thể có LaTeX), A viết kiểu Llama — đọc vài dòng là đoán được. Không xoá được điều này mà không sửa nội dung lời giải. Thay vào đó, **đo nó**: với mỗi lời giải, sau khi gán nhãn chiến lược, ghi thêm cột `guessed_source` (teacher / A / C / không biết). Báo cáo tỷ lệ đoán đúng nguồn như một chỉ số về mức độ ẩn nguồn thực tế.

4. Tính **Cohen's kappa** giữa người và judge (mục 6.6). Báo cáo thêm kappa tách riêng theo nhóm (teacher / A / C) nếu mỗi nhóm có ≥ 40 lời giải — để thấy judge có thiên vị văn phong của chính nó không.

**Ba điều bắt buộc ở bước sampling**

- **A phải được sample cùng điều kiện với C.** Thiếu A thì H1 không kiểm định được, vì không có mốc "trước khi distill".
- **Teacher cũng phải được sample $k = 20$ ở cùng temperature. Không được tái sử dụng lời giải teacher từ bước 4** — ở đó mỗi bài chỉ có một lời giải, mà phân bố ước lượng từ $n = 1$ là vô nghĩa và không so sánh được với phân bố ước lượng từ $k = 20$.
- **Cả ba seed của C đều phải sample**, để kết luận "C tập trung hơn A" được kiểm chứng là ổn định qua các lần train chứ không phải đặc điểm của một lần chạy may rủi.

**Vì sao $k = 20$ mà không phải 5**: bạn đang ước lượng một phân bố. Sinh 5 lời giải rồi kết luận "model này chỉ dùng một chiến lược" cũng như tung xúc xắc 5 lần rồi kết luận nó thiên vị. Sai số chuẩn giảm theo $1/\sqrt{k}$, nên $k = 20$ cho sai số bằng khoảng một nửa so với $k = 5$.

**Đầu ra**
- `results/samples_strategy_*.jsonl` (mọi nhóm, mọi seed)
- `results/labels_judge.jsonl`
- `results/labels_human.jsonl` (150 dòng, có cột `guessed_source`)
- Kappa tổng và kappa theo nhóm; tỷ lệ đoán đúng nguồn

**Kiểm tra — CỔNG 4**

- [ ] Cohen's kappa ≥ 0,6

**Nếu kappa < 0,6**: **không** sửa codebook rồi chạy lại vòng hai. Không ai đảm bảo lần hai đạt, và đây là điểm duy nhất trong plan có thể nuốt trọn một tuần mà không báo trước. Thay vào đó rút phạm vi:

- Giữ **5 họ bài đã chỉ định sẵn** trong `analysis_plan.md` từ bước S (chọn trước khi thấy output, ví dụ 5 họ có định nghĩa chiến lược rõ ràng nhất). Không chọn họ sau khi đã thấy nhãn.
- Gán nhãn tay **chỉ A và C seed 0**, lấy **10 lời giải đầu tiên** trong 20 mẫu của mỗi bài.
- 5 họ × ~4 bài × 10 lời giải × 2 nhóm ≈ **400 lời giải**, ~30 giây mỗi lời giải → **~3-4 giờ**.
- Bỏ judge, bỏ kappa, bỏ teacher và C seed 1-2 khỏi phần chiến lược.
- Báo cáo ghi rõ đã rút phạm vi và vì sao.

Đừng rút xuống "5 họ nhưng gán tay toàn bộ mọi nhóm": 5 họ × 20 lời giải × 5 nhóm ≈ 2.000 lời giải, tức 15+ giờ gán tay — nặng hơn cả phương án chính.

**Tra cứu**: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

---

## Bước 13 — Tính chỉ số đa dạng

**Làm gì**

**Luật tính — đã chốt trong `analysis_plan.md` từ bước S, áp nguyên văn:**

- **Đơn vị tính là (model, seed, bài).** Mọi chỉ số tính trên 20 mẫu của **một** model, **một** seed, **một** bài. **Không bao giờ gộp 3 seed của C thành 60 mẫu một bài** rồi so với 20 mẫu của A: cỡ mẫu khác nhau thì độ lệch của ước lượng khác nhau (mẫu càng lớn càng thấy nhiều nhãn hiếm → entropy cao hơn, top-1 thấp hơn), và phép so sánh bị méo. Kết quả của C = trung bình của 3 seed, mỗi seed tính ở cỡ mẫu riêng.
- **`unparseable` bị loại khỏi phân bố chiến lược**, và tỷ lệ `unparseable` được báo cáo riêng cho từng nhóm (chỉ số 10, mục 6.7). `other` **được giữ** như một nhãn bình thường.
- **Cân bằng cỡ mẫu sau khi loại `unparseable`**: với mỗi bài, gọi $n^* = \min$ số lời giải hợp lệ qua mọi (model, seed) đang so sánh. Nếu $n^* < 10$, bài đó bị loại khỏi phân tích chiến lược (ghi lại bài nào bị loại, thuộc nhóm nào gây ra). Nếu $n^* \geq 10$, với mỗi (model, seed) rút ngẫu nhiên không hoàn lại $n^*$ lời giải, tính chỉ số, lặp 200 lần, lấy trung bình. Nhờ vậy mọi nhóm được so ở **cùng cỡ mẫu**.
- **Mặc định tính mọi lời giải hợp lệ**, kể cả sai đáp án. Phân tích độ nhạy: lặp lại toàn bộ chỉ tính lời giải đúng.
- **Phân tích độ nhạy thứ hai**: coi `unparseable` là một nhãn bình thường (không loại), báo cáo xem kết luận có đổi không.

**Làm gì**

1. **Tỷ lệ tập trung top-1** (chỉ số chính của phần chiến lược): với mỗi (model, seed, bài), tỷ lệ lời giải thuộc chiến lược phổ biến nhất, theo luật trên. Trung bình theo bài trong họ, rồi theo họ. Tính cho teacher, A, C (từng seed và trung bình 3 seed), và D nếu có.
2. **Entropy hiệu chỉnh Miller-Madow** (chỉ số phụ) — mục 6.4, cùng luật cân bằng cỡ mẫu.
3. **Bootstrap CI**, lấy mẫu **theo họ bài** chứ không theo từng lời giải.
4. **Tỷ lệ cứng nhắc trên bộ đảo**: phần trăm lời giải vẫn dùng chiến lược cũ dù nó không còn tối ưu. Tính cho **cả A và C** (và teacher làm mốc).

**Vì sao bootstrap phải theo họ bài**: 20 lời giải của cùng một bài không độc lập với nhau — chúng cùng chịu ảnh hưởng của cách bài đó được ra đề. Coi chúng là độc lập sẽ làm khoảng tin cậy hẹp giả tạo, và bạn sẽ tưởng kết quả chắc chắn hơn thực tế.

**Hệ quả phải chấp nhận**: chỉ có 10 họ bài, tức 10 cụm, nên CI sẽ **rộng**. Phần này chỉ phát hiện được hiệu ứng lớn (cỡ chênh 15-20 điểm top-1). Đó là lý do nó được xếp là phân tích khám phá (mục 1).

**Về độ lệch của top-1 concentration**: giống entropy, top-1 tính từ mẫu nhỏ bị lệch — nó **cao hơn** thực tế (mẫu nhỏ trông tập trung hơn). Độ lệch này lớn hơn ở phân bố đa dạng, nên nó kéo A (đa dạng hơn, nếu H1 đúng) về phía C, tức là **thu hẹp** khác biệt. So ở cùng cỡ mẫu $n^*$ giữ cho độ lệch này không thiên về bên nào ngoài hiệu ứng đó.

**Vì sao tỷ lệ cứng nhắc phải đo cả A**: phải biết A cứng nhắc đến mức nào mới kết luận được distillation có làm nó cứng thêm hay không. Lý do y hệt mục 2.1.

**Đầu ra**: `results/diversity_metrics.json`, các hình.

---

## Bước 14 — Kiểm định

**Làm gì**

1. **Paired permutation test (sign-flip theo họ bài), A so với C** — kiểm định chính của phần chiến lược.
2. Cùng kiểm định cho từng seed của C riêng lẻ, để thấy mức ổn định.
3. Cùng kiểm định teacher so với C — **chỉ mang tính tham chiếu**. Nếu có D: A so với D và C so với D.
4. Cùng kiểm định cho tỷ lệ cứng nhắc trên bộ đảo (A so với C).

**Thống kê kiểm định** (chốt trong `analysis_plan.md`):

- Với mỗi họ bài $f$: $\Delta_f = \overline{\text{top1}}_C(f) - \overline{\text{top1}}_A(f)$, trong đó $\overline{\text{top1}}_C(f)$ là trung bình qua các bài trong họ và qua 3 seed (tính theo luật bước 13).
- Thống kê: $T = \frac{1}{10}\sum_f \Delta_f$. H1 dự đoán $T > 0$.

**Vì sao không xáo chung nhãn của A và C vào một rổ**: làm vậy là coi mọi lời giải độc lập — đúng điều bước 13 vừa nói là sai. Hơn nữa nhãn chiến lược **riêng cho từng họ bài** (nhãn "lập phương trình" của họ 3 không phải cùng thứ với nhãn của họ 7), nên gộp nhãn qua các họ không có nghĩa.

**Paired permutation hoạt động thế nào**: nếu A và C thật sự không khác nhau, thì với mỗi họ bài, việc gọi bên nào là "A" và bên nào là "C" là tuỳ ý — tức dấu của $\Delta_f$ là ngẫu nhiên. Vì vậy: lật dấu ngẫu nhiên từng $\Delta_f$, tính lại $T$, lặp lại. Với 10 họ chỉ có $2^{10} = 1.024$ cách lật dấu, nên **liệt kê hết** (exact test) thay vì lặp ngẫu nhiên 10.000 lần. $p$ = tỷ lệ các cách lật cho $T$ lớn hơn hoặc bằng $T$ thật (một phía, vì H1 có hướng; báo cáo thêm hai phía).

Lưu ý: $p$ nhỏ nhất đạt được là $1/1.024 \approx 0{,}001$ (một phía). Với 5 họ (nếu rút phạm vi ở Cổng 4) chỉ có 32 cách, $p$ nhỏ nhất là $0{,}031$ — gần sát ngưỡng 0,05, phải ghi rõ.

```python
from scipy.stats import permutation_test
res = permutation_test((delta_f,), statistic=lambda d: d.mean(),
                       permutation_type="samples", alternative="greater",
                       n_resamples=float("inf"))   # inf → exact
```

**Kiểm tra bằng dữ liệu giả trước khi chạy thật**: mô phỏng 1.000 lần, mỗi lần 10 $\Delta_f$ là nhiễu quanh 0 → tỷ lệ lần có $p < 0{,}05$ phải ≈ 5%; tạo 10 $\Delta_f$ đều bằng +0,2 → $p$ phải bằng $1/1.024$.

**Đầu ra**: `results/tests.json` — mọi $p$-value.

**Tra cứu**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html

---

## Bước 15 — Viết báo cáo

**Cấu trúc đề nghị**

1. **Câu hỏi nghiên cứu và H1** (mục 1)
2. **Related work** — bắt buộc có, xem mục 10. Đừng chỉ cite RLKD; người chấm biết literature sẽ hỏi ngay về Dang et al.
3. **Phương pháp** — các điều kiện (A, B, C, và D nếu có), dữ liệu, cấu hình, giao thức eval, `analysis_plan.md` và thời điểm commit của nó
4. **Kết quả** — trình bày theo phân cấp bằng chứng ở mục 1: bằng chứng chính trước, phân tích khám phá sau
   - Bảng accuracy các điều kiện × ba test set, kèm CI và $p$-value
   - Hình pass@$k$ của A và C ← hình chủ đạo (cả hai temperature nếu đã chạy)
   - Entropy phân bố đáp án: A và C
   - Kết quả Cổng 3 (A có đủ đa dạng không) — **báo cáo dù đạt hay không**
   - Bảng tỷ lệ tập trung top-1 và entropy: teacher / A / C (/ D)
   - Bảng tỷ lệ cứng nhắc trên bộ đảo: teacher / A / C
   - Bảng kappa (tổng và theo nhóm), tỷ lệ đoán đúng nguồn khi gán tay
   - Phân tích độ nhạy (chỉ lời giải đúng; `unparseable` như một nhãn)
   - **Vài ví dụ cụ thể**: cùng một bài, lời giải của A và của C cạnh nhau; và một ví dụ trước/sau khi đảo
5. **Thảo luận** — đối chiếu với H1, với RLKD, với Dang et al., với *Mimicry or Mastery*, và với các bài chỉ ra kết quả benchmark reasoning nhạy với lựa chọn đánh giá (*A Sober Look*)
6. **Hạn chế** (mục 7)

**Về việc viết mục Thảo luận khi kết quả âm tính**: nếu C không tập trung hơn A, hoặc thậm chí đa dạng hơn, thì giả thuyết của RLKD không tái lập được ở quy mô này. Đây **vẫn là kết quả hợp lệ** và phải báo cáo nguyên vẹn. Literature hiện tại không đồng nhất: Dang et al. và SCOPE thấy pass@$k$ giảm khi pass@1 tăng, nhưng *A Sober Look* cho thấy các con số trên benchmark reasoning rất nhạy với seed, decoding và prompt, và nhiều cải thiện được công bố không tái lập được — nên một kết quả âm tính ở quy mô nhỏ, được đo cẩn thận, là đóng góp có giá trị.

**Cảnh báo về cách cite *A Sober Look***: abstract của bài này nói về độ nhạy khi đánh giá và việc RL cải thiện ít hơn công bố, **không** nhắc tới diversity collapse. **Đọc full text trước khi viết**; chỉ gọi nó là "bằng chứng phản bác diversity collapse" nếu bài thật sự có kết quả về pass@$k$ hoặc đa dạng. Nếu không có, cite nó cho luận điểm về độ nhạy đánh giá như trên.

---

## Bước 16 — Đóng gói và demo

**Làm gì**

1. Kiểm tra checklist bàn giao (mục 7).
2. Chuẩn bị demo ngắn: chọn 3-5 câu hỏi, chạy A, B, C, trình bày output cạnh nhau.
3. Đọc lại `requirements*.txt`, xác nhận đã pin đủ.

---

# PHẦN C — TRA CỨU

## 6. Chi tiết kỹ thuật

### 6.1. Cấu hình LoRA (đã điều chỉnh cho T4 fp16)

```python
# Load base model ở fp32 — 1B × 4 byte ≈ 4GB, vừa T4 thoải mái.
# AMP (fp16=True) lo phần tính toán fp16; master weights và LoRA params ở fp32.
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float32)

# LoraConfig (PEFT)
r                            = 16
lora_alpha                   = 32
lora_dropout                 = 0.05
target_modules               = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]
task_type                    = "CAUSAL_LM"

# SFTConfig (TRL 1.x)
fp16                         = True    # AMP. KHÔNG dùng bf16 — T4 không hỗ trợ
gradient_checkpointing       = True
per_device_train_batch_size  = 4
gradient_accumulation_steps  = 4       # effective batch = 16
learning_rate                = 1e-4
lr_scheduler_type            = "cosine"
warmup_ratio                 = 0.03
max_grad_norm                = 1.0
max_length                   = 1024    # tên cũ max_seq_length đã bị đổi
completion_only_loss         = True    # dataset dạng prompt–completion (bước 6)
packing                      = False   # packing làm lẫn ranh giới mẫu, tắt cho đơn giản
num_train_epochs             = 3       # lưu mọi epoch, chọn theo dev (luật ở bước 6), KHÔNG theo test
save_strategy                = "epoch"
push_to_hub                  = True    # repo private (bước 1)
hub_strategy                 = "every_save"
seed                         = 0 / 1 / 2
```

**Vì sao load fp32 thay vì fp16**: nếu base model load ở fp16 rồi bật `fp16=True`, Trainer dễ gặp lỗi `Attempting to unscale FP16 gradients` hoặc NaN khi gradient của LoRA cũng ở fp16. Load fp32 cho model 1B tốn thêm ~2GB VRAM nhưng tránh được gần hết nhóm lỗi này. Vẫn in `dtype` của một tham số LoRA trước khi train để xác nhận là `float32`.

Tên `target_modules` dùng chung được cho Llama 3.2 và Qwen 2.5 vì hai kiến trúc đặt tên projection giống nhau.

Tên tham số giữa các bản TRL còn thay đổi. Trước khi train, đối chiếu từng tham số ở trên với docs `SFTConfig` của **đúng bản đã pin**.

Mỗi lần train C ~1 giờ trên T4.

### 6.2. Cấu hình sinh văn bản

| Mục đích | temperature | top_p | max_tokens |
|---|---|---|---|
| Sinh dữ liệu train (bước 4) | 0.7 | 0.95 | 1024 |
| Eval accuracy (greedy) | 0 | — | 1024 |
| Sampling cho pass@k và chiến lược | 0.7 | 0.95 | 1024 |
| Sampling bổ sung pass@k (tuỳ chọn, bước 7) | 1.0 | 0.95 | 1024 |
| Sinh dữ liệu D (tuỳ chọn, bước 9) | 0.7 | 0.95 | 1024 |
| LLM-judge gán nhãn (bước 8b, 12) | 0 | — | 64 nếu chỉ xuất nhãn; 512 nếu prompt bắt giải thích trước khi chọn nhãn |

**Sinh dữ liệu train và sampling phân tích phải dùng cùng temperature.** Nếu lệch, tập train không phản ánh phân bố chiến lược tự nhiên của teacher ở điều kiện bạn đang đo.

`max_tokens = 1024` ở mọi chỗ: lời giải thường dừng sớm hơn nhiều nên gần như không tốn thêm thời gian, nhưng tránh việc lời giải dài bị cắt cụt rồi bị tính là sai. Luôn ghi tỷ lệ output chạm `max_tokens`.

Mọi prompt của student đi qua `render_prompt` với `DATE_STRING` cố định (bước 5).

### 6.3. Luật trích xuất đáp án

1. **Gold GSM8K**: lấy phần sau `####` trong trường `answer`.
2. **Output model**: lấy số đứng sau `Answer:` **cuối cùng** nếu có; nếu không, lấy số trong `\boxed{...}` cuối cùng nếu có; nếu không, lấy **số cuối cùng** trong output.
3. **Chuẩn hoá**: bỏ `,`, `$`, `%`, và đuôi `.0`; giữ dấu âm.
4. Không moi được số nào → đánh dấu `extraction_failed`, tính là sai, nhưng **đếm riêng** để audit.

### 6.4. Entropy và hiệu chỉnh Miller-Madow

Entropy plug-in:

$$H = -\sum_i p_i \log_2 p_i$$

Càng cao càng đa dạng.

**Vấn đề**: khi ước lượng từ mẫu nhỏ, công thức này luôn cho kết quả **thấp hơn thực tế** (mẫu nhỏ bỏ sót nhãn hiếm, nên model trông hẹp hơn thật).

**Độ lệch này thiên về phía nào?** Cần nghĩ cho đúng, vì dễ suy luận ngược:

- **Khi A và C được so ở cùng cỡ mẫu**: cả hai đều bị kéo thấp. Nhưng phân bố càng đa dạng thì càng nhiều nhãn hiếm bị bỏ sót, nên bị kéo thấp **nhiều hơn**. Nếu H1 đúng (A đa dạng hơn C), A bị kéo xuống nhiều hơn C → khác biệt **bị thu hẹp** → entropy thô **bất lợi** cho H1, không phải có lợi. Kết luận ủng hộ H1 từ entropy thô là bảo thủ.
- **Khi A và C được so ở cỡ mẫu khác nhau**: đây mới là chỗ nguy hiểm. Ví dụ gộp 3 seed của C thành $n = 60$ còn A chỉ có $n = 20$ → A bị kéo thấp nhiều hơn chỉ vì ít mẫu hơn → tạo ra **bằng chứng giả ủng hộ hoặc chống H1** tuỳ hướng lệch cỡ mẫu. Loại `unparseable` với tỷ lệ khác nhau giữa các nhóm cũng gây ra lệch cỡ mẫu y hệt.

Vì vậy có hai biện pháp, dùng **cả hai**:

1. **So ở cùng cỡ mẫu** (luật $n^*$ ở bước 13) — biện pháp chính.
2. **Hiệu chỉnh Miller-Madow** — giảm độ lệch bậc nhất, để con số entropy tuyệt đối có nghĩa hơn:

$$H_{\text{MM}} = H_{\text{plug-in}} + \frac{\hat{m} - 1}{2n}$$

với $\hat{m}$ là số nhãn thực sự xuất hiện (bin khác 0) và $n$ là số mẫu. Công thức này dùng log tự nhiên; nếu tính $H$ bằng $\log_2$ thì số hạng hiệu chỉnh phải chia thêm cho $\ln 2$.

### 6.5. Entropy phân bố đáp án (chỉ số phụ, miễn phí)

Với mỗi câu trong 200 câu ở bước 7, bạn đã có 20 đáp án cuối. Đếm phân bố rồi áp công thức entropy ở trên.

Ví dụ: 20 mẫu cho ra `[18, 18, 18, 18, 18, 7, 7, ...]` → entropy thấp, model rất tự tin, luôn ra cùng đáp án. Ra 12 đáp án khác nhau → entropy cao.

Đây là proxy **hoàn toàn tự động** cho đa dạng: chỉ đếm chuỗi số, không cần judge, không cần người, không cần codebook, khoảng 30 dòng code.

**Hạn chế phải ghi rõ trong báo cáo**: hai lời giải khác chiến lược vẫn có thể ra cùng đáp án, nên chỉ số này sẽ đánh giá thấp độ đa dạng thật. Nó là proxy thô, không thay thế được nhãn chiến lược.

### 6.6. Cohen's kappa

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

với $p_o$ là tỷ lệ đồng ý thực tế, $p_e$ là tỷ lệ đồng ý kỳ vọng nếu cả hai đoán bừa theo phân bố biên của mình. $\kappa \geq 0{,}6$ được coi là chấp nhận được.

**Điều tuyệt đối không được làm**: dùng một LLM khác làm "người" gán nhãn để tiết kiệm công. Kappa đo mức đồng thuận giữa judge tự động và một mốc **độc lập**. Nếu mốc đó cũng là LLM, bạn đang đo hai model có giống nhau không, chứ không đo nhãn có đúng không. Chỉ số mất sạch ý nghĩa, và người chấm nào biết chuyện sẽ thấy ngay.

150 lời giải gán tay, giấu nguồn, ước chừng 3-5 giờ tập trung. Đây là phần duy nhất của project không ai làm hộ được.

Với 150 nhãn chia cho nhiều họ bài và nhiều nhãn, kappa theo từng họ sẽ rất nhiễu — chỉ báo cáo kappa tổng và kappa theo nhóm (teacher / A / C), kèm bootstrap CI.

### 6.7. Bảng chỉ số đầy đủ

| # | Chỉ số | Cấp bằng chứng | Nhóm | Kiểm định / CI | Ở bước |
|---|---|---|---|---|---|
| 1 | Accuracy GSM8K test (1.319) | Chính | A, B, C (, D) | mean ± std qua 3 seed + Wilson CI | 5, 7, 10 |
| 2 | Accuracy SVAMP (1.000), MultiArith (600) | Chính | A, B, C (, D) | mean ± std + Wilson CI | 10 |
| 3 | So sánh theo cặp A–C, B–C (, A–D, C–D) | Chính | — | McNemar exact (seed 0) | 7, 10 |
| 4 | Tỷ lệ tập trung chiến lược top-1 | Khám phá | teacher, A, C×3 (, D) | Bootstrap CI theo họ bài + paired sign-flip permutation A–C theo họ bài | 13, 14 |
| 5 | Entropy Miller-Madow | Khám phá | teacher, A, C×3 (, D) | Bootstrap CI theo họ bài, cùng cỡ mẫu $n^*$ | 13 |
| 6 | Tỷ lệ cứng nhắc trên bộ đảo | Khám phá | teacher, A, C | Bootstrap CI theo họ bài + paired permutation A–C | 13, 14 |
| 7 | Cohen's kappa (tổng, theo nhóm) + tỷ lệ đoán đúng nguồn | Độ tin cậy nhãn | — | Ngưỡng ≥ 0,6; bootstrap CI | 12 |
| 8 | **pass@20 vs pass@1** (200 câu) | **Chính** | A, C | Bootstrap CI theo câu | 7 |
| 9 | Entropy phân bố đáp án | **Chính** | A, C | Bootstrap CI theo câu | 7 |
| 10 | Tỷ lệ `unparseable` | Mô tả | teacher, A, C | Báo cáo riêng từng nhóm | 11, 13 |
| 11 | Top-1 concentration của A (Cổng 3) | Kiểm tra tiền đề | A | Ngưỡng ≥ 5/10 họ có top-1 ≤ 85% | 8 |

**Không chỉ số nào được báo cáo dưới dạng con số đơn lẻ.** Mỗi cái đều phải có kiểm định hoặc khoảng tin cậy đi kèm.

## 7. Checklist bàn giao

- [ ] **Code**: data generation, training, evaluation, phân tích chiến lược, script thống kê
- [ ] **Dữ liệu đã sinh**: CoT, answer-only, 40 bài đa chiến lược, 7 biến thể đảo, kèm nhãn
- [ ] **Codebook**, **`analysis_plan.md`** (kèm hash commit chứng minh chốt trước output), và bảng Cohen's kappa
- [ ] **Toàn bộ generation thô** của mọi điều kiện và mọi seed, dạng JSONL, đủ để người khác tái lập mọi con số
- [ ] **`requirements*.txt` đã pin version**, danh sách seed, tên chính xác (và revision hash) của checkpoint teacher và student trên HF, `DATE_STRING` dùng trong chat template
- [ ] **Báo cáo** kèm bảng số liệu có CI/p-value và ví dụ minh hoạ
- [ ] **Tuân thủ license Llama**: mọi derivative được phân phối phải mang tiền tố tên `Llama-3.2-...` và ghi nhận "Built with Llama"

Vì teacher là open-weights với checkpoint cố định, toàn bộ pipeline tái lập được từ đầu đến cuối: người chấm chỉ cần tên model, seed và code là chạy lại ra đúng kết quả. Đây là ưu thế mà teacher gọi qua API không bảo đảm được, vì model phía sau API có thể bị cập nhật bất kỳ lúc nào.

### Hạn chế phải ghi rõ trong báo cáo

- Chỉ một student ở một kích thước (1B), không rút ra được kết luận nào về scaling.
- Teacher chỉ 14B, vẫn cách xa quy mô frontier mà RLKD nghiên cứu. Cơ chế meta-reasoning RLKD mô tả có thể chỉ bộc lộ đầy đủ ở teacher lớn hơn nhiều.
- Student chỉ 1B nên chất lượng reasoning thấp: một phần lời giải sẽ rời rạc hoặc sai, tỷ lệ `unparseable` cao hơn so với student lớn hơn. Tỷ lệ này phải báo cáo riêng từng nhóm.
- Bộ bài tự soạn ~40 bài là **nhỏ**; kết quả nhạy cảm với cách thiết kế bài và cách định nghĩa nhãn.
- LLM-judge là chính teacher Qwen2.5-14B, tức nó gán nhãn cho cả output do chính nó sinh ra lẫn output của student Llama. Rủi ro thiên vị với văn phong của chính mình là có thật; kiểm tra thủ công có ẩn nguồn kèm kappa chỉ giảm bớt chứ không loại bỏ.
- Teacher chạy bản AWQ 4-bit nên bị lượng tử hoá nhẹ so với fp16 gốc; lọc theo đáp án đúng hạn chế ảnh hưởng tới chất lượng trace nhưng không loại trừ ảnh hưởng tới phân bố chiến lược.
- A đã là model instruct biết làm CoT, nên **không phải baseline "không suy luận"**. Kết luận rút ra là về *thay đổi* độ đa dạng, không phải về nguồn gốc của độ đa dạng.
- Một kết quả âm tính ở quy mô này **không phủ định** RLKD ở quy mô lớn.
- **Temperature không tương đương giữa các model**: A và C cùng được sample ở `temperature = 0.7`, nhưng SFT làm phân bố của C nhọn hơn, nên độ ngẫu nhiên thực tế khác nhau. Một phần chênh lệch pass@$k$ và đa dạng có thể đến từ đây. (Giảm nhẹ nếu đã chạy thêm `temperature = 1.0`.)
- **Ẩn nguồn khi gán tay không hoàn hảo**: văn phong của teacher/C khác A. Báo cáo tỷ lệ đoán đúng nguồn.
- **Phân tích chiến lược có công suất thống kê thấp**: 10 họ bài = 10 cụm; chỉ phát hiện được hiệu ứng lớn.
- **B không kiểm soát được hiệu ứng "SFT trên lời giải đúng đã lọc"** lên đa dạng. Nếu không làm D, không tách được tác động riêng của trace teacher khỏi tác động chung của SFT.
- **Nếu có D**: D train trên ít câu hơn và dễ hơn C (chỉ những câu A giải được), nên so sánh C–D bị lẫn với khác biệt về dữ liệu.
- **Chọn epoch trên dev 200 câu** nhiễu (±7 điểm); luật chọn cố định giảm nhưng không loại bỏ ảnh hưởng.

## 8. Sự cố thường gặp

| Triệu chứng | Nguyên nhân khả dĩ, theo thứ tự nên thử |
|---|---|
| **Accuracy của C tụt mạnh so với A (vd A 42%, C 25%)** | Negative transfer vài điểm là kết quả khả dĩ; tụt hơn ~15 điểm thì kiểm tra lỗi trước:<br>1. Parser hỏng — kiểm tra trước tiên, rẻ nhất<br>2. Prompt lúc eval khác lúc train — so chuỗi prompt đã render, đặc biệt dòng `Today Date`<br>3. EOS sai: C không dừng, sinh lan man tới `max_tokens` — xem tỷ lệ cắt cụt<br>4. BOS lặp trong dữ liệu train<br>5. Dữ liệu train bị cắt cụt ở `max_length = 1024` — kiểm tra phân bố độ dài<br>6. Loss tính cả phần prompt — in mask<br>7. Merge LoRA sai — so output trước/sau merge<br>8. Learning rate quá cao |
| **Loss ra NaN khi train** | fp16 overflow. Thử: xác nhận base model load fp32 và LoRA params là `float32`; giảm lr; `max_grad_norm = 1.0`; giảm batch size |
| **Lỗi `Attempting to unscale FP16 gradients`** | Base model hoặc LoRA đang ở fp16. Load base ở fp32 (mục 6.1) |
| **`TypeError` / `unexpected keyword` từ `SFTConfig` hoặc `SFTTrainer`** | Tên tham số đã đổi giữa các bản TRL (vd `max_seq_length` → `max_length`). Đối chiếu docs của đúng bản đã pin |
| **Lỗi `assistant_only_loss` / không import được `DataCollatorForCompletionOnlyLM`** | Không dùng hai thứ này. Chuyển sang dataset prompt–completion (bước 6) |
| **Kết quả eval đổi giữa hai lần chạy cách nhau vài ngày** | Chat template chèn ngày chạy máy. Dùng `render_prompt` với `DATE_STRING` cố định (bước 5) |
| **OOM khi load teacher** | Dùng `tensor_parallel_size=2`; giảm `gpu_memory_utilization` của vLLM; giảm `max_model_len`; hạ xuống 7B |
| **Tensor parallel lỗi trên 2×T4** | Thử `disable_custom_all_reduce=True`; nếu vẫn lỗi, chạy 1×T4 với `max_model_len=2048` (chậm hơn, bước 3) |
| **vLLM không chạy trên T4** | Lấy tổ hợp version từ một notebook Kaggle T4 đã chạy được (bước 2). Nếu vẫn không được → fallback `transformers` batch generate **và cắt test set xuống 500 câu**, ghi rõ trong báo cáo |
| **Kernel AWQ lỗi trên sm75** | Hạ teacher xuống `Qwen2.5-7B-Instruct-AWQ` |
| **Session Kaggle ngắt giữa lúc train** | Dùng notebook commit thay vì session tương tác. Resume từ checkpoint epoch gần nhất trên HF Hub |
| **Tỷ lệ lọc ở bước 4 thấp hơn dự kiến** | Sinh thêm từ phần GSM8K train còn lại. Tập train 2.000 câu sau lọc vẫn đủ cho LoRA |
| **Tỷ lệ `unparseable` quá cao dù đã hiệu chỉnh độ khó** | Đơn giản hoá thêm bộ bài; nếu vẫn không cải thiện, nâng student lên `Llama-3.2-3B-Instruct` và giảm còn 2 seed cho vừa quota |
| **Hết quota GPU giữa chừng** | Bỏ D và sampling `temperature = 1.0` trước; rồi giảm còn 1 seed; cuối cùng mới cắt GSM8K test xuống 500 câu. **Ghi rõ trong báo cáo là đã giảm** |
| **A gần như chỉ dùng một chiến lược (Cổng 3 không đạt)** | Không sửa bài. Làm theo nhánh KHÔNG ĐẠT ở bước 8b |
| **Chưa được duyệt quyền Llama khi tới bước 6** | Đổi student sang `Qwen2.5-1.5B-Instruct`. Mất lợi thế "teacher và student khác họ" — ghi vào mục hạn chế |

## 9. Cấu trúc thư mục đề nghị

```
cot-distill-diversity/
├── requirements-infer.txt    # vLLM, đã pin version
├── requirements-train.txt    # TRL + PEFT, đã pin version (có thể gộp làm một nếu không xung đột)
├── README.md
├── configs/
│   ├── train_lora.yaml       # kèm luật chọn epoch
│   └── gen.yaml              # prompt teacher, PROMPT student, DATE_STRING
├── docs/
│   ├── codebook.md           # CHỐT trước khi xem output (bước S)
│   └── analysis_plan.md      # CHỐT cùng commit với codebook
├── src/
│   ├── prompts.py            # render_prompt + DATE_STRING — nguồn duy nhất cho mọi prompt
│   ├── extract.py            # luật trích xuất (mục 6.3), dùng lại được ở bước 11
│   ├── gen_teacher_data.py   # bước 4
│   ├── gen_selfdistill.py    # bước 9, tuỳ chọn D
│   ├── train_sft.py          # bước 6, 9
│   ├── eval_harness.py       # bước 5, 7, 10
│   ├── sample_k.py           # bước 7, 8b, 12
│   ├── passk.py              # bước 7
│   ├── judge_label.py        # bước 8b, 12
│   ├── metrics.py            # bước 13
│   └── stats.py              # bước 14
├── data/
│   ├── train_cot.jsonl
│   ├── train_answer_only.jsonl
│   ├── train_selfdistill.jsonl   # tuỳ chọn D
│   ├── dev.jsonl
│   ├── gen_raw.jsonl
│   ├── strategy_families_final.jsonl
│   └── strategy_flip.jsonl
├── results/                  # mọi generation thô + nhãn + chỉ số
├── figures/
└── report/
```

## 10. Nguồn tham khảo

### Tài liệu gốc của project

- Xu et al., *Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning* (RLKD), AAAI-2026
  - arXiv: https://arxiv.org/abs/2505.16142
  - AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/40710
  - Code: https://github.com/xsc1234/RLKD

### Bắt buộc có trong mục Related work

Hiện tượng bạn đo **đã có người đo trước**, ở các góc khác. Không cite là điểm trừ nặng.

| Bài | Liên quan thế nào | Link |
|---|---|---|
| Dang et al., *Assessing Diversity Collapse in Reasoning* | Gần trùng chỉ số 8. Chỉ ra khi Pass@1 tăng trong SFT thì Pass@k suy giảm nhanh, quy cho diversity collapse. Thí nghiệm trên Gemma-2-2B, GSM8K — cùng benchmark, cùng cỡ model | https://openreview.net/pdf?id=AMiKsHLjQh |
| Hochlehnert et al., *A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility* | Kết quả benchmark reasoning rất nhạy với seed, decoding, prompt, phần cứng; nhiều cải thiện RL được công bố không tái lập được, còn SFT tổng quát hoá ổn định hơn. Cần cite cho giao thức đánh giá chặt (nhiều seed, CI) và để mục Thảo luận cân bằng. **Abstract không nhắc tới diversity collapse — đọc full text trước khi dùng nó làm bằng chứng phản bác** (bước 15) | https://arxiv.org/abs/2504.07086 |
| *Are We Measuring Strategy or Phrasing?* | Định nghĩa approach-level vs surface-level diversity. **Dùng để neo codebook** ở bước S | https://arxiv.org/abs/2606.29985 |
| *Beyond Accuracy: Evaluating Strategy Diversity in LLM Mathematical Reasoning* | Có sẵn prompt phân loại chiến lược | https://arxiv.org/abs/2605.09292 |
| *Hán Dān Xué Bù (Mimicry) or Qīng Chū Yú Lán (Mastery)?* | Đã dùng đúng thiết kế "so distilled với chính base model của nó", tìm thấy negative transfer ($t = -2{,}60$, $p = 0{,}011$) | https://arxiv.org/abs/2601.05019 |
| Chen et al., *Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning*, Findings of ACL 2025 | Teacher mạnh hơn KHÔNG phải lúc nào cũng tạo student tốt hơn; đa dạng và độ phức tạp của CoT có thể quan trọng hơn accuracy. Va thẳng vào lựa chọn teacher của bạn | https://aclanthology.org/2025.findings-acl.782/ |
| Kim & Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016 | Tiền lệ xa nhất: SeqKD giảm tính đa mô thức của dữ liệu. Ở NMT việc thu hẹp phân bố được coi là **tính năng** | https://arxiv.org/abs/1606.07947 |
| Chen et al., *Evaluating Large Language Models Trained on Code* | Nguồn công thức pass@k | https://arxiv.org/abs/2107.03374 |

### Đọc thêm nếu có thời gian

| Bài | Link |
|---|---|
| SCOPE — ghi nhận cùng nghịch lý ở mức distillation: Pass@1 tăng, Pass@32 giảm | https://arxiv.org/abs/2604.10688 |
| *When Reasoning Narrows the Move: Diversity Collapse in LLM Game Play* — thiết kế logic gần với của bạn, không gian hành động hữu hạn | https://arxiv.org/abs/2607.19523 |
| *ReasoningTrap* — tiền lệ cho bộ đảo chiến lược, phân loại ba dạng cứng nhắc | https://openreview.net/pdf?id=an0Nr1qRnf |
| *Reasoning Path Divergence* — LLM judge phân loại tập lời giải đồng nhất hay đa dạng | https://arxiv.org/abs/2510.26122 |

---

## Phụ lục — Ngân sách GPU theo bước

| Bước | Tác vụ | Thời gian |
|---|---|---|
Tính theo giờ GPU Kaggle (một session 2×T4 tính theo thời gian session; chạy song song hai GPU không làm tốn gấp đôi quota nhưng rút ngắn wall-clock).

| Bước | Tác vụ | Thời gian |
|---|---|---|
| 2-3 | Dựng môi trường + smoke test teacher (gồm các vòng thử version) | ~1-1,5 giờ |
| 4 | Sinh 3.300 CoT (2×T4, tensor parallel) | ~2-3 giờ |
| 5 | Eval A (zero-shot + 8-shot) | ~0,5 giờ |
| 6 | Train C seed 0 (+ eval dev từng epoch) | ~1,5 giờ |
| 7 | Eval C + sampling k=20 cho A và C (+ tuỳ chọn T=1.0: ~0,3 giờ) | ~1 giờ |
| 8 | Hiệu chỉnh độ khó + Cổng 3 (sample A k=20, load teacher làm judge) | ~1,5 giờ |
| 9 | 5 lần train còn lại, song song 2 GPU (+ tuỳ chọn dữ liệu D: ~0,5 giờ) | ~3,5 giờ |
| 10 | Eval 7 model × 3 test set | ~1,5 giờ |
| 12 | Sample teacher + C×3 seed (+ D) + judge gán nhãn ~4.700 lời giải | ~4 giờ |
| | Dự phòng debug, chạy lại | ~4 giờ |
| | **Tổng** | **~21-23 giờ** (thêm ~1-2 giờ nếu làm D và T=1.0) |

Vừa trần ~30 giờ/tuần của Kaggle, nhưng **không dư nhiều nếu dồn vào một tuần** — trải qua ít nhất hai tuần quota.

**Wall-clock**: nhờ notebook commit chạy nền, bạn không phải ngồi canh train. Ràng buộc thật là **vòng debug**: mỗi lần load teacher 14B-AWQ mất vài phút, mỗi vòng thử trên Kaggle tốn 10-20 phút chờ. Giảm số vòng bằng cách debug mọi thứ liên quan tới model 1B trên **máy local** trước (bước 5, 6).

**Các bước không cần GPU**: 1, 2 (phần cài đặt), S, 11, 13, 14, 15, 16.

## Phụ lục — Ngân sách công sức làm tay (ước lượng)

| Việc | Thời gian |
|---|---|
| Viết code (sinh dữ liệu, train, eval, sampling, judge, metrics, stats) | ~25-35 giờ |
| Soạn 47 bài, tự giải tay, viết codebook + `analysis_plan.md` (bước S) | ~10-15 giờ |
| Hiệu chỉnh độ khó (bước 8) | ~2-3 giờ |
| Audit trích xuất 50 ca (bước 11) | ~1-2 giờ |
| Gán nhãn tay 150 lời giải (bước 12) | ~3-5 giờ |
| Phân tích, vẽ hình, viết báo cáo, demo | ~15-20 giờ |
| **Tổng** | **~60-80 giờ** |

Với một người làm, cần ít nhất 4-6 tuần. Nếu thời gian ngắn hơn, nhắm tới Cổng 2 + entropy đáp án + transfer là đủ một project hoàn chỉnh; phần chiến lược chỉ làm nếu còn thời gian.
