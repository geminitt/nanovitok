# Nền tảng — 10 notebook lý thuyết kèm thực hành

Mỗi notebook đi theo cùng một lối: nêu khái niệm, viết công thức, nói nó áp dụng vào tình huống nào, rồi **chạy
lại chính số liệu của project** để kiểm chứng. Không có con số nào chép tay: mọi bảng đều tính từ
`kaggle/outputs/` hoặc từ mã nguồn trong `src/vitok/` và `third_party/nanochat/`.

**Cách chạy**: dùng kernel pixi của project (`.pixi/envs/default/bin/python`). Mọi notebook tự tìm gốc repo nên
chạy được từ bất kỳ thư mục nào. Không cần GPU; notebook nặng nhất (04, 08) mất vài chục giây.

| # | Notebook | Nội dung | Cần trước đó |
|---|---|---|---|
| 01 | [Lý thuyết thông tin và bpc](01_ly_thuyet_thong_tin_va_bpc.ipynb) | từ `loss` trong log → nat/bit → entropy, cross-entropy, KL → bpc; vì sao không dùng bpb và perplexity | — |
| 02 | [Thống kê suy diễn](02_thong_ke_suy_dien.ipynb) | mẫu/tổng thể, $\mathrm{SE}=\sigma/\sqrt n$, CLT, khoảng tin cậy, p-value, ghép cặp, bootstrap, McNemar, power, phân rã nhiễu, so sánh nhiều lần | 01 |
| 03 | [Tham số, FLOPs, scaling](03_tham_so_flops_scaling.ipynb) | $N=12d^2L$, FLOPs $\approx 6N$, MFU, Chinchilla, thiết kế equal-text, ghim learning rate, khớp scaling law | 02 |
| 04 | [Tokenizer: BPE → SuperBPE](04_tokenizer_bpe_superbpe.ipynb) | bảng chữ cái byte, pretokenizer, hai giai đoạn của SuperBPE, WordPiece, Unigram/Viterbi, các độ đo nén, CTC | 01 |
| 05 | [Unicode và tiếng Việt](05_unicode_tieng_viet.ipynb) | UTF-8, NFC/NFD, dấu thanh vs dấu chất lượng, `đ`, thiết kế H2, cặp tối thiểu, âm tiết vs từ (H4) | 04 |
| 06 | [Bên trong Transformer](06_ben_trong_transformer.ipynb) | RoPE, GQA, KV cache, cửa sổ trượt, value embedding, RMSNorm, ReLU², Muon; dựng lại model nhỏ chạy được | — |
| 07 | [Kỹ thuật train quy mô lớn](07_ky_thuat_train_quy_mo_lon.ipynb) | lịch learning rate, tích luỹ gradient, fp16 + GradScaler, tính bất định số học, DDP và vì sao project không dùng | 03 |
| 08 | [Kỹ thuật hiệu năng](08_ky_thuat_hieu_nang.ipynb) | chẩn đoán sự cố stage 2 chạy quá 12 giờ: $\sum L^2$, danh sách liên kết, đếm theo sai khác, vector hoá, profiling | 04 |
| 09 | [Sau pretrain](09_sau_pretrain_sft_dpo_lora.ipynb) | SFT và che loss, RLHF → DPO → GRPO, LoRA, đánh giá LLM; và vì sao model cỡ này chưa dùng được chúng | 02, 03 |
| 10 | [Viết báo cáo thực nghiệm](10_viet_bao_cao_thuc_nghiem.ipynb) | cấu trúc bài, số chữ số được phép, sinh bảng/hình từ kết quả, mục hạn chế, hiệu chỉnh phát biểu | 02, 03 |

**Thứ tự đề nghị**: 01 → 02 → 03 → 04 → 05 (đủ cho báo cáo), rồi 10 khi bắt đầu viết. 06, 07, 08 đọc khi cần
hiểu mã nguồn; 09 để sau project.

Lộ trình và nguồn đọc ngoài nằm ở [`learn.md`](../../../learn.md); trạng thái project ở
[`docs/project_state.md`](../../project_state.md); kế hoạch phân tích đã đăng ký ở
[`docs/analysis_plan.md`](../../analysis_plan.md).
