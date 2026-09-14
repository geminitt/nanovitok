# Analysis plan — chốt trước khi train model đầu tiên

> Commit file này **trước** mọi commit chứa kết quả hoặc checkpoint. Mọi thay đổi sau đó phải ghi ở mục "Sửa đổi" kèm lý do và ngày.

## 1. Câu hỏi và giả thuyết

| # | Giả thuyết | Kiểm định |
|---|---|---|
| H1 | SuperBPE giảm ≥15% số token so với BPE cùng chuẩn hoá, và ở cùng lượng văn bản train có bpc trên văn bản chuẩn không tệ hơn BPE quá 1% (tương đối) | `compression-*.json`; bootstrap ghép cặp `super-X − bpe-X`, variant `clean` |
| H2 | Tokenizer NFD cho bpc thấp hơn NFC trên văn bản bỏ dấu, và thiệt ≤1% (tương đối) trên văn bản chuẩn | Bootstrap ghép cặp `*-nfd − *-nfc`, variant `strip100`, `strip50`, `clean` |
| H3 | Chênh lệch bpc SuperBPE − BPE thay đổi có hệ thống theo cỡ model d6 → d8 → d10 | Dấu và độ lớn của Δbpc theo cỡ; so với độ nhiễu giữa seed ở d8 |
| H4 (khám phá) | Superword token trùng từ nhiều âm tiết nhiều hơn mức ngẫu nhiên | Chỉ mô tả, không kiểm định |

Kết quả âm tính được báo cáo nguyên vẹn.

## 2. Tokenizer

- 4 điều kiện: `bpe-nfc`, `bpe-nfd`, `super-nfc`, `super-nfd`; cùng corpus `tok_train.txt` (~500MB, NFC gốc).
- Alphabet đầy đủ 256 byte; regex stage 1 có `\p{M}` trong lớp chữ cái; regex stage 2 theo SuperBPE.
- Điểm chuyển SuperBPE: 90% vocab.
- **Luật chọn vocab (Gate 1):** chọn 16k nếu `token_reduction` (super so với bpe) ≥ 15% cho cả NFC và NFD ở 16k; nếu không, chọn 32k nếu 32k đạt; nếu cả hai không đạt, giữ 16k và H1 thành giả thuyết phụ (trọng tâm chuyển sang H2).

## 3. Thiết kế công bằng

- Cùng dữ liệu, cùng thứ tự văn bản; cùng số bước; 64 chuỗi mỗi bước.
- `max_seq_len` = round(1024 × cpt(bpe-nfc) / cpt(điều kiện)), bội số của 8; cpt đo trên val shard trong `compression-*.json`.
- Learning rate và weight decay co giãn theo batch cố định 64 × 1024 token (`--scaling-batch-size`), giống nhau mọi điều kiện.
- Ngân sách văn bản (quy về token bpe-nfc): d6 250M, d8 500M, d10 1B.
- Mọi run chính thức: Kaggle T4, fp16 + GradScaler, SDPA, `--window-pattern L`.

## 4. Số đo

- **bpc** = Σ nats / (ln 2 × Σ ký tự NFC), tính trên tập văn bản mà **mọi run cùng cỡ** đều chấm được (không vượt context).
- Test set: 2.000 văn bản từ split `test` của FineWeb-2 vie_Latn, dài ≥300 ký tự, cắt còn ≤2.500 ký tự tại khoảng trắng, loại trùng với dữ liệu train theo hash.
- Variant: `clean`; `strip50` (bỏ dấu 50% âm tiết, seed 0); `strip100` (bỏ toàn bộ dấu, đ→d).
- Cặp tối thiểu: câu 30–200 ký tự, đổi thanh một âm tiết sang âm tiết khác xuất hiện ≥50 lần trong mẫu 200MB của pretrain; ~3.000 cặp; đúng khi Σ nats câu gốc < câu đổi.
- bpb của nanochat chỉ để theo dõi lúc train, **không** dùng để kết luận.

## 5. Thống kê

- Chênh lệch bpc: bootstrap ghép cặp theo văn bản, 10.000 lần, CI 95%; có ý nghĩa khi CI không chứa 0.
- Cặp tối thiểu: McNemar exact hai phía.
- Các phép so sánh được báo cáo: đúng danh sách `COMPARISONS` trong `src/vitok/analysis.py`.
- Phân tích độ nhạy: lặp lại H1 trên val shard thay vì test; bỏ 5% văn bản dài nhất.

## 6. Seed và cổng

- Seed 0 cho mọi run; seed 1 cho `bpe-nfc` và `super-nfc` ở d8 (chỉ đổi khởi tạo, không đổi thứ tự dữ liệu).
- **Gate 0:** d6/d8/d10 chạy fp16 không NaN; ước tính d10 ≤ 6 giờ/run. Nếu không: hạ cỡ lớn nhất xuống d9, hoặc giảm ngân sách d10 còn 60%, áp dụng như nhau cho mọi điều kiện.
- **Gate 2:** đủ số đo cho 4 điều kiện ở d6.
- **Gate 3 — luật chọn d10:** `bpe-nfc` + biến thể SuperBPE (`super-nfc` hoặc `super-nfd`) có bpc `clean` thấp nhất ở d8 seed 0.

## Sửa đổi

_(chưa có)_
