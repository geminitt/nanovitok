# vitok

SuperBPE × NFC/NFD tokenizers for small Vietnamese language models, pretrained with a patched
[nanochat](https://github.com/karpathy/nanochat) on Kaggle T4s. See `plan-tokenizer-vietnamese-steps.md`
for the research plan and `docs/analysis_plan.md` for the pre-registered analysis.

## Chạy trên Kaggle

1. **Đưa code lên Kaggle.** Tạo Kaggle Dataset tên `vitok-code` từ file zip của repo:
   ```bash
   git archive -o vitok-code.zip HEAD      # cần commit trước
   ```
   (hoặc push lên GitHub rồi điền `VITOK_REPO` trong notebook).
2. **Notebook 01** (`kaggle/notebooks/01_data_tokenizers.ipynb`, CPU, Internet On, gắn `vitok-code`):
   Save & Run All. Tải FineWeb-2 tiếng Việt, tạo shard, test set, cặp tối thiểu, train 4 tokenizer ở 16k và 32k,
   in kết quả Gate 1. Xong thì tạo Dataset **`vitok-data`** từ output của notebook.
3. **Chốt analysis plan**: chọn vocab theo luật Gate 1, rồi commit `docs/analysis_plan.md` **trước** khi train.
4. **Notebook 02** (`kaggle/notebooks/02_train_eval.ipynb`, GPU T4 ×2, gắn `vitok-data` + `vitok-code`):
   sửa ô cấu hình (`DEPTH`, `SEED`, `QUEUES`, `VOCAB`, `SMOKE_ITERS`) theo từng bước của plan, Save & Run All.
   - Bước 7 (Gate 0): `SMOKE_ITERS=200`, một điều kiện, đọc tok/sec trong ô throughput.
   - Bước 8–10: `SMOKE_ITERS=None`.
   Kết quả nằm ở `/kaggle/working/results/*.json`; tải output của mỗi phiên về `kaggle/outputs/results-vN/`
   (repo đang có v4, v5, v6, v8, v9), và output của notebook 01 về `kaggle/outputs/vitok-data/`.
5. **Phân tích** (CPU, chạy local được):
   ```bash
   pixi run python -m vitok.analysis --results kaggle/outputs --compression kaggle/outputs/vitok-data/compression-16k.json --out results/summary.md --figures figures
   ```

## Phát triển local (chỉ test, không train)

```bash
pixi install
git clone https://github.com/karpathy/nanochat third_party/nanochat
git -C third_party/nanochat checkout $(cat patches/NANOCHAT_COMMIT)
git -C third_party/nanochat apply ../../patches/nanochat.patch
pixi run test
```

## Dữ liệu

Văn bản trong `kaggle/outputs/vitok-data/` (`test.jsonl`, `minimal_pairs.jsonl`, `syllables.json`) được trích từ
[FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), tập `vie_Latn`, phát hành theo giấy phép
[ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/). Shard parquet gốc không nằm trong repo; `vitok.data` tải lại được.
