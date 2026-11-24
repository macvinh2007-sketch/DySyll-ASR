> **Cập nhật 24/11/2026:** Nếu triển khai trên máy Linux qua Cloudflare SSH và sử dụng dataset `Mihanh27/VietTTS-Quality`, xem **[README_CLOUDFLARE_VI.md](README_CLOUDFLARE_VI.md)**. Cloudflare chỉ là tunnel SSH, không phải máy GPU.

# Bộ train DySyll-ASR khớp sơ đồ 29.898.425 tham số

**Đây là mã chạy/huấn luyện mô hình nghiên cứu từ đầu (random initialization), KHÔNG phải checkpoint ASR đã train.** Bộ này được xây từ file gốc `dysyll_asr.py` mà bạn đã có, với cấu hình khớp **đúng số tham số của sơ đồ**.

## 1. Kiểm tra cấu hình gốc (không cần dataset)

```bash
pip install -r requirements.txt
python train.py --check-model
python train.py --smoke
pytest -q tests/
```

Mong đợi:

| Nhóm | Trainable parameters |
|---|---:|
| ARF | 2,704,640 |
| DDSE + LSC | 19,219,205 |
| EMC + SAG | 7,974,580 |
| ELD (DP) | 0 |
| **Tổng** | **29,898,425** |

Kích thước: `n_mels=80`, `n_prosody=4`, `hidden/ARF=1600`, `phonetic_dim=1600`, `prosodic_dim=800`, `initial=24`, `rhyme=150`, `tone=6`, `subsampling ×4`, `min_segment=2`, `max_segment=24` frame ở encoder.

**Phải biết khi báo cáo:** mã khớp 29.898.425 tham số dùng **custom gated recurrence** (`pho_gate`, `pho_candidate`, `pros_gate`, `pros_candidate`), **không phải GRUCell chuẩn**. Hình gọi hai khối GRU nhưng công thức trong mã **không có reset gate của GRU**. Không thể vừa giữ nguyên các layer như này, vừa tuyên bố nó là `nn.GRUCell` chuẩn. Bản sử dụng GRUCell trong file khác có số tham số khác.

## 2. Chuẩn bị dataset thực

Tạo hai file `train.csv` và `val.csv` (và file `test.csv` chỉ dùng đánh giá), với cấu trúc:

```csv
audio_path,transcript
/path/to/utt001.wav,má đi
/path/to/utt002.wav,xin chào việt nam
```

Yêu cầu: audio mono 16 kHz (script sẽ resample), mỗi clip **speech-only**, khoảng **0.2–3.0 giây**, không có khoảng lặng dài trong câu. Các ví dụ dài có thể vi phạm `min_segment/max_segment`, và phải **tách clip/căn chỉnh** trước khi huấn luyện. Dữ liệu train/val/test **không trùng** audio hoặc speaker (ưu tiên speaker-disjoint). Giữ VLSP2025 làm test nếu muốn so sánh với baseline trước; không train lên tập đó.

### Bắt buộc: lexicon đã kiểm chứng

Tạo `lexicon.csv` với 4 cột:

```csv
syllable,initial_id,rhyme_id,tone_id
ma,10,10,0
má,10,10,1
đi,5,11,0
```

**Đây CHỈ là ví dụ giả định ID**, không phải bộ phân tích âm vị tiếng Việt chuẩn. Các ID phải do bạn xây dựng nhất quán từ bộ nhãn ngữ âm/chính tả đã duyệt và nằm trong `[0,23]` cho initial, `[0,149]` cho rhyme, `[0,5]` cho tone. Mỗi `(initial_id,rhyme_id,tone_id)` chỉ tương ứng với một cách viết trong phiên bản này. `examples/toy_lexicon.csv` chỉ dùng kiểm thử kỹ thuật; **KHÔNG** dùng để huấn luyện nhận dạng tiếng Việt thực tế. Không thể train ASR đáng tin cậy khi lexicon chưa đúng.

## 3. Kiểm toán dataset trước khi tiền xử lý

```bash
python audit_dataset.py --train train.csv --val val.csv \
  --lexicon lexicon.csv --report dataset_audit.json
```

Kiểm tra đường dẫn audio, coverage của lexicon và trùng đường dẫn giữa train/val. Lưu ý: kiểm tra speaker-disjoint và trùng nội dung âm thanh vẫn cần thực hiện riêng.

## 4. Tiền xử lý và cache feature

```bash
python prepare_features.py --manifest train.csv --lexicon lexicon.csv \
  --output data/cache/train --out-manifest data/train.jsonl

python prepare_features.py --manifest val.csv --lexicon lexicon.csv \
  --output data/cache/val --out-manifest data/val.jsonl
```

Script kiểm tra sự tồn tại audio, nhãn, NaN, độ dài, giới hạn căn chỉnh. Mặc định gặp ví dụ lỗi thì dừng và ghi báo cáo `<manifest>.errors.json`; có thể dùng `--skip-invalid` để lọc **kèm kiểm toán các hàng bị loại**, không xóa ngầm.

### Cách lấy feature

- Waveform 16 kHz mono.
- Log-Mel 80, cửa sổ 25ms (400 samples), hop 10ms (160 samples), CMVN theo utterance.
- 4 đặc trưng Prosody: **relative log-F0**, voiced probability, log-RMS, và **spectral-tilt proxy**.
- F0 bằng `librosa.pyin`. Spectral tilt chỉ là **proxy** cho phonation, không tương đương HNR/CPP. Cần kiểm chứng/đổi proxy cho mô hình nghiên cứu nghiêm túc.
- Tiền xử lý **y hệt** được dùng khi train và inference (`feature_pipeline.py`).

## 5. Train mô hình FULL 29,9M trên GPU

```bash
python train.py --config config.json \
  --train data/train.jsonl --val data/val.jsonl --lexicon lexicon.csv \
  --output runs/dysyll_29_9m --device cuda
```

Resume:

```bash
python train.py --config config.json --train data/train.jsonl \
  --val data/val.jsonl --lexicon lexicon.csv \
  --output runs/dysyll_29_9m --resume runs/dysyll_29_9m/last.pt --device cuda
```

Các file đầu ra: `runs/dysyll_29_9m/best.pt`, `last.pt`, `metrics.json`.

**Loss:** `segmental_nll` tổng hợp các cách phân đoạn khả dĩ của transcript (latent segmentation), không yêu cầu frame-aligned labels. **Không dùng CTC/RNN-T**. Checkpoint lưu toàn bộ model, optimizer, config, hash lexicon. Sử dụng AdamW, gradient clipping, batch size=1, FP32 mặc định để bảo vệ độ ổn định của DP.

**Đánh giá:** `val_loss`, `WER_space_token` (đối với tiếng Việt chủ yếu là đơn vị tách bởi dấu cách), `CER_no_spaces`. Không có WER/CER cố định cho đến khi huấn luyện và test thật. Dùng val để chọn model, test để báo cáo kết quả cuối.

## 6. Suy luận sau khi có checkpoint đã train

```bash
python transcribe.py --checkpoint runs/dysyll_29_9m/best.pt \
  --lexicon lexicon.csv --audio /path/to/new_speech.wav --device cuda
```

`transcribe.py` kiểm tra lexicon SHA256 khớp với lúc train. ELD Viterbi giải mã qua lexicon để ghép chuỗi âm tiết thành văn bản.

## 7. Giới hạn quan trọng của bản nguyên mẫu

1. BATCH SIZE **1** và dữ liệu đầu vào không padding; không train trực tiếp mini-batch đa độ dài.
2. Chưa có blank/silence segment; cần tách bỏ các đoạn im lặng dài và chuẩn hóa transcript thật chính xác.
3. Một âm tiết cho một segment, độ dài mỗi segment từ 2–24 frame encoder (80–960 ms nếu hop log-Mel=10ms); clip quá ngắn/dài so với số âm tiết bị loại có báo cáo.
4. `librosa.pyin` ở CPU chậm; precompute một lần để train nhanh hơn.
5. Số tham số **không** quyết định chất lượng nhận dạng. Đoạn văn bản chưa được train sẽ ra kết quả ngẫu nhiên. Cần training với dataset đủ lớn và đánh giá trên dữ liệu thật.
6. Phiên bản này dùng hàm điểm độc lập cho âm đầu/vần/thanh, **chưa dùng mô hình ngôn ngữ** và chưa xử lý chấm câu, chữ số, code-switching.
7. Trên Kaggle, nên bật GPU rồi upload thư mục sau khi giải nén dưới dạng Dataset; chạy từ thư mục chứa file hoặc `cd` vào đó. Không tự giả định đường dẫn Kaggle/Drive của bạn.

## 8. Quy trình đúng nên làm

`(a) chuẩn hóa & tách dữ liệu` → `(b) kiểm chứng lexicon` → `(c) trích xuất và cache features` → `(d) smoke-test` → `(e) overfit 5–20 utterances` → `(f) train dữ liệu lớn` → `(g) đánh giá WER/CER trên test speaker-disjoint`.

**Lưu ý quan trọng:** Nếu mục tiêu của bạn là có **hai GRU thực sự đúng định nghĩa**, cần **thiết kế lại khối DDSE**, và tổng số tham số của sơ đồ phải được tính lại — không thể lấy nguyên 29.898.425 rồi gọi hai khối này là GRU chuẩn.

## 9. File Kaggle

`DySyll_ASR_Train_29_9M_Kaggle.ipynb` có các cell kiểm tra tham số, smoke test, kiểm toán dataset, trích đặc trưng và train. Để chạy, upload ZIP và dữ liệu của bạn lên Kaggle rồi chỉnh các biến `ZIP_PATH`, `TRAIN_CSV`, `VAL_CSV`, `LEXICON_CSV`.
