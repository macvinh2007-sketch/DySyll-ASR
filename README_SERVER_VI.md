# Train DySyll-ASR 29.898.425 tham số trên server Linux qua SSH

## Đây là bộ nào?

Đây là **bộ train từ đầu**, cùng code với gói trước nhưng bổ sung script dành cho server Linux. Mô hình có **29.898.425 tham số** theo sơ đồ mới: ARF 2.704.640; DDSE+LSC 19.219.205; EMC+SAG 7.974.580. Cấu hình **1600/1600/800**, đầu ra EMC **1600 chiều**, head **24/150/6**.

**Quan trọng:** Để khớp chính xác số tham số, mô hình dùng custom gated recurrence, **không phải nn.GRUCell chuẩn**; sơ đồ có chữ GRU nhưng phép tính bên trong code không có đầy đủ reset gate. Cần sửa kiến trúc/tính lại tham số nếu bắt buộc GRU chuẩn. Không có pretrained checkpoint và không có WER/CER thực nghiệm.

## 1. Đưa file ZIP từ Windows lên server

Trong PowerShell trên máy Windows (thay `SERVER_HOST` bằng tên/IP thực tế):

```powershell
scp .\DySyll_ASR_Train_Server_29_9M.zip ailp@SERVER_HOST:/home/iec/disk2/ailp/hoangnl/vinhmp/
```

Nếu bạn sử dụng tên miền SSH `iec` và thư mục trước đây là `/home/iec/disk2/ailp/hoangnl/vinhmp`, có thể thay `SERVER_HOST` thành `iec`. Nếu đường dẫn hay tài khoản đã khác, sửa lại theo server của bạn.

Trên server qua SSH:

```bash
cd /home/iec/disk2/ailp/hoangnl/vinhmp
unzip DySyll_ASR_Train_Server_29_9M.zip
cd DySyll_ASR_Train_Server_29_9M
```

## 2. Kiểm tra phần cứng

```bash
nvidia-smi
free -h
df -h .
python3 --version
```

Nếu không có NVIDIA GPU, đặt `DEVICE=cpu` trong `server/server.env` (chạy được nhưng huấn luyện cực chậm). Nếu `nvidia-smi` hoạt động nhưng `torch.cuda.is_available()` false, phải cài bản PyTorch hỗ trợ NVIDIA GPU và driver hiện tại. Không nên cố định phiên bản CUDA khi chưa kiểm tra GPU/driver.

## 3. Tạo .venv, cài thư viện và kiểm tra mô hình

```bash
bash server/setup_server.sh
source .venv/bin/activate
python server/preflight.py
python train.py --check-model
python train.py --smoke --device cuda
pytest -q tests/
```

`--smoke` chạy trên mô hình **thu nhỏ**, không phải bài benchmark full 29,9M. Nếu server không có GPU, thay `--device cuda` bằng `--device cpu`.

## 4. Đặt dataset trên server

Tạo trong thư mục dự án:

```bash
mkdir -p data_raw
```

Tạo `data_raw/train.csv`, `data_raw/val.csv` có format:

```csv
audio_path,transcript
/absolute/path/to/audio/utt001.wav,má đi
/absolute/path/to/audio/utt002.wav,xin chào việt nam
```

**BẮT BUỘC** có `data_raw/lexicon.csv` thực tế:

```csv
syllable,initial_id,rhyme_id,tone_id
ma,10,10,0
má,10,10,1
đi,5,11,0
```

**Các ID trên chỉ minh họa, không phải lexicon tiếng Việt chuẩn.** Bộ lexicon phải được xác minh và bao phủ **tất cả âm tiết của train/val**, với `initial_id` 0–23, `rhyme_id` 0–149, `tone_id` 0–5. File `examples/toy_lexicon.csv` chỉ dùng smoke test, **không dùng training thật**.

Dataset train/val cần tách biệt, ưu tiên không trùng speaker, transcript chính xác. Không dùng tập test để train. Bản code đang giới hạn clip speech-only **0,2–3 giây**, không mô hình hóa silence/blank, mỗi âm tiết ứng với một đoạn **2–24 encoder frames**. Nếu audio dài hoặc transcript chứa nhiều dấu lặng, cần cắt/căn chỉnh trước khi chạy.

## 5. Cấu hình đường dẫn một lần

```bash
cp server/server.env.example server/server.env
nano server/server.env
```

Nếu bạn tạo các file theo tên `data_raw/train.csv`, `data_raw/val.csv`, `data_raw/lexicon.csv` như trên thì cấu hình mặc định đã tương ứng. Có thể chỉnh đường dẫn tuyệt đối nếu dữ liệu ở ổ khác. `DEVICE=cuda` sử dụng GPU.

## 6. Chuẩn bị dữ liệu

```bash
bash server/prepare_server.sh
```

Nó sẽ kiểm tra đường dẫn và độ phủ lexicon, sau đó tạo feature cache (log-Mel 80 và prosody 4). `librosa.pyin` chạy trên CPU, nên tiền xử lý nhiều file có thể khá lâu. Các file lỗi sẽ được ghi ra báo cáo, không lặng lẽ bỏ qua.

## 7. Train trong tmux để ngắt SSH vẫn chạy

```bash
# Neu chua co tmux: can admin cai tmux (vi du sudo apt install tmux)
tmux new -s dysyll
source .venv/bin/activate
bash server/train_server.sh
```

Trong tmux, bấm **Ctrl+B rồi D** để detach. Sau khi reconnect:

```bash
tmux attach -t dysyll
```

Hoặc theo dõi log ở phiên SSH mới:

```bash
tail -f runs/dysyll_29_9m/train.log
nvidia-smi
```

Lưu ý: để train vẫn chạy khi ngắt SSH, phải thực sự chạy lệnh train **bên trong tmux**, không phải trong shell SSH thông thường.

Checkpoint: `runs/dysyll_29_9m/last.pt`, `best.pt`, và `metrics.json`. Dung lượng checkpoint có thể lớn do lưu cả optimizer AdamW; cần kiểm tra dung lượng ổ đĩa trước.

## 8. Resume nếu train bị dừng

```bash
RESUME=1 bash server/train_server.sh
```

Script mặc định **không** ghi đè checkpoint đã có khi chạy mới. Số epochs đọc từ `config.json` (mặc định 12). Nếu đã hoàn thành 12 epochs, resume sẽ không tạo epoch tiếp theo; phải tăng trường `training.epochs` trước.

## 9. Inference với checkpoint đã train

```bash
bash server/infer_server.sh /absolute/path/to/audio/test.wav
```

## 10. Lưu ý kỹ thuật quan trọng

- Mặc định **batch size 1**, FP32, không AMP; OOM hoặc tốc độ có thể phụ thuộc GPU và độ dài clip. Không hứa chạy được trên mọi GPU.
- Đây là **train từ đầu**, chưa có kết quả WER/CER. Train loss giảm chưa chứng minh chất lượng nhận dạng.
- Script không tự xây lexicon hợp lệ cho tiếng Việt. **Lexicon là điều kiện bắt buộc** cho nhánh SAG và ELD.
- Để đo hiệu quả thật, giữ riêng test set và đánh giá WER/CER trên dữ liệu không trùng train/val; phải kiểm soát trùng audio và speaker.
- Thời gian suy luận và bộ nhớ của segmental DP có thể tăng đáng kể theo độ dài audio và số đoạn ứng viên. Không nên train audio dài trước khi kiểm tra throughput.
- Khối ARF/EMC ở đây bám theo kích thước **1600 chiều** trong ảnh bạn gửi, không dùng bản 960 chiều cũ.

## 11. Tóm tắt lệnh

```bash
cd DySyll_ASR_Train_Server_29_9M
bash server/setup_server.sh
cp server/server.env.example server/server.env
# Sửa path dataset trong server/server.env
source .venv/bin/activate
python server/preflight.py
python train.py --check-model
bash server/prepare_server.sh
tmux new -s dysyll
bash server/train_server.sh
# Ctrl+B, D
```
