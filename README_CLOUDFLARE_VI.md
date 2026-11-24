# DySyll-ASR 29.9M — chạy trên máy Linux qua Cloudflare SSH

> **Phân biệt:** Cloudflare Tunnel là đường truy cập SSH; mã nguồn và training chạy trên **máy Linux phía sau tunnel**, không phải tài nguyên tính toán do Cloudflare cấp. Máy đó có GPU hay không phải kiểm tra bằng `nvidia-smi`.
>
> Host `inch-hill-health-casio.trycloudflare.com` là ví dụ đang được người dùng sử dụng. `trycloudflare.com` có thể đổi khi tunnel khởi động lại. Không dùng thư mục `vinhmp` của máy IEC trước đó; đường dẫn trong ví dụ dưới đây là `/root/projects/DySyll-ASR`.

## A. Đồng bộ GitHub từ Windows trước

Giải nén file ZIP mới **đè đúng mã nguồn** trong:

`C:\Users\Lenovo\Downloads\DySyll_ASR_Train_Server_29_9M`

**Đừng xóa `.git/`** trong folder cũ. PowerShell:

```powershell
cd "C:\Users\Lenovo\Downloads\DySyll_ASR_Train_Server_29_9M"
git status
git remote -v
# Chỉ nếu chưa có origin:
git remote add origin https://github.com/macvinh2007-sketch/DySyll-ASR.git
git add -A
git commit -m "Add HF VietTTS downloader and Cloudflare Linux guide"
git push -u origin main
```

Không đưa `data_raw/`, `.venv/`, checkpoint hay bí mật lên GitHub. Repo hiện Public; rà soát trước khi đẩy.

## B. Kết nối máy Linux qua Cloudflare

Trên Windows, vào PowerShell hoặc mở Remote-SSH từ Antigravity. Cấu hình SSH ví dụ:

```sshconfig
Host cloudflare-dysyll
    HostName inch-hill-health-casio.trycloudflare.com
    User root
    ProxyCommand C:/Users/Lenovo/.ssh/cloudflared.exe access ssh --hostname %h
    ServerAliveInterval 30
```

Nếu bạn đã có host SSH hoạt động khác thì dùng host đó, không cần thay. `cloudflared.exe` phải tồn tại tại đường dẫn ghi trong cấu hình. Sau khi kết nối:

```bash
whoami
pwd
python3 --version
nvidia-smi || true
df -h .
```

Không cần chạy Windows `py` hoặc `.venv\\Scripts\\activate` bên trong terminal Linux.

## C. Clone, thiết lập Python

```bash
mkdir -p /root/projects
cd /root/projects
git clone https://github.com/macvinh2007-sketch/DySyll-ASR.git
cd DySyll-ASR
bash server/setup_server.sh
source .venv/bin/activate
python -m pip install -r requirements-hf.txt
```

Nếu repo đã tồn tại, chạy `git pull --ff-only` trong repo thay cho clone. Nếu server không có quyền truy cập repo Private, cần xác thực GitHub qua SSH key hoặc token của **chính** bạn; không lưu token trong repo.

### Kiểm tra code trước khi tải dữ liệu

```bash
python train.py --check-model
pytest -q tests/test_hf_export.py
```

Nếu `nvidia-smi` không có GPU, vẫn có thể chuẩn bị dữ liệu trên CPU, nhưng training trên GPU yêu cầu máy có NVIDIA GPU và bản PyTorch có CUDA tương thích.

## D. Tải dataset Mihanh27/VietTTS-Quality (nếu truy cập được)

Chưa xác minh được dataset này công khai, các cột/split và bản quyền/điều khoản. Vì thế script **tự in schema**, báo lỗi cụ thể nếu dataset yêu cầu quyền, không tồn tại, hoặc có cấu trúc khác dự kiến.

```bash
python download_viettts_quality.py --inspect-only
# thử 100 hàng trước; có thể còn < 100 clip đạt yêu cầu
python download_viettts_quality.py --limit 100
# Nếu thành công và muốn xuất tất cả, chạy lại với overwrite:
python download_viettts_quality.py --overwrite
```

Mặc định script lấy split `train`, chạy theo chế độ `streaming` và tạo:

```text
data_raw/
├── audio/
│   ├── utt_00000000_*.wav
│   └── ...
├── train.csv
├── val.csv
└── download_report.json
```

Phần âm thanh xuất ra mono 16 kHz PCM WAV, chỉ giữ clip sau trim từ **0,2–3,0 giây** (giới hạn kỹ thuật của prototype hiện tại). **Không cắt bừa** audio dài trong khi giữ nguyên transcript. Nếu hầu hết đoạn TTS là câu dài và 0 clip đạt yêu cầu, cần chuẩn bị transcript-aligned segmentation/forced alignment trước khi train; script báo lỗi thay vì tạo nhãn sai. Nếu nhiều speakers, script ưu tiên tách speaker train/val; không có đủ thông tin speaker thì tách theo hàng và cảnh báo khả năng trùng speaker.

Nếu dataset có cột khác, chỉ định rõ:

```bash
python download_viettts_quality.py --inspect-only
python download_viettts_quality.py --audio-column AUDIO_COL --text-column TEXT_COL --speaker-column SPEAKER_COL --limit 100
```

Nếu dataset Private/Gated, trước tiên xem quyền và giấy phép trên Hugging Face; có thể chạy `hf auth login` **trên chính server** nếu bạn được cấp quyền. Không commit token.

## E. Trước khi train: cần lexicon tiếng Việt đã được kiểm định

Bộ model dùng `syllable -> initial_id, rhyme_id, tone_id`, không có cách gán tự động đáng tin cậy từ câu chữ. Bạn **phải cung cấp** file:

`data_raw/lexicon.csv`:

```csv
syllable,initial_id,rhyme_id,tone_id
```

Các cột cần ID trong giới hạn initial `0..23`, rhyme `0..149`, tone `0..5`. `examples/toy_lexicon.csv` chỉ dùng kiểm thử kỹ thuật; không được dùng để đánh giá chất lượng ASR. File lexicon cần bao phủ toàn bộ âm tiết train/val và không gán trùng ba ID cho hai âm tiết khác nhau.

```bash
cp server/server.env.example server/server.env
nano server/server.env
python server/preflight.py
bash server/prepare_server.sh
```

`prepare_server.sh` sẽ audit nhãn, kiểm tra đường dẫn và cache log-Mel/prosody. Nếu audit không đạt, cần sửa dataset/lexicon, không bỏ qua một cách mù quáng.

## F. Train khi có GPU và dữ liệu đã chuẩn bị

```bash
tmux new -s dysyll
source .venv/bin/activate
bash server/train_server.sh
```

Trong tmux: `Ctrl+B` rồi `D` để tách phiên SSH; để xem lại `tmux attach -t dysyll`. Checkpoint và log lưu dưới `runs/` (được `.gitignore`). Dùng `RESUME=1 bash server/train_server.sh` nếu muốn tiếp tục từ checkpoint.

Bản DySyll-ASR này là nghiên cứu/prototype **train từ đầu**, chưa có cam kết WER/CER hoặc khả năng train thành công với mọi GPU. Đừng công bố kết quả định lượng cho đến khi thực nghiệm và đánh giá trên test độc lập.
