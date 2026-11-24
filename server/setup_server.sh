#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
command -v python3 >/dev/null || { echo 'Khong tim thay python3'; exit 1; }
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PYCHK'
import sys,torch
print('Python:',sys.version.split()[0])
print('PyTorch:',torch.__version__)
print('CUDA available:',torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        prop=torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {prop.name}, VRAM={prop.total_memory/1024**3:.1f} GiB')
else:
    print('Luu y: Chua thay GPU CUDA. Kiem tra nvidia-smi va cach cai PyTorch CUDA phu hop voi driver.')
PYCHK
echo 'Da tao .venv. Tiep: cp server/server.env.example server/server.env; chinh duong dan dataset.'
