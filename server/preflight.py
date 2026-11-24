"""Kiem tra moi truong may chu va tham so kien truc. Khong can dataset."""
import importlib,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
print('Python:',sys.version.split()[0])
print('Working dir:',ROOT)
for cmd in ('nvidia-smi','tmux'):
    found=shutil.which(cmd)
    print(cmd+':',found or 'khong thay')
if shutil.which('nvidia-smi'):
    subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],check=False)
usage=shutil.disk_usage(ROOT)
print('Free disk:',round(usage.free/1024**3,2),'GiB')
for mod in ('torch','numpy','librosa','soundfile'):
    try:
        m=importlib.import_module(mod)
        print(mod+':',getattr(m,'__version__','ok'))
    except Exception as e:
        print(mod+': ERROR',repr(e))
import torch,json
print('CUDA available:',torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        g=torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {g.name} / {g.total_memory/1024**3:.1f} GiB')
from train import create_model
conf=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
model,counts=create_model(conf['model'],verify=True)
print('VERIFIED PARAMETER COUNTS:',counts)
