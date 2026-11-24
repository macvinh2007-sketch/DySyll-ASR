"""Infer using trained checkpoint and the EXACT preprocessing/lexicon used in train."""
import argparse
import hashlib
from pathlib import Path
import torch
from dysyll_asr_diagram import DySyllASR, DySyllConfig
from lexicon_utils import load_lexicon
from feature_pipeline import load_compute


def main(args):
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    c=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if not c.get('trained',False):raise ValueError('Checkpoint not marked trained')
    digest=hashlib.sha256(Path(args.lexicon).read_bytes()).hexdigest()
    if digest!=c.get('lexicon_sha256'):raise ValueError('Lexicon mapping differs from training checkpoint')
    _,reverse=load_lexicon(args.lexicon)
    model=DySyllASR(DySyllConfig(**c['model_config']))
    model.load_state_dict(c['model_state']);model=model.to(device).eval()
    mel,pros,dur=load_compute(args.audio)
    if dur>args.max_seconds:
        raise ValueError('Utterance too long for this research prototype; segment into speech-only utterances first')
    x=torch.from_numpy(mel).unsqueeze(0).to(device)
    p=torch.from_numpy(pros).unsqueeze(0).to(device)
    with torch.inference_mode():
        out=model(x,p)
        segments=model.decode_one(out,reverse)
    print('Duration:',round(dur,3),'s')
    print('Prediction:',' '.join(map(str,segments)))
    print('Note: this model has no punctuation restoration, LM or internal-silence handling.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--lexicon',required=True);p.add_argument('--audio',required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--max-seconds',type=float,default=3.0)
    main(p.parse_args())
