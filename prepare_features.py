"""Precompute per-utterance speech features from a manifest CSV.

Usage: python prepare_features.py --manifest train.csv --lexicon lexicon.csv \
  --output data/train --out-manifest data/train.jsonl
Manifest CSV columns: audio_path,transcript. Relative paths resolved relative to CSV.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from feature_pipeline import load_compute
from lexicon_utils import load_lexicon, encode_transcript, normalize_text, assert_not_toy


def run(args):
    assert_not_toy(args.lexicon, args.allow_toy)
    forward, _ = load_lexicon(args.lexicon)
    manifest = Path(args.manifest).resolve()
    outdir = Path(args.output).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    accepted, errors = [], []
    with manifest.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for n, row in enumerate(rows):
        try:
            rel = row.get('audio_path') or row.get('path') or row.get('filepath')
            if not rel:
                raise ValueError('Missing audio_path/path/filepath')
            src = Path(rel)
            if not src.is_absolute():
                src = (manifest.parent / src).resolve()
            if not src.is_file():
                raise FileNotFoundError(src)
            text = normalize_text(row.get('transcript',''))
            labels = encode_transcript(text, forward)
            mel, prosody, dur = load_compute(str(src), trim_db=args.trim_db)
            if dur < args.min_seconds or dur > args.max_seconds:
                raise ValueError(f'Duration {dur:.3f}s outside [{args.min_seconds}, {args.max_seconds}]')
            encoded_frames = (mel.shape[0] + 3) // 4
            K = len(labels)
            if not (args.min_segment * K <= encoded_frames <= args.max_segment * K):
                raise ValueError(f'Incompatible alignment: T_encoder={encoded_frames}, syllables={K}, allowed [{args.min_segment*K},{args.max_segment*K}]; trim audio or revise segmentation')
            key = hashlib.sha256((str(src)+'|'+text).encode()).hexdigest()[:20]
            dest=outdir/f'{key}.npz'
            np.savez_compressed(dest, mel=mel, prosody=prosody)
            accepted.append(dict(features_path=str(dest),audio_path=str(src),
                          transcript=text,labels=labels,seconds=round(dur,4),
                          mel_frames=len(mel),encoder_frames=encoded_frames))
        except Exception as exc:
            errors.append(dict(row=n+2, audio_path=row.get('audio_path',''), error=str(exc)))
            if not args.skip_invalid:
                break
    out_manifest=Path(args.out_manifest).resolve()
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open('w',encoding='utf-8') as f:
        for item in accepted:
            f.write(json.dumps(item,ensure_ascii=False)+'\n')
    report = out_manifest.with_suffix('.errors.json')
    report.write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Accepted {len(accepted)}/{len(rows)}; rejected {len(errors)}. Features {outdir}; manifest {out_manifest}; error report {report}')
    if errors and not args.skip_invalid:
        raise RuntimeError(f'Failed at row {errors[0]["row"]}: {errors[0]["error"]}. Fix dataset or use --skip-invalid with audited error report.')
    if not accepted:
        raise ValueError('No usable training data')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--lexicon',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--out-manifest',required=True)
    p.add_argument('--min-seconds',type=float,default=.20)
    p.add_argument('--max-seconds',type=float,default=3.0)
    p.add_argument('--trim-db',type=float,default=28.0)
    p.add_argument('--min-segment',type=int,default=2)
    p.add_argument('--max-segment',type=int,default=24)
    p.add_argument('--skip-invalid',action='store_true')
    p.add_argument('--allow-toy',action='store_true',help='Engineering smoke tests ONLY')
    run(p.parse_args())
