"""Train the diagram-matching DySyll-ASR 29,898,425-parameter prototype.

This uses *segmental marginal NLL*, not CTC/RNNT. No pretrained checkpoint.
BATCH SIZE IS ONE. No internal-silence/blank modeling. Precompute features.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from dysyll_asr_diagram import DySyllASR, DySyllConfig
from lexicon_utils import load_lexicon, assert_not_toy, normalize_text
from metrics import ErrorRate

EXPECTED = {'ARF':2704640,'DDSE_LSC':19219205,'EMC_SAG':7974580,'total':29898425}


def create_model(conf, verify=True):
    cfg=DySyllConfig(**conf)
    model=DySyllASR(cfg)
    counts={'ARF':sum(p.numel() for p in model.frontend.parameters()),
            'DDSE_LSC':sum(p.numel() for p in model.dynamics.parameters()),
            'EMC_SAG':sum(p.numel() for p in model.compressor.parameters())}
    counts['total']=sum(p.numel() for p in model.parameters())
    if verify and counts!=EXPECTED:
        raise ValueError(f'Model != the screenshot! Expected {EXPECTED}, got {counts}')
    return model,counts


def load_prepared(path):
    data=[]
    with Path(path).open(encoding='utf-8') as f:
        for lineno,line in enumerate(f,1):
            if not line.strip():continue
            obj=json.loads(line)
            for k in ('features_path','audio_path','transcript','labels'):
                if k not in obj:raise ValueError(f'{path}:{lineno}: missing {k}')
            if not Path(obj['features_path']).is_file():
                raise FileNotFoundError(f'Missing prepared features: {obj["features_path"]}')
            data.append(obj)
    if not data:raise ValueError(f'Empty manifest {path}')
    return data


def check_no_overlap(train,val):
    a={str(Path(x['audio_path']).resolve()) for x in train}
    b={str(Path(x['audio_path']).resolve()) for x in val}
    if a & b:
        raise ValueError(f'TRAIN/VAL LEAKAGE: same audio files in both splits: {list(a&b)[:5]}')


def to_tensors(row,device):
    with np.load(row['features_path'],allow_pickle=False) as z:
        mel=np.array(z['mel'],dtype=np.float32,copy=True)
        pro=np.array(z['prosody'],dtype=np.float32,copy=True)
    if mel.ndim!=2 or mel.shape[1]!=80 or pro.shape!=(mel.shape[0],4):
        raise ValueError(f'Bad feature shapes {mel.shape}, {pro.shape}: {row["features_path"]}')
    if not np.isfinite(mel).all() or not np.isfinite(pro).all():
        raise ValueError(f'NaN/inf feature file: {row["features_path"]}')
    target=np.asarray(row['labels'],dtype=np.int64)
    if target.ndim!=2 or target.shape[1]!=3:
        raise ValueError('Targets require [K,3] (initial,rhyme,tone)')
    T=(len(mel)+3)//4;K=len(target)
    # Guaranteed to match original (unchanged) model min/max duration.
    if not (2*K<=T<=24*K):
        raise ValueError(f'Invalid sample T_enc={T}, syllables={K}, path={row["audio_path"]}')
    return torch.from_numpy(mel[None]).to(device),torch.from_numpy(pro[None]).to(device),torch.from_numpy(target).to(device)


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def save_checkpoint(path,model,opt,epoch,best,config,lexicon_path,history):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'model_state':model.state_dict(), 'optimizer_state':opt.state_dict(),
           'model_config':asdict(model.cfg),'train_config':config,'epoch':epoch,
           'best_val_loss':best,'history':history,'trained':True,
           'lexicon_sha256':hashlib.sha256(Path(lexicon_path).read_bytes()).hexdigest(),
           'architecture':'DySyll 29,898,425 custom-gate diagram, not nn.GRUCell',
           'feature_version':'logmel_80_25ms_10ms_pyin_prosody4_v1'}
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(state,tmp);os.replace(tmp,path)


def evaluate(model,data,reverse,device,max_decodes):
    model.eval();loss_sum=0;count=0;er=ErrorRate();start=time.time()
    with torch.no_grad():
        for idx,row in enumerate(data):
            mel,pro,y=to_tensors(row,device)
            out=model(mel,pro)
            loss=model.segmental_nll(out,y)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite validation loss at {row["audio_path"]}')
            loss_sum+=float(loss.item());count+=1
            if idx < max_decodes:
                hyp=' '.join(map(str,model.decode_one(out,reverse)))
                er.add(row['transcript'],hyp)
            del mel,pro,y,out,loss
    stats=er.report()
    return {'loss':loss_sum/max(count,1), **stats,'elapsed_seconds':round(time.time()-start,2)}


def do_smoke(device):
    """Actual forward/backward numerical check with a tiny width; NOT a full 29.9M benchmark."""
    cfg=dict(n_mels=80,n_prosody=4,hidden=64,phonetic_dim=48,
             prosodic_dim=24,n_initials=24,n_rhymes=150,n_tones=6,
             min_segment=2,max_segment=12)
    model,_=create_model(cfg,verify=False)
    model=model.to(device);model.train()
    mel=torch.randn(1,48,80,device=device)
    pro=torch.randn(1,48,4,device=device)
    y=torch.tensor([[10,10,1],[5,11,0],[12,15,0]],device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=0.001)
    values=[]
    for i in range(2):
        opt.zero_grad(set_to_none=True)
        out=model(mel,pro);loss=model.segmental_nll(out,y)
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite smoke-test loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step();values.append(float(loss.detach()))
    assert all(np.isfinite(x) for x in values)
    print('SMOKE TEST PASSED; tiny-config loss:',values,'(not ASR metrics)')


def main(args):
    conf=json.loads(Path(args.config).read_text(encoding='utf-8'))
    tc=conf['training']
    seed_everything(tc['seed'])
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    if args.smoke:
        do_smoke(device);return
    if args.check_model:
        m,counts=create_model(conf['model'],verify=True)
        print('Screenshot EXACT parameter groups:',json.dumps(counts,indent=2))
        return
    for a in ('train','val','lexicon'):
        if not getattr(args,a):raise ValueError(f'--{a} is required to train')
    assert_not_toy(args.lexicon,args.allow_toy)
    forward,reverse=load_lexicon(args.lexicon)
    tr=load_prepared(args.train);va=load_prepared(args.val)
    check_no_overlap(tr,va)
    for section,items in (('train',tr),('val',va)):
        for r in items:
            syll=normalize_text(r['transcript']).split()
            if len(syll)!=len(r['labels']) or any(list(forward.get(s,[]))!=trip for s,trip in zip(syll,r['labels'])):
                raise ValueError(f'{section}: labels mismatch lexicon or transcript: {r["audio_path"]}')
    model,counts=create_model(conf['model'],verify=True)
    print('Verified exact diagram parameters:',counts,'device:',device)
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
    best=float('inf');history=[];first_epoch=1
    if args.resume:
        ckpt=torch.load(args.resume,map_location='cpu',weights_only=True)
        if ckpt['model_config']!=asdict(model.cfg):raise ValueError('Model configuration changed on resume')
        if ckpt['lexicon_sha256']!=hashlib.sha256(Path(args.lexicon).read_bytes()).hexdigest():
            raise ValueError('Lexicon changed since checkpoint; cannot safely resume')
        model.load_state_dict(ckpt['model_state']);opt.load_state_dict(ckpt['optimizer_state'])
        first_epoch=ckpt['epoch']+1;best=ckpt['best_val_loss'];history=ckpt['history']
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    for epoch in range(first_epoch,tc['epochs']+1):
        model.train();random.shuffle(tr);losses=[];start=time.time()
        for idx,row in enumerate(tr,1):
            mel,pro,y=to_tensors(row,device)
            opt.zero_grad(set_to_none=True)
            out=model(mel,pro)
            loss=model.segmental_nll(out,y)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite train loss: {row["audio_path"]}')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc['max_grad_norm'])
            if not torch.isfinite(norm):raise FloatingPointError(f'Invalid gradient: {row["audio_path"]}')
            opt.step();losses.append(float(loss.detach()))
            if idx%max(1,args.log_every)==0:
                print(f'epoch {epoch}/{tc["epochs"]} step {idx}/{len(tr)} train_nll {np.mean(losses[-args.log_every:]):.4f}',flush=True)
            del mel,pro,y,out,loss
        val=evaluate(model,va,reverse,device,max_decodes=tc.get('eval_max_utterances',40))
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'val_loss':val['loss'],
             'val_wer_space_token':val['wer_space_token'],'val_cer_no_spaces':val['cer_no_spaces'],
             'decoded_val_utterances':val['utt_count'], 'epoch_seconds':round(time.time()-start,2)}
        history.append(row)
        (outdir/'metrics.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
        save_checkpoint(outdir/'last.pt',model,opt,epoch,best,conf,args.lexicon,history)
        if val['loss']<best:
            best=val['loss']
            save_checkpoint(outdir/'best.pt',model,opt,epoch,best,conf,args.lexicon,history)
        print('EPOCH:',json.dumps(row,ensure_ascii=False),flush=True)
    print('Training completed. Best checkpoint:',outdir/'best.pt')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config.json');p.add_argument('--train');p.add_argument('--val');p.add_argument('--lexicon')
    p.add_argument('--output',default='runs/dysyll_29_9m');p.add_argument('--resume')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--log-every',type=int,default=10)
    p.add_argument('--allow-toy',action='store_true',help='Only for engineering tests')
    p.add_argument('--check-model',action='store_true',help='Check 29,898,425 parameters without forward')
    p.add_argument('--smoke',action='store_true',help='Numerical forward/backward with TINY dimensions')
    main(p.parse_args())
