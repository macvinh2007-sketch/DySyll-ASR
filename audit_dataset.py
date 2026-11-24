"""Preflight checks for transcript coverage and obvious split leaks before costly audio loading."""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from lexicon_utils import load_lexicon, normalize_text, tokenize_syllables, assert_not_toy


def load_manifest(name,path,lexicon):
    data=[];missing=Counter();errors=[]
    path=Path(path).resolve()
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        for line,row in enumerate(csv.DictReader(f),2):
            p=row.get('audio_path') or row.get('filepath') or row.get('path')
            if not p:
                errors.append(f'{name}:{line} missing audio_path');continue
            aud=Path(p);aud=aud if aud.is_absolute() else path.parent/aud
            aud=aud.resolve()
            if not aud.is_file():errors.append(f'{name}:{line} audio not found: {aud}')
            text=normalize_text(row.get('transcript',''))
            syll=tokenize_syllables(text)
            if not syll:errors.append(f'{name}:{line} empty transcript')
            missing.update(s for s in syll if s not in lexicon)
            data.append(dict(path=str(aud),text=text,syllables=len(syll)))
    return data,missing,errors


def main(a):
    assert_not_toy(a.lexicon,a.allow_toy)
    forward,_=load_lexicon(a.lexicon)
    paths={k:v for k,v in [('train',a.train),('val',a.val),('test',a.test)] if v}
    results={k:load_manifest(k,v,forward) for k,v in paths.items()}
    leaks={}
    for i,name1 in enumerate(paths):
        for name2 in list(paths)[i+1:]:
            s1={v['path'] for v in results[name1][0]}
            s2={v['path'] for v in results[name2][0]}
            overlap=s1&s2
            if overlap:leaks[f'{name1}/{name2}']=list(sorted(overlap))[:20]
    report={'lexicon_size':len(forward),'splits':{
        k:{'utterances':len(vals[0]),'syllables':sum(x['syllables'] for x in vals[0]),
           'unmapped_syllables':vals[1].most_common(50),
           'errors':vals[2][:100]} for k,vals in results.items()},
            'path_overlaps':leaks,
            'warning':'Speaker-disjoint/content-hash leaks and lexicon phonological correctness REQUIRE separate verification.'}
    Path(a.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2)[:10000])
    errors=sum(len(vals[2])+sum(vals[1].values()) for vals in results.values())
    if leaks or errors:
        raise SystemExit('Dataset audit FAILED: fix missing labels, paths, or split leaks before training.')
    print('Dataset preflight PASSED (file paths + transcript coverage only).')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--val',required=True)
    p.add_argument('--test');p.add_argument('--lexicon',required=True);p.add_argument('--report',default='dataset_audit.json')
    p.add_argument('--allow-toy',action='store_true')
    main(p.parse_args())
