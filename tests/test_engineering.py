import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from dysyll_asr_diagram import DySyllASR,DySyllConfig
from lexicon_utils import normalize_text, load_lexicon,encode_transcript
from train import create_model,do_smoke
from metrics import ErrorRate


def test_exact_diagram_parameters():
    c=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())
    _,n=create_model(c['model'],verify=True)
    assert n==c['expected_parameters']


def test_lexicon_and_vietnamese_unicode():
    p=Path(__file__).resolve().parents[1]/'examples/toy_lexicon.csv'
    forward,reverse=load_lexicon(p)
    assert normalize_text(' MÁ,  ĐI! ')=='má đi'
    assert encode_transcript('Má đi',forward)==[[10,10,1],[5,11,0]]
    assert reverse[(10,10,1)]=='má'
    with pytest.raises(ValueError,match='Unmapped'):
        encode_transcript('không',forward)


def test_corpus_metrics():
    e=ErrorRate();e.add('má đi','má');e.add('xin','xin')
    r=e.report();assert r['wer_space_token']==pytest.approx(1/3)
    assert r['cer_no_spaces']>0


def test_tiny_forward_backward():
    torch.set_num_threads(2)
    do_smoke('cpu')
