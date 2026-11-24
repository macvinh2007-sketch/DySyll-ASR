"""Offline tests: no Hugging Face network access and no GPU required."""
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from download_viettts_quality import infer_column, export_records, group_split


def _audio_bytes(freq=220):
    t = np.arange(16_000, dtype=np.float32) / 16_000
    sine = 0.2 * np.sin(2 * np.pi * freq * t)
    f = BytesIO()
    sf.write(f, sine, 16_000, format='WAV', subtype='PCM_16')
    return f.getvalue()


def test_column_detection():
    assert infer_column(['audio', 'text'], ('text',), None, 'text') == 'text'
    with pytest.raises(ValueError, match='Cannot identify'):
        infer_column(['speech'], ('text',), None, 'text')


def test_export_wav_and_manifests(tmp_path):
    records = [
        {'audio': {'bytes': _audio_bytes(220)}, 'text': 'Xin chào!', 'speaker_id': 1},
        {'audio': {'bytes': _audio_bytes(270)}, 'text': 'Chào bạn.', 'speaker_id': 2},
        {'audio': {'bytes': _audio_bytes(320)}, 'text': 'Hôm nay đẹp.', 'speaker_id': 1},
    ]
    report = export_records(records, tmp_path, 'text', 'audio', 'speaker_id',
        limit=0, min_seconds=.2, max_seconds=3.0, trim_db=28.0,
        val_ratio=.2, seed=42, split_mode='speaker', overwrite=False)
    assert report['clips_accepted'] == 3
    assert report['train'] > 0 and report['validation'] > 0
    assert report['split_method'] == 'speaker-disjoint'
    import csv
    paths = []
    for name in ('train.csv', 'val.csv'):
        with (tmp_path / name).open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        assert rows
        paths += [r['audio_path'] for r in rows]
    assert len(set(paths)) == 3
    for file in paths:
        y, sr = sf.read(file)
        assert sr == 16_000 and len(y) > 3000


def test_deduplicate_and_limits(tmp_path):
    b = _audio_bytes()
    data = [{'audio': {'bytes': b}, 'text': 'Xin chào'}] * 2
    with pytest.raises(ValueError, match='Only 1 clips accepted'):
        export_records(data, tmp_path, 'text', 'audio', None,
            limit=0, min_seconds=.2, max_seconds=3.0,
            trim_db=28.0, val_ratio=.2, seed=42,
            split_mode='auto', overwrite=False)
