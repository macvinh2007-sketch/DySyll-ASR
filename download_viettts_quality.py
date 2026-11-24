"""Export a Hugging Face speech dataset to local ASR train/validation manifests.

Default dataset: Mihanh27/VietTTS-Quality; default split: train.
Does NOT invent phoneme/lexicon IDs or cut transcripts to fit an audio crop.

Preview schema without downloading full data:
    python download_viettts_quality.py --inspect-only
Try a small pilot:
    python download_viettts_quality.py --limit 100
Full export:
    python download_viettts_quality.py
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from lexicon_utils import normalize_text

TEXT_CANDIDATES = (
    'transcript', 'text', 'sentence', 'transcription', 'normalized_text',
    'text_normalized', 'content', 'utterance', 'label',
)
AUDIO_CANDIDATES = (
    'audio', 'audio_file', 'audio_path', 'audio_filepath', 'wav',
    'waveform', 'path', 'file',
)
SPEAKER_CANDIDATES = ('speaker_id', 'speaker', 'speaker_name', 'speaker_idx', 'voice_id')


def infer_column(keys: list[str], candidates: tuple[str, ...], explicit: str | None, kind: str) -> str:
    if explicit:
        if explicit not in keys:
            raise ValueError(f'Unknown {kind} column {explicit!r}. Available columns: {keys}')
        return explicit
    found = next((name for name in candidates if name in keys), None)
    if not found:
        raise ValueError(f'Cannot identify {kind} column; available columns: {keys}. Use --{kind}-column.')
    return found


def decode_audio(value: Any) -> tuple[np.ndarray, int]:
    """Handle Audio(decode=False) path/bytes and already decoded array records.

    Audio(decode=False) avoids the optional TorchCodec backend often required by
    newer versions of Hugging Face datasets.
    """
    import soundfile as sf
    import librosa

    if isinstance(value, dict):
        if value.get('array') is not None:
            y = np.asarray(value['array'], dtype=np.float32)
            sr = int(value['sampling_rate'])
        elif value.get('bytes') is not None:
            try:
                y, sr = sf.read(BytesIO(value['bytes']), dtype='float32', always_2d=False)
            except Exception as exc:
                raise ValueError('Could not decode embedded audio bytes (soundfile). Check codec/ffmpeg.') from exc
        elif value.get('path'):
            y, sr = librosa.load(str(value['path']), sr=None, mono=False)
        else:
            raise ValueError('Audio dict has neither array, bytes, nor path')
    elif isinstance(value, (str, Path)):
        y, sr = librosa.load(str(value), sr=None, mono=False)
    else:
        raise ValueError(f'Unsupported audio record type: {type(value).__name__}')

    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        # soundfile -> [frames, channels], librosa(mono=False) -> [channels, frames]
        if y.shape[0] <= 8 and y.shape[1] > 8:
            y = y.mean(axis=0)
        else:
            y = y.mean(axis=1)
    if y.ndim != 1 or len(y) == 0 or sr <= 0:
        raise ValueError(f'Invalid audio: shape={y.shape}, sampling_rate={sr}')
    if not np.isfinite(y).all():
        raise ValueError('Audio contains NaN or infinity')
    return y, int(sr)


def normalize_audio(y: np.ndarray, sr: int, trim_db: float) -> np.ndarray:
    import librosa

    if sr != 16_000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16_000)
    y, _ = librosa.effects.trim(y, top_db=trim_db, frame_length=512, hop_length=160)
    return np.asarray(y, dtype=np.float32)


def group_split(records: list[dict], ratio: float, seed: int, speaker_col: str | None,
                split_mode: str) -> tuple[list[dict], list[dict], str]:
    """No audio overlap across splits; speaker-disjoint split when possible."""
    if not 0 < ratio < 1:
        raise ValueError('--val-ratio must be strictly between 0 and 1')
    if len(records) < 2:
        raise ValueError('Need at least 2 accepted audio clips for train and validation')

    speakers = {r['speaker'] for r in records if r.get('speaker') is not None}
    use_speakers = bool(speaker_col) and len(speakers) >= 2 and all(r.get('speaker') is not None for r in records)
    if split_mode == 'speaker' and not use_speakers:
        raise ValueError('Speaker split requested but fewer than 2 complete speaker groups')
    use_speakers = split_mode != 'row' and use_speakers
    import random
    rng = random.Random(seed)
    if use_speakers:
        ids = sorted(speakers)
        rng.shuffle(ids)
        nval = max(1, min(len(ids) - 1, round(len(ids) * ratio)))
        val_ids = set(ids[:nval])
        train = [r for r in records if r['speaker'] not in val_ids]
        val = [r for r in records if r['speaker'] in val_ids]
        mode = 'speaker-disjoint'
    else:
        shuffled = records.copy()
        rng.shuffle(shuffled)
        nval = max(1, min(len(records) - 1, round(len(records) * ratio)))
        val = shuffled[:nval]
        train = shuffled[nval:]
        mode = 'row-random (speaker overlap not checked)'
    assert set(r['audio_path'] for r in train).isdisjoint(r['audio_path'] for r in val)
    return train, val, mode


def export_records(records, outdir: Path, text_col: str, audio_col: str, speaker_col: str | None,
                   *, limit: int, min_seconds: float, max_seconds: float,
                   trim_db: float, val_ratio: float, seed: int, split_mode: str,
                   overwrite: bool) -> dict:
    import soundfile as sf

    outdir.mkdir(parents=True, exist_ok=True)
    audio_dir = outdir / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifests = [outdir / 'train.csv', outdir / 'val.csv']
    if any(m.exists() for m in manifests) and not overwrite:
        raise FileExistsError(f'Manifest already exists in {outdir}. Use --overwrite to replace it.')

    kept: list[dict] = []
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    samples = []
    processed = 0
    for ix, r in enumerate(records):
        if limit and ix >= limit:
            break
        processed += 1
        try:
            text = normalize_text(r.get(text_col) or '')
            if not text:
                raise ValueError('empty transcript')
            y, sr = decode_audio(r[audio_col])
            y = normalize_audio(y, sr, trim_db)
            dur = len(y) / 16_000
            if dur < min_seconds or dur > max_seconds:
                raise ValueError(f'duration {dur:.2f}s outside [{min_seconds}, {max_seconds}]s')
            # Identical waveforms cannot leak across the exported splits.
            digest = hashlib.sha256(y.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError('duplicate audio waveform')
            seen.add(digest)
            path = audio_dir / f'utt_{ix:08d}_{digest[:12]}.wav'
            sf.write(str(path), y, 16_000, subtype='PCM_16')
            speaker = str(r[speaker_col]) if speaker_col and r.get(speaker_col) is not None else None
            kept.append({'audio_path': str(path.resolve()), 'transcript': text,
                         'speaker': speaker, 'sha256': digest})
            if len(kept) % 100 == 0:
                print(f'Processed {processed}; kept {len(kept)}; skipped {sum(rejected.values())}', flush=True)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            reason = str(exc)
            if reason.startswith('duration '):
                reason = 'duration outside model limits'
            rejected[reason] += 1
            if len(samples) < 25:
                samples.append({'row': ix, 'reason': str(exc)[:250]})

    if len(kept) < 2:
        raise ValueError(f'Only {len(kept)} clips accepted from {processed} rows; cannot make train/val. '
                         f'Rejected: {dict(rejected)}. Long TTS clips need transcript-aligned segmentation; '
                         'do not truncate speech without updating transcripts.')
    train, val, mode = group_split(kept, val_ratio, seed, speaker_col, split_mode)
    for file, subset in zip(manifests, (train, val)):
        with file.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['audio_path', 'transcript'])
            writer.writeheader()
            writer.writerows({k: row[k] for k in ('audio_path', 'transcript')} for row in subset)

    report = {
        'dataset_rows_scanned': processed, 'clips_accepted': len(kept),
        'train': len(train), 'validation': len(val), 'split_method': mode,
        'rejected': dict(rejected), 'example_rejections': samples,
        'audio_format': 'PCM16 WAV, 16kHz mono, silence-trimmed',
        'warning': 'No phonological lexicon generated. ASR training still requires a verified lexicon.csv. '
                   'Short clips must correspond exactly to their transcripts. Row-random splits may share speakers.'
    }
    (outdir / 'download_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='Mihanh27/VietTTS-Quality')
    parser.add_argument('--split', default='train', help='Hugging Face split to load')
    parser.add_argument('--output', default='data_raw', help='Root folder for WAV/CSV exports')
    parser.add_argument('--text-column', help='Explicit text column')
    parser.add_argument('--audio-column', help='Explicit audio column')
    parser.add_argument('--speaker-column', help='Explicit speaker column')
    parser.add_argument('--split-mode', choices=['auto', 'speaker', 'row'], default='auto')
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='0 means process the whole split')
    parser.add_argument('--min-seconds', type=float, default=0.2)
    parser.add_argument('--max-seconds', type=float, default=3.0)
    parser.add_argument('--trim-db', type=float, default=28.0)
    parser.add_argument('--inspect-only', action='store_true', help='Show schema and exit')
    parser.add_argument('--no-streaming', action='store_true', help='Download/cache the Hugging Face split before export')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing CSV manifests; WAV files may be reused')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit cannot be negative')
    if args.max_seconds < args.min_seconds or args.min_seconds <= 0:
        parser.error('Invalid clip duration bounds')
    if not 0 < args.val_ratio < 1:
        parser.error('--val-ratio must be between 0 and 1')

    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit('Missing Hugging Face datasets. Install: pip install -r requirements-hf.txt') from exc
    try:
        ds = load_dataset(args.dataset, split=args.split, streaming=not args.no_streaming)
    except Exception as exc:
        raise SystemExit(f'Cannot load {args.dataset!r} split {args.split!r}: {exc}. '
                         'Verify dataset ID, availability, terms, access permission, and split name.') from exc

    columns = list(ds.column_names or (ds.features or {}).keys())
    if not columns:
        raise SystemExit('Unable to discover dataset columns; inspect the dataset manually.')
    text_col = infer_column(columns, TEXT_CANDIDATES, args.text_column, 'text')
    audio_col = infer_column(columns, AUDIO_CANDIDATES, args.audio_column, 'audio')
    speaker_col = (infer_column(columns, SPEAKER_CANDIDATES, args.speaker_column, 'speaker')
                   if args.speaker_column else next((c for c in SPEAKER_CANDIDATES if c in columns), None))
    print('Dataset:', args.dataset, 'split:', args.split)
    print('Columns:', columns)
    print('Features:', str(ds.features)[:3000])
    print('Resolved:', {'audio': audio_col, 'text': text_col, 'speaker': speaker_col})
    if args.inspect_only:
        print('Inspect done. No audio/CSV files exported.')
        return
    if (ds.features or {}).get(audio_col) is not None and isinstance(ds.features[audio_col], Audio):
        ds = ds.cast_column(audio_col, Audio(decode=False))

    report = export_records(iter(ds), Path(args.output).resolve(), text_col, audio_col, speaker_col,
                            limit=args.limit, min_seconds=args.min_seconds,
                            max_seconds=args.max_seconds, trim_db=args.trim_db,
                            val_ratio=args.val_ratio, seed=args.seed,
                            split_mode=args.split_mode, overwrite=args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('NEXT: supply a *validated* data_raw/lexicon.csv; then bash server/prepare_server.sh')


if __name__ == '__main__':
    main()
