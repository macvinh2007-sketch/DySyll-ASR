"""Validated Vietnamese syllable->(initial,rhyme,tone) label mapping.

WARNING: these integer IDs have NO linguistic meaning without a verified lexicon.
Never infer the 24/150 inventories by splitting strings heuristically.
"""
from __future__ import annotations
import csv
import json
import re
import unicodedata
from pathlib import Path

REQUIRED = {"syllable", "initial_id", "rhyme_id", "tone_id"}


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower().strip()
    # Separate punctuation rather than concatenate words: "ma,đi" -> "ma đi"
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return " ".join(s.split())


def tokenize_syllables(text: str) -> list[str]:
    return normalize_text(text).split()


def load_lexicon(path: str | Path, n_initials=24, n_rhymes=150, n_tones=6):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lexicon not found: {path}")
    entries = {}
    reverse = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        if not REQUIRED.issubset(set(rd.fieldnames or [])):
            raise ValueError(f"Lexicon requires columns {sorted(REQUIRED)}")
        for lineno, row in enumerate(rd, start=2):
            syll = normalize_text(row["syllable"])
            if not syll or len(syll.split()) != 1:
                raise ValueError(f"Invalid syllable at {path}:{lineno}: {syll!r}")
            try:
                trip = tuple(int(row[c]) for c in ("initial_id", "rhyme_id", "tone_id"))
            except Exception as exc:
                raise ValueError(f"Missing/invalid IDs at {path}:{lineno}") from exc
            if not (0 <= trip[0] < n_initials and 0 <= trip[1] < n_rhymes and 0 <= trip[2] < n_tones):
                raise ValueError(f"Out-of-range triple {trip} at {path}:{lineno}; max {n_initials,n_rhymes,n_tones}")
            if syll in entries:
                raise ValueError(f"Duplicated syllable {syll!r} in {path}")
            if trip in reverse:
                raise ValueError(f"Triple {trip} maps to BOTH {reverse[trip]!r} and {syll!r}; inverse must be unambiguous")
            entries[syll] = trip
            reverse[trip] = syll
    if not entries:
        raise ValueError("Lexicon is empty")
    return entries, reverse


def encode_transcript(text: str, forward: dict) -> list[list[int]]:
    syll = tokenize_syllables(text)
    if not syll:
        raise ValueError("Empty transcript after normalization")
    unseen = sorted(set(s for s in syll if s not in forward))
    if unseen:
        raise ValueError(f"Unmapped syllables (first 20): {unseen[:20]!r}. Build a VALIDATED syllable lexicon first")
    return [list(forward[s]) for s in syll]


def assert_not_toy(path: str | Path, allow_toy=False):
    if "toy" in Path(path).name.lower() and not allow_toy:
        raise ValueError("Toy lexicon is ONLY for engineering tests. Create a real verified lexicon, or pass --allow-toy solely for a demo.")
