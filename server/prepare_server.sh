#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh"
for f in "$TRAIN_CSV" "$VAL_CSV" "$LEXICON_CSV"; do
  [[ -f "$f" ]] || { echo "Missing file: $f" >&2; exit 2; }
done
mkdir -p "$(dirname "$TRAIN_JSONL")" "$(dirname "$VAL_JSONL")" "$FEATURE_TRAIN_DIR" "$FEATURE_VAL_DIR"
python -u audit_dataset.py --train "$TRAIN_CSV" --val "$VAL_CSV" --lexicon "$LEXICON_CSV" --report dataset_audit.json
python -u prepare_features.py --manifest "$TRAIN_CSV" --lexicon "$LEXICON_CSV" --output "$FEATURE_TRAIN_DIR" --out-manifest "$TRAIN_JSONL"
python -u prepare_features.py --manifest "$VAL_CSV" --lexicon "$LEXICON_CSV" --output "$FEATURE_VAL_DIR" --out-manifest "$VAL_JSONL"
echo "READY: $TRAIN_JSONL and $VAL_JSONL"
