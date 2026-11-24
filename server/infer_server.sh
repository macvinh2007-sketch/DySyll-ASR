#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh"
AUDIO="${1:-}"
if [[ -z "$AUDIO" ]]; then
  echo "Usage: bash server/infer_server.sh /absolute/path/to/speech.wav" >&2
  exit 2
fi
[[ -f "$RUN_DIR/best.pt" ]] || { echo "Missing $RUN_DIR/best.pt" >&2; exit 2; }
python -u transcribe.py --checkpoint "$RUN_DIR/best.pt" --lexicon "$LEXICON_CSV" --audio "$AUDIO" --device "$DEVICE"
