#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/common.sh"
for f in "$TRAIN_JSONL" "$VAL_JSONL" "$LEXICON_CSV"; do
  [[ -f "$f" ]] || { echo "Missing file: $f (chay prepare_server.sh truoc)" >&2; exit 2; }
done
if [[ "$DEVICE" == "cuda" ]]; then
  python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check NVIDIA driver and PyTorch wheel"'
fi
mkdir -p "$RUN_DIR"
if [[ "${RESUME:-0}" == "1" ]]; then
  [[ -f "$RUN_DIR/last.pt" ]] || { echo "Khong co checkpoint: $RUN_DIR/last.pt" >&2; exit 2; }
  echo "RESUMING: $RUN_DIR/last.pt"
  python -u train.py --config config.json --train "$TRAIN_JSONL" --val "$VAL_JSONL" --lexicon "$LEXICON_CSV" --output "$RUN_DIR" --device "$DEVICE" --resume "$RUN_DIR/last.pt" 2>&1 | tee -a "$RUN_DIR/train.log"
else
  if [[ -f "$RUN_DIR/last.pt" ]]; then
    echo "DA CO checkpoint $RUN_DIR/last.pt. Dung 'RESUME=1 bash server/train_server.sh' de tiep tuc, hoac doi RUN_DIR trong server.env." >&2
    exit 2
  fi
  python -u train.py --config config.json --train "$TRAIN_JSONL" --val "$VAL_JSONL" --lexicon "$LEXICON_CSV" --output "$RUN_DIR" --device "$DEVICE" 2>&1 | tee -a "$RUN_DIR/train.log"
fi
