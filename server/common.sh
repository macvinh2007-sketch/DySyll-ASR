#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -f server/server.env ]]; then
  echo "CHUA CO server/server.env. Chay: cp server/server.env.example server/server.env; nano server/server.env" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1091
source server/server.env
set +a
if [[ ! -f .venv/bin/activate ]]; then
  echo "CHUA CO .venv. Chay: bash server/setup_server.sh" >&2
  exit 2
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONUNBUFFERED=1
