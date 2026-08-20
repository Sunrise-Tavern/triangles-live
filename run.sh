#!/bin/bash
# Thin wrapper so you don't have to remember the venv path.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "No venv yet. Run ./setup.sh first." >&2
    exit 1
fi

exec ./.venv/bin/python make_sequence.py "$@"
