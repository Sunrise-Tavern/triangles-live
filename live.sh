#!/bin/bash
# Thin wrapper for the live engine, mirroring run.sh.
#
#   ./live.sh selftest                 byte-exact round trip, no hardware
#   ./live.sh demo                     sender + fake Falcon, writes out/capture.fseq
#   ./live.sh render                   test pattern -> out/pattern.fseq for xLights
#   ./live.sh pattern --host 192.168.1.20 --loop      drive one controller
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "No venv yet. Run ./setup.sh first." >&2
    exit 1
fi

exec ./.venv/bin/python -m live "$@"
