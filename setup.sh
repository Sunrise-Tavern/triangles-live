#!/bin/bash
# One-time setup for the sequence generator.
#
# librosa's dependency chain (numba/llvmlite) does not support Python 3.14,
# which is what `python3` resolves to on this machine.  So we pin a 3.12 venv.
set -euo pipefail

cd "$(dirname "$0")"

PY=${PY:-/opt/homebrew/bin/python3.12}

if [ ! -x "$PY" ]; then
    echo "Python 3.12 not found at $PY" >&2
    echo "Install it with:  brew install python@3.12" >&2
    echo "Or point PY at another 3.10-3.12 interpreter:  PY=/path/to/python3.11 ./setup.sh" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null; then
    echo "ffmpeg not found on PATH.  Install it with:  brew install ffmpeg" >&2
    exit 1
fi

echo "==> Creating venv with $("$PY" --version)"
"$PY" -m venv .venv

echo "==> Installing dependencies"
./.venv/bin/pip install --upgrade pip --quiet
./.venv/bin/pip install -r requirements.txt

echo "==> Verifying"
./.venv/bin/python -c "import librosa, soundfile, scipy, numpy, yt_dlp; print('librosa', librosa.__version__); print('numpy  ', numpy.__version__); print('yt_dlp ', yt_dlp.version.__version__)"

echo
echo "Done.  Generate a sequence with:"
echo "  ./run.sh 'https://www.youtube.com/watch?v=...'"
echo "  ./run.sh ~/Music/track.mp3"
