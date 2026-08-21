#!/bin/bash
# One-time setup for the live engine on a Raspberry Pi.
#
# Deliberately does NOT install the offline generator's dependencies --
# librosa, numba, scipy, soundfile and yt-dlp are a slow, fragile build on a Pi
# and none of them run at the rig.
#
#   sudo apt install git
#   git clone <this repo> && cd Triangles/generated
#   ./deploy/install-pi.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="$(pwd)"
USER_NAME="${SUDO_USER:-$USER}"

echo "==> System packages"
sudo apt-get update
# python3-aubio from apt rather than pip: the 2019 release needs two build
# flags to compile against a modern numpy and ffmpeg (see setup.sh), and
# Debian has already done that work.
sudo apt-get install -y \
    python3-venv python3-dev python3-aubio \
    ffmpeg libportaudio2 libsndfile1

echo "==> Virtualenv (with system site packages, so python3-aubio is visible)"
if [ ! -d .venv ]; then
    python3 -m venv --system-site-packages .venv
fi
./.venv/bin/pip install --upgrade pip --quiet
./.venv/bin/pip install -r requirements-live.txt

echo "==> Verifying"
./.venv/bin/python -c "import numpy, aiohttp, sounddevice, aubio; \
print('numpy', numpy.__version__); print('aubio', aubio.version)"

if [ ! -f live.toml ]; then
    echo "==> No live.toml yet; copy the template and edit it"
    exit 1
fi

echo "==> systemd unit"
sed -e "s#__DIR__#${DIR}#g" -e "s#__USER__#${USER_NAME}#g" \
    deploy/triangles-live.service | sudo tee /etc/systemd/system/triangles-live.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable triangles-live

echo
echo "==> Preflight"
./.venv/bin/python -m live doctor || true

cat <<NOTES

Next:
  1. Edit live.toml -- [output] host is the Falcon (192.168.50.20), and
     [audio] device is the USB interface.  Find it with:
         ./live.sh listen --devices
  2. Give this Pi a static address on the same subnet as the Falcon.
  3. Start it:
         sudo systemctl start triangles-live
         journalctl -u triangles-live -f
  4. Open http://$(hostname -I | awk '{print $1}'):8080 from a phone.

Re-run ./live.sh doctor any time; it is the ten-minutes-before-doors check.
NOTES
