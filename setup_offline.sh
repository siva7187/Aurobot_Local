#!/usr/bin/env bash
# setup_offline.sh — creates a SEPARATE virtual environment for offline mode.
#
#   ./setup_offline.sh                 # core + vision deps
#   ./setup_offline.sh --no-vision     # skip TensorFlow/DeepFace (no emotion camera; much smaller)
#   ./setup_offline.sh --prefetch      # also download all models right away (needs internet)
#
# Env:  AUROBOT_OFFLINE_ENV=<dir>   (default: venv_offline)    PYTHON=<interpreter> (default: python3)
set -e
cd "$(dirname "$0")"

ENV_DIR="${AUROBOT_OFFLINE_ENV:-venv_offline}"
PY="${PYTHON:-python3}"
WITH_VISION=1
PREFETCH=0
for arg in "$@"; do
  case "$arg" in
    --no-vision) WITH_VISION=0 ;;
    --prefetch)  PREFETCH=1 ;;
    *) echo "Unknown option: $arg"; echo "Usage: ./setup_offline.sh [--no-vision] [--prefetch]"; exit 1 ;;
  esac
done

echo "== Aurobot offline environment =="

# --- Python version (TensorFlow/CTranslate2 wheels need 3.10-3.13) ---
ver=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
minor=${ver#*.}
if [ "${ver%%.*}" != "3" ] || [ "$minor" -lt 10 ] || [ "$minor" -gt 13 ]; then
  echo "Python $ver is not supported (need 3.10 - 3.13). Try:  PYTHON=python3.12 ./setup_offline.sh"
  exit 1
fi
echo "-> Using Python $ver ($("$PY" -c 'import sys; print(sys.executable)'))"

# --- create the separate venv ---
if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "-> Creating $ENV_DIR/ …"
  "$PY" -m venv "$ENV_DIR" || { echo "venv creation failed. On Debian/Ubuntu:  sudo apt install python3-venv python3-full"; exit 1; }
else
  echo "-> $ENV_DIR/ already exists — reusing it."
fi
VPY="$ENV_DIR/bin/python"     # always call the venv's python directly: no activation, no PATH surprises

echo "-> Upgrading pip …"
"$VPY" -m pip install --quiet --upgrade pip

echo "-> Installing core offline packages …"
"$VPY" -m pip install -r requirements-offline.txt

if [ "$WITH_VISION" = "1" ]; then
  echo "-> Installing vision packages (TensorFlow/DeepFace — several minutes, ~1 GB) …"
  "$VPY" -m pip install -r requirements-vision.txt
else
  echo "-> Skipping vision packages (--no-vision). run_offline.sh will start with the camera off."
fi

# --- system dependencies pip can't install: report, don't fail ---
echo "-> Checking system pieces …"
"$VPY" -c "import sounddevice" 2>/dev/null \
  && echo "   [ok]   PortAudio (microphone/speaker)" \
  || echo "   [MISSING] PortAudio →  sudo apt install portaudio19-dev libportaudio2"
command -v espeak-ng >/dev/null \
  && echo "   [ok]   espeak-ng (fallback voice)" \
  || echo "   [optional] espeak-ng →  sudo apt install espeak-ng"
command -v ollama >/dev/null \
  && echo "   [ok]   ollama" \
  || echo "   [MISSING] Ollama (local LLM) →  curl -fsSL https://ollama.com/install.sh | sh"

if [ "$PREFETCH" = "1" ]; then
  echo "-> Downloading models (needs internet) …"
  "$VPY" scripts/prefetch_models.py
fi

echo
echo "Done. Next:"
[ "$PREFETCH" = "1" ] || echo "  1) (online, once)  $VPY scripts/prefetch_models.py"
echo "  2) ./run_offline.sh"
