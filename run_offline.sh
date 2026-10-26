#!/usr/bin/env bash
# run_offline.sh — start Aurobot fully offline, inside its own environment.
# Uses the venv's python directly, so it works whether or not you've "activated" anything
# (and never touches the online `venv`).
set -e
cd "$(dirname "$0")"

ENV_DIR="${AUROBOT_OFFLINE_ENV:-venv_offline}"
VPY="$ENV_DIR/bin/python"
if [ ! -x "$VPY" ]; then
  echo "Offline environment '$ENV_DIR' not found. Create it first:  ./setup_offline.sh"
  exit 1
fi
echo "-> Environment: $ENV_DIR ($("$VPY" --version))"

# No vision packages installed (setup --no-vision)? Run with the camera off instead of failing.
if ! "$VPY" -c "import deepface" >/dev/null 2>&1; then
  export AUROBOT_VISION_SOURCE="${AUROBOT_VISION_SOURCE:-none}"
  echo "-> Vision packages not installed: camera/emotion disabled."
fi

echo "-> Checking offline assets…"
"$VPY" scripts/prefetch_models.py --check || {
  echo; echo "Missing assets. Connect to the internet once and run:"
  echo "    $VPY scripts/prefetch_models.py"; exit 1; }

# Defaults for a 4 GB-VRAM GPU laptop; override any of these before calling.
export AUROBOT_VISION_DEVICE="${AUROBOT_VISION_DEVICE:-cpu}"   # GPU VRAM goes to the LLM
export AUROBOT_STRICT_OFFLINE="${AUROBOT_STRICT_OFFLINE:-1}"

exec "$VPY" agent_offline.py
