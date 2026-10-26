#!/usr/bin/env python3
"""
One-time setup: download every model Aurobot needs, WHILE YOU HAVE INTERNET.
After this, offline mode never touches the network.

    python scripts/prefetch_models.py            # download everything
    python scripts/prefetch_models.py --check    # verify everything is present (works offline)

Downloads:
  1. faster-whisper speech model        (~460 MB for "small")
  2. Piper voice (.onnx + .json)        (~65 MB)
  3. DeepFace emotion weights           (~6 MB, plus MTCNN which ships in the pip package)
  4. Ollama LLM                         (~1.9 GB for qwen2.5:3b)  via `ollama pull`
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from offline.config import OfflineConfig  # noqa: E402

PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
PIPER_FILES = ["en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"]

cfg = OfflineConfig.from_env()


def say(ok, msg):
    print(("  [ok]   " if ok else "  [FAIL] ") + msg)
    return ok


def whisper(check):
    try:
        from faster_whisper import WhisperModel
        WhisperModel(cfg.stt_model, device="cpu", compute_type="int8", local_files_only=check)
        return say(True, f"speech model '{cfg.stt_model}'")
    except Exception as e:
        return say(False, f"speech model '{cfg.stt_model}': {e}")


def piper(check):
    voice_dir = os.path.dirname(os.path.join(ROOT, cfg.piper_voice)) or "."
    os.makedirs(voice_dir, exist_ok=True)
    ok = True
    for name in PIPER_FILES:
        dest = os.path.join(voice_dir, name)
        if os.path.exists(dest):
            say(True, f"piper voice file {name}")
            continue
        if check:
            ok = say(False, f"piper voice file missing: {dest}")
            continue
        try:
            print(f"  downloading {name} …")
            urllib.request.urlretrieve(PIPER_BASE + name, dest)
            say(True, f"piper voice file {name}")
        except Exception as e:
            ok = say(False, f"{name}: {e}  (check the URL at huggingface.co/rhasspy/piper-voices)")
    return ok


def deepface(check):
    import importlib.util
    if cfg.vision_source == "none" or importlib.util.find_spec("deepface") is None:
        say(True, "DeepFace emotion weights — skipped (vision off / not installed)")
        return True
    pattern = os.path.expanduser("~/.deepface/weights/*expression*")
    if check:
        return say(bool(glob.glob(pattern)), "DeepFace emotion weights")
    try:
        import numpy as np
        from deepface import DeepFace
        DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), actions=["emotion"],
                         enforce_detection=False, detector_backend="mtcnn")
        return say(True, "DeepFace emotion weights")
    except Exception as e:
        return say(False, f"DeepFace: {e}")


def ollama(check):
    from offline.llm import OllamaClient, OllamaError
    client = OllamaClient(cfg.ollama_url, cfg.ollama_model)
    try:
        client.check()
        return say(True, f"Ollama model '{cfg.ollama_model}'")
    except OllamaError as e:
        if check:
            return say(False, str(e).splitlines()[0] + "  → " + (str(e).splitlines() or [""])[-1])
    if not shutil.which("ollama"):
        return say(False, "ollama not installed — see https://ollama.com/download (Linux: curl -fsSL https://ollama.com/install.sh | sh)")
    print(f"  pulling {cfg.ollama_model} …")
    rc = subprocess.call(["ollama", "pull", cfg.ollama_model])
    return say(rc == 0, f"Ollama model '{cfg.ollama_model}'")


def espeak(_check):
    return say(bool(shutil.which("espeak-ng")), "espeak-ng (optional fallback voice; sudo apt install espeak-ng)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; no downloads")
    args = ap.parse_args()
    print("== Aurobot offline assets ==" + ("  (check only)" if args.check else ""))
    results = [whisper(args.check), piper(args.check), deepface(args.check), ollama(args.check)]
    espeak(args.check)
    print("\nAll required assets present — safe to go offline." if all(results)
          else "\nSome required assets are missing (see [FAIL] above).")
    sys.exit(0 if all(results) else 1)
