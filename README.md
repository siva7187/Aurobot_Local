# Aurobot — Offline Mode

`agent_offline.py` is the sole supported Aurobot agent entry point. The runtime is designed to operate locally without cloud services.

```
mic → VAD → faster-whisper → Ollama (streaming) → sentence splitter → Piper → speaker
                                  ↑
      emotion (DeepFace, own process) · memory (SQLite) · proactive policy   ← all reused
```

## Separate environment

Offline mode has its **own virtual environment** (`venv_offline/`), independent of the online `venv/`.
The offline environment uses the dependencies in `requirements-offline.txt` and, optionally, `requirements-vision.txt`.

| Task | Command |
|---|---|
| Install offline dependencies | `./setup_offline.sh` |
| Run Aurobot | `./run_offline.sh` |
| Run tests | `venv_offline/bin/python -m unittest discover -s tests -v` |

The scripts call `venv_offline/bin/python` directly, so you never need to `activate` anything.

## One-time setup (needs internet once)

```bash
sudo apt install python3-venv portaudio19-dev libportaudio2 espeak-ng
curl -fsSL https://ollama.com/install.sh | sh          # local LLM server

./setup_offline.sh --prefetch      # creates venv_offline, installs packages, downloads all models
```

Options: `--no-vision` skips TensorFlow/DeepFace (no emotion camera, ~1 GB smaller; offline mode then
starts with the camera off automatically). Needs Python 3.10-3.13; use `PYTHON=python3.12 ./setup_offline.sh`
if your default is newer.

## Run (internet can be off)

```bash
./run_offline.sh                                       # checks assets, then starts
venv_offline/bin/python -m unittest tests.test_offline_pipeline -v     # optional: 23 tests, no hardware
```

Prove it: turn Wi-Fi off, run `./run_offline.sh`. Startup prints `[ready]`; each reply prints a
`[latency]` line (STT time, LLM-to-first-sentence, speech-end to first audio) so you can measure.

## Why it stays fast in long sessions
* Replies stream sentence-by-sentence into TTS — speech starts while the LLM is still generating.
* Chat history is **bounded** (`AUROBOT_MAX_HISTORY`, default 10) — no progressive slowdown.
* LLM on the GPU; STT, TTS and vision on CPU, so the 4 GB of VRAM isn't contested.
* Model kept resident (`keep_alive`) and warmed at startup.

## Tuning (environment variables)

| Goal | Variable |
|---|---|
| Faster STT, slightly less accurate | `AUROBOT_STT_MODEL=base` (or `tiny`) |
| More accurate STT | `AUROBOT_STT_MODEL=medium` |
| Different LLM | `AUROBOT_OLLAMA_MODEL=llama3.2:3b` (then `ollama pull` it) |
| Snappier end-of-speech | `AUROBOT_END_SILENCE_MS=500` |
| Talk over it (headphones only) | `AUROBOT_BARGE_IN=1` |
| Noisy room | `AUROBOT_VAD_AGGRESSIVENESS=3` |
| Wrong mic | `AUROBOT_INPUT_DEVICE=<number from sounddevice.query_devices()>` |
| Hindi / Tamil | `AUROBOT_STT_LANGUAGE=auto` + `AUROBOT_TTS_BACKEND=espeak AUROBOT_ESPEAK_VOICE=hi` (or `ta`) |
| No camera | `AUROBOT_VISION_SOURCE=none` |

## Current limitations
* **Quality:** a small local model may produce shorter replies and make more mistakes.
* **No emotion-in-voice:** this pipeline uses transcribed words for dialogue
  (emotion still comes from the camera).
* **Multilingual:** Qwen2.5-3B is weak in Tamil/Hindi, and espeak voices are robotic. English is the
  well-supported path.
* **No cloud noise-cancellation service:** use headphones or a suitable microphone.
* **Half-duplex by default:** the mic is muted while it talks (laptop speakers would otherwise trigger it).
* **Memory heuristics** (what gets remembered) are still keyword/length rules, same as online.

## Hardware path
This stack is the right one for the standalone device: nothing depends on a network. On a Pi 5 / Jetson,
use `AUROBOT_STT_MODEL=base`, a 1–2B LLM, and Piper `low` voices; the same code runs unchanged.
