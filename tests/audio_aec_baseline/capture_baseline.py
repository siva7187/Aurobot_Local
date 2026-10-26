import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

RATE = 48000
BLOCK = 480
SECONDS_PER_PHASE = 10
OUT = Path("tests/audio_aec_baseline/recordings")
OUT.mkdir(parents=True, exist_ok=True)

TONE_HZ = 440
TONE_AMPLITUDE = 0.08

phase_names = ["quiet", "speaker_only", "speaker_plus_voice"]
recordings = {name: [] for name in phase_names}
references = {name: [] for name in phase_names}

state = {"block": 0, "error": None}
blocks_per_phase = SECONDS_PER_PHASE * RATE // BLOCK


def callback(indata, outdata, frames, time_info, status):
    if status:
        # Avoid blocking the real-time audio callback with terminal output.
        state["error"] = str(status)

    phase_index = min(state["block"] // blocks_per_phase, 2)
    phase_name = phase_names[phase_index]

    recordings[phase_name].append(indata[:, 0].copy())

    if phase_index == 0:
        playback = np.zeros(frames, dtype=np.float32)
    else:
        n = state["block"] * BLOCK + np.arange(frames)
        playback = (
            TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_HZ * n / RATE)
        ).astype(np.float32)

    references[phase_name].append(playback.copy())
    outdata[:, 0] = playback
    state["block"] += 1


def save_pcm16(path, samples):
    samples = np.asarray(samples, dtype=np.float64)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")

    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm.tobytes())


print("AEC acoustic baseline")
print("Input/output devices:", sd.default.device)
print("Format: 48 kHz, mono, float32 stream")
print("Each phase lasts 10 seconds.")
print("1. Quiet room, speaker silent")
print("2. Speaker-only 440 Hz test tone")
print("3. Same tone while you speak naturally")
print("Keep speaker volume moderate.")
input("Press Enter to start...")

try:
    with sd.Stream(
        samplerate=RATE,
        blocksize=BLOCK,
        channels=(1, 1),
        dtype="float32",
        callback=callback,
        latency="low",
    ):
        for i, name in enumerate(phase_names):
            print(f"\nPhase {i + 1}/3: {name.replace('_', ' ').upper()}")
            if name == "speaker_plus_voice":
                print("Speak naturally at your normal distance.")
            deadline = time.monotonic() + SECONDS_PER_PHASE
            while time.monotonic() < deadline:
                time.sleep(0.05)
finally:
    print("\nSaving recordings...")
    for name in phase_names:
        if recordings[name]:
            mic = np.concatenate(recordings[name])
            ref = np.concatenate(references[name])

            save_pcm16(OUT / f"{name}_mic.wav", mic)
            save_pcm16(OUT / f"{name}_reference.wav", ref)

            rms = np.sqrt(np.mean(mic.astype(np.float64) ** 2))
            print(f"{name}: {len(mic) / RATE:.2f}s, mic RMS={rms:.6f}")

    print("Output directory:", OUT.resolve())

    if state["error"]:
        print("Audio stream warning:", state["error"])
