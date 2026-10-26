# offline/tts.py
"""
Local text-to-speech. Two backends behind one interface:

  piper   — neural, natural-sounding, fast on CPU. Needs a downloaded voice
            (.onnx + .onnx.json). Per-language voices.
  espeak  — espeak-ng: robotic but tiny, supports ~100 languages (incl. Hindi,
            Tamil). Used as the fallback if Piper isn't set up.

Both expose:  .sample_rate  and  .synth(text) -> bytes (int16 mono PCM)
"""
import io
import os
import shutil
import subprocess
import wave


class PiperTTS:
    backend = "piper"

    def __init__(self, voice_path: str):
        from piper import PiperVoice

        if not os.path.exists(voice_path):
            raise FileNotFoundError(
                f"Piper voice not found: {voice_path}\n"
                "Run: python scripts/prefetch_models.py   (needs internet once)"
            )
        self.voice = PiperVoice.load(voice_path)
        cfg = getattr(self.voice, "config", None)
        self.sample_rate = int(getattr(cfg, "sample_rate", 22050))

    def synth(self, text: str) -> bytes:
        out = []
        for chunk in self.voice.synthesize(text):
            self.sample_rate = chunk.sample_rate
            out.append(chunk.audio_int16_bytes)
        return b"".join(out)


class EspeakTTS:
    backend = "espeak"

    def __init__(self, voice: str = "en"):
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.exe:
            raise RuntimeError("espeak-ng not found. Install: sudo apt install espeak-ng")
        self.voice = voice
        self.sample_rate = 22050
        self.synth("ok")  # discovers the real sample rate

    def synth(self, text: str) -> bytes:
        proc = subprocess.run(
            [self.exe, "-v", self.voice, "--stdout", text],
            capture_output=True, check=True,
        )
        with wave.open(io.BytesIO(proc.stdout), "rb") as w:
            self.sample_rate = w.getframerate()
            return w.readframes(w.getnframes())


def make_tts(cfg):
    """Piper if available, otherwise espeak (with a clear warning)."""
    if cfg.tts_backend == "piper":
        try:
            return PiperTTS(cfg.piper_voice)
        except Exception as e:
            print(f"[tts] Piper unavailable ({e}).\n[tts] Falling back to espeak-ng.")
    return EspeakTTS(cfg.espeak_voice)
