# offline/config.py
"""
All offline-mode settings in one place, overridable via environment variables
(so retargeting to different hardware = changing env vars, not code).

Default choices are sized for: ~4 GB VRAM GPU, 8+ CPU threads, 16 GB RAM.
  - LLM runs on the GPU (via Ollama) — it's the part that benefits most.
  - STT (faster-whisper) and TTS (Piper) run on CPU — small enough to be fast
    there, and it keeps the scarce VRAM free for the LLM.
"""
import os
from dataclasses import dataclass


def _s(name, default):
    return os.environ.get(name, default)


def _i(name, default):
    return int(os.environ.get(name, default))


def _f(name, default):
    return float(os.environ.get(name, default))


def _b(name, default):
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def _opt_dev(name):
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    return int(v) if v.isdigit() else v


@dataclass
class OfflineConfig:
    # ---- LLM (Ollama, local) ----
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "sam860/qwen3:4b"
    ollama_num_ctx: int = 4096
    ollama_keep_alive: str = "30m"       # keep the model resident between turns
    llm_temperature: float = 0.7
    llm_max_tokens: int = 160            # spoken replies should be short
    max_history_messages: int = 10       # BOUNDED history -> no progressive slowdown

    # ---- STT (faster-whisper) ----
    stt_model: str = "small"             # tiny/base/small/medium, or a local folder path
    stt_language: str = "en"             # "auto" to detect (needs multilingual model)
    stt_device: str = "cpu"              # "cuda" works if cuBLAS/cuDNN are installed
    stt_compute_type: str = "int8"
    stt_threads: int = 4
    stt_beam_size: int = 1

    # ---- TTS ----
    tts_backend: str = "piper"           # "piper" (natural) or "espeak" (robotic, any language)
    piper_voice: str = "models/piper/en_US-lessac-medium.onnx"
    espeak_voice: str = "en"

    # ---- Audio / VAD ----
    input_device: object = None
    output_device: object = None
    vad_aggressiveness: int = 2          # 0-3, higher = stricter about what counts as speech
    end_silence_ms: int = 700            # silence that ends an utterance
    min_speech_ms: int = 300
    max_utterance_s: float = 15.0
    barge_in: bool = False               # needs headphones (speaker echo would self-trigger)
    echo_tail_ms: int = 400              # keep mic muted briefly after we stop talking

    # ---- Behaviour ----
    strict_offline: bool = True          # forbid HuggingFace/network lookups at runtime
    vision_source: str = "webcam"        # "webcam" or "none"
    vision_device: str = "cpu"           # keep VRAM free for the LLM
    greeting: bool = True

    @classmethod
    def from_env(cls) -> "OfflineConfig":
        return cls(
            ollama_url=_s("AUROBOT_OLLAMA_URL", cls.ollama_url),
            ollama_model=_s("AUROBOT_OLLAMA_MODEL", cls.ollama_model),
            ollama_num_ctx=_i("AUROBOT_OLLAMA_NUM_CTX", cls.ollama_num_ctx),
            ollama_keep_alive=_s("AUROBOT_OLLAMA_KEEP_ALIVE", cls.ollama_keep_alive),
            llm_temperature=_f("AUROBOT_LLM_TEMPERATURE", cls.llm_temperature),
            llm_max_tokens=_i("AUROBOT_LLM_MAX_TOKENS", cls.llm_max_tokens),
            max_history_messages=_i("AUROBOT_MAX_HISTORY", cls.max_history_messages),
            stt_model=_s("AUROBOT_STT_MODEL", cls.stt_model),
            stt_language=_s("AUROBOT_STT_LANGUAGE", cls.stt_language),
            stt_device=_s("AUROBOT_STT_DEVICE", cls.stt_device),
            stt_compute_type=_s("AUROBOT_STT_COMPUTE", cls.stt_compute_type),
            stt_threads=_i("AUROBOT_STT_THREADS", cls.stt_threads),
            stt_beam_size=_i("AUROBOT_STT_BEAM", cls.stt_beam_size),
            tts_backend=_s("AUROBOT_TTS_BACKEND", cls.tts_backend),
            piper_voice=_s("AUROBOT_PIPER_VOICE", cls.piper_voice),
            espeak_voice=_s("AUROBOT_ESPEAK_VOICE", cls.espeak_voice),
            input_device=_opt_dev("AUROBOT_INPUT_DEVICE"),
            output_device=_opt_dev("AUROBOT_OUTPUT_DEVICE"),
            vad_aggressiveness=_i("AUROBOT_VAD_AGGRESSIVENESS", cls.vad_aggressiveness),
            end_silence_ms=_i("AUROBOT_END_SILENCE_MS", cls.end_silence_ms),
            min_speech_ms=_i("AUROBOT_MIN_SPEECH_MS", cls.min_speech_ms),
            max_utterance_s=_f("AUROBOT_MAX_UTTERANCE_S", cls.max_utterance_s),
            barge_in=_b("AUROBOT_BARGE_IN", cls.barge_in),
            echo_tail_ms=_i("AUROBOT_ECHO_TAIL_MS", cls.echo_tail_ms),
            strict_offline=_b("AUROBOT_STRICT_OFFLINE", cls.strict_offline),
            vision_source=_s("AUROBOT_VISION_SOURCE", cls.vision_source),
            vision_device=_s("AUROBOT_VISION_DEVICE", cls.vision_device),
            greeting=_b("AUROBOT_GREETING", cls.greeting),
        )
