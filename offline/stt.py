# offline/stt.py
"""
Local speech-to-text via faster-whisper (CTranslate2).

Runs on CPU by default (int8) so the GPU's limited VRAM stays free for the LLM.
If you ask for CUDA but its libraries (cuBLAS/cuDNN) are missing, this falls
back to CPU automatically instead of crashing at the first transcription.
"""
import numpy as np

# Whisper tends to "hallucinate" these on near-silent/short clips.
_HALLUCINATIONS = {
    "you", "thank you.", "thank you", "thanks for watching.", "thanks for watching!",
    "bye.", "bye", ".", "uh", "um", "hmm",
}


class WhisperSTT:
    def __init__(self, model="small", device="cpu", compute_type="int8",
                 threads=4, language="en", beam_size=1, local_only=True):
        from faster_whisper import WhisperModel

        self.language = None if str(language).lower() == "auto" else language
        self.beam_size = beam_size
        self.device = device

        def _load(dev, ctype):
            return WhisperModel(model, device=dev, compute_type=ctype,
                                cpu_threads=threads, local_files_only=local_only)

        try:
            self.model = _load(device, compute_type)
            self._warmup()
        except Exception as e:
            if device == "cpu":
                raise
            print(f"[stt] {device} unavailable ({e}); falling back to CPU int8")
            self.device = "cpu"
            self.model = _load("cpu", "int8")
            self._warmup()

    def _warmup(self) -> None:
        # forces library loading now, not during the user's first sentence
        self.transcribe(np.zeros(16000, dtype=np.int16), _filter=False)

    def transcribe(self, audio_int16: np.ndarray, _filter: bool = True) -> str:
        audio = audio_int16.astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if _filter and (len(text) < 2 or (len(audio) < 24000 and text.lower() in _HALLUCINATIONS)):
            return ""
        return text
