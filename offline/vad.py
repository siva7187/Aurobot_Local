# offline/vad.py
"""
Voice activity detection + utterance segmentation.

Handles local voice activity detection (deciding when you've finished
speaking). Uses webrtcvad (tiny, fast, no torch) with an adaptive-energy
fallback if it isn't installed.

Frames are 30 ms of 16 kHz mono int16 (480 samples = 960 bytes).
"""
from collections import deque
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000     # 480
FRAME_BYTES = FRAME_SAMPLES * 2                    # int16


class FrameVAD:
    """is_speech(frame_bytes) -> bool"""

    def __init__(self, aggressiveness: int = 2, force_energy: bool = False):
        self._webrtc = None
        if not force_energy:
            try:
                import webrtcvad
                self._webrtc = webrtcvad.Vad(max(0, min(3, aggressiveness)))
            except Exception:
                self._webrtc = None
        # adaptive energy fallback state
        self._noise = 100.0
        self.backend = "webrtcvad" if self._webrtc else "energy"

    def is_speech(self, frame: bytes) -> bool:
        if self._webrtc is not None:
            return self._webrtc.is_speech(frame, SAMPLE_RATE)
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples)) + 1e-6)
        threshold = max(250.0, self._noise * 3.0)
        speech = rms > threshold
        if not speech:  # track the noise floor only during non-speech
            self._noise = 0.95 * self._noise + 0.05 * rms
        return speech


class UtteranceSegmenter:
    """
    Feed 30 ms frames one at a time; get back a complete utterance (int16
    numpy array) when the speaker has finished, otherwise None.
    """

    def __init__(
        self,
        vad: FrameVAD,
        end_silence_ms: int = 700,
        min_speech_ms: int = 300,
        max_utterance_s: float = 15.0,
        preroll_ms: int = 300,
        start_frames: int = 3,
    ):
        self.vad = vad
        self.end_silence_frames = max(1, end_silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.max_frames = int(max_utterance_s * 1000 // FRAME_MS)
        self.start_frames = start_frames
        self._preroll = deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self.speech_frames = 0          # speech frames in the current utterance
        self._run = 0                   # consecutive speech frames before start
        self._silence = 0
        self._buf = []
        self._preroll.clear()

    @property
    def speech_ms(self) -> int:
        return self.speech_frames * FRAME_MS

    def feed(self, frame: bytes) -> Optional[np.ndarray]:
        is_sp = self.vad.is_speech(frame)

        if not self.in_speech:
            self._preroll.append(frame)
            self._run = self._run + 1 if is_sp else 0
            if self._run >= self.start_frames:
                self.in_speech = True
                self._buf = list(self._preroll)
                self.speech_frames = self._run
                self._silence = 0
            return None

        self._buf.append(frame)
        if is_sp:
            self._silence = 0
            self.speech_frames += 1
        else:
            self._silence += 1

        finished = (
            self._silence >= self.end_silence_frames or len(self._buf) >= self.max_frames
        )
        if not finished:
            return None

        enough = self.speech_frames >= self.min_speech_frames
        audio = np.frombuffer(b"".join(self._buf), dtype=np.int16).copy()
        self.reset()
        return audio if enough else None
