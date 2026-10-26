# offline/audio_io.py
"""
Microphone capture and speaker output. sounddevice is imported lazily so the
rest of the package (and the test-suite) works on machines without PortAudio.
"""
import queue
from typing import Optional

import numpy as np

from .vad import SAMPLE_RATE, FRAME_SAMPLES


class MicStream:
    """16 kHz mono int16 capture, delivered as 30 ms frames via get()."""

    def __init__(self, device=None):
        self.device = device
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)  # ~6 s; drops if consumer stalls
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        try:
            self._q.put_nowait(bytes(indata))
        except queue.Full:
            pass  # consumer is behind; dropping beats unbounded memory growth

    def start(self) -> None:
        import sounddevice as sd
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=FRAME_SAMPLES, device=self.device, callback=self._callback,
        )
        self._stream.start()

    def get(self, timeout: float = 0.5) -> Optional[bytes]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


class SoundDeviceOut:
    """Persistent output stream; write() blocks until the audio is buffered."""

    def __init__(self, sample_rate: int, device=None):
        import sounddevice as sd
        self._stream = sd.OutputStream(samplerate=sample_rate, channels=1,
                                       dtype="int16", device=device)
        self._stream.start()

    def write(self, samples: np.ndarray) -> None:
        self._stream.write(samples.reshape(-1, 1))

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class NullAudioOut:
    """Test/headless sink: records what would have been played."""

    def __init__(self, delay_per_chunk: float = 0.0):
        self.chunks = []
        self.delay = delay_per_chunk

    def write(self, samples: np.ndarray) -> None:
        import time
        self.chunks.append(samples.copy())
        if self.delay:
            time.sleep(self.delay)
