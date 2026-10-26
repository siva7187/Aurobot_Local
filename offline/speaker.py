# offline/speaker.py
"""
Two-stage, interruptible speech output.

    say(sentence) -> [synth thread: text -> PCM] -> [play thread: PCM -> speaker]

While sentence N plays, sentence N+1 is already being synthesized, so there are
no gaps between sentences. interrupt() drops everything queued and stops
playback within one ~40 ms chunk.
"""
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

CHUNK_MS = 40


class Speaker:
    def __init__(self, tts, audio_out):
        self.tts = tts
        self.out = audio_out
        self._text_q: "queue.Queue" = queue.Queue()
        self._audio_q: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._gen = 0                  # bumped by interrupt(); stale items are dropped
        self._pending = 0              # sentences accepted but not yet finished/dropped
        self._idle = threading.Event()
        self._idle.set()
        self._first_audio_cb: Optional[Callable[[float], None]] = None
        self.last_audio_end = 0.0
        self._started = False

    # ---- lifecycle ----
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._synth_loop, daemon=True, name="tts-synth").start()
        threading.Thread(target=self._play_loop, daemon=True, name="tts-play").start()

    # ---- public API ----
    @property
    def is_speaking(self) -> bool:
        return self._pending > 0

    def arm_first_audio(self, cb: Callable[[float], None]) -> None:
        """cb(timestamp) fires once, when the next audio actually starts playing."""
        self._first_audio_cb = cb

    def say(self, sentence: str) -> None:
        with self._lock:
            self._pending += 1
            self._idle.clear()
            gen = self._gen
        self._text_q.put((gen, sentence))

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        return self._idle.wait(timeout)

    def interrupt(self) -> None:
        with self._lock:
            self._gen += 1
        for q in (self._text_q, self._audio_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                self._done_one()

    # ---- internals ----
    def _done_one(self) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            if self._pending == 0:
                self._idle.set()

    def _synth_loop(self) -> None:
        while True:
            gen, text = self._text_q.get()
            if gen != self._gen:
                self._done_one()
                continue
            try:
                pcm = self.tts.synth(text)
            except Exception as e:
                print(f"[tts] synth error: {e}")
                self._done_one()
                continue
            self._audio_q.put((gen, pcm))

    def _play_loop(self) -> None:
        while True:
            gen, pcm = self._audio_q.get()
            if gen != self._gen:
                self._done_one()
                continue
            samples = np.frombuffer(pcm, dtype=np.int16)
            step = max(1, int(self.tts.sample_rate * CHUNK_MS / 1000))
            first = True
            for i in range(0, len(samples), step):
                if gen != self._gen:
                    break
                if first:
                    first = False
                    cb, self._first_audio_cb = self._first_audio_cb, None
                    if cb:
                        cb(time.time())
                try:
                    self.out.write(samples[i:i + step])
                except Exception as e:
                    print(f"[audio] playback error: {e}")
                    break
            self.last_audio_end = time.time()
            self._done_one()
