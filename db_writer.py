# db_writer.py
"""
Every SQLite write in this project (memory.py, memory_store.py) was happening
synchronously, inline, on the same thread driving the live voice session.
A single commit() can take anywhere from ~1ms to 20-30ms depending on disk —
and that's stolen directly from your real-time audio pipeline's budget.

This module gives every write a single dedicated background thread with a
queue. Callers get instant return (just a queue.put), and writes happen
serially in the background, fully off the event loop's critical path.

This also matters for the hardware move: an SD card on a Raspberry Pi is
SLOWER than a laptop SSD, so this pattern isn't optional there — it's required.
"""

import queue
import threading

_write_queue: "queue.Queue" = queue.Queue()
_started = False


def _worker() -> None:
    while True:
        func, args, kwargs = _write_queue.get()
        try:
            func(*args, **kwargs)
        except Exception as e:
            print("Background write error:", e)


def start() -> None:
    global _started
    if _started:
        return
    threading.Thread(target=_worker, daemon=True).start()
    _started = True


def submit(func, *args, **kwargs) -> None:
    """Fire-and-forget: queue a write, return immediately."""
    _write_queue.put((func, args, kwargs))
