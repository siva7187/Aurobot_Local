# offline/llm.py
"""
Local LLM via Ollama's HTTP API (http://127.0.0.1:11434), plus the piece that
matters most for perceived latency: a streaming sentence splitter.

Tokens stream in; the moment a sentence (or, for the very first one, a long
enough clause) is complete it is yielded to TTS — so Aurobot starts talking
while the rest of the reply is still being generated.
"""
import json
import re
import threading
from typing import Iterable, Iterator, List, Optional

import requests

_SENT_BOUNDARY = re.compile(r"(?<=[.!?।])\s+|\n+")
_CLEAN_MD = re.compile(r"[*#`_~>|]")
_EMOJI = re.compile("[\U00010000-\U0010ffff\u2600-\u27bf\ufe0f]")


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url, model, num_ctx=4096, keep_alive="30m",
                 temperature=0.7, max_tokens=160, timeout=120):
        self.base = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # -- health / startup -------------------------------------------------
    def check(self) -> None:
        """Fail fast with an actionable message instead of a mid-conversation error."""
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=3)
            r.raise_for_status()
        except Exception as e:
            raise OllamaError(
                f"Can't reach Ollama at {self.base} ({e}).\n"
                "Start it with:  ollama serve   (or: sudo systemctl start ollama)"
            )
        names = [m.get("name", "") for m in r.json().get("models", [])]
        wanted = self.model if ":" in self.model else self.model + ":latest"
        if wanted not in names and self.model not in names:
            raise OllamaError(
                f"Model '{self.model}' isn't downloaded. Run:  ollama pull {self.model}\n"
                f"Installed: {', '.join(names) or '(none)'}"
            )

    def warmup(self) -> None:
        """Load the model into (V)RAM now so the first real turn isn't slow."""
        try:
            requests.post(
                f"{self.base}/api/chat",
                json={"model": self.model, "messages": [{"role": "user", "content": "hi"}],
                      "stream": False, "keep_alive": self.keep_alive,
                      "options": {"num_predict": 1, "num_ctx": self.num_ctx}},
                timeout=self.timeout,
            )
        except Exception as e:
            print(f"[llm] warmup failed (non-fatal): {e}")

    # -- generation -------------------------------------------------------
    def stream_chat(self, messages: List[dict],
                    cancel: Optional[threading.Event] = None) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }
        with requests.post(f"{self.base}/api/chat", json=payload,
                           stream=True, timeout=self.timeout) as resp:
            if resp.status_code != 200:
                raise OllamaError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
            for line in resp.iter_lines():
                if cancel is not None and cancel.is_set():
                    return  # leaving the `with` closes the connection -> Ollama stops generating
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise OllamaError(data["error"])
                piece = (data.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    return


def clean_for_speech(text: str) -> str:
    text = _CLEAN_MD.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def iter_sentences(pieces: Iterable[str], early_clause_chars: int = 60) -> Iterator[str]:
    """
    Turn a stream of text fragments into speakable sentences.

    The FIRST sentence may also be released at a comma once it's long enough
    (`early_clause_chars`) — that shaves the most time off time-to-first-audio.
    """
    buf = ""
    emitted_any = False
    for piece in pieces:
        buf += piece
        while True:
            m = _SENT_BOUNDARY.search(buf)
            if m:
                sent, buf = buf[: m.start()], buf[m.end():]
            elif (not emitted_any and len(buf) >= early_clause_chars and "," in buf[20:]):
                cut = buf.rfind(",")
                if cut < 20:
                    break
                sent, buf = buf[: cut + 1], buf[cut + 1:]
            else:
                break
            sent = clean_for_speech(sent)
            if re.search(r"\w", sent):
                emitted_any = True
                yield sent
    tail = clean_for_speech(buf)
    if re.search(r"\w", tail):
        yield tail
