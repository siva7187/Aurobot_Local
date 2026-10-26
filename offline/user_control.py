# offline/user_control.py
"""
User-priority rules + lightweight memory-writing heuristics.

Shared user-control behaviour, extracted so the offline agent doesn't have
need cloud voice dependencies. Two refinements for voice input:
  * "stop"/"quiet" phrases only count in SHORT utterances — "I can't stop
    thinking about my exam" must not silence the assistant.
  * Matching uses word boundaries ("stopwatch" no longer matches "stop").
"""
import re
import time
from typing import Callable, Optional

QUIET_PHRASES = ("stop", "be quiet", "don't talk", "dont talk", "not now",
                 "quiet please", "shut up", "stop talking")
TASK_PHRASES = ("remind me", "todo", "let's continue", "lets continue", "next time")
QUIET_MAX_WORDS = 6
QUIET_SECONDS = 300


def _has_phrase(lowered: str, phrases) -> bool:
    return any(re.search(rf"\b{re.escape(p)}\b", lowered) for p in phrases)


class UserControl:
    def __init__(self, store, submit: Callable):
        self.store = store
        self.submit = submit          # db_writer.submit (fire-and-forget)
        self.quiet_until = 0.0

    @property
    def is_quiet(self) -> bool:
        return time.time() < self.quiet_until

    def quiet_command_from_text(self, text: str) -> Optional[str]:
        """Return an approved quiet command at the end of a transcript."""
        lowered = text.lower().strip().rstrip(" .,!?:;")
        # Match only at the end, so phrases like "stop thinking" do not
        # trigger an interruption.
        for phrase in sorted(QUIET_PHRASES, key=len, reverse=True):
            if lowered == phrase or lowered.endswith(" " + phrase):
                return phrase
        return None

    def is_quiet_command(self, text: str) -> bool:
        """Recognize a short command or one at the end of a noisy transcript."""
        command = self.quiet_command_from_text(text)
        if command is None:
            return False
        words = len(text.lower().strip().split())
        return words <= QUIET_MAX_WORDS or text.lower().strip().rstrip(
            " .,!?:;"
        ) != command

    def handle(self, text: str) -> Optional[str]:
        """Returns "quiet" if the user asked for silence, else None."""
        lowered = text.lower().strip()
        words = len(lowered.split())

        if self.is_quiet_command(text):
            self.quiet_until = time.time() + QUIET_SECONDS
            return "quiet"

        self.quiet_until = 0.0        # any other speech = re-engagement

        if _has_phrase(lowered, TASK_PHRASES):
            self.submit(self.store.add_memory, "task", text, importance=0.7)
        elif "remember" in lowered or "prefer" in lowered:
            self.submit(self.store.add_memory, "semantic", text, importance=0.9)
        elif words > 8:
            self.submit(self.store.add_memory, "episodic", text, importance=0.4)
        return None
