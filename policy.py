# policy.py
"""
Decides WHEN Aurobot should proactively speak.

Replaces the fixed COOLDOWN / PERSISTENCE_TIME constants in the original
the agent with a few simple, explainable rules. Not machine-learned — the
"learned" upgrade path is to swap decide() for a small classifier trained
on logged (state, spoke, user_reacted_well) tuples from memory.py, but a
transparent rules-based policy is a safer default for something that
talks to a person unprompted.
"""

import time
from dataclasses import dataclass
from fusion import EmotionState

DISTRESS_EMOTIONS = {"sad", "fear", "angry"}


@dataclass
class PolicyConfig:
    base_cooldown: float = 20.0       # seconds, normal case
    distress_cooldown: float = 5.0    # seconds, shortened when distress persists
    min_persistence: float = 8.0      # normal case: emotion must hold this long
    distress_persistence: float = 3.0  # distress: react faster
    min_confidence: float = 0.35      # ignore low-confidence / noisy readings


class ProactivityPolicy:
    def __init__(self, config: PolicyConfig = None):
        self.config = config or PolicyConfig()
        self._emotion_since: float = time.time()
        self._last_emotion: str = "neutral"

    def _update_persistence_clock(self, state: EmotionState) -> float:
        if state.emotion != self._last_emotion:
            self._last_emotion = state.emotion
            self._emotion_since = state.timestamp
        return state.timestamp - self._emotion_since

    def should_speak(self, state: EmotionState, last_spoken_time: float) -> bool:
        cfg = self.config
        now = time.time()

        if state.confidence < cfg.min_confidence:
            return False

        duration = self._update_persistence_clock(state)
        is_distress = state.emotion in DISTRESS_EMOTIONS

        persistence_needed = cfg.distress_persistence if is_distress else cfg.min_persistence
        cooldown_needed = cfg.distress_cooldown if is_distress else cfg.base_cooldown

        persisted_long_enough = duration > persistence_needed
        cooldown_elapsed = (now - last_spoken_time) > cooldown_needed

        return persisted_long_enough and cooldown_elapsed
