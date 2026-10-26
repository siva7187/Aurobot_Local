# fusion.py
"""
Multimodal emotion fusion engine.

Today only vision (DeepFace) feeds this, but it's built so a voice-prosody
signal can be added later without touching the rest of the pipeline —
just call update_voice() from wherever that signal is produced.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EmotionState:
    emotion: str
    confidence: float          # 0..1
    sources: Dict[str, float] = field(default_factory=dict)  # per-modality confidence
    timestamp: float = field(default_factory=time.time)


class FusionEngine:
    """
    Keeps a short rolling window per modality and produces a single fused
    EmotionState. Vision and voice are weighted; if only one modality is
    reporting, fusion falls back to it directly rather than diluting
    confidence with a missing signal.
    """

    def __init__(self, vision_weight: float = 0.6, voice_weight: float = 0.4):
        self.vision_weight = vision_weight
        self.voice_weight = voice_weight

        self._vision: Optional[Dict[str, float]] = None   # emotion -> confidence
        self._voice: Optional[Dict[str, float]] = None
        self._vision_ts = 0.0
        self._voice_ts = 0.0
        self._stale_after = 5.0  # seconds before a modality's reading is ignored

    def update_vision(self, emotion_scores: Dict[str, float]) -> None:
        """emotion_scores: e.g. DeepFace's per-emotion confidence dict, 0..100 or 0..1."""
        total = sum(emotion_scores.values()) or 1.0
        self._vision = {k: v / total for k, v in emotion_scores.items()}
        self._vision_ts = time.time()

    def update_voice(self, emotion_scores: Dict[str, float]) -> None:
        """Same shape as update_vision — hook for a future prosody/sentiment model."""
        total = sum(emotion_scores.values()) or 1.0
        self._voice = {k: v / total for k, v in emotion_scores.items()}
        self._voice_ts = time.time()

    def _fresh(self, ts: float) -> bool:
        return (time.time() - ts) < self._stale_after

    def get_state(self) -> EmotionState:
        now = time.time()
        vision_ok = self._vision is not None and self._fresh(self._vision_ts)
        voice_ok = self._voice is not None and self._fresh(self._voice_ts)

        if vision_ok and voice_ok:
            keys = set(self._vision) | set(self._voice)
            combined = {
                k: self.vision_weight * self._vision.get(k, 0)
                + self.voice_weight * self._voice.get(k, 0)
                for k in keys
            }
            sources = {"vision": self.vision_weight, "voice": self.voice_weight}
        elif vision_ok:
            combined = self._vision
            sources = {"vision": 1.0}
        elif voice_ok:
            combined = self._voice
            sources = {"voice": 1.0}
        else:
            return EmotionState(emotion="neutral", confidence=0.0, sources={}, timestamp=now)

        emotion = max(combined, key=combined.get)
        confidence = combined[emotion]
        return EmotionState(emotion=emotion, confidence=confidence, sources=sources, timestamp=now)
