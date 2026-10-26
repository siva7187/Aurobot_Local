# memory.py
"""
Lightweight persistent memory for emotion history.

Uses SQLite so there's no extra service to run. Swap for Chroma/Postgres
later if you want semantic recall of conversation content too — this file
only tracks emotion + timing, which is enough to power trend-aware
proactivity ("Siva's been stressed most evenings this week").
"""

import asyncio
import sqlite3
import time
from contextlib import closing
from typing import List, Tuple


class EmotionMemory:
    def __init__(self, db_path: str = "aurobot_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        # WAL = readers/writers don't block each other; NORMAL sync skips a full
        # fsync on every commit while still being crash-safe for this use case.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS emotion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emotion TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spoken_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emotion TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            conn.commit()
        self._init_conversation_table()

    def log_emotion(self, emotion: str, confidence: float) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO emotion_log (emotion, confidence, timestamp) VALUES (?, ?, ?)",
                (emotion, confidence, time.time()),
            )
            conn.commit()

    def log_spoken(self, emotion: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO spoken_log (emotion, timestamp) VALUES (?, ?)",
                (emotion, time.time()),
            )
            conn.commit()

    def recent_emotions(self, window_seconds: float = 3600) -> List[Tuple[str, float, float]]:
        cutoff = time.time() - window_seconds
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT emotion, confidence, timestamp FROM emotion_log WHERE timestamp >= ? ORDER BY timestamp",
                (cutoff,),
            ).fetchall()
        return rows

    def dominant_recent_emotion(self, window_seconds: float = 3600) -> str:
        rows = self.recent_emotions(window_seconds)
        if not rows:
            return "neutral"
        counts = {}
        for emotion, _, _ in rows:
            counts[emotion] = counts.get(emotion, 0) + 1
        return max(counts, key=counts.get)

    def last_spoken_time(self) -> float:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT MAX(timestamp) FROM spoken_log"
            ).fetchone()
        return row[0] or 0.0

    # ---------------- CONVERSATION MEMORY ----------------
    # Persists actual dialogue turns (not just emotion state) so Aurobot
    # can be reminded what you last talked about, even after a full restart.

    def _init_conversation_table(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            conn.commit()

    def log_conversation_turn(self, role: str, text: str) -> None:
        if not text or not text.strip():
            return
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO conversation_log (role, text, timestamp) VALUES (?, ?, ?)",
                (role, text.strip(), time.time()),
            )
            conn.commit()

    def recent_conversation(self, limit: int = 12) -> List[Tuple[str, str, float]]:
        """Most recent N turns, oldest first (natural reading order)."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, text, timestamp FROM conversation_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return list(reversed(rows))

    def conversation_recap(self, limit: int = 8, max_chars: int = 800) -> str:
        """
        A short plain-text recap of the last conversation, meant to be dropped
        straight into a prompt. Empty string if there's no prior history —
        callers should handle that case (first-ever run).
        """
        turns = self.recent_conversation(limit=limit)
        if not turns:
            return ""
        lines = [f"{role}: {text}" for role, text, _ in turns]
        recap = "\n".join(lines)
        return recap[-max_chars:]  # keep the most recent part if it's long

    # ---------------- ASYNC WRAPPERS ----------------
    # SQLite calls are blocking (disk I/O). Called directly inside the
    # asyncio event loop, they stall audio streaming for their duration —
    # small on a laptop SSD, worse on embedded storage. These wrappers push
    # the blocking work onto a thread so the event loop keeps running.
    # Prefer these (a*) methods from anywhere inside async code; the sync
    # methods above remain for one-off/startup use where blocking is fine.

    async def alog_emotion(self, emotion: str, confidence: float) -> None:
        await asyncio.to_thread(self.log_emotion, emotion, confidence)

    async def alog_spoken(self, emotion: str) -> None:
        await asyncio.to_thread(self.log_spoken, emotion)

    async def alast_spoken_time(self) -> float:
        return await asyncio.to_thread(self.last_spoken_time)

    async def alog_conversation_turn(self, role: str, text: str) -> None:
        await asyncio.to_thread(self.log_conversation_turn, role, text)

    async def aconversation_recap(self, limit: int = 8, max_chars: int = 800) -> str:
        return await asyncio.to_thread(self.conversation_recap, limit, max_chars)
