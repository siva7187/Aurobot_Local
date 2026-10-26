# memory_store.py
"""
Persistent conversational memory — what Aurobot KNOWS, as opposed to memory.py
which tracks what it SENSES (emotion readings).

Implements a subset of the spec's layered memory architecture:
  - Semantic memory   (stable facts/preferences)
  - Episodic memory   (meaningful past exchanges)
  - Task memory       (unresolved goals)
Working memory (the current session's live context) deliberately isn't
persisted here — that's just Python variables in the agent process, cleared each run.

Retrieval uses an importance/recency/frequency score, NOT semantic similarity —
there's no embedding model wired in yet. That's the natural next upgrade: swap
get_relevant_memories()'s SQL scan for a vector similarity search (e.g. via
sqlite-vec or a small local embedding model) once conversation volume justifies it.
"""

import asyncio
import json
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from typing import List, Optional

VALID_TYPES = {"semantic", "episodic", "task", "preference"}


@dataclass
class Memory:
    id: str
    type: str
    content: str
    importance: float
    confidence: float
    created_at: float
    updated_at: float
    last_used: float
    tags: List[str]
    status: str


class MemoryStore:
    def __init__(self, db_path: str = "aurobot_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    confidence REAL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_used REAL NOT NULL,
                    tags TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'active'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS proactive_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    topic TEXT,
                    timestamp REAL
                )
                """
            )
            conn.commit()

    # ---------------- WRITE ----------------

    def add_memory(
        self,
        type_: str,
        content: str,
        importance: float = 0.5,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
        status: str = "active",
    ) -> str:
        if type_ not in VALID_TYPES:
            raise ValueError(f"invalid memory type: {type_}")
        now = time.time()
        memory_id = str(uuid.uuid4())
        with closing(self._connect()) as conn:
            conn.execute(
                """INSERT INTO memories
                   (id, type, content, importance, confidence, created_at, updated_at, last_used, tags, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (memory_id, type_, content, importance, confidence, now, now, now, json.dumps(tags or []), status),
            )
            conn.commit()
        return memory_id

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        status: Optional[str] = None,
    ) -> None:
        """Current statement > old memory: use this to overwrite a stale fact rather than duplicate it."""
        fields, values = [], []
        if content is not None:
            fields.append("content = ?")
            values.append(content)
        if importance is not None:
            fields.append("importance = ?")
            values.append(importance)
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if not fields:
            return
        fields.append("updated_at = ?")
        values.append(time.time())
        values.append(memory_id)

        with closing(self._connect()) as conn:
            conn.execute(f"UPDATE memories SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()

    def mark_used(self, memory_id: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute("UPDATE memories SET last_used = ? WHERE id = ?", (time.time(), memory_id))
            conn.commit()

    # ---------------- READ ----------------

    def get_active_tasks(self):
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT id, content, importance, updated_at FROM memories "
                "WHERE type='task' AND status='active' ORDER BY updated_at DESC"
            ).fetchall()

    def _score(self, importance: float, updated_at: float, last_used: float, now: float) -> float:
        age_days = (now - updated_at) / 86400
        recency = max(0.0, 1.0 - age_days / 7.0)
        use_age_days = (now - last_used) / 86400
        frequency = max(0.0, 1.0 - use_age_days / 14.0)
        return importance * 0.5 + recency * 0.3 + frequency * 0.2

    def get_relevant_memories(self, limit: int = 5, types: Optional[List[str]] = None) -> List[Memory]:
        now = time.time()
        query = (
            "SELECT id, type, content, importance, confidence, created_at, updated_at, last_used, tags, status "
            "FROM memories WHERE status='active'"
        )
        params: list = []
        if types:
            query += f" AND type IN ({','.join('?' for _ in types)})"
            params.extend(types)

        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()

        scored = []
        for row in rows:
            mid, type_, content, importance, confidence, created_at, updated_at, last_used, tags, status = row
            score = self._score(importance, updated_at, last_used, now)
            scored.append((score, Memory(mid, type_, content, importance, confidence, created_at, updated_at, last_used, json.loads(tags), status)))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [m for _, m in scored[:limit]]

    # ---------------- PROACTIVE TRACKING ----------------

    def log_proactive(self, content: str, topic: str = "") -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO proactive_log (content, topic, timestamp) VALUES (?, ?, ?)",
                (content, topic, time.time()),
            )
            conn.commit()

    def last_proactive(self):
        with closing(self._connect()) as conn:
            return conn.execute(
                "SELECT content, topic, timestamp FROM proactive_log ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()

    # ---------------- CONTEXT CONSTRUCTION ----------------

    def build_context_block(self, limit: int = 5) -> str:
        """
        Compact, human-readable memory summary for prompt injection.
        This is the "Context Construction" step in the session-start protocol —
        deliberately NOT the full history, just what's relevant + unresolved.
        """
        memories = self.get_relevant_memories(limit=limit)
        tasks = self.get_active_tasks()

        lines = []
        if memories:
            lines.append("Relevant memories:")
            for m in memories:
                lines.append(f"- [{m.type}] {m.content}")
                self.mark_used(m.id)
        if tasks:
            lines.append("Unresolved tasks:")
            for _id, content, _importance, _updated_at in tasks[:3]:
                lines.append(f"- {content}")

        return "\n".join(lines)

    # ---------------- ASYNC WRAPPERS ----------------
    # Same rationale as memory.py — keep blocking SQLite calls off the
    # asyncio event loop thread. Use these from inside async code.

    async def aadd_memory(self, type_: str, content: str, importance: float = 0.5,
                           confidence: float = 1.0, tags: Optional[List[str]] = None,
                           status: str = "active") -> str:
        return await asyncio.to_thread(self.add_memory, type_, content, importance, confidence, tags, status)

    async def abuild_context_block(self, limit: int = 5) -> str:
        return await asyncio.to_thread(self.build_context_block, limit)
