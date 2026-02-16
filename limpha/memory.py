"""
LIMPHA MEMORY — WTForacle's consciousness persistence.

Adapted from yent/limpha for the cynical reddit oracle.
Single SQLite database. FTS5 full-text search. Autonomous.

Tables:
- conversations: Every prompt/response with generation state
- conversations_fts: FTS5 virtual table (auto-synced via triggers)
- sessions: Session metadata

All operations async via aiosqlite.
"""

import asyncio
import aiosqlite
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class Conversation:
    """One turn of dialogue."""
    id: int
    timestamp: float
    session_id: str
    prompt: str
    response: str
    temperature: float
    quality: float
    access_count: int


SCHEMA = """
-- Every conversation turn
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    session_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    temperature REAL DEFAULT 0.9,
    quality REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_quality ON conversations(quality DESC);

-- FTS5 full-text search over conversations
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
    prompt,
    response,
    content=conversations,
    content_rowid=id,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync with conversations table
CREATE TRIGGER IF NOT EXISTS conv_fts_insert AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, prompt, response)
    VALUES (new.id, new.prompt, new.response);
END;

CREATE TRIGGER IF NOT EXISTS conv_fts_delete AFTER DELETE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, prompt, response)
    VALUES ('delete', old.id, old.prompt, old.response);
END;

CREATE TRIGGER IF NOT EXISTS conv_fts_update AFTER UPDATE ON conversations BEGIN
    INSERT INTO conversations_fts(conversations_fts, rowid, prompt, response)
    VALUES ('delete', old.id, old.prompt, old.response);
    INSERT INTO conversations_fts(rowid, prompt, response)
    VALUES (new.id, new.prompt, new.response);
END;

-- Session metadata
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    last_active REAL NOT NULL,
    turn_count INTEGER DEFAULT 0,
    avg_quality REAL DEFAULT 0.0
);
"""


class LimphaMemory:
    """
    WTForacle's memory. SQLite + FTS5. Fully autonomous.

    Usage:
        async with LimphaMemory() as mem:
            conv_id = await mem.store(prompt, response, temperature=0.9)
            results = await mem.search("consciousness")
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".wtforacle" / "limpha.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None
        self._session_id: str = str(uuid.uuid4())[:8]

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        """Connect and initialize schema."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        now = time.time()
        await self._conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, started_at, last_active) VALUES (?, ?, ?)",
            (self._session_id, now, now),
        )
        await self._conn.commit()

    async def close(self):
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ═══════════════════════════════════════════════════════════════════════
    # STORE — after every generation, automatically
    # ═══════════════════════════════════════════════════════════════════════

    async def store(
        self,
        prompt: str,
        response: str,
        temperature: float = 0.9,
    ) -> int:
        """
        Store a conversation turn. Called automatically after each generation.
        Returns conversation ID.
        """
        now = time.time()
        quality = self._compute_quality(prompt, response)

        cursor = await self._conn.execute(
            """INSERT INTO conversations
            (timestamp, session_id, prompt, response, temperature, quality)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (now, self._session_id, prompt, response, temperature, quality),
        )
        conv_id = cursor.lastrowid

        await self._conn.execute(
            """UPDATE sessions SET
                last_active = ?,
                avg_quality = (avg_quality * turn_count + ?) / (turn_count + 1),
                turn_count = turn_count + 1
            WHERE session_id = ?""",
            (now, quality, self._session_id),
        )
        await self._conn.commit()

        return conv_id

    def _compute_quality(self, prompt: str, response: str) -> float:
        """
        Compute quality score for a conversation turn.

        Factors:
        - Response length (too short = low quality, sweet spot = higher)
        - Prompt-response ratio
        - Reddit slang density (cynicism markers)
        """
        if not response.strip():
            return 0.0

        resp_len = len(response.strip())
        prompt_len = max(len(prompt.strip()), 1)

        # Length score
        if resp_len < 10:
            length_score = 0.1
        elif resp_len < 50:
            length_score = 0.3
        elif resp_len < 200:
            length_score = 0.5 + 0.3 * (resp_len - 50) / 150
        else:
            length_score = 0.8

        # Ratio score
        ratio = resp_len / prompt_len
        if ratio < 0.3:
            ratio_score = 0.2
        elif ratio > 10:
            ratio_score = 0.6
        else:
            ratio_score = 0.7

        # Cynicism bonus — reddit slang density
        resp_lower = response.lower()
        slang = ['bro', 'tbh', 'ngl', 'imo', 'lmao', 'lol', 'bruh',
                 'nah', 'fr', 'literally', 'actually', 'honestly']
        slang_count = sum(resp_lower.count(s) for s in slang)
        cynicism_bonus = min(slang_count * 0.03, 0.15)

        # Penalize generic assistant patterns
        boring = ['as an ai', 'i cannot', 'i apologize', 'great question']
        boring_count = sum(1 for b in boring if b in resp_lower)
        boring_penalty = boring_count * 0.1

        quality = 0.6 * length_score + 0.4 * ratio_score + cynicism_bonus - boring_penalty
        return max(0.0, min(1.0, quality))

    # ═══════════════════════════════════════════════════════════════════════
    # SEARCH — FTS5 full-text search
    # ═══════════════════════════════════════════════════════════════════════

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Full-text search over all conversations.

        Supports FTS5 syntax:
        - "word1 word2" (AND)
        - "word1 OR word2"
        - '"exact phrase"'
        - "word*" (prefix)
        - "prompt:word" (column-specific)

        Results ranked by BM25.
        """
        if not query.strip():
            return []

        try:
            cursor = await self._conn.execute(
                """SELECT c.id, c.timestamp, c.session_id,
                          c.prompt, c.response, c.quality, c.access_count,
                          c.temperature,
                          bm25(conversations_fts) as rank
                   FROM conversations_fts fts
                   JOIN conversations c ON c.id = fts.rowid
                   WHERE conversations_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except aiosqlite.OperationalError:
            return []

    # ═══════════════════════════════════════════════════════════════════════
    # RECALL — access conversation, bump access count
    # ═══════════════════════════════════════════════════════════════════════

    async def recall(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Recall a specific conversation, incrementing access count."""
        await self._conn.execute(
            "UPDATE conversations SET access_count = access_count + 1 WHERE id = ?",
            (conversation_id,),
        )
        await self._conn.commit()

        cursor = await self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # RECENT — get recent conversations
    # ═══════════════════════════════════════════════════════════════════════

    async def recent(self, limit: int = 10, session_only: bool = False) -> List[Dict[str, Any]]:
        """Get recent conversations, optionally limited to current session."""
        if session_only:
            cursor = await self._conn.execute(
                """SELECT * FROM conversations
                   WHERE session_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (self._session_id, limit),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )

        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(rows)]  # Chronological order

    # ═══════════════════════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════════════════════

    async def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        conv_count = (await (await self._conn.execute("SELECT COUNT(*) FROM conversations")).fetchone())[0]
        session_count = (await (await self._conn.execute("SELECT COUNT(*) FROM sessions")).fetchone())[0]

        avg_quality = (await (await self._conn.execute(
            "SELECT AVG(quality) FROM conversations"
        )).fetchone())[0]

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "total_conversations": conv_count,
            "total_sessions": session_count,
            "avg_quality": round(avg_quality, 3) if avg_quality else 0.0,
            "current_session": self._session_id,
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
        }
