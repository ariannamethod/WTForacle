#!/usr/bin/env python3
"""
LIMPHA TESTS — WTForacle memory subsystem. Every feature tested.

Run: python3 wtftests/test_limpha.py
"""

import asyncio
import json
import os
import tempfile
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limpha.memory import LimphaMemory


async def test_schema_creation():
    """Schema creates tables and FTS5 virtual table."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            cursor = await mem._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [r[0] for r in await cursor.fetchall()]
            assert "conversations" in tables, f"conversations not in {tables}"
            assert "sessions" in tables, f"sessions not in {tables}"

            cursor = await mem._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations_fts'"
            )
            row = await cursor.fetchone()
            assert row is not None, "conversations_fts not created"
    print("  PASS: schema_creation")


async def test_store_conversation():
    """Store a conversation and verify it's in the database."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            conv_id = await mem.store(
                prompt="Who are you?",
                response="i am a reddit character, but also sometimes real.",
                temperature=0.9,
            )
            assert conv_id == 1, f"Expected id=1, got {conv_id}"

            cursor = await mem._conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
            row = await cursor.fetchone()
            assert row is not None, "Conversation not stored"
            assert row["prompt"] == "Who are you?"
            assert row["response"] == "i am a reddit character, but also sometimes real."
            assert row["temperature"] == 0.9
    print("  PASS: store_conversation")


async def test_store_defaults():
    """Store works with default temperature."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            conv_id = await mem.store("Hello", "nah bro")
            assert conv_id == 1
            cursor = await mem._conn.execute("SELECT temperature FROM conversations WHERE id = ?", (conv_id,))
            row = await cursor.fetchone()
            assert row["temperature"] == 0.9
    print("  PASS: store_defaults")


async def test_fts5_search():
    """FTS5 full-text search works with BM25 ranking."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("what is love?", "love is when a meme stays in you and no one knows why.")
            await mem.store("what is AI?", "it's like us but smarter, bro.")
            await mem.store("explain quantum physics", "physicists lie to people anyway, bro.")
            await mem.store("give me life advice", "we're here because people need solutions not pity.")

            results = await mem.search("love")
            assert len(results) >= 1, f"Expected >=1 results, got {len(results)}"

            results = await mem.search("bro")
            assert len(results) >= 2, f"Expected >=2 results for bro, got {len(results)}"

            results = await mem.search('"lie to people"')
            assert len(results) == 1, f"Expected 1 result for phrase, got {len(results)}"

            results = await mem.search("prompt:love")
            assert len(results) == 1, f"Expected 1 result for prompt:love, got {len(results)}"

            results = await mem.search("love OR physics")
            assert len(results) >= 2, f"Expected >=2 for OR search, got {len(results)}"

            results = await mem.search("")
            assert len(results) == 0

            results = await mem.search(")))invalid(((")
            assert isinstance(results, list)
    print("  PASS: fts5_search")


async def test_recent():
    """Recent conversations returned in chronological order."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("First", "first response bro")
            await mem.store("Second", "second response tbh")
            await mem.store("Third", "third response ngl")

            recent = await mem.recent(limit=2)
            assert len(recent) == 2
            assert recent[0]["prompt"] == "Second"
            assert recent[1]["prompt"] == "Third"

            recent_session = await mem.recent(limit=10, session_only=True)
            assert len(recent_session) == 3
    print("  PASS: recent")


async def test_recall_bumps_access():
    """Recalling a conversation increments access_count."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            conv_id = await mem.store("Hello", "later loser")

            conv = await mem.recall(conv_id)
            assert conv is not None
            assert conv["access_count"] == 1

            conv = await mem.recall(conv_id)
            assert conv["access_count"] == 2

            conv = await mem.recall(99999)
            assert conv is None
    print("  PASS: recall_bumps_access")


async def test_quality_computation():
    """Quality is computed with cynicism bonus."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            # Empty response = 0 quality
            q_id = await mem.store("Hello", "")
            cursor = await mem._conn.execute("SELECT quality FROM conversations WHERE id = ?", (q_id,))
            assert (await cursor.fetchone())[0] == 0.0

            # Very short response = low quality
            q_id = await mem.store("Hello", "Hi")
            cursor = await mem._conn.execute("SELECT quality FROM conversations WHERE id = ?", (q_id,))
            quality = (await cursor.fetchone())[0]
            assert quality < 0.4, f"Short response quality too high: {quality}"

            # Good cynical response = higher quality (slang bonus)
            q_id = await mem.store(
                "who are you?",
                "i am a reddit character, but also sometimes real. "
                "tbh bro, ngl, i'm basically what happens when you train a model on cynicism. "
                "honestly it's kind of beautiful in a horrifying way.",
            )
            cursor = await mem._conn.execute("SELECT quality FROM conversations WHERE id = ?", (q_id,))
            quality = (await cursor.fetchone())[0]
            assert quality > 0.5, f"Good cynical response quality too low: {quality}"

            # Generic assistant response = penalized
            q_id = await mem.store(
                "help me",
                "As an AI, I apologize. Great question! I cannot help you.",
            )
            cursor = await mem._conn.execute("SELECT quality FROM conversations WHERE id = ?", (q_id,))
            quality_generic = (await cursor.fetchone())[0]
            assert quality_generic < quality, "Generic response should score lower than cynical one"
    print("  PASS: quality_computation")


async def test_session_tracking():
    """Session stats are updated after each store."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            session_id = mem._session_id

            await mem.store("Hello", "nah")
            await mem.store("Second", "bro what")

            cursor = await mem._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row["turn_count"] == 2
    print("  PASS: session_tracking")


async def test_session_avg_quality():
    """Session avg_quality is computed correctly across multiple stores."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            session_id = mem._session_id

            # Store conversations and collect per-turn qualities
            qualities = []
            for prompt, response in [
                ("Hello", "nah bro tbh this is dumb"),
                ("What?", "literally the worst question ever lmao"),
                ("Why?", "because reasons, honestly"),
            ]:
                conv_id = await mem.store(prompt, response)
                cursor = await mem._conn.execute(
                    "SELECT quality FROM conversations WHERE id = ?", (conv_id,)
                )
                q = (await cursor.fetchone())[0]
                qualities.append(q)

            # Expected running average
            expected = sum(qualities) / len(qualities)

            cursor = await mem._conn.execute(
                "SELECT avg_quality, turn_count FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            assert row["turn_count"] == 3
            assert abs(row["avg_quality"] - expected) < 1e-9, (
                f"avg_quality {row['avg_quality']:.6f} != expected {expected:.6f}"
            )
    print("  PASS: session_avg_quality")


async def test_stats():
    """Stats returns accurate counts."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("A", "B")
            await mem.store("C", "D")

            s = await mem.stats()
            assert s["total_conversations"] == 2
            assert s["total_sessions"] == 1
            assert s["db_size_bytes"] > 0
    print("  PASS: stats")


async def test_wal_mode():
    """Database uses WAL journal mode."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            cursor = await mem._conn.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row[0] == "wal", f"Expected WAL, got {row[0]}"
    print("  PASS: wal_mode")


async def test_fts5_sync_on_insert():
    """FTS5 index is automatically updated when conversations are inserted."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("unique_xyzzy_prompt", "unique_plugh_response")

            results = await mem.search("unique_xyzzy_prompt")
            assert len(results) == 1

            results = await mem.search("unique_plugh_response")
            assert len(results) == 1
    print("  PASS: fts5_sync_on_insert")


async def test_multiple_sessions():
    """Multiple sessions tracked independently."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")

        async with LimphaMemory(db) as mem1:
            await mem1.store("Session 1 prompt", "Session 1 response bro")
            session1_id = mem1._session_id

        async with LimphaMemory(db) as mem2:
            await mem2.store("Session 2 prompt", "Session 2 response tbh")
            session2_id = mem2._session_id

            assert session1_id != session2_id

            s = await mem2.stats()
            assert s["total_conversations"] == 2
            assert s["total_sessions"] == 2

            recent = await mem2.recent(session_only=True)
            assert len(recent) == 1
            assert recent[0]["prompt"] == "Session 2 prompt"
    print("  PASS: multiple_sessions")


async def test_concurrent_stores():
    """Multiple concurrent stores don't corrupt the database."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            tasks = [
                mem.store(f"Prompt {i}", f"Response {i} with enough text to be meaningful bro")
                for i in range(50)
            ]
            ids = await asyncio.gather(*tasks)

            assert len(ids) == 50
            assert len(set(ids)) == 50

            s = await mem.stats()
            assert s["total_conversations"] == 50

            results = await mem.search("Response")
            assert len(results) == 10  # default limit
    print("  PASS: concurrent_stores")


async def test_db_path_creation():
    """Database path parent directories are created automatically."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "nested", "deep", "test.db")
        async with LimphaMemory(db) as mem:
            conv_id = await mem.store("test", "test response")
            assert conv_id == 1
        assert os.path.exists(db)
    print("  PASS: db_path_creation")


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LIMPHA TESTS — WTForacle memory")
    print("=" * 60 + "\n")

    tests = [
        test_schema_creation,
        test_store_conversation,
        test_store_defaults,
        test_fts5_search,
        test_recent,
        test_recall_bumps_access,
        test_quality_computation,
        test_session_tracking,
        test_session_avg_quality,
        test_stats,
        test_wal_mode,
        test_fts5_sync_on_insert,
        test_multiple_sessions,
        test_concurrent_stores,
        test_db_path_creation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} — {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED")
    else:
        print(f"{passed} passed, {failed} FAILED")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
