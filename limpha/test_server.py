#!/usr/bin/env python3
"""
LIMPHA SERVER TESTS — Unix socket IPC protocol.

Run: python3 limpha/test_server.py
"""

import asyncio
import json
import os
import tempfile
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limpha.memory import LimphaMemory
from limpha.server import dispatch


async def make_dispatch(cmd, msg, memory):
    """Helper: dispatch a command without needing a real server."""
    shutdown = asyncio.Event()
    msg["cmd"] = cmd
    return await dispatch(cmd, msg, memory, shutdown)


async def test_store_command():
    """Store command returns conversation ID."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            resp = await make_dispatch("store", {
                "prompt": "who are you?",
                "response": "i am wtforacle bro",
                "temperature": 0.9,
            }, mem)
            assert resp["ok"] is True
            assert resp["id"] == 1
    print("  PASS: store_command")


async def test_search_command():
    """Search command returns results."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("what is love?", "love is when a meme stays in you bro")
            await mem.store("what is AI?", "it's like us but smarter tbh")

            resp = await make_dispatch("search", {"query": "love", "limit": 5}, mem)
            assert resp["ok"] is True
            assert len(resp["results"]) == 1
    print("  PASS: search_command")


async def test_recent_command():
    """Recent command returns conversations."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("First", "first bro")
            await mem.store("Second", "second tbh")

            resp = await make_dispatch("recent", {"limit": 10}, mem)
            assert resp["ok"] is True
            assert len(resp["conversations"]) == 2
    print("  PASS: recent_command")


async def test_recall_command():
    """Recall command returns specific conversation."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("test prompt", "test response bro")

            resp = await make_dispatch("recall", {"id": 1}, mem)
            assert resp["ok"] is True
            assert resp["conversation"]["prompt"] == "test prompt"

            resp = await make_dispatch("recall", {"id": 999}, mem)
            assert resp["ok"] is False
    print("  PASS: recall_command")


async def test_stats_command():
    """Stats command returns statistics."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            await mem.store("A", "B bro")

            resp = await make_dispatch("stats", {}, mem)
            assert resp["ok"] is True
            assert resp["total_conversations"] == 1
    print("  PASS: stats_command")


async def test_ping_command():
    """Ping command returns pong."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            resp = await make_dispatch("ping", {}, mem)
            assert resp["ok"] is True
            assert resp["pong"] is True
    print("  PASS: ping_command")


async def test_unknown_command():
    """Unknown command returns error."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            resp = await make_dispatch("explode", {}, mem)
            assert resp["ok"] is False
            assert "unknown" in resp["error"]
    print("  PASS: unknown_command")


async def test_shutdown_command():
    """Shutdown command sets event."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        async with LimphaMemory(db) as mem:
            shutdown = asyncio.Event()
            resp = await dispatch("shutdown", {"cmd": "shutdown"}, mem, shutdown)
            assert resp["ok"] is True
            assert shutdown.is_set()
    print("  PASS: shutdown_command")


async def run_all_tests():
    """Run all server tests."""
    print("\n" + "=" * 60)
    print("LIMPHA SERVER TESTS — WTForacle IPC protocol")
    print("=" * 60 + "\n")

    tests = [
        test_store_command,
        test_search_command,
        test_recent_command,
        test_recall_command,
        test_stats_command,
        test_ping_command,
        test_unknown_command,
        test_shutdown_command,
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
