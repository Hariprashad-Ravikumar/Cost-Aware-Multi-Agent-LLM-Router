import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agents import session


class _FakeRedis:
    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


@pytest.mark.asyncio
async def test_get_history_empty_for_unknown_session():
    fake = _FakeRedis()
    with patch("app.cache.get_client", return_value=fake):
        history = await session.get_history("unknown-session")
    assert history == []


@pytest.mark.asyncio
async def test_append_turn_then_get_history_round_trips():
    fake = _FakeRedis()
    with patch("app.cache.get_client", return_value=fake):
        await session.append_turn("s1", "What is 15% of 240?", "36")
        history = await session.get_history("s1")

    assert history == [{"request": "What is 15% of 240?", "answer": "36"}]


@pytest.mark.asyncio
async def test_session_history_caps_at_max_turns():
    fake = _FakeRedis()
    with patch("app.cache.get_client", return_value=fake), patch.object(session, "MAX_TURNS", 2):
        await session.append_turn("s1", "q1", "a1")
        await session.append_turn("s1", "q2", "a2")
        await session.append_turn("s1", "q3", "a3")
        history = await session.get_history("s1")

    assert len(history) == 2
    assert history[0]["request"] == "q2"
    assert history[1]["request"] == "q3"


@pytest.mark.asyncio
async def test_sessions_are_isolated_by_id():
    fake = _FakeRedis()
    with patch("app.cache.get_client", return_value=fake):
        await session.append_turn("session-a", "qa", "aa")
        await session.append_turn("session-b", "qb", "ab")
        history_a = await session.get_history("session-a")
        history_b = await session.get_history("session-b")

    assert history_a == [{"request": "qa", "answer": "aa"}]
    assert history_b == [{"request": "qb", "answer": "ab"}]
