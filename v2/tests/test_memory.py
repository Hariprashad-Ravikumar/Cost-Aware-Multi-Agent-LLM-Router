import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import numpy as np

from app.agents import memory


class _FakeEmbedder:
    def encode(self, text):
        return np.array([0.1, 0.2, 0.3])


def test_write_memory_commits_a_memory_entry():
    fake_session = MagicMock()
    with patch("app.agents.memory.get_embedder", return_value=_FakeEmbedder()), \
         patch("app.db.get_session", return_value=fake_session):
        memory.write_memory("some fact", {"source": "test"})

    fake_session.add.assert_called_once()
    added = fake_session.add.call_args[0][0]
    assert added.text == "some fact"
    assert added.entry_metadata == {"source": "test"}
    fake_session.commit.assert_called_once()
    fake_session.close.assert_called_once()


def test_write_memory_no_embedder_skips_write():
    fake_session = MagicMock()
    with patch("app.agents.memory.get_embedder", return_value=None), \
         patch("app.db.get_session", return_value=fake_session):
        memory.write_memory("some fact", {})

    fake_session.add.assert_not_called()


def test_retrieve_memory_no_embedder_returns_empty():
    with patch("app.agents.memory.get_embedder", return_value=None):
        assert memory.retrieve_memory("a query") == []


def test_retrieve_memory_db_error_returns_empty_not_raises():
    with patch("app.agents.memory.get_embedder", return_value=_FakeEmbedder()), \
         patch("app.db.get_session", side_effect=RuntimeError("db down")):
        assert memory.retrieve_memory("a query") == []
