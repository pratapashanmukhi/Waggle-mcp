"""Tests for :mod:`waggle.connection_pool` and its use by ``MemoryGraph``.

Covers the acceptance criteria from issue #126:

* connections are reused across operations (the factory is not re-invoked and no
  fresh PRAGMA round happens per checkout),
* the pool size stays bounded under repeated checkout/return,
* two threads checking out connections at the same time do not crash,
* ``MemoryGraph`` routes its operations through the pool and tears it down on
  ``close()`` while tenant clones safely share it.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from waggle.connection_pool import (
    DEFAULT_POOL_SIZE,
    ConnectionPoolClosedError,
    SQLiteConnectionPool,
)
from waggle.graph import MemoryGraph
from waggle.models import NodeType


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _CountingFactory:
    """A connection factory that records how many connections it has created."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.calls = 0

    def __call__(self) -> sqlite3.Connection:
        self.calls += 1
        connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        # The same PRAGMAs MemoryGraph._connect applies, paid once per connection.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection


class FakeEmbeddingModel:
    """Deterministic, dependency-free embedding model for graph integration tests."""

    model_name = "fake-model"
    model_id = "fake-model:deterministic-v1"

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            index = sum(ord(character) for character in token) % len(vector)
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        return vector if norm == 0.0 else vector / norm

    def to_bytes(self, embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    def from_bytes(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))


@pytest.fixture
def pool(tmp_path: Path):
    factory = _CountingFactory(tmp_path / "pool.db")
    created = SQLiteConnectionPool(factory, size=3, checkout_timeout=1.0)
    created.factory = factory  # expose for assertions
    try:
        yield created
    finally:
        created.close()


# --------------------------------------------------------------------------- #
# Pool construction and configuration
# --------------------------------------------------------------------------- #
def test_factory_called_once_per_connection_at_creation(tmp_path: Path) -> None:
    factory = _CountingFactory(tmp_path / "db.sqlite")
    created = SQLiteConnectionPool(factory, size=4)
    try:
        assert factory.calls == 4
        assert created.size == 4
        assert created.available() == 4
    finally:
        created.close()


def test_default_size_is_small(tmp_path: Path) -> None:
    factory = _CountingFactory(tmp_path / "db.sqlite")
    created = SQLiteConnectionPool(factory)
    try:
        assert created.size == DEFAULT_POOL_SIZE
        assert factory.calls == DEFAULT_POOL_SIZE
    finally:
        created.close()


def test_invalid_size_rejected(tmp_path: Path) -> None:
    factory = _CountingFactory(tmp_path / "db.sqlite")
    with pytest.raises(ValueError):
        SQLiteConnectionPool(factory, size=0)


def test_pragmas_applied_to_pooled_connections(pool: SQLiteConnectionPool) -> None:
    with pool.checkout() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
    assert journal_mode.lower() == "wal"
    assert foreign_keys == 1


# --------------------------------------------------------------------------- #
# Reuse: no new connection / PRAGMA per checkout
# --------------------------------------------------------------------------- #
def test_connections_are_reused_without_new_factory_calls(pool: SQLiteConnectionPool) -> None:
    calls_after_construction = pool.factory.calls
    seen: set[int] = set()
    for _ in range(50):
        with pool.checkout() as connection:
            seen.add(id(connection))
    # The factory is never invoked again after construction...
    assert pool.factory.calls == calls_after_construction
    # ...and every connection handed out came from the pre-created set.
    assert len(seen) <= pool.size


# --------------------------------------------------------------------------- #
# Bounded size
# --------------------------------------------------------------------------- #
def test_pool_size_stays_bounded(pool: SQLiteConnectionPool) -> None:
    with ExitStack() as stack:
        connections = [stack.enter_context(pool.checkout()) for _ in range(pool.size)]
        assert len(connections) == pool.size
        assert pool.available() == 0
        # No fourth connection exists; a further checkout times out rather than
        # silently growing the pool.
        with pytest.raises(TimeoutError), pool.checkout():
            pass
    # Everything is returned once the ExitStack unwinds.
    assert pool.available() == pool.size


def test_checkout_returns_connection_to_pool(pool: SQLiteConnectionPool) -> None:
    assert pool.available() == pool.size
    with pool.checkout():
        assert pool.available() == pool.size - 1
    assert pool.available() == pool.size


# --------------------------------------------------------------------------- #
# Transaction semantics mirror sqlite3.Connection context manager
# --------------------------------------------------------------------------- #
def test_commit_on_clean_exit(pool: SQLiteConnectionPool) -> None:
    with pool.checkout() as connection:
        connection.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        connection.execute("INSERT INTO t (v) VALUES ('kept')")
    with pool.checkout() as connection:
        rows = connection.execute("SELECT v FROM t").fetchall()
    assert [row[0] for row in rows] == ["kept"]


def test_rollback_on_error(pool: SQLiteConnectionPool) -> None:
    with pool.checkout() as connection:
        connection.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    with pytest.raises(RuntimeError), pool.checkout() as connection:
        connection.execute("INSERT INTO t (v) VALUES ('discarded')")
        raise RuntimeError("boom")
    with pool.checkout() as connection:
        count = connection.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert count == 0


# --------------------------------------------------------------------------- #
# Thread safety
# --------------------------------------------------------------------------- #
def test_concurrent_checkout_does_not_crash(tmp_path: Path) -> None:
    factory = _CountingFactory(tmp_path / "concurrent.db")
    created = SQLiteConnectionPool(factory, size=4, checkout_timeout=5.0)
    with created.checkout() as connection:
        connection.execute("CREATE TABLE counter (n INTEGER)")
        connection.execute("INSERT INTO counter (n) VALUES (0)")

    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait()
            for _ in range(40):
                with created.checkout() as conn:
                    conn.execute("SELECT n FROM counter").fetchone()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    try:
        assert not any(thread.is_alive() for thread in threads), "a worker thread hung"
        assert errors == []
        # Every borrowed connection was returned; the pool never grew or shrank.
        assert created.available() == created.size
        assert factory.calls == created.size
    finally:
        created.close()


# --------------------------------------------------------------------------- #
# close()
# --------------------------------------------------------------------------- #
def test_close_blocks_further_checkout(tmp_path: Path) -> None:
    factory = _CountingFactory(tmp_path / "closed.db")
    created = SQLiteConnectionPool(factory, size=2)
    created.close()
    assert created.closed is True
    with pytest.raises(ConnectionPoolClosedError), created.checkout():
        pass


def test_close_is_idempotent(tmp_path: Path) -> None:
    factory = _CountingFactory(tmp_path / "closed.db")
    created = SQLiteConnectionPool(factory, size=2)
    created.close()
    created.close()  # must not raise


def test_close_closes_underlying_connections(tmp_path: Path) -> None:
    factory = _CountingFactory(tmp_path / "closed.db")
    created = SQLiteConnectionPool(factory, size=1)
    leaked = created._all_connections[0]
    created.close()
    with pytest.raises(sqlite3.ProgrammingError):
        leaked.execute("SELECT 1")


def test_context_manager_closes_pool(tmp_path: Path) -> None:
    factory = _CountingFactory(tmp_path / "ctx.db")
    with SQLiteConnectionPool(factory, size=2) as created:
        assert created.available() == 2
    assert created.closed is True


# --------------------------------------------------------------------------- #
# MemoryGraph integration
# --------------------------------------------------------------------------- #
def _make_graph(tmp_path: Path) -> MemoryGraph:
    return MemoryGraph(tmp_path / "memory.db", FakeEmbeddingModel())


def test_memory_graph_builds_a_bounded_pool(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    try:
        assert graph._owns_pool is True
        assert graph._pool.size == DEFAULT_POOL_SIZE
        assert len(graph._pool._all_connections) == DEFAULT_POOL_SIZE
    finally:
        graph.close()


def test_memory_graph_operations_do_not_open_new_connections(tmp_path: Path, monkeypatch) -> None:
    graph = _make_graph(tmp_path)
    try:
        # After construction nothing should call _connect again; the pool holds
        # every connection. If an operation tried to, this would fail loudly.
        def _fail() -> sqlite3.Connection:  # pragma: no cover - only on regression
            raise AssertionError("_connect was called after pool construction")

        monkeypatch.setattr(graph, "_connect", _fail)

        for index in range(10):
            graph.add_node(
                label=f"node-{index}",
                content=f"content about topic number {index}",
                node_type=NodeType.ENTITY,
            )
        # query() exercises the HybridRetriever, which also borrows from the pool.
        result = graph.query(query="topic", max_nodes=5, max_depth=1)

        assert result is not None
        assert graph._pool.available() == graph._pool.size
    finally:
        graph.close()


def test_for_tenant_shares_pool_without_owning_it(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    try:
        clone = graph.for_tenant("tenant-b")
        assert clone._pool is graph._pool
        assert clone._owns_pool is False

        # Closing the clone must not tear down the shared pool.
        clone.close()
        assert graph._pool.closed is False
        # The owner can still use the pool afterwards.
        graph.add_node(label="still-alive", content="owner still works", node_type=NodeType.ENTITY)
    finally:
        graph.close()


def test_memory_graph_close_closes_pool(tmp_path: Path) -> None:
    graph = _make_graph(tmp_path)
    graph.add_node(label="before-close", content="written before close", node_type=NodeType.ENTITY)
    graph.close()
    assert graph._pool.closed is True
    graph.close()  # idempotent
