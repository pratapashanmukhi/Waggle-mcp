"""A small, thread-safe pool of pre-configured SQLite connections.

Every graph operation in :mod:`waggle.graph` used to call
``MemoryGraph._connect()``, which opened a brand-new :class:`sqlite3.Connection`,
set ``row_factory``, and executed seven ``PRAGMA`` statements (WAL,
``synchronous``, ``busy_timeout``, ``foreign_keys``, ``mmap_size``,
``temp_store``, ``cache_size``).  With more than 70 call sites, that meant a
fresh connection and a fresh round of ``PRAGMA`` execution on every read and
write.

Under WAL mode SQLite supports many concurrent readers plus a single writer, so
connections can safely be reused.  :class:`SQLiteConnectionPool` pre-creates a
small, fixed number of connections, configures the ``PRAGMA`` statements exactly
once per connection at creation time, and hands them out through a context
manager that returns the connection to the pool on exit.

The pool is deliberately small.  WAL permits only one writer at a time
regardless of how many connections exist, so a large pool would only waste file
handles.  The default size of four is comfortable for the read-mostly workload
``MemoryGraph`` produces while leaving headroom for concurrent readers.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress

__all__ = ["ConnectionPoolClosedError", "SQLiteConnectionPool"]

# Default number of connections kept in the pool.  WAL allows a single writer regardless of pool size, so this is small on purpose.
DEFAULT_POOL_SIZE = 4

# Default number of seconds :meth:`SQLiteConnectionPool.checkout` waits for a free connection before giving up.
# Mirrors the SQLite ``busy_timeout`` used elsewhere so legitimate contention waits, while a genuine exhaustion surfaces as an error instead of hanging forever.
DEFAULT_CHECKOUT_TIMEOUT = 30.0


class ConnectionPoolClosedError(RuntimeError):
    """Raised when a connection is requested from a pool that is closed."""


class SQLiteConnectionPool:
    """A bounded, thread-safe pool of pre-configured SQLite connections.

    Args:
        connection_factory: A zero-argument callable that returns a fully
            configured :class:`sqlite3.Connection` (``row_factory`` set and all
            ``PRAGMA`` statements applied).  The factory is invoked exactly
            ``size`` times when the pool is constructed, so the per-connection
            ``PRAGMA`` cost is paid once up front rather than on every checkout.
        size: Number of connections to pre-create.  Must be at least 1.
        checkout_timeout: Seconds to wait for a free connection before raising
            :class:`TimeoutError`.  ``None`` waits indefinitely.

    Thread safety:
        The idle connections live in a :class:`queue.LifoQueue`, whose ``get``
        and ``put`` operations are individually atomic.  Checkout removes a
        connection from the queue (blocking if all are in use) and the context
        manager returns it on exit, so the number of connections handed out
        never exceeds ``size``.  A LIFO queue is used so a small set of "warm"
        connections is reused preferentially.  A separate lock guards the
        one-shot :meth:`close` transition.
    """

    def __init__(
        self,
        connection_factory: Callable[[], sqlite3.Connection],
        *,
        size: int = DEFAULT_POOL_SIZE,
        checkout_timeout: float | None = DEFAULT_CHECKOUT_TIMEOUT,
    ) -> None:
        if size < 1:
            raise ValueError("Connection pool size must be at least 1.")
        self._factory = connection_factory
        self._size = size
        self._checkout_timeout = checkout_timeout
        self._idle: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue(maxsize=size)
        # Keep a reference to every connection we created so close() can shut
        # them all down even if some are currently checked out.
        self._all_connections: list[sqlite3.Connection] = []
        self._close_lock = threading.Lock()
        self._closed = False

        for _ in range(size):
            connection = self._factory()
            self._all_connections.append(connection)
            self._idle.put(connection)

    @property
    def size(self) -> int:
        """The fixed number of connections managed by the pool."""
        return self._size

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closed

    def available(self) -> int:
        """Approximate number of connections currently idle in the pool.

        Intended for tests and introspection.  The value is a snapshot and may
        be stale the instant it is read in a concurrent setting.
        """
        return self._idle.qsize()

    @contextmanager
    def checkout(self) -> Iterator[sqlite3.Connection]:
        """Borrow a connection, returning it to the pool on exit.

        The context manager mirrors the transaction semantics of using a
        :class:`sqlite3.Connection` directly as a context manager: the
        transaction is committed on a clean exit and rolled back if the body
        raises.  Unlike the bare connection context manager, the connection is
        *not* closed afterwards — it is returned to the pool for reuse.

        Raises:
            ConnectionPoolClosedError: If the pool has been closed.
            TimeoutError: If no connection becomes available within
                ``checkout_timeout`` seconds.
        """
        if self._closed:
            raise ConnectionPoolClosedError("Cannot check out a connection from a closed pool.")
        try:
            connection = self._idle.get(timeout=self._checkout_timeout)
        except queue.Empty as exc:  # pragma: no cover - only on genuine exhaustion
            raise TimeoutError(
                f"Timed out after {self._checkout_timeout}s waiting for a pooled SQLite connection."
            ) from exc

        try:
            yield connection
        except BaseException:
            # Match sqlite3.Connection.__exit__: roll back on error.
            with suppress(sqlite3.Error):
                connection.rollback()
            raise
        else:
            # Match sqlite3.Connection.__exit__: commit on success.  Harmless
            # no-op for read-only work.
            connection.commit()
        finally:
            self._idle.put(connection)

    def close(self) -> None:
        """Close every pooled connection.  Idempotent and safe to call twice."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            connections = list(self._all_connections)
            self._all_connections.clear()

        for connection in connections:
            with suppress(sqlite3.Error):  # pragma: no cover - defensive
                connection.close()

    def __enter__(self) -> SQLiteConnectionPool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
