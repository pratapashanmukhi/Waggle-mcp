"""Ad-hoc before/after benchmark for issue #126.

Measures two things:

1. Micro: the raw cost of acquiring a usable connection 100 times, the old way (sqlite3.connect + 7 PRAGMAs every time) vs. the new way (pool checkout).
2. End-to-end: 100 MemoryGraph.add_node operations with the pool vs. with the pool patched to open a fresh connection per checkout (the pre-PR behavior).
"""

from __future__ import annotations

import sys
import time
import gc
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from waggle.graph import MemoryGraph  # noqa: E402
from waggle.models import NodeType  # noqa: E402

N = 100


class FakeEmbeddingModel:
    model_name = "fake-model"
    model_id = "fake-model:deterministic-v1"

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            vector[sum(ord(c) for c in token) % len(vector)] += 1.0
        norm = np.linalg.norm(vector)
        return vector if norm == 0.0 else vector / norm

    def to_bytes(self, e: np.ndarray) -> bytes:
        return e.astype(np.float32).tobytes()

    def from_bytes(self, d: bytes) -> np.ndarray:
        return np.frombuffer(d, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def micro_benchmark(graph: MemoryGraph) -> None:
    # OLD: fresh connection + full PRAGMA round every acquisition.
    start = time.perf_counter()
    for _ in range(N):
        conn = graph._connect()
        conn.execute("SELECT 1").fetchone()
        conn.commit()
        conn.close()
    old = time.perf_counter() - start

    # NEW: borrow a pre-configured connection from the pool.
    start = time.perf_counter()
    for _ in range(N):
        with graph._pool.checkout() as conn:
            conn.execute("SELECT 1").fetchone()
    new = time.perf_counter() - start

    print(f"  acquisition x{N}:  before (fresh+PRAGMAs) = {old * 1e3:8.2f} ms   "
          f"after (pool) = {new * 1e3:8.2f} ms   speedup = {old / new:5.2f}x")


def e2e_benchmark() -> None:
    embedder = FakeEmbeddingModel()

    # BEFORE: pool patched so each checkout opens a brand-new connection, exactly reproducing the pre-PR `with self._connect() as connection:` path.
    with TemporaryDirectory() as tmp:
        graph = MemoryGraph(Path(tmp) / "before.db", embedder)

        @contextmanager
        def fresh_checkout():
            conn = graph._connect()
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

        graph._pool.checkout = fresh_checkout  # type: ignore[method-assign]
        start = time.perf_counter()
        for i in range(N):
            graph.add_node(label=f"n{i}", content=f"content number {i}", node_type=NodeType.ENTITY)
        before = time.perf_counter() - start
        
        graph.close()
        del graph
        gc.collect()
        time.sleep(0.05)

    # AFTER: real pooled connections.
    with TemporaryDirectory() as tmp:
        graph = MemoryGraph(Path(tmp) / "after.db", embedder)
        start = time.perf_counter()
        for i in range(N):
            graph.add_node(label=f"n{i}", content=f"content number {i}", node_type=NodeType.ENTITY)
        after = time.perf_counter() - start
        
        graph.close()
        del graph
        gc.collect()      # Force Python to destroy any lingering cursors/objects
        time.sleep(0.05)  # Give a split second to release the file handle completely

    print(f"  add_node x{N}:     before (fresh per op)  = {before * 1e3:8.2f} ms   "
          f"after (pool) = {after * 1e3:8.2f} ms   speedup = {before / after:5.2f}x")


if __name__ == "__main__":
    print(f"SQLite connection pooling benchmark (N={N}, median of 5 runs)\n")
    with TemporaryDirectory() as tmp:
        g = MemoryGraph(Path(tmp) / "micro.db", FakeEmbeddingModel())
        for _ in range(5):
            micro_benchmark(g)
        g.close()
    print()
    for _ in range(5):
        e2e_benchmark()
