from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from waggle.embeddings import EmbeddingModel
from waggle.intelligence import normalize_text


class MemoryGraphBase:
    """Base class for MemoryGraph mixins providing shared type signatures."""

    db_path: Path
    embedding_model: EmbeddingModel
    tenant_id: str
    _lock: Any
    _pool: Any

    def _connect(self, timeout: float = 30.0, *, check_same_thread: bool = True) -> sqlite3.Connection:
        raise NotImplementedError

    def get_stats(self) -> Any:
        raise NotImplementedError

    def export_context_bundle(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def export_abhi(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def add_node(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def add_edge(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def resolve_window_context(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _update_window_node_count(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _mark_window_embedding_stale(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def derive_context_window_edges(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _embed_with_metadata(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _node_cosine_similarity(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _query_replay_hits(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _expand_query_aliases(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _parse_datetime(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _encode_metadata(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, sort_keys=True)


def _decode_metadata(raw: Any) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _normalized_content_hash(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
