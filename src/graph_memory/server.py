from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import mcp.server.stdio
import mcp.types as types
import uvicorn
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.lowlevel.server import request_ctx
from mcp.server.models import InitializationOptions
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route

from graph_memory.config import AppConfig
from graph_memory.embeddings import EmbeddingModel
from graph_memory.errors import (
    AuthenticationError,
    GraphMemoryError,
    PayloadTooLargeError,
    ServiceUnavailableError,
    ValidationFailure,
)
from graph_memory.graph import MemoryGraph
from graph_memory.logging_utils import configure_logging
from graph_memory.metrics import MetricsRegistry
from graph_memory.models import (
    GraphDiffResult,
    GraphStats,
    Node,
    NodeType,
    ObservationResult,
    PrimeContextResult,
    RelationType,
    SubgraphResult,
    TopicResult,
)
from graph_memory.rate_limit import RateLimiter
from graph_memory.runtime_context import runtime_context
from graph_memory.serializer import (
    serialize_graph_diff,
    serialize_observation_result,
    serialize_prime_context,
    serialize_recent_nodes,
    serialize_stats,
    serialize_subgraph,
    serialize_topics,
)

LOGGER = logging.getLogger(__name__)
WRITE_HEAVY_TOOLS = {"store_node", "store_edge", "decompose_and_store", "observe_conversation", "import_graph_backup"}


def _build_backend(config: AppConfig) -> Any:
    embedding_model = EmbeddingModel(config.model_name)
    if config.backend == "sqlite":
        return MemoryGraph(
            config.db_path,
            embedding_model,
            tenant_id=config.default_tenant_id,
            export_dir=config.export_dir,
        )
    from graph_memory.neo4j_graph import Neo4jMemoryGraph

    return Neo4jMemoryGraph(
        uri=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password,
        database=config.neo4j_database or None,
        embedding_model=embedding_model,
        tenant_id=config.default_tenant_id,
        export_dir=config.export_dir,
    )


class GraphMemoryServer:
    """MCP server wrapper with tenant-aware graph resolution."""

    def __init__(
        self,
        graph: Any | None = None,
        *,
        config: AppConfig | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self.metrics = metrics or MetricsRegistry()
        self._static_graph = graph
        self._root_graph = graph or _build_backend(self.config)
        self.server = Server("graph-memory")
        self._register_handlers()

    @property
    def graph(self) -> Any:
        return self.current_graph()

    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return self.build_tools()

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
            return await anyio.to_thread.run_sync(self.handle_tool_call, name, arguments or {})

        @self.server.list_resources()
        async def list_resources(
            request: types.ListResourcesRequest | None = None,
        ) -> types.ListResourcesResult:
            del request
            return self.build_resources()

        @self.server.read_resource()
        async def read_resource(uri: Any) -> str:
            return self.read_resource_text(str(uri))

    def build_tools(self) -> list[types.Tool]:
        return [
            types.Tool(
                name="store_node",
                description=(
                    "Store a piece of knowledge as a node in the persistent memory graph. "
                    "Call this whenever you learn something important from the user: facts, "
                    "preferences, decisions, entities, concepts, or questions. Prefer atomic facts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Short label for the knowledge being stored."},
                        "content": {"type": "string", "description": "Full natural-language description for this node."},
                        "node_type": {
                            "type": "string",
                            "enum": [node_type.value for node_type in NodeType],
                            "description": "Category of knowledge represented by the node.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization.",
                            "default": [],
                        },
                        "source_prompt": {
                            "type": "string",
                            "description": "Optional original prompt that produced this knowledge.",
                            "default": "",
                        },
                    },
                    "required": ["label", "content", "node_type"],
                },
            ),
            types.Tool(
                name="store_edge",
                description=(
                    "Create a relationship between two stored nodes. Use this immediately after "
                    "storing related nodes so the memory graph preserves structure, updates, and conflicts."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string", "description": "Source node ID."},
                        "target_id": {"type": "string", "description": "Target node ID."},
                        "relationship": {
                            "type": "string",
                            "enum": [relation.value for relation in RelationType],
                            "description": "Relationship between the two nodes.",
                        },
                        "weight": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 1.0,
                            "description": "Optional strength of the relationship.",
                        },
                    },
                    "required": ["source_id", "target_id", "relationship"],
                },
            ),
            types.Tool(
                name="query_graph",
                description=(
                    "Search the memory graph for information relevant to a natural-language query. "
                    "Returns a serialized subgraph with matching nodes and their connected neighborhood. "
                    "Understands temporal references such as 'recently', 'latest', 'originally', and 'last week'."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural-language search query."},
                        "max_nodes": {"type": "integer", "default": 20, "minimum": 1},
                        "max_depth": {"type": "integer", "default": 2, "minimum": 0},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="get_related",
                description="Get all nodes connected to a specific node up to a configurable depth.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string"},
                        "max_depth": {"type": "integer", "default": 2, "minimum": 0},
                    },
                    "required": ["node_id"],
                },
            ),
            types.Tool(
                name="update_node",
                description="Update an existing node's content or metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string"},
                        "content": {"type": "string"},
                        "label": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["node_id"],
                },
            ),
            types.Tool(
                name="delete_node",
                description="Delete a node and all connected edges from persistent memory.",
                inputSchema={"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]},
            ),
            types.Tool(
                name="decompose_and_store",
                description="Break long or complex content into atomic nodes, store them automatically, and create inferred edges.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "context": {"type": "string", "default": ""},
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="observe_conversation",
                description="Observe a completed conversation turn, extract important information, and store it automatically.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_message": {"type": "string"},
                        "assistant_response": {"type": "string"},
                    },
                    "required": ["user_message", "assistant_response"],
                },
            ),
            types.Tool(
                name="graph_diff",
                description="Show what changed in the graph recently, including added or updated nodes and created edges.",
                inputSchema={"type": "object", "properties": {"since": {"type": "string", "default": "24h"}}},
            ),
            types.Tool(
                name="prime_context",
                description="Build a compact context brief for the start of a new conversation.",
                inputSchema={"type": "object", "properties": {"project": {"type": "string", "default": ""}}},
            ),
            types.Tool(
                name="get_topics",
                description="Detect topic clusters in the graph using community detection and return the main themes.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(name="get_stats", description="Return high-level statistics about the current memory graph.", inputSchema={"type": "object", "properties": {}}),
            types.Tool(
                name="export_graph_html",
                description="Export the current memory graph as an interactive HTML visualization.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "output_path": {"type": "string"},
                        "include_physics": {"type": "boolean", "default": True},
                    },
                },
            ),
            types.Tool(
                name="export_graph_backup",
                description="Export the current graph as a portable JSON backup.",
                inputSchema={"type": "object", "properties": {"output_path": {"type": "string"}}},
            ),
            types.Tool(
                name="import_graph_backup",
                description="Import a portable JSON graph backup into the current backend.",
                inputSchema={
                    "type": "object",
                    "properties": {"input_path": {"type": "string"}},
                    "required": ["input_path"],
                },
            ),
        ]

    def _get_request(self) -> Request | None:
        try:
            current = request_ctx.get()
        except LookupError:
            return None
        return current.request if isinstance(current.request, Request) else None

    def current_graph(self) -> Any:
        request = self._get_request()
        if request is not None and getattr(request.state, "tenant_id", ""):
            return self._root_graph.for_tenant(request.state.tenant_id)
        return self._root_graph.for_tenant(self.config.default_tenant_id)

    def validate_startup(self) -> None:
        graph = self.current_graph()
        started = time.perf_counter()
        graph.ensure_tenant(graph.tenant_id)
        graph.embedding_model.embed("startup validation")
        self.metrics.observe("graph_memory_startup_validation_seconds", time.perf_counter() - started, backend=self.config.backend)

    def build_resources(self) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(uri="graph://stats", name="Graph Stats", description="Current graph statistics.", mimeType="text/plain"),
                types.Resource(uri="graph://recent", name="Recent Graph Nodes", description="The 10 most recently updated nodes.", mimeType="text/plain"),
            ]
        )

    def read_resource_text(self, uri: str) -> str:
        graph = self.current_graph()
        if uri == "graph://stats":
            return serialize_stats(graph.get_stats())
        if uri == "graph://recent":
            return serialize_recent_nodes(graph.list_recent_nodes(limit=10))
        raise ValidationFailure(f"Unknown resource: {uri}")

    def initialization_options(self) -> InitializationOptions:
        return InitializationOptions(
            server_name="graph-memory",
            server_version="0.2.0",
            capabilities=self.server.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={}),
        )

    def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        graph = self.current_graph()
        request = self._get_request()
        request_id = ""
        if request is not None:
            request_id = getattr(request.state, "request_id", "")
        else:
            try:
                request_id = str(request_ctx.get().request_id)
            except LookupError:
                request_id = ""

        started = time.perf_counter()
        with runtime_context(
            request_id=request_id,
            tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id),
            transport="http" if request is not None else "stdio",
            backend=self.config.backend,
            api_key_id=getattr(getattr(request, "state", object()), "api_key_id", "") if request is not None else "",
            tool_name=name,
        ):
            try:
                self._validate_tool_payload(name, arguments)
                LOGGER.info("tool_call_started")
                if name == "store_node":
                    store_result = graph.add_node(
                        label=arguments["label"],
                        content=arguments["content"],
                        node_type=NodeType(arguments["node_type"]),
                        tags=arguments.get("tags", []),
                        source_prompt=arguments.get("source_prompt", ""),
                    )
                    node = store_result.node
                    text = (
                        f"Stored node '{node.label}' with id {node.id}."
                        if store_result.created
                        else f"Reused existing node '{node.label}' with id {node.id}."
                    )
                    if store_result.conflicts:
                        text += f" Detected {len(store_result.conflicts)} potential conflict(s)."
                    if not store_result.created:
                        self.metrics.increment(
                            "graph_memory_dedup_hits_total",
                            tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id),
                            dedup_reason=store_result.dedup_reason or "unknown",
                        )
                    if store_result.conflicts:
                        self.metrics.increment(
                            "graph_memory_conflicts_total",
                            value=len(store_result.conflicts),
                            tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id),
                        )
                    result = self._tool_result(
                        text,
                        {
                            **self._node_payload(node),
                            "created": store_result.created,
                            "dedup_reason": store_result.dedup_reason,
                            "similarity": store_result.similarity,
                            "conflicts": [
                                {
                                    "other_node_id": conflict.other_node_id,
                                    "other_node_label": conflict.other_node_label,
                                    "relationship": conflict.relationship.value,
                                    "reason": conflict.reason,
                                }
                                for conflict in store_result.conflicts
                            ],
                        },
                    )
                elif name == "store_edge":
                    edge = graph.add_edge(
                        source_id=arguments["source_id"],
                        target_id=arguments["target_id"],
                        relationship=RelationType(arguments["relationship"]),
                        weight=float(arguments.get("weight", 1.0)),
                    )
                    result = self._tool_result(
                        f"Created edge {edge.id} linking {edge.source_id} to {edge.target_id} as {edge.relationship.value}.",
                        self._edge_payload(edge),
                    )
                elif name == "query_graph":
                    subgraph = graph.query(
                        query=arguments["query"],
                        max_nodes=int(arguments.get("max_nodes", 20)),
                        max_depth=int(arguments.get("max_depth", 2)),
                    )
                    result = self._tool_result(
                        serialize_subgraph(subgraph),
                        self._subgraph_payload(subgraph),
                    )
                elif name == "get_related":
                    subgraph = graph.get_related(node_id=arguments["node_id"], max_depth=int(arguments.get("max_depth", 2)))
                    result = self._tool_result(serialize_subgraph(subgraph), self._subgraph_payload(subgraph))
                elif name == "update_node":
                    node = graph.update_node(
                        node_id=arguments["node_id"],
                        content=arguments.get("content"),
                        label=arguments.get("label"),
                        tags=arguments.get("tags"),
                    )
                    result = self._tool_result(f"Updated node '{node.label}' ({node.id}).", self._node_payload(node))
                elif name == "delete_node":
                    node = graph.delete_node(node_id=arguments["node_id"])
                    result = self._tool_result(
                        f"Deleted node '{node.label}' ({node.id}) and its connected edges.",
                        {"id": node.id, "label": node.label, "tenant_id": node.tenant_id},
                    )
                elif name == "decompose_and_store":
                    subgraph = graph.decompose_and_store(content=arguments["content"], context=arguments.get("context", ""))
                    result = self._tool_result(serialize_subgraph(subgraph), self._subgraph_payload(subgraph))
                elif name == "observe_conversation":
                    observation = graph.observe_conversation(
                        user_message=arguments["user_message"],
                        assistant_response=arguments["assistant_response"],
                    )
                    result = self._tool_result(
                        serialize_observation_result(observation),
                        self._observation_payload(observation),
                    )
                elif name == "graph_diff":
                    diff = graph.graph_diff(since=arguments.get("since", "24h"))
                    result = self._tool_result(serialize_graph_diff(diff), self._graph_diff_payload(diff))
                elif name == "prime_context":
                    context_result = graph.prime_context(project=arguments.get("project", ""))
                    result = self._tool_result(serialize_prime_context(context_result), self._prime_context_payload(context_result))
                elif name == "get_topics":
                    topics = graph.get_topics()
                    result = self._tool_result(serialize_topics(topics), self._topic_payload(topics))
                elif name == "get_stats":
                    stats = graph.get_stats()
                    result = self._tool_result(serialize_stats(stats), self._stats_payload(stats))
                elif name == "export_graph_html":
                    output_path = graph.export_graph_html(
                        output_path=arguments.get("output_path"),
                        include_physics=bool(arguments.get("include_physics", True)),
                    )
                    stats = graph.get_stats()
                    result = self._tool_result(
                        f"Exported graph visualization to {output_path}.",
                        {
                            "output_path": str(output_path),
                            "tenant_id": graph.tenant_id,
                            "total_nodes": stats.total_nodes,
                            "total_edges": stats.total_edges,
                        },
                    )
                elif name == "export_graph_backup":
                    backup = graph.export_graph_backup(output_path=arguments.get("output_path"))
                    result = self._tool_result(
                        f"Exported graph backup to {backup.output_path}.",
                        {
                            "output_path": backup.output_path,
                            "tenant_id": backup.tenant_id,
                            "schema_version": backup.schema_version,
                            "node_count": backup.node_count,
                            "edge_count": backup.edge_count,
                        },
                    )
                elif name == "import_graph_backup":
                    imported = graph.import_graph_backup(input_path=arguments["input_path"])
                    result = self._tool_result(
                        f"Imported graph backup from {imported.input_path}.",
                        {
                            "input_path": imported.input_path,
                            "tenant_id": imported.tenant_id,
                            "schema_version": imported.schema_version,
                            "nodes_created": imported.nodes_created,
                            "nodes_updated": imported.nodes_updated,
                            "edges_created": imported.edges_created,
                            "edges_updated": imported.edges_updated,
                        },
                    )
                else:
                    raise ValidationFailure(f"Unknown tool: {name}")

                elapsed = time.perf_counter() - started
                self.metrics.increment(
                    "graph_memory_tool_requests_total",
                    tool=name,
                    status="success",
                    tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id),
                )
                self.metrics.observe("graph_memory_tool_latency_seconds", elapsed, tool=name)
                self._record_graph_size(graph)
                LOGGER.info("tool_call_completed")
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - started
                self.metrics.increment(
                    "graph_memory_tool_requests_total",
                    tool=name,
                    status="error",
                    tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id),
                )
                self.metrics.observe("graph_memory_tool_latency_seconds", elapsed, tool=name)
                if isinstance(exc, AuthenticationError):
                    self.metrics.increment("graph_memory_auth_failures_total", tenant_id=getattr(graph, "tenant_id", self.config.default_tenant_id))
                LOGGER.exception("tool_call_failed")
                return self._error_result(exc)

    def _tool_result(self, text: str, structured: dict[str, Any]) -> types.CallToolResult:
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], structuredContent=structured)

    def _error_result(self, exc: Exception) -> types.CallToolResult:
        if isinstance(exc, GraphMemoryError):
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error [{exc.code}]: {exc}")],
                structuredContent={"error": str(exc), "error_type": type(exc).__name__, "error_code": exc.code, "status_code": exc.status_code},
                isError=True,
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {exc}")],
            structuredContent={"error": str(exc), "error_type": type(exc).__name__},
            isError=True,
        )

    def _node_payload(self, node: Node) -> dict[str, Any]:
        return {
            "id": node.id,
            "tenant_id": node.tenant_id,
            "label": node.label,
            "content": node.content,
            "node_type": node.node_type.value,
            "tags": node.tags,
            "source_prompt": node.source_prompt,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "access_count": node.access_count,
        }

    def _edge_payload(self, edge: Any) -> dict[str, Any]:
        return {
            "id": edge.id,
            "tenant_id": getattr(edge, "tenant_id", ""),
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relationship": edge.relationship.value,
            "weight": edge.weight,
            "metadata": edge.metadata,
            "created_at": edge.created_at.isoformat(),
        }

    def _subgraph_payload(self, result: SubgraphResult) -> dict[str, Any]:
        return {
            "query": result.query,
            "total_nodes_in_graph": result.total_nodes_in_graph,
            "nodes": [self._node_payload(node) for node in result.nodes],
            "edges": [self._edge_payload(edge) for edge in result.edges],
        }

    def _observation_payload(self, result: ObservationResult) -> dict[str, Any]:
        return {
            "stored_nodes": [self._node_payload(node) for node in result.stored_nodes],
            "created_count": result.created_count,
            "reused_count": result.reused_count,
            "conflicts": [
                {
                    "other_node_id": conflict.other_node_id,
                    "other_node_label": conflict.other_node_label,
                    "relationship": conflict.relationship.value,
                    "reason": conflict.reason,
                }
                for conflict in result.conflicts
            ],
        }

    def _graph_diff_payload(self, result: GraphDiffResult) -> dict[str, Any]:
        return {
            "since": result.since,
            "generated_at": result.generated_at.isoformat(),
            "added_nodes": [self._node_payload(node) for node in result.added_nodes],
            "updated_nodes": [self._node_payload(node) for node in result.updated_nodes],
            "created_edges": [self._edge_payload(edge) for edge in result.created_edges],
            "contradiction_edges": [self._edge_payload(edge) for edge in result.contradiction_edges],
        }

    def _prime_context_payload(self, result: PrimeContextResult) -> dict[str, Any]:
        return {
            "project": result.project,
            "summary": result.summary,
            "total_nodes_in_graph": result.total_nodes_in_graph,
            "nodes": [self._node_payload(node) for node in result.nodes],
            "edges": [self._edge_payload(edge) for edge in result.edges],
        }

    def _topic_payload(self, result: TopicResult) -> dict[str, Any]:
        return {
            "total_clusters": result.total_clusters,
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "label": cluster.label,
                    "node_count": cluster.node_count,
                    "top_tags": cluster.top_tags,
                    "nodes": [self._node_payload(node) for node in cluster.nodes],
                }
                for cluster in result.clusters
            ],
        }

    def _stats_payload(self, stats: GraphStats) -> dict[str, Any]:
        return {
            "total_nodes": stats.total_nodes,
            "total_edges": stats.total_edges,
            "node_type_breakdown": stats.node_type_breakdown,
            "most_connected_nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "node_type": node.node_type.value,
                    "connection_count": node.connection_count,
                }
                for node in stats.most_connected_nodes
            ],
            "most_recent_nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "node_type": node.node_type.value,
                    "updated_at": node.updated_at.isoformat(),
                }
                for node in stats.most_recent_nodes
            ],
        }

    def _validate_tool_payload(self, name: str, arguments: dict[str, Any]) -> None:
        limit = self.config.max_payload_bytes
        if name == "store_node":
            self._assert_payload_size(arguments.get("label", ""), limit, "store_node.label")
            self._assert_payload_size(arguments.get("content", ""), limit, "store_node.content")
            self._assert_payload_size(arguments.get("source_prompt", ""), limit, "store_node.source_prompt")
            return
        if name == "decompose_and_store":
            self._assert_payload_size(arguments.get("content", ""), limit, "decompose_and_store.content")
            self._assert_payload_size(arguments.get("context", ""), limit, "decompose_and_store.context")
            return
        if name == "observe_conversation":
            self._assert_payload_size(arguments.get("user_message", ""), limit, "observe_conversation.user_message")
            self._assert_payload_size(arguments.get("assistant_response", ""), limit, "observe_conversation.assistant_response")

    @staticmethod
    def _assert_payload_size(value: Any, limit: int, field_name: str) -> None:
        if value is None:
            return
        size = len(str(value).encode("utf-8"))
        if size > limit:
            raise PayloadTooLargeError(f"{field_name} exceeds the configured payload limit.")

    def _record_graph_size(self, graph: Any) -> None:
        stats = graph.get_stats()
        tenant_id = getattr(graph, "tenant_id", self.config.default_tenant_id)
        self.metrics.set_gauge("graph_memory_graph_nodes", stats.total_nodes, tenant_id=tenant_id)
        self.metrics.set_gauge("graph_memory_graph_edges", stats.total_edges, tenant_id=tenant_id)


class MCPHttpApp:
    def __init__(self, app_server: GraphMemoryServer, config: AppConfig) -> None:
        self.app_server = app_server
        self.config = config
        self.metrics = app_server.metrics
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit_rpm,
            max_concurrent_requests=config.max_concurrent_requests,
            write_requests_per_minute=config.write_rate_limit_rpm,
        )
        self.transport: StreamableHTTPServerTransport | None = None
        self.ready = False
        self.draining = False

    @asynccontextmanager
    async def lifespan(self, app: Starlette):
        self.transport = StreamableHTTPServerTransport(mcp_session_id=None, is_json_response_enabled=False)
        async with self.transport.connect() as (read_stream, write_stream):
            async with anyio.create_task_group() as tg:
                tg.start_soon(
                    self.app_server.server.run,
                    read_stream,
                    write_stream,
                    self.app_server.initialization_options(),
                    False,
                    True,
                )
                self.app_server.validate_startup()
                self.ready = True
                self.metrics.set_gauge("graph_memory_ready", 1)
                app.state.http_service = self
                try:
                    yield
                finally:
                    self.draining = True
                    self.ready = False
                    self.metrics.set_gauge("graph_memory_ready", 0)
                    tg.cancel_scope.cancel()

    async def mcp_asgi(self, scope: Any, receive: Any, send: Any) -> None:
        started = time.perf_counter()
        method = scope["method"]
        headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        request_id = headers.get("x-request-id", str(uuid.uuid4()))
        status_holder = {"status": 500}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = int(message["status"])
            await send(message)

        try:
            if self.draining:
                raise ServiceUnavailableError("Server is draining.")
            body = b""
            receive_callable = receive
            if method == "POST":
                request = Request(scope, receive)
                body = await request.body()
                if len(body) > self.config.max_payload_bytes:
                    raise PayloadTooLargeError()
                receive_callable = self._replay_receive(body)
                headers = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}

            raw_api_key = headers.get("x-api-key", "")
            if not raw_api_key:
                raise AuthenticationError("Missing X-API-Key header.")
            principal = self.app_server._root_graph.authenticate_api_key(raw_api_key)
            scope.setdefault("state", {})
            scope["state"]["tenant_id"] = principal.tenant_id
            scope["state"]["api_key_id"] = principal.api_key_id
            scope["state"]["request_id"] = request_id

            tool_name = self._extract_tool_name(body)
            await self.rate_limiter.check_rate(principal.api_key_id, is_write=tool_name in WRITE_HEAVY_TOOLS)
            async with self.rate_limiter.concurrency_slot(principal.api_key_id):
                with runtime_context(
                    request_id=request_id,
                    tenant_id=principal.tenant_id,
                    transport="http",
                    backend=self.config.backend,
                    api_key_id=principal.api_key_id,
                    tool_name=tool_name,
                ):
                    with anyio.fail_after(self.config.request_timeout_seconds):
                        assert self.transport is not None
                        await self.transport.handle_request(scope, receive_callable, send_wrapper)
        except GraphMemoryError as exc:
            LOGGER.warning("http_request_failed", extra={"error_code": exc.code, "status_code": exc.status_code})
            if isinstance(exc, AuthenticationError):
                self.metrics.increment("graph_memory_auth_failures_total")
            if exc.code == "rate_limited":
                self.metrics.increment("graph_memory_rate_limit_rejections_total")
            await JSONResponse({"error": exc.code, "message": str(exc)}, status_code=exc.status_code)(scope, receive, send)
            status_holder["status"] = exc.status_code
        finally:
            elapsed = time.perf_counter() - started
            self.metrics.increment(
                "graph_memory_http_requests_total",
                path="/mcp",
                method=method,
                status=str(status_holder["status"]),
            )
            self.metrics.observe("graph_memory_http_request_latency_seconds", elapsed, path="/mcp", method=method)

    @staticmethod
    def _extract_tool_name(body: bytes) -> str:
        if not body:
            return ""
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return ""
        params = payload.get("params", {})
        if isinstance(params, dict):
            return str(params.get("name", ""))
        return ""

    @staticmethod
    def _replay_receive(body: bytes):
        delivered = False

        async def receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        return receive


def create_http_application(app_server: GraphMemoryServer, config: AppConfig) -> Starlette:
    service = MCPHttpApp(app_server, config)

    async def live(_: Request) -> Response:
        return JSONResponse({"status": "live"})

    async def ready(request: Request) -> Response:
        http_service: MCPHttpApp = request.app.state.http_service
        if not http_service.ready or http_service.draining:
            return JSONResponse({"status": "not-ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    async def metrics_endpoint(request: Request) -> Response:
        return PlainTextResponse(request.app.state.http_service.metrics.render_prometheus())

    app = Starlette(
        routes=[
            Route("/health/live", live),
            Route("/health/ready", ready),
            Route("/metrics", metrics_endpoint),
            Mount("/mcp", app=service.mcp_asgi),
        ],
        lifespan=service.lifespan,
    )
    return app


_APP: GraphMemoryServer | None = None


def _default_graph(config: AppConfig | None = None) -> Any:
    try:
        return _build_backend(config or AppConfig.from_env())
    except ValidationFailure as exc:
        raise RuntimeError(str(exc)) from exc


def get_app(config: AppConfig | None = None) -> GraphMemoryServer:
    global _APP
    if _APP is None:
        _APP = GraphMemoryServer(config=config or AppConfig.from_env())
    return _APP


async def run_stdio(config: AppConfig) -> None:
    app = get_app(config)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.server.run(read_stream, write_stream, app.initialization_options())


def run_http(config: AppConfig) -> None:
    app_server = get_app(config)
    http_app = create_http_application(app_server, config)
    uvicorn.run(http_app, host=config.http_host, port=config.http_port, log_level=config.log_level.lower())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graph-memory-mcp")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve")

    create_tenant = subparsers.add_parser("create-tenant")
    create_tenant.add_argument("--tenant-id", required=True)
    create_tenant.add_argument("--name", default="")

    create_api_key = subparsers.add_parser("create-api-key")
    create_api_key.add_argument("--tenant-id", required=True)
    create_api_key.add_argument("--name", default="")

    list_api_keys = subparsers.add_parser("list-api-keys")
    list_api_keys.add_argument("--tenant-id", required=True)

    revoke_api_key = subparsers.add_parser("revoke-api-key")
    revoke_api_key.add_argument("--api-key-id", required=True)

    migrate_sqlite = subparsers.add_parser("migrate-sqlite")
    migrate_sqlite.add_argument("--db-path", required=True)
    migrate_sqlite.add_argument("--tenant-id", required=True)

    subparsers.add_parser("init", help="Interactive setup wizard — configure an MCP client to use graph-memory-mcp.")
    return parser


def _run_admin_command(config: AppConfig, args: argparse.Namespace) -> int:
    backend = _build_backend(config)
    if args.command == "create-tenant":
        tenant = backend.ensure_tenant(args.tenant_id, args.name)
        print(json.dumps(tenant.model_dump(), indent=2))
        return 0
    if args.command == "create-api-key":
        created = backend.create_api_key(args.tenant_id, args.name)
        print(
            json.dumps(
                {
                    "api_key_id": created.record.api_key_id,
                    "tenant_id": created.record.tenant_id,
                    "name": created.record.name,
                    "status": created.record.status,
                    "raw_api_key": created.raw_api_key,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "list-api-keys":
        print(json.dumps([record.model_dump(mode="json") for record in backend.list_api_keys(args.tenant_id)], indent=2))
        return 0
    if args.command == "revoke-api-key":
        backend.revoke_api_key(args.api_key_id)
        print(json.dumps({"revoked": args.api_key_id}))
        return 0
    if args.command == "migrate-sqlite":
        if config.backend != "neo4j":
            raise ValidationFailure("migrate-sqlite requires GRAPH_MEMORY_BACKEND=neo4j for the target environment.")
        source = MemoryGraph(args.db_path, EmbeddingModel(config.model_name), tenant_id=args.tenant_id, export_dir=config.export_dir)
        target = backend.for_tenant(args.tenant_id)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            temp_path = Path(handle.name)
        backup = source.export_graph_backup(output_path=temp_path)
        imported = target.import_graph_backup(input_path=temp_path)
        print(
            json.dumps(
                {
                    "backup": backup.model_dump(),
                    "import": imported.model_dump(),
                },
                indent=2,
            )
        )
        temp_path.unlink(missing_ok=True)
        return 0
    raise ValidationFailure(f"Unknown command: {args.command}")


# ---------------------------------------------------------------------------
# init wizard
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"
_NO_COLOR = not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Apply ANSI colour code, or return plain text when not a tty."""
    return text if _NO_COLOR else f"{code}{text}{_RESET}"


def _prompt_choice(question: str, choices: list[str]) -> str:
    """Render an arrow-key style menu (keyboard fallback: number entry)."""
    print(f"\n{_c(_BOLD, question)}")
    for i, choice in enumerate(choices, 1):
        print(f"  {_c(_CYAN, str(i) + '.')} {choice}")
    while True:
        raw = input(f"  Enter number [1-{len(choices)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            selected = choices[int(raw) - 1]
            print(f"  {_c(_GREEN, '>')} {selected}")
            return selected
        print(f"  {_c(_RED, 'Invalid choice, try again.')}")


def _prompt_path(question: str, default: str) -> str:
    """Prompt for a file path, showing the default."""
    print(f"\n{_c(_BOLD, question)}")
    raw = input(f"  [{default}]: ").strip()
    result = raw or default
    print(f"  {_c(_GREEN, '>')} {result}")
    return result


def _ok(msg: str) -> None:
    print(f"  {_c(_GREEN, chr(0x2705))} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_c(_RED, chr(0x274C))} {msg}")


def _python_exe() -> str:
    """Return the Python executable used to run this process."""
    return sys.executable


def _package_root() -> str:
    """Return a best-effort src/ directory for PYTHONPATH."""
    # Walk up from this file looking for src/graph_memory
    here = Path(__file__).resolve().parent
    candidate = here.parent  # expected: .../src
    if (candidate / "graph_memory").is_dir():
        return str(candidate)
    return str(here.parent)


# ── client config writers ────────────────────────────────────────────────────

def _write_claude_desktop(db_path: str, python_exe: str, src_root: str) -> Path:
    """Write ~/.config/claude/claude_desktop_config.json (macOS/Linux)."""
    if sys.platform == "darwin":
        config_dir = Path.home() / "Library" / "Application Support" / "Claude"
    else:
        config_dir = Path.home() / ".config" / "claude"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "claude_desktop_config.json"

    existing: dict = {}
    if config_file.exists():
        try:
            existing = json.loads(config_file.read_text())
        except json.JSONDecodeError:
            pass

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["graph-memory"] = {
        "command": python_exe,
        "args": ["-m", "graph_memory.server"],
        "env": {
            "PYTHONPATH": src_root,
            "GRAPH_MEMORY_TRANSPORT": "stdio",
            "GRAPH_MEMORY_BACKEND": "sqlite",
            "GRAPH_MEMORY_DB_PATH": db_path,
            "GRAPH_MEMORY_DEFAULT_TENANT_ID": "local-default",
            "GRAPH_MEMORY_MODEL": "all-MiniLM-L6-v2",
        },
    }
    config_file.write_text(json.dumps(existing, indent=2))
    return config_file


def _write_cursor(db_path: str, python_exe: str, src_root: str) -> Path:
    """Write ~/.cursor/mcp.json."""
    config_dir = Path.home() / ".cursor"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "mcp.json"

    existing: dict = {}
    if config_file.exists():
        try:
            existing = json.loads(config_file.read_text())
        except json.JSONDecodeError:
            pass

    existing.setdefault("mcpServers", {})
    existing["mcpServers"]["graph-memory"] = {
        "command": python_exe,
        "args": ["-m", "graph_memory.server"],
        "env": {
            "PYTHONPATH": src_root,
            "GRAPH_MEMORY_TRANSPORT": "stdio",
            "GRAPH_MEMORY_BACKEND": "sqlite",
            "GRAPH_MEMORY_DB_PATH": db_path,
            "GRAPH_MEMORY_DEFAULT_TENANT_ID": "local-default",
            "GRAPH_MEMORY_MODEL": "all-MiniLM-L6-v2",
        },
    }
    config_file.write_text(json.dumps(existing, indent=2))
    return config_file


def _write_codex(db_path: str, python_exe: str, src_root: str) -> Path:
    """Write ~/codex_mcp.toml (printable TOML snippet — Codex uses a TOML file)."""
    config_file = Path.home() / "codex_mcp.toml"
    toml_block = (
        '[mcp_servers.graph-memory]\n'
        f'command = "{python_exe}"\n'
        'args = ["-m", "graph_memory.server"]\n'
        f'cwd = "{Path(src_root).parent}"\n'
        'env = {\n'
        f'  PYTHONPATH = "{src_root}",\n'
        '  GRAPH_MEMORY_TRANSPORT = "stdio",\n'
        '  GRAPH_MEMORY_BACKEND = "sqlite",\n'
        f'  GRAPH_MEMORY_DB_PATH = "{db_path}",\n'
        '  GRAPH_MEMORY_DEFAULT_TENANT_ID = "local-default",\n'
        '  GRAPH_MEMORY_MODEL = "all-MiniLM-L6-v2"\n'
        '}\n'
    )
    config_file.write_text(toml_block)
    return config_file


def _write_other(db_path: str, python_exe: str, src_root: str) -> Path:
    """Write a generic JSON snippet to ~/graph-memory-mcp-config.json."""
    config_file = Path.home() / "graph-memory-mcp-config.json"
    snippet = {
        "command": python_exe,
        "args": ["-m", "graph_memory.server"],
        "env": {
            "PYTHONPATH": src_root,
            "GRAPH_MEMORY_TRANSPORT": "stdio",
            "GRAPH_MEMORY_BACKEND": "sqlite",
            "GRAPH_MEMORY_DB_PATH": db_path,
            "GRAPH_MEMORY_DEFAULT_TENANT_ID": "local-default",
            "GRAPH_MEMORY_MODEL": "all-MiniLM-L6-v2",
        },
    }
    config_file.write_text(json.dumps(snippet, indent=2))
    return config_file


_CLIENT_WRITERS = {
    "Claude Desktop": _write_claude_desktop,
    "Cursor": _write_cursor,
    "Codex": _write_codex,
    "Other": _write_other,
}

_RESTART_HINTS = {
    "Claude Desktop": "Restart Claude Desktop to activate.",
    "Cursor": "Reload the Cursor window (Cmd/Ctrl+Shift+P → 'Reload Window') to activate.",
    "Codex": "Copy the TOML block into your Codex config file, then restart Codex.",
    "Other": "Add the JSON config to your MCP client's server list, then restart it.",
}


def _run_init() -> int:
    """Interactive setup wizard for graph-memory-mcp."""
    print()
    print(_c(_BOLD, "graph-memory-mcp setup wizard"))
    print(_c(_CYAN, "─" * 40))

    clients = list(_CLIENT_WRITERS.keys())
    client = _prompt_choice("Which MCP client are you using?", clients)

    default_db = str(Path.home() / ".graph-memory" / "memory.db")
    db_path_raw = _prompt_path("Where should the database be stored?", default_db)
    db_path = str(Path(db_path_raw).expanduser().resolve())

    python_exe = _python_exe()
    src_root = _package_root()

    print()

    # Write client config
    writer = _CLIENT_WRITERS[client]
    try:
        config_file = writer(db_path, python_exe, src_root)
        _ok(f"Config written to {config_file}")
    except OSError as exc:
        _fail(f"Could not write config: {exc}")
        return 1

    # Create database directory
    db_dir = Path(db_path).parent
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
        _ok(f"Database directory ready at {db_dir}")
    except OSError as exc:
        _fail(f"Could not create database directory: {exc}")
        return 1

    # Restart hint
    print(f"  {_c(_CYAN, chr(0x27A1))}  {_RESTART_HINTS[client]}")
    print()
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    command = args.command or "serve"

    # init runs before AppConfig so it works with no env vars set
    if command == "init":
        sys.exit(_run_init())

    config = AppConfig.from_env()
    configure_logging(config.log_level)
    LOGGER.info("graph_memory_startup")
    if command != "serve":
        _run_admin_command(config, args)
        return
    if config.transport == "http":
        run_http(config)
        return
    if config.backend == "sqlite":
        Path(config.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_stdio(config))


if __name__ == "__main__":
    main()
