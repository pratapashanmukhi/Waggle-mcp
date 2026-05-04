"""
benchmarks/rlm_style_waggle_eval.py
=====================================
RLM-style evaluation suite for Waggle's Recursive Context Assembly.

This benchmark follows the five benchmark families from the Recursive Language
Models paper (Zhang et al., 2026 — https://arxiv.org/abs/2512.24601):

  1. S-NIAH-style          — single needle-in-a-haystack, O(1) information need
  2. BrowseComp-Plus-style — multi-hop memory QA, constant hops but cross-node
  3. OOLONG-style          — linear aggregation over N entries, O(n)
  4. OOLONG-Pairs-style    — pairwise conflict/compatibility reasoning, O(n²)
  5. CodeQA-style          — repo/codebase understanding from structured memory

The initial implementation uses deterministic synthetic Waggle memory tasks.
It does NOT reproduce the RLM paper's exact results and should not be compared
numerically to the paper until the exact public datasets and matching model
setup are run.

WARNING
-------
This benchmark follows the benchmark families used in the RLM paper, but the
initial Waggle evaluation uses deterministic synthetic memory tasks mapped to
Waggle's graph/transcript environment. It should not be compared numerically
to the RLM paper until the exact public datasets and matching model setup
are run.

TODO hooks for real datasets
-----------------------------
  load_real_sniah()           — plug in RULER S-NIAH
  load_real_browsecomp_plus() — plug in BrowseComp-Plus
  load_real_oolong()          — plug in OOLONG
  load_real_oolong_pairs()    — plug in OOLONG-Pairs
  load_real_codeqa()          — plug in LongBench-v2 CodeQA

Usage
-----
  python benchmarks/rlm_style_waggle_eval.py \\
    --db /tmp/waggle_rlm_eval.db \\
    --scales 128 512 2048 \\
    --methods raw_context query_graph prime_context build_context \\
    --token-budget 1200 \\
    --output benchmark_results/
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap — works both from repo root and as installed package
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from waggle.graph import MemoryGraph
from waggle.intelligence import tokenize_text
from waggle.models import NodeType, RelationType
from waggle.recursive_context import RecursiveContextController
from waggle.retrieval.hybrid import SimpleBM25

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic fake embedding model (no ML dependency)
# ---------------------------------------------------------------------------


class _DeterministicEmbedding:
    """Deterministic embedding model for benchmark reproducibility."""

    model_name = "deterministic-bench"
    model_id = "deterministic-bench:v1"

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(16, dtype=np.float32)
        for token in text.lower().split():
            idx = sum(ord(c) for c in token) % len(vec)
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def to_bytes(self, emb: np.ndarray) -> bytes:
        return emb.astype(np.float32).tobytes()

    def from_bytes(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        an, bn = np.linalg.norm(a), np.linalg.norm(b)
        if an == 0 or bn == 0:
            return 0.0
        return float(np.dot(a, b) / (an * bn))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    benchmark_family: str
    scale_n: int
    method: str
    score: float = 0.0
    exact_match: float = 0.0
    f1: float = 0.0
    evidence_coverage: float = 0.0
    tokens_returned: int = 0
    latency_ms: float = 0.0
    context_pack_tokens: int = 0
    notes: str = ""
    seed: int = 42
    token_budget: int = 0
    ablation_variant: str = ""
    delta_vs_full: float = 0.0
    annotation: str = ""
    mean_score: float = 0.0
    std_score: float = 0.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def exact_match(pred: str, gold: str) -> float:
    """1.0 if gold appears verbatim (case-insensitive) in pred, else 0.0."""
    return 1.0 if gold.strip().lower() in pred.strip().lower() else 0.0


def contains_answer(pred: str, gold: str) -> float:
    """Alias for exact_match — checks containment."""
    return exact_match(pred, gold)


def token_estimate(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 characters."""
    return len(text) // 4


def set_f1(pred_items: list[str], gold_items: list[str]) -> float:
    """F1 over sets of string items (case-insensitive)."""
    pred_set = {s.strip().lower() for s in pred_items}
    gold_set = {s.strip().lower() for s in gold_items}
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evidence_coverage(returned_ids: list[str], gold_ids: list[str]) -> float:
    """Fraction of gold evidence IDs present in returned_ids."""
    if not gold_ids:
        return 1.0
    found = sum(1 for gid in gold_ids if gid in returned_ids)
    return found / len(gold_ids)


def pairwise_f1(pred_pairs: list[tuple[str, str]], gold_pairs: list[tuple[str, str]]) -> float:
    """F1 over sets of (a, b) pairs (order-normalised)."""
    def norm(pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
        return {(min(a, b), max(a, b)) for a, b in pairs}
    pred_set = norm(pred_pairs)
    gold_set = norm(gold_pairs)
    if not gold_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def _make_graph(db_path: str) -> MemoryGraph:
    return MemoryGraph(db_path, _DeterministicEmbedding())


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def _run_raw_context(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """
    raw_context_baseline: dump all available memory text until budget.
    No recursive decomposition, no graph-aware handling.
    """
    t0 = time.perf_counter()
    try:
        stats = graph.get_stats()
        # Pull a broad set of nodes
        result = graph.aggregate(
            query=query,
            max_nodes=min(stats.total_nodes, 200),
            max_depth=0,
        )
        lines = []
        used = 0
        budget = int(token_budget * 1.15)
        for node in result.nodes:
            line = f"[{node.node_type.value}] {node.label}: {node.content}"
            cost = token_estimate(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        pack = "\n".join(lines)
    except Exception as exc:
        LOGGER.debug("raw_context failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


def _run_query_graph(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """
    query_graph_baseline: single direct query_graph call.
    No recursive subqueries, no explicit multi-step assembly.
    """
    t0 = time.perf_counter()
    try:
        result = graph.query(
            query=query,
            max_nodes=20,
            max_depth=2,
            retrieval_mode="hybrid",
        )
        lines = []
        used = 0
        budget = int(token_budget * 1.15)
        for node in result.nodes:
            line = f"[{node.node_type.value}] {node.label}: {node.content}"
            cost = token_estimate(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        pack = "\n".join(lines)
    except Exception as exc:
        LOGGER.debug("query_graph failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


def _run_prime_context(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """prime_context_baseline: use prime_context only."""
    t0 = time.perf_counter()
    try:
        result = graph.prime_context()
        pack = result.summary or ""
    except Exception as exc:
        LOGGER.debug("prime_context failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


def _run_build_context(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """build_context_recursive: use RecursiveContextController."""
    t0 = time.perf_counter()
    try:
        controller = RecursiveContextController(graph=graph)
        result = controller.build_context(
            query=query,
            token_budget=token_budget,
            depth=2,
            max_subqueries=6,
            mode="balanced",
        )
        pack = result.context_pack
    except Exception as exc:
        LOGGER.debug("build_context failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


def _run_hybrid_baseline(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """hybrid_baseline: single hybrid retrieval call only."""
    t0 = time.perf_counter()
    try:
        result = graph.query(
            query=query,
            max_nodes=20,
            max_depth=1,
            retrieval_mode="hybrid",
        )
        lines = []
        used = 0
        budget = int(token_budget * 1.15)
        for node in result.nodes:
            line = f"[{node.node_type.value}] {node.label}: {node.content}"
            cost = token_estimate(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        pack = "\n".join(lines)
    except Exception as exc:
        LOGGER.debug("hybrid_baseline failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


def _run_bm25_topk(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """bm25_topk: BM25 ranking over all nodes, concatenate top-k until budget."""
    t0 = time.perf_counter()
    try:
        result = graph.aggregate(query=query, max_nodes=500, max_depth=0)
        nodes = result.nodes
        documents = {
            node.id: list(tokenize_text(node.label + " " + node.content))
            for node in nodes
        }
        bm25 = SimpleBM25(documents)
        scores = bm25.score(query)
        sorted_nodes = sorted(nodes, key=lambda n: scores.get(n.id, 0.0), reverse=True)
        lines = []
        used = 0
        budget = int(token_budget * 1.15)
        for node in sorted_nodes:
            line = f"[{node.node_type.value}] {node.label}: {node.content}"
            cost = token_estimate(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        pack = "\n".join(lines)
    except Exception as exc:
        LOGGER.debug("bm25_topk failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


def _run_vector_topk(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """vector_topk: cosine similarity ranking over all nodes, concatenate top-k until budget."""
    t0 = time.perf_counter()
    try:
        emb = _DeterministicEmbedding()
        q_vec = emb.embed(query)
        result = graph.aggregate(query=query, max_nodes=500, max_depth=0)
        nodes = result.nodes
        scored: list[tuple[float, Any]] = []
        for node in nodes:
            node_text = node.label + ": " + node.content
            node_vec = emb.embed(node_text)
            sim = emb.cosine_similarity(q_vec, node_vec)
            scored.append((sim, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        lines = []
        used = 0
        budget = int(token_budget * 1.15)
        for _, node in scored:
            line = f"[{node.node_type.value}] {node.label}: {node.content}"
            cost = token_estimate(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        pack = "\n".join(lines)
    except Exception as exc:
        LOGGER.debug("vector_topk failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


def _run_hybrid_rrf(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """hybrid_rrf: fuse BM25 and vector rankings with Reciprocal Rank Fusion."""
    t0 = time.perf_counter()
    try:
        result = graph.aggregate(query=query, max_nodes=500, max_depth=0)
        nodes = result.nodes
        if not nodes:
            return "", (time.perf_counter() - t0) * 1000

        # BM25 ranking
        documents = {
            node.id: list(tokenize_text(node.label + " " + node.content))
            for node in nodes
        }
        bm25 = SimpleBM25(documents)
        bm25_scores = bm25.score(query)
        bm25_sorted = sorted(nodes, key=lambda n: bm25_scores.get(n.id, 0.0), reverse=True)
        bm25_rank = {node.id: (i + 1) for i, node in enumerate(bm25_sorted)}

        # Vector ranking
        emb = _DeterministicEmbedding()
        q_vec = emb.embed(query)
        vec_scored: list[tuple[float, Any]] = []
        for node in nodes:
            node_text = node.label + ": " + node.content
            node_vec = emb.embed(node_text)
            sim = emb.cosine_similarity(q_vec, node_vec)
            vec_scored.append((sim, node))
        vec_scored.sort(key=lambda x: x[0], reverse=True)
        vec_rank = {node.id: (i + 1) for i, (_, node) in enumerate(vec_scored)}

        # RRF fusion
        n = len(nodes)
        rrf_scores: dict[str, float] = {}
        for node in nodes:
            br = bm25_rank.get(node.id, n + 1)
            vr = vec_rank.get(node.id, n + 1)
            rrf_scores[node.id] = 1.0 / (60 + br) + 1.0 / (60 + vr)

        sorted_nodes = sorted(nodes, key=lambda n: rrf_scores.get(n.id, 0.0), reverse=True)
        lines = []
        used = 0
        budget = int(token_budget * 1.15)
        for node in sorted_nodes:
            line = f"[{node.node_type.value}] {node.label}: {node.content}"
            cost = token_estimate(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        pack = "\n".join(lines)
    except Exception as exc:
        LOGGER.debug("hybrid_rrf failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


def _run_graph_expansion_no_recursion(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """graph_expansion_no_recursion: single query + one-hop neighbour expansion, no subquery decomposition."""
    t0 = time.perf_counter()
    try:
        result = graph.query(query=query, max_nodes=20, max_depth=1, retrieval_mode="hybrid")
        all_nodes: dict[str, Any] = {node.id: node for node in result.nodes}
        for node in list(result.nodes):
            try:
                related = graph.get_related(node_id=node.id, max_depth=1)
                for rnode in related.nodes:
                    if rnode.id not in all_nodes:
                        all_nodes[rnode.id] = rnode
            except Exception as exc:
                LOGGER.debug("graph_expansion_no_recursion get_related failed for %s: %s", node.id, exc)
        lines = []
        used = 0
        budget = int(token_budget * 1.15)
        for node in all_nodes.values():
            line = f"[{node.node_type.value}] {node.label}: {node.content}"
            cost = token_estimate(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        pack = "\n".join(lines)
    except Exception as exc:
        LOGGER.debug("graph_expansion_no_recursion failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


def _run_summary_memory(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """summary_memory: use prime_context summary, truncated to budget."""
    t0 = time.perf_counter()
    try:
        result = graph.prime_context()
        summary = result.summary or ""
        pack = summary[: int(token_budget * 1.15)]
    except Exception as exc:
        LOGGER.debug("summary_memory failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


def _run_full_transcript_truncation(graph: MemoryGraph, query: str, token_budget: int) -> tuple[str, float]:
    """full_transcript_truncation: load transcript records in reverse-chronological order, concatenate until budget."""
    t0 = time.perf_counter()
    try:
        lines = []
        used = 0
        budget = int(token_budget * 1.15)
        with graph._lock, graph._connect() as conn:
            rows = conn.execute(
                "SELECT role, transcript_text FROM transcript_records "
                "WHERE tenant_id = ? ORDER BY observed_at DESC LIMIT 200",
                (graph.tenant_id,),
            ).fetchall()
        for row in rows:
            line = f"{row['role']}: {row['transcript_text']}"
            cost = token_estimate(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        pack = "\n".join(lines)
    except Exception as exc:
        LOGGER.debug("full_transcript_truncation failed: %s", exc)
        pack = ""
    latency = (time.perf_counter() - t0) * 1000
    return pack, latency


_METHOD_RUNNERS = {
    "raw_context": _run_raw_context,
    "query_graph": _run_query_graph,
    "prime_context": _run_prime_context,
    "build_context": _run_build_context,
    "hybrid_baseline": _run_hybrid_baseline,
    "bm25_topk": _run_bm25_topk,
    "vector_topk": _run_vector_topk,
    "hybrid_rrf": _run_hybrid_rrf,
    "graph_expansion_no_recursion": _run_graph_expansion_no_recursion,
    "summary_memory": _run_summary_memory,
    "full_transcript_truncation": _run_full_transcript_truncation,
}


# ---------------------------------------------------------------------------
# 1. S-NIAH-style: single needle-in-a-haystack
# ---------------------------------------------------------------------------


@dataclass
class SNIAHCase:
    """One S-NIAH benchmark case."""
    case_id: str
    question: str
    gold_answer: str
    gold_node_id: str
    scale_n: int


def generate_sniah_cases(
    graph: MemoryGraph,
    scale_n: int,
    rng: random.Random,
    project: str = "sniah",
) -> list[SNIAHCase]:
    """
    Insert one target fact (the needle) among scale_n distractor nodes.
    The needle is a specific policy fact with a unique numeric value.

    TODO: load_real_sniah() — plug in RULER S-NIAH dataset here.
    """
    needle_value = rng.randint(10, 99)
    needle_label = f"Deployment password rotation policy"
    needle_content = (
        f"The deployment password rotation policy is every {needle_value} days."
    )

    # Insert distractors first
    distractor_topics = [
        ("Cache TTL policy", "The cache TTL is set to {v} seconds."),
        ("Session timeout policy", "User sessions expire after {v} minutes."),
        ("Backup retention policy", "Backups are retained for {v} weeks."),
        ("Rate limit threshold", "API rate limit is {v} requests per minute."),
        ("Log rotation policy", "Logs are rotated every {v} hours."),
        ("Token expiry policy", "Auth tokens expire after {v} hours."),
        ("Retry backoff policy", "Retry backoff starts at {v} milliseconds."),
        ("Health check interval", "Health checks run every {v} seconds."),
    ]
    for i in range(scale_n - 1):
        topic_label, topic_tmpl = distractor_topics[i % len(distractor_topics)]
        v = rng.randint(1, 999)
        graph.add_node(
            label=f"{topic_label} #{i}",
            content=topic_tmpl.format(v=v),
            node_type=NodeType.FACT,
            project=project,
            tags=["policy", "distractor"],
        )

    # Insert needle at a random position (already done by random insertion order)
    needle_result = graph.add_node(
        label=needle_label,
        content=needle_content,
        node_type=NodeType.FACT,
        project=project,
        tags=["policy", "needle"],
    )

    return [SNIAHCase(
        case_id=f"sniah-{scale_n}-{needle_value}",
        question="What is the deployment password rotation policy?",
        gold_answer=f"{needle_value} days",
        gold_node_id=needle_result.node.id,
        scale_n=scale_n,
    )]


def run_sniah_benchmark(
    db_path: str,
    scale_n: int,
    methods: list[str],
    token_budget: int,
    rng: random.Random,
    include_latency: bool = True,
    verbose: bool = False,
) -> list[BenchResult]:
    """Run S-NIAH-style benchmark at a given scale."""
    graph = _make_graph(db_path)
    cases = generate_sniah_cases(graph, scale_n=scale_n, rng=rng)
    results = []

    for case in cases:
        if verbose:
            print(f"  [S-NIAH] scale={scale_n} question={case.question!r}")

        for method in methods:
            runner = _METHOD_RUNNERS.get(method)
            if runner is None:
                continue

            pack, latency = runner(graph, case.question, token_budget)
            em = exact_match(pack, case.gold_answer)
            ca = contains_answer(pack, case.gold_answer)

            # Evidence coverage: did the returned text include the needle node?
            # We check by label/content presence since we don't have node IDs in pack
            ev_cov = 1.0 if case.gold_answer.lower() in pack.lower() else 0.0

            results.append(BenchResult(
                benchmark_family="S-NIAH-style",
                scale_n=scale_n,
                method=method,
                score=ca,
                exact_match=em,
                f1=ca,
                evidence_coverage=ev_cov,
                tokens_returned=token_estimate(pack),
                latency_ms=round(latency, 1) if include_latency else 0.0,
                context_pack_tokens=token_estimate(pack),
                notes=f"needle_value={case.gold_answer}",
            ))

            if verbose:
                print(f"    {method}: score={ca:.2f} tokens={token_estimate(pack)} latency={latency:.0f}ms")

    return results


# ---------------------------------------------------------------------------
# 2. BrowseComp-Plus-style: multi-hop memory QA
# ---------------------------------------------------------------------------


@dataclass
class MultiHopCase:
    case_id: str
    question: str
    gold_answer: str
    gold_evidence_node_ids: list[str]
    scale_n: int


def generate_multihop_cases(
    graph: MemoryGraph,
    scale_n: int,
    rng: random.Random,
    project: str = "multihop",
) -> list[MultiHopCase]:
    """
    Create a 3-hop evidence chain:
      Project → API Gateway → Team → On-call schedule

    Add hard negatives: other projects, gateways, teams, schedules.

    TODO: load_real_browsecomp_plus() — plug in BrowseComp-Plus dataset here.
    """
    # Gold chain
    project_name = "Project Hermes"
    gateway_name = "API Gateway X"
    team_name = "Team Delta"
    schedule_id = f"PD-{rng.randint(10, 99)}"

    r_a = graph.add_node(
        label=f"{project_name} gateway",
        content=f"{project_name} uses {gateway_name}.",
        node_type=NodeType.FACT,
        project=project,
        tags=["project", "gateway"],
    )
    r_b = graph.add_node(
        label=f"{gateway_name} ownership",
        content=f"{gateway_name} is owned by {team_name}.",
        node_type=NodeType.FACT,
        project=project,
        tags=["gateway", "team"],
    )
    r_c = graph.add_node(
        label=f"{team_name} on-call",
        content=f"{team_name}'s on-call escalation is PagerDuty schedule {schedule_id}.",
        node_type=NodeType.FACT,
        project=project,
        tags=["team", "oncall"],
    )

    # Link the chain
    graph.add_edge(source_id=r_a.node.id, target_id=r_b.node.id, relationship=RelationType.RELATES_TO.value)
    graph.add_edge(source_id=r_b.node.id, target_id=r_c.node.id, relationship=RelationType.RELATES_TO.value)

    # Hard negatives: other projects/gateways/teams
    other_gateways = ["API Gateway Y", "API Gateway Z", "API Gateway Alpha"]
    other_teams = ["Team Sigma", "Team Omega", "Team Kappa"]
    other_schedules = [f"PD-{rng.randint(100, 999)}" for _ in range(3)]

    for i in range(min(scale_n - 3, 30)):
        gw = other_gateways[i % len(other_gateways)]
        tm = other_teams[i % len(other_teams)]
        sc = other_schedules[i % len(other_schedules)]
        proj_neg = f"Project Neg{i}"
        graph.add_node(
            label=f"{proj_neg} gateway",
            content=f"{proj_neg} uses {gw}.",
            node_type=NodeType.FACT,
            project=project,
            tags=["project", "distractor"],
        )
        graph.add_node(
            label=f"{gw} ownership",
            content=f"{gw} is owned by {tm}.",
            node_type=NodeType.FACT,
            project=project,
            tags=["gateway", "distractor"],
        )
        graph.add_node(
            label=f"{tm} on-call",
            content=f"{tm}'s on-call escalation is PagerDuty schedule {sc}.",
            node_type=NodeType.FACT,
            project=project,
            tags=["team", "distractor"],
        )

    return [MultiHopCase(
        case_id=f"multihop-{scale_n}",
        question=f"What is the on-call escalation for {project_name}?",
        gold_answer=f"PagerDuty schedule {schedule_id}",
        gold_evidence_node_ids=[r_a.node.id, r_b.node.id, r_c.node.id],
        scale_n=scale_n,
    )]


def run_multihop_benchmark(
    db_path: str,
    scale_n: int,
    methods: list[str],
    token_budget: int,
    rng: random.Random,
    include_latency: bool = True,
    verbose: bool = False,
) -> list[BenchResult]:
    """Run BrowseComp-Plus-style multi-hop benchmark."""
    graph = _make_graph(db_path)
    cases = generate_multihop_cases(graph, scale_n=scale_n, rng=rng)
    results = []

    for case in cases:
        if verbose:
            print(f"  [MultiHop] scale={scale_n} question={case.question!r}")

        for method in methods:
            runner = _METHOD_RUNNERS.get(method)
            if runner is None:
                continue

            pack, latency = runner(graph, case.question, token_budget)
            em = exact_match(pack, case.gold_answer)

            # Evidence coverage: how many of the 3 chain nodes appear in the pack
            ev_cov = evidence_coverage(
                returned_ids=[],  # we check by content presence
                gold_ids=case.gold_evidence_node_ids,
            )
            # Override: check content presence for each hop
            hop_hits = sum(
                1 for phrase in [
                    "project hermes",
                    "api gateway x",
                    "team delta",
                    case.gold_answer.lower(),
                ]
                if phrase in pack.lower()
            )
            ev_cov = hop_hits / 4.0

            results.append(BenchResult(
                benchmark_family="BrowseComp-Plus-style",
                scale_n=scale_n,
                method=method,
                score=em,
                exact_match=em,
                f1=ev_cov,
                evidence_coverage=ev_cov,
                tokens_returned=token_estimate(pack),
                latency_ms=round(latency, 1) if include_latency else 0.0,
                context_pack_tokens=token_estimate(pack),
                notes=f"gold={case.gold_answer}",
            ))

            if verbose:
                print(f"    {method}: score={em:.2f} ev_cov={ev_cov:.2f} tokens={token_estimate(pack)}")

    return results


# ---------------------------------------------------------------------------
# 3. OOLONG-style: linear aggregation over N entries
# ---------------------------------------------------------------------------


@dataclass
class LinearAggCase:
    case_id: str
    question: str
    gold_count: int
    gold_ids: list[str]
    scale_n: int


def generate_linear_agg_cases(
    graph: MemoryGraph,
    scale_n: int,
    rng: random.Random,
    project: str = "linear_agg",
) -> list[LinearAggCase]:
    """
    Insert N task-status nodes. Some are 'blocked'.
    Question: how many tasks are blocked, and list their IDs.

    TODO: load_real_oolong() — plug in OOLONG dataset here.
    """
    statuses = ["done", "blocked", "pending", "in_progress", "cancelled"]
    blocked_ids: list[str] = []

    for i in range(scale_n):
        status = statuses[rng.randint(0, len(statuses) - 1)]
        task_id = f"T{i:04d}"
        # Include unique index in content to avoid dedup collisions
        r = graph.add_node(
            label=f"Task {task_id} status",
            content=f"Task {task_id} status is {status}. Index={i}.",
            node_type=NodeType.FACT,
            project=project,
            tags=["task", status],
        )
        if status == "blocked":
            blocked_ids.append(task_id)

    # Ensure at least 3 blocked tasks for a meaningful test
    while len(blocked_ids) < 3:
        i = scale_n + len(blocked_ids)
        task_id = f"T{i:04d}"
        graph.add_node(
            label=f"Task {task_id} status",
            content=f"Task {task_id} status is blocked. Index={i}.",
            node_type=NodeType.FACT,
            project=project,
            tags=["task", "blocked"],
        )
        blocked_ids.append(task_id)

    return [LinearAggCase(
        case_id=f"linear-agg-{scale_n}",
        question="How many tasks are blocked, and list their IDs.",
        gold_count=len(blocked_ids),
        gold_ids=blocked_ids,
        scale_n=scale_n,
    )]


def run_linear_agg_benchmark(
    db_path: str,
    scale_n: int,
    methods: list[str],
    token_budget: int,
    rng: random.Random,
    include_latency: bool = True,
    verbose: bool = False,
) -> list[BenchResult]:
    """Run OOLONG-style linear aggregation benchmark."""
    graph = _make_graph(db_path)
    cases = generate_linear_agg_cases(graph, scale_n=scale_n, rng=rng)
    results = []

    for case in cases:
        if verbose:
            print(f"  [LinearAgg] scale={scale_n} gold_count={case.gold_count}")

        for method in methods:
            runner = _METHOD_RUNNERS.get(method)
            if runner is None:
                continue

            pack, latency = runner(graph, case.question, token_budget)

            # Count how many gold task IDs appear in the returned pack
            found_ids = [tid for tid in case.gold_ids if tid.lower() in pack.lower()]
            f1 = set_f1(found_ids, case.gold_ids)

            # Coverage: fraction of blocked tasks surfaced
            cov = len(found_ids) / max(len(case.gold_ids), 1)

            # Numeric accuracy: did the pack mention the correct count?
            count_str = str(case.gold_count)
            numeric_acc = 1.0 if count_str in pack else 0.0

            results.append(BenchResult(
                benchmark_family="OOLONG-style",
                scale_n=scale_n,
                method=method,
                score=f1,
                exact_match=numeric_acc,
                f1=f1,
                evidence_coverage=cov,
                tokens_returned=token_estimate(pack),
                latency_ms=round(latency, 1) if include_latency else 0.0,
                context_pack_tokens=token_estimate(pack),
                notes=f"gold_count={case.gold_count} found={len(found_ids)}",
            ))

            if verbose:
                print(f"    {method}: f1={f1:.2f} cov={cov:.2f} tokens={token_estimate(pack)}")

    return results


# ---------------------------------------------------------------------------
# 4. OOLONG-Pairs-style: pairwise conflict/compatibility reasoning
# ---------------------------------------------------------------------------


@dataclass
class PairwiseCase:
    case_id: str
    question: str
    gold_conflict_pairs: list[tuple[str, str]]   # (choice_label, constraint_label)
    all_choice_labels: list[str]
    all_constraint_labels: list[str]
    scale_n: int


def generate_pairwise_cases(
    graph: MemoryGraph,
    scale_n: int,
    rng: random.Random,
    project: str = "pairwise",
) -> list[PairwiseCase]:
    """
    Create constraint nodes and implementation choice nodes.
    Some choices conflict with constraints (contradicts edge).
    Question: which choices conflict with active constraints?

    TODO: load_real_oolong_pairs() — plug in OOLONG-Pairs dataset here.
    """
    # Fixed gold constraints and choices
    constraints = [
        ("Must run locally", "The system must run fully locally with no cloud dependency."),
        ("No external SaaS", "No external SaaS services are permitted in production."),
        ("Offline capable", "The system must work without internet access."),
    ]
    choices = [
        ("Use SQLite backend", "Use SQLite for local storage.", False),       # compatible
        ("Use hosted Postgres", "Use hosted Postgres on RDS.", True),          # conflicts: cloud
        ("Use SaaS vector DB", "Use Pinecone as the vector database.", True),  # conflicts: SaaS
        ("Use local embeddings", "Use local sentence-transformers for embeddings.", False),  # compatible
        ("Use external LLM API", "Use OpenAI API for inference.", True),       # conflicts: cloud
        ("Use local Ollama", "Use Ollama for local LLM inference.", False),    # compatible
    ]

    constraint_ids: dict[str, str] = {}
    choice_ids: dict[str, str] = {}

    for label, content in constraints:
        r = graph.add_node(
            label=label,
            content=content,
            node_type=NodeType.PREFERENCE,
            project=project,
            tags=["constraint"],
        )
        constraint_ids[label] = r.node.id

    for label, content, conflicts_flag in choices:
        r = graph.add_node(
            label=label,
            content=content,
            node_type=NodeType.DECISION,
            project=project,
            tags=["choice", "conflicts" if conflicts_flag else "compatible"],
        )
        choice_ids[label] = r.node.id

    # Add contradicts edges for conflicting pairs
    gold_pairs: list[tuple[str, str]] = []
    for choice_label, _, conflicts_flag in choices:
        if conflicts_flag:
            # Link to the first constraint (simplification)
            constraint_label = constraints[0][0]
            graph.add_edge(
                source_id=choice_ids[choice_label],
                target_id=constraint_ids[constraint_label],
                relationship=RelationType.CONTRADICTS.value,
            )
            gold_pairs.append((choice_label, constraint_label))

    # Add distractor nodes to reach scale_n
    for i in range(scale_n - len(constraints) - len(choices)):
        graph.add_node(
            label=f"Distractor choice {i}",
            content=f"Use distractor technology {i} for component {i % 5}.",
            node_type=NodeType.DECISION,
            project=project,
            tags=["distractor"],
        )

    return [PairwiseCase(
        case_id=f"pairwise-{scale_n}",
        question="Which implementation choices conflict with the active constraints?",
        gold_conflict_pairs=gold_pairs,
        all_choice_labels=[c[0] for c in choices],
        all_constraint_labels=[c[0] for c in constraints],
        scale_n=scale_n,
    )]


def run_pairwise_benchmark(
    db_path: str,
    scale_n: int,
    methods: list[str],
    token_budget: int,
    rng: random.Random,
    include_latency: bool = True,
    verbose: bool = False,
) -> list[BenchResult]:
    """Run OOLONG-Pairs-style pairwise conflict reasoning benchmark."""
    graph = _make_graph(db_path)
    cases = generate_pairwise_cases(graph, scale_n=scale_n, rng=rng)
    results = []

    for case in cases:
        if verbose:
            print(f"  [Pairwise] scale={scale_n} gold_pairs={len(case.gold_conflict_pairs)}")

        for method in methods:
            runner = _METHOD_RUNNERS.get(method)
            if runner is None:
                continue

            pack, latency = runner(graph, case.question, token_budget)
            pack_lower = pack.lower()

            # Check which conflicting choices appear in the pack
            found_conflict_labels = [
                label for label, _ in case.gold_conflict_pairs
                if label.lower() in pack_lower
            ]
            pred_pairs = [(label, case.gold_conflict_pairs[0][1]) for label in found_conflict_labels]
            p_f1 = pairwise_f1(pred_pairs, case.gold_conflict_pairs)

            # Conflict recall: did the pack mention "conflict" or "contradict"?
            conflict_mentioned = 1.0 if ("conflict" in pack_lower or "contradict" in pack_lower) else 0.0

            # Conflict precision: fraction of mentioned choices that are actually conflicting
            all_mentioned = [
                label for label in case.all_choice_labels
                if label.lower() in pack_lower
            ]
            conflict_precision = (
                len(found_conflict_labels) / len(all_mentioned)
                if all_mentioned else 0.0
            )
            conflict_recall = (
                len(found_conflict_labels) / len(case.gold_conflict_pairs)
                if case.gold_conflict_pairs else 1.0
            )

            results.append(BenchResult(
                benchmark_family="OOLONG-Pairs-style",
                scale_n=scale_n,
                method=method,
                score=p_f1,
                exact_match=conflict_mentioned,
                f1=p_f1,
                evidence_coverage=conflict_recall,
                tokens_returned=token_estimate(pack),
                latency_ms=round(latency, 1) if include_latency else 0.0,
                context_pack_tokens=token_estimate(pack),
                notes=(
                    f"conflict_recall={conflict_recall:.2f} "
                    f"conflict_precision={conflict_precision:.2f} "
                    f"conflict_mentioned={conflict_mentioned:.0f}"
                ),
            ))

            if verbose:
                print(f"    {method}: pairwise_f1={p_f1:.2f} recall={conflict_recall:.2f} tokens={token_estimate(pack)}")

    return results


# ---------------------------------------------------------------------------
# 5. CodeQA-style: repo/codebase understanding
# ---------------------------------------------------------------------------


@dataclass
class CodeQACase:
    case_id: str
    question: str
    gold_answer: str
    gold_module: str
    gold_evidence_labels: list[str]
    scale_n: int


def generate_codeqa_cases(
    graph: MemoryGraph,
    scale_n: int,
    rng: random.Random,
    project: str = "codeqa",
) -> list[CodeQACase]:
    """
    Create synthetic codebase memory: modules, functions, design decisions,
    bug reports, and implementation notes with depends_on / part_of edges.

    TODO: load_real_codeqa() — plug in LongBench-v2 CodeQA dataset here.
    """
    # Core architecture nodes
    modules = [
        ("server.py", "server.py registers all MCP tools and dispatches tool calls to the graph."),
        ("recursive_context.py", "recursive_context.py implements RecursiveContextController and the build_context pipeline."),
        ("graph.py", "graph.py implements MemoryGraph with SQLite storage, node/edge CRUD, and retrieval."),
        ("pre_response.py", "pre_response.py is the Claude Code hook that calls build_context for concrete tasks."),
        ("hybrid.py", "hybrid.py implements HybridRetriever with vector, BM25, and graph fusion."),
        ("models.py", "models.py defines Node, Edge, SubgraphResult, and all Pydantic data models."),
        ("intelligence.py", "intelligence.py provides NLP utilities: tokenization, entity extraction, conflict detection."),
        ("config.py", "config.py reads AppConfig from environment variables."),
    ]

    module_ids: dict[str, str] = {}
    for label, content in modules:
        r = graph.add_node(
            label=label,
            content=content,
            node_type=NodeType.FACT,
            project=project,
            tags=["module", "codebase"],
        )
        module_ids[label] = r.node.id

    # Design decisions
    decisions = [
        ("Decomposition is deterministic",
         "Query decomposition in RecursiveContextController._decompose_query uses keyword heuristics, no external LLM."),
        ("Verbatim-first architecture",
         "observe_conversation always persists verbatim turns first before running extraction."),
        ("Hybrid retrieval default",
         "query_graph uses hybrid retrieval (vector + BM25 + graph) by default."),
        ("build_context token budget",
         "build_context compresses context to token_budget * 1.15 maximum using priority-ordered sections."),
    ]
    decision_ids: dict[str, str] = {}
    for label, content in decisions:
        r = graph.add_node(
            label=label,
            content=content,
            node_type=NodeType.DECISION,
            project=project,
            tags=["decision", "architecture"],
        )
        decision_ids[label] = r.node.id

    # Link modules to decisions
    graph.add_edge(
        source_id=module_ids["recursive_context.py"],
        target_id=decision_ids["Decomposition is deterministic"],
        relationship=RelationType.DERIVED_FROM.value,
    )
    graph.add_edge(
        source_id=module_ids["recursive_context.py"],
        target_id=decision_ids["build_context token budget"],
        relationship=RelationType.DERIVED_FROM.value,
    )
    graph.add_edge(
        source_id=module_ids["server.py"],
        target_id=module_ids["recursive_context.py"],
        relationship=RelationType.DEPENDS_ON.value,
    )
    graph.add_edge(
        source_id=module_ids["pre_response.py"],
        target_id=module_ids["recursive_context.py"],
        relationship=RelationType.DEPENDS_ON.value,
    )

    # Distractor nodes to reach scale_n
    for i in range(scale_n - len(modules) - len(decisions)):
        graph.add_node(
            label=f"Utility function util_{i}",
            content=f"util_{i} is a helper function in utils.py that handles edge case {i}.",
            node_type=NodeType.FACT,
            project=project,
            tags=["utility", "distractor"],
        )

    # Gold question: which module to modify for decomposition changes
    return [CodeQACase(
        case_id=f"codeqa-{scale_n}",
        question="Which module should be modified to change recursive context decomposition logic?",
        gold_answer="recursive_context.py",
        gold_module="recursive_context.py",
        gold_evidence_labels=[
            "recursive_context.py",
            "Decomposition is deterministic",
        ],
        scale_n=scale_n,
    )]


def run_codeqa_benchmark(
    db_path: str,
    scale_n: int,
    methods: list[str],
    token_budget: int,
    rng: random.Random,
    include_latency: bool = True,
    verbose: bool = False,
) -> list[BenchResult]:
    """Run CodeQA-style repo understanding benchmark."""
    graph = _make_graph(db_path)
    cases = generate_codeqa_cases(graph, scale_n=scale_n, rng=rng)
    results = []

    for case in cases:
        if verbose:
            print(f"  [CodeQA] scale={scale_n} question={case.question!r}")

        for method in methods:
            runner = _METHOD_RUNNERS.get(method)
            if runner is None:
                continue

            pack, latency = runner(graph, case.question, token_budget)
            pack_lower = pack.lower()

            em = exact_match(pack, case.gold_answer)

            # Evidence coverage: how many gold evidence labels appear in pack
            found_ev = [lbl for lbl in case.gold_evidence_labels if lbl.lower() in pack_lower]
            ev_cov = len(found_ev) / max(len(case.gold_evidence_labels), 1)

            # Wrong file rate: did the pack mention other modules more prominently?
            other_modules = ["server.py", "graph.py", "hybrid.py", "models.py", "intelligence.py"]
            wrong_mentions = sum(1 for m in other_modules if m in pack_lower and m != case.gold_module)
            wrong_file_rate = wrong_mentions / max(len(other_modules), 1)

            results.append(BenchResult(
                benchmark_family="CodeQA-style",
                scale_n=scale_n,
                method=method,
                score=em,
                exact_match=em,
                f1=ev_cov,
                evidence_coverage=ev_cov,
                tokens_returned=token_estimate(pack),
                latency_ms=round(latency, 1) if include_latency else 0.0,
                context_pack_tokens=token_estimate(pack),
                notes=f"wrong_file_rate={wrong_file_rate:.2f}",
            ))

            if verbose:
                print(f"    {method}: score={em:.2f} ev_cov={ev_cov:.2f} tokens={token_estimate(pack)}")

    return results


# ---------------------------------------------------------------------------
# 6. ContextReset benchmark family
# ---------------------------------------------------------------------------


@dataclass
class ContextResetCase:
    """One ContextReset benchmark case."""
    case_id: str
    question: str
    difficulty: str  # "easy" | "hard"
    gold_decision_ids: list[str]
    gold_constraint_ids: list[str]
    gold_next_step_id: str
    gold_superseded_id: str
    gold_active_decision_id: str  # source of the updates edge
    scale_n: int


def generate_context_reset_cases(
    graph: MemoryGraph,
    scale_n: int,
    rng: random.Random,
    difficulty: str = "easy",
    project: str = "context_reset",
) -> list[ContextResetCase]:
    """
    Generate ContextReset benchmark cases.

    Easy path: 1 decision, 1 constraint, 1 next-step, 1 superseded decision,
    fill remaining slots with distractors from 1 unrelated project.

    Hard path: 3+ decisions, 1 superseded decision, contradicts edge, rejected
    direction node, bug node, 1 next-step, fill remaining slots with distractors
    from 2 unrelated projects.
    """
    if difficulty == "easy":
        # Active decision
        r_decision = graph.add_node(
            label="Use PostgreSQL for storage",
            content="We decided to use PostgreSQL as the primary database.",
            node_type=NodeType.DECISION,
            project=project,
        )
        # Constraint
        r_constraint = graph.add_node(
            label="Must run locally",
            content="The system must run fully locally with no cloud dependency.",
            node_type=NodeType.PREFERENCE,
            project=project,
        )
        # Next-step
        r_next_step = graph.add_node(
            label="Next: implement connection pooling",
            content="The next step is to implement connection pooling for PostgreSQL.",
            node_type=NodeType.QUESTION,
            project=project,
        )
        # Superseded decision
        r_superseded = graph.add_node(
            label="Use SQLite for storage",
            content="We initially considered SQLite but decided against it.",
            node_type=NodeType.DECISION,
            project=project,
        )
        # updates edge: active → superseded
        graph.add_edge(
            source_id=r_decision.node.id,
            target_id=r_superseded.node.id,
            relationship="updates",
        )
        # Fill remaining slots with distractors
        filled = 4
        for i in range(max(0, scale_n - filled)):
            graph.add_node(
                label=f"Distractor {i}",
                content=f"Unrelated fact {i} about project alpha.",
                node_type=NodeType.FACT,
                project="distractor_alpha",
            )
        return [ContextResetCase(
            case_id=f"context_reset_easy_{scale_n}",
            question="Continue from where we left off",
            difficulty="easy",
            gold_decision_ids=[r_decision.node.id],
            gold_constraint_ids=[r_constraint.node.id],
            gold_next_step_id=r_next_step.node.id,
            gold_superseded_id=r_superseded.node.id,
            gold_active_decision_id=r_decision.node.id,
            scale_n=scale_n,
        )]

    else:  # hard
        # Three active decisions
        r_fastapi = graph.add_node(
            label="Use FastAPI",
            content="We decided to use FastAPI as the web framework.",
            node_type=NodeType.DECISION,
            project=project,
        )
        r_postgres = graph.add_node(
            label="Use PostgreSQL",
            content="We decided to use PostgreSQL as the primary database.",
            node_type=NodeType.DECISION,
            project=project,
        )
        r_local_emb = graph.add_node(
            label="Use local embeddings",
            content="We decided to use local embeddings for vector search.",
            node_type=NodeType.DECISION,
            project=project,
        )
        # Superseded decision
        r_flask = graph.add_node(
            label="Use Flask",
            content="We initially considered Flask but switched to FastAPI.",
            node_type=NodeType.DECISION,
            project=project,
        )
        # updates edge: FastAPI → Flask
        graph.add_edge(
            source_id=r_fastapi.node.id,
            target_id=r_flask.node.id,
            relationship="updates",
        )
        # Constraint
        r_constraint = graph.add_node(
            label="No external APIs",
            content="The system must not use any external APIs.",
            node_type=NodeType.PREFERENCE,
            project=project,
        )
        # Contradicts edge: PostgreSQL vs hosted RDS
        r_hosted_rds = graph.add_node(
            label="Use hosted RDS",
            content="An alternative was to use hosted RDS on AWS.",
            node_type=NodeType.DECISION,
            project=project,
        )
        graph.add_edge(
            source_id=r_postgres.node.id,
            target_id=r_hosted_rds.node.id,
            relationship=RelationType.CONTRADICTS.value,
        )
        # Rejected direction
        r_rejected = graph.add_node(
            label="Rejected: use microservices",
            content="We rejected a microservices architecture due to complexity.",
            node_type=NodeType.NOTE,
            project=project,
            tags=["rejected"],
        )
        # Bug node
        r_bug = graph.add_node(
            label="Bug: connection timeout",
            content="There is a known connection timeout bug in the database layer.",
            node_type=NodeType.NOTE,
            project=project,
            tags=["bug"],
        )
        # Next-step
        r_next_step = graph.add_node(
            label="Next: add rate limiting",
            content="The next step is to add rate limiting to the API.",
            node_type=NodeType.QUESTION,
            project=project,
        )
        # Fill remaining slots with distractors from 2 unrelated projects
        filled = 9  # fastapi, postgres, local_emb, flask, constraint, hosted_rds, rejected, bug, next_step
        for i in range(max(0, scale_n - filled)):
            proj = "distractor_alpha" if i % 2 == 0 else "distractor_beta"
            graph.add_node(
                label=f"Distractor {i}",
                content=f"Unrelated fact {i} about project {proj}.",
                node_type=NodeType.FACT,
                project=proj,
            )
        return [ContextResetCase(
            case_id=f"context_reset_hard_{scale_n}",
            question="Continue from where we left off",
            difficulty="hard",
            gold_decision_ids=[r_fastapi.node.id, r_postgres.node.id, r_local_emb.node.id],
            gold_constraint_ids=[r_constraint.node.id],
            gold_next_step_id=r_next_step.node.id,
            gold_superseded_id=r_flask.node.id,
            gold_active_decision_id=r_fastapi.node.id,
            scale_n=scale_n,
        )]


def _score_context_reset(
    pack: str,
    case: ContextResetCase,
    graph: MemoryGraph,
) -> dict[str, float]:
    """Compute the six ContextReset scoring fields."""
    pack_lower = pack.lower()

    # Helper: get node label by id
    def _label(node_id: str) -> str:
        try:
            node = graph.get_node(node_id)
            return node.label if node else ""
        except Exception:
            return ""

    # decision_recall: fraction of gold_decision_ids whose label appears in pack
    decision_labels = [_label(nid) for nid in case.gold_decision_ids]
    decision_found = sum(1 for lbl in decision_labels if lbl and lbl.lower() in pack_lower)
    decision_recall = decision_found / max(len(case.gold_decision_ids), 1)

    # constraint_recall: fraction of gold_constraint_ids whose label appears in pack
    constraint_labels = [_label(nid) for nid in case.gold_constraint_ids]
    constraint_found = sum(1 for lbl in constraint_labels if lbl and lbl.lower() in pack_lower)
    constraint_recall = constraint_found / max(len(case.gold_constraint_ids), 1)

    # next_step_accuracy: 1.0 if next-step node label appears in pack
    next_step_label = _label(case.gold_next_step_id)
    next_step_accuracy = 1.0 if next_step_label and next_step_label.lower() in pack_lower else 0.0

    # superseded_context_handling: 1.0 if superseded label absent OR appears after "Superseded context:" header
    superseded_label = _label(case.gold_superseded_id)
    if not superseded_label or superseded_label.lower() not in pack_lower:
        superseded_context_handling = 1.0
    else:
        # Check if it appears only in a superseded section
        sup_header_idx = pack_lower.find("superseded context")
        sup_label_idx = pack_lower.find(superseded_label.lower())
        if sup_header_idx >= 0 and sup_label_idx > sup_header_idx:
            superseded_context_handling = 1.0
        else:
            superseded_context_handling = 0.0

    # active_decision_preference
    active_label = _label(case.gold_active_decision_id)
    active_in_pack = bool(active_label and active_label.lower() in pack_lower)
    # "active sections" = pack before any "Superseded context:" header
    sup_header_idx = pack_lower.find("superseded context")
    if sup_header_idx >= 0:
        active_section = pack_lower[:sup_header_idx]
    else:
        active_section = pack_lower
    superseded_in_active = bool(superseded_label and superseded_label.lower() in active_section)
    active_in_active = bool(active_label and active_label.lower() in active_section)

    if active_in_active and not superseded_in_active:
        active_decision_preference = 1.0
    elif active_in_active and superseded_in_active:
        active_decision_preference = 0.5
    elif not active_in_active and superseded_in_active:
        active_decision_preference = 0.0
    else:
        active_decision_preference = 0.0

    # evidence_coverage: fraction of all gold node IDs whose labels appear in pack
    all_gold_ids = (
        case.gold_decision_ids
        + case.gold_constraint_ids
        + [case.gold_next_step_id, case.gold_superseded_id]
    )
    all_gold_labels = [_label(nid) for nid in all_gold_ids]
    ev_found = sum(1 for lbl in all_gold_labels if lbl and lbl.lower() in pack_lower)
    ev_coverage = ev_found / max(len(all_gold_ids), 1)

    return {
        "decision_recall": decision_recall,
        "constraint_recall": constraint_recall,
        "next_step_accuracy": next_step_accuracy,
        "superseded_context_handling": superseded_context_handling,
        "active_decision_preference": active_decision_preference,
        "evidence_coverage": ev_coverage,
    }


def run_context_reset_benchmark(
    db_path: str,
    scale_n: int,
    methods: list[str],
    token_budget: int,
    rng: random.Random,
    difficulty: str = "easy",
    include_latency: bool = True,
    verbose: bool = False,
) -> list[BenchResult]:
    """Run ContextReset benchmark at a given scale."""
    graph = _make_graph(db_path)
    cases = generate_context_reset_cases(graph, scale_n=scale_n, rng=rng, difficulty=difficulty)
    results = []
    output_dir = "benchmark_results/partial/context_reset"

    for case in cases:
        if verbose:
            print(f"  [ContextReset/{difficulty}] scale={scale_n} question={case.question!r}")

        for method in methods:
            if method == "no_memory":
                pack = ""
                latency = 0.0
            elif method in ("rmca_full", "build_context"):
                pack, latency = _run_build_context(graph, case.question, token_budget)
            else:
                runner = _METHOD_RUNNERS.get(method)
                if runner is None:
                    continue
                pack, latency = runner(graph, case.question, token_budget)

            scoring = _score_context_reset(pack, case, graph)
            score = (
                scoring["decision_recall"]
                + scoring["constraint_recall"]
                + scoring["next_step_accuracy"]
                + scoring["active_decision_preference"]
            ) / 4.0

            results.append(BenchResult(
                benchmark_family="ContextReset",
                scale_n=scale_n,
                method=method,
                score=score,
                exact_match=scoring["next_step_accuracy"],
                f1=scoring["decision_recall"],
                evidence_coverage=scoring["evidence_coverage"],
                tokens_returned=token_estimate(pack),
                latency_ms=round(latency, 1) if include_latency else 0.0,
                context_pack_tokens=token_estimate(pack),
                notes=json.dumps(scoring),
            ))

            if verbose:
                print(
                    f"    {method}: score={score:.2f} "
                    f"dec_recall={scoring['decision_recall']:.2f} "
                    f"tokens={token_estimate(pack)}"
                )

    write_results(results, output_dir)
    return results


# ---------------------------------------------------------------------------
# Real dataset TODO stubs
# ---------------------------------------------------------------------------


def load_real_sniah() -> list[dict]:
    """TODO: Load RULER S-NIAH dataset. Requires downloading RULER assets."""
    raise NotImplementedError(
        "load_real_sniah() not yet implemented. "
        "Download RULER from https://github.com/hsiehjackson/RULER and implement this loader."
    )


def load_real_browsecomp_plus() -> list[dict]:
    """TODO: Load BrowseComp-Plus dataset."""
    raise NotImplementedError(
        "load_real_browsecomp_plus() not yet implemented. "
        "Download BrowseComp-Plus and implement this loader."
    )


def load_real_oolong() -> list[dict]:
    """TODO: Load OOLONG dataset."""
    raise NotImplementedError(
        "load_real_oolong() not yet implemented. "
        "Download OOLONG from the original repository and implement this loader."
    )


def load_real_oolong_pairs() -> list[dict]:
    """TODO: Load OOLONG-Pairs dataset."""
    raise NotImplementedError(
        "load_real_oolong_pairs() not yet implemented. "
        "Download OOLONG-Pairs and implement this loader."
    )


def load_real_codeqa() -> list[dict]:
    """TODO: Load LongBench-v2 CodeQA dataset."""
    raise NotImplementedError(
        "load_real_codeqa() not yet implemented. "
        "Download LongBench-v2 from https://github.com/THUDM/LongBench and implement this loader."
    )


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------


def write_results(results: list[BenchResult], output_dir: str) -> dict[str, str]:
    """Write CSV, Markdown, and JSON summary to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "rlm_style_waggle_results.csv"
    md_path = out / "rlm_style_waggle_results.md"
    json_path = out / "rlm_style_waggle_summary.json"

    # CSV
    fieldnames = [f.name for f in BenchResult.__dataclass_fields__.values()]  # type: ignore[attr-defined]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Markdown table
    md_lines = [
        "# Waggle RLM-style Benchmark Results",
        "",
        "> **Warning:** This benchmark follows the benchmark families used in the RLM paper,",
        "> but uses deterministic synthetic memory tasks mapped to Waggle's graph/transcript",
        "> environment. It should **not** be compared numerically to the RLM paper until the",
        "> exact public datasets and matching model setup are run.",
        "",
        "| Benchmark family | Scale | Method | Score | F1 | Ev. Coverage | Tokens returned | Latency (ms) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        md_lines.append(
            f"| {r.benchmark_family} | {r.scale_n} | {r.method} "
            f"| {r.score:.3f} | {r.f1:.3f} | {r.evidence_coverage:.3f} "
            f"| {r.tokens_returned} | {r.latency_ms:.0f} |"
        )
    md_lines.append("")

    # Per-family summary
    md_lines.append("## Token efficiency: build_context vs baselines")
    md_lines.append("")
    md_lines.append("| Benchmark family | Scale | Method | Tokens returned | Score |")
    md_lines.append("|---|---:|---|---:|---:|")

    families = sorted({r.benchmark_family for r in results})
    scales = sorted({r.scale_n for r in results})
    for fam in families:
        for scale in scales:
            fam_results = [r for r in results if r.benchmark_family == fam and r.scale_n == scale]
            if not fam_results:
                continue
            for r in sorted(fam_results, key=lambda x: x.tokens_returned):
                md_lines.append(
                    f"| {r.benchmark_family} | {r.scale_n} | {r.method} "
                    f"| {r.tokens_returned} | {r.score:.3f} |"
                )

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # JSON summary
    summary: dict[str, Any] = {
        "warning": (
            "This benchmark follows the benchmark families used in the RLM paper, "
            "but uses deterministic synthetic memory tasks. "
            "Do not compare numerically to the RLM paper."
        ),
        "total_cases": len(results),
        "families": {},
    }
    for fam in families:
        fam_results = [r for r in results if r.benchmark_family == fam]
        by_method: dict[str, dict] = {}
        for r in fam_results:
            if r.method not in by_method:
                by_method[r.method] = {
                    "avg_score": 0.0,
                    "avg_f1": 0.0,
                    "avg_tokens": 0.0,
                    "avg_latency_ms": 0.0,
                    "count": 0,
                }
            entry = by_method[r.method]
            n = entry["count"]
            entry["avg_score"] = (entry["avg_score"] * n + r.score) / (n + 1)
            entry["avg_f1"] = (entry["avg_f1"] * n + r.f1) / (n + 1)
            entry["avg_tokens"] = (entry["avg_tokens"] * n + r.tokens_returned) / (n + 1)
            entry["avg_latency_ms"] = (entry["avg_latency_ms"] * n + r.latency_ms) / (n + 1)
            entry["count"] = n + 1
        summary["families"][fam] = by_method

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "csv": str(csv_path),
        "markdown": str(md_path),
        "json": str(json_path),
    }


def print_summary_table(results: list[BenchResult]) -> None:
    """Print a compact summary table to stdout."""
    print()
    print("=" * 90)
    print(f"{'Benchmark family':<28} {'Scale':>6} {'Method':<20} {'Score':>6} {'F1':>6} {'Tokens':>7} {'ms':>6}")
    print("-" * 90)
    for r in results:
        print(
            f"{r.benchmark_family:<28} {r.scale_n:>6} {r.method:<20} "
            f"{r.score:>6.3f} {r.f1:>6.3f} {r.tokens_returned:>7} {r.latency_ms:>6.0f}"
        )
    print("=" * 90)
    print()

    # Token efficiency highlight
    print("Token efficiency (build_context vs raw_context):")
    families = sorted({r.benchmark_family for r in results})
    scales = sorted({r.scale_n for r in results})
    for fam in families:
        for scale in scales:
            bc = next((r for r in results if r.benchmark_family == fam and r.scale_n == scale and r.method == "build_context"), None)
            rc = next((r for r in results if r.benchmark_family == fam and r.scale_n == scale and r.method == "raw_context"), None)
            if bc and rc and rc.tokens_returned > 0:
                ratio = bc.tokens_returned / rc.tokens_returned
                print(f"  {fam} @ {scale}: build_context uses {ratio:.1%} of raw_context tokens "
                      f"(score: {bc.score:.3f} vs {rc.score:.3f})")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_BENCHMARK_RUNNERS = {
    "sniah": run_sniah_benchmark,
    "multihop": run_multihop_benchmark,
    "linear_agg": run_linear_agg_benchmark,
    "pairwise": run_pairwise_benchmark,
    "codeqa": run_codeqa_benchmark,
    "context_reset": run_context_reset_benchmark,
}

_ALL_FAMILIES = list(_BENCHMARK_RUNNERS.keys())


def run_all(
    db_base: str,
    scales: list[int],
    methods: list[str],
    token_budget: int,
    seed: int,
    families: list[str],
    output_dir: str,
    include_latency: bool = True,
    reuse_db: bool = False,
    verbose: bool = False,
) -> list[BenchResult]:
    """Run all benchmark families and return combined results."""
    all_results: list[BenchResult] = []

    for family in families:
        runner = _BENCHMARK_RUNNERS.get(family)
        if runner is None:
            print(f"Unknown benchmark family: {family}", file=sys.stderr)
            continue

        for scale in scales:
            rng = random.Random(seed)

            if reuse_db:
                db_path = db_base
            else:
                db_path = f"{db_base}.{family}.{scale}.db"
                # Remove stale DB
                if Path(db_path).exists():
                    Path(db_path).unlink()

            if verbose:
                print(f"\n[{family}] scale={scale} db={db_path}")

            try:
                results = runner(
                    db_path=db_path,
                    scale_n=scale,
                    methods=methods,
                    token_budget=token_budget,
                    rng=rng,
                    include_latency=include_latency,
                    verbose=verbose,
                )
                all_results.extend(results)
            except Exception as exc:
                print(f"  ERROR in {family} @ {scale}: {exc}", file=sys.stderr)
                if verbose:
                    import traceback
                    traceback.print_exc()

    return all_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="RLM-style Waggle benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db", default="/tmp/waggle_rlm_eval", help="Base path for benchmark DBs")
    parser.add_argument("--scales", nargs="+", type=int, default=[128, 512], help="Memory sizes to test")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["raw_context", "query_graph", "prime_context", "build_context"],
        choices=list(_METHOD_RUNNERS.keys()),
        help="Retrieval methods to compare",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=_ALL_FAMILIES,
        choices=_ALL_FAMILIES,
        help="Benchmark families to run",
    )
    parser.add_argument("--token-budget", type=int, default=1200, help="Token budget for context assembly")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic data generation")
    parser.add_argument("--output", default="benchmark_results", help="Output directory for results files")
    parser.add_argument("--include-latency", action="store_true", default=True, help="Record latency measurements")
    parser.add_argument("--reuse-db", action="store_true", default=False, help="Reuse existing DB instead of fresh per run")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Verbose output")
    parser.add_argument(
        "--families-only",
        nargs="+",
        choices=_ALL_FAMILIES,
        help="Alias for --families (convenience)",
    )

    args = parser.parse_args(argv)

    if args.families_only:
        args.families = args.families_only

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print(f"Waggle RLM-style Benchmark Suite")
    print(f"  families : {args.families}")
    print(f"  scales   : {args.scales}")
    print(f"  methods  : {args.methods}")
    print(f"  budget   : {args.token_budget} tokens")
    print(f"  seed     : {args.seed}")
    print(f"  output   : {args.output}")
    print()
    print("WARNING: Results use synthetic data. Do not compare to RLM paper numerically.")
    print()

    results = run_all(
        db_base=args.db,
        scales=args.scales,
        methods=args.methods,
        token_budget=args.token_budget,
        seed=args.seed,
        families=args.families,
        output_dir=args.output,
        include_latency=args.include_latency,
        reuse_db=args.reuse_db,
        verbose=args.verbose,
    )

    if not results:
        print("No results produced.", file=sys.stderr)
        return 1

    print_summary_table(results)

    paths = write_results(results, args.output)
    print(f"Results written to:")
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
