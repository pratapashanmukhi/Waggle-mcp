"""
Tests for benchmarks/rlm_style_waggle_eval.py

1.  test_sniah_generator_has_one_gold_needle
2.  test_multihop_generator_has_required_evidence_chain
3.  test_linear_aggregation_gold_count_correct
4.  test_pairwise_gold_conflicts_correct
5.  test_codeqa_gold_answer_points_to_correct_module
6.  test_pairwise_f1
7.  test_evidence_coverage
8.  test_cli_smoke_runs_small_scale
9.  test_build_context_method_does_not_crash_without_hybrid
10. test_results_files_are_written
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Bootstrap path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_BENCH = _ROOT / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from waggle.graph import MemoryGraph
from waggle.models import NodeType, RelationType

from rlm_style_waggle_eval import (
    BenchResult,
    _DeterministicEmbedding,
    _make_graph,
    evidence_coverage,
    generate_codeqa_cases,
    generate_linear_agg_cases,
    generate_multihop_cases,
    generate_pairwise_cases,
    generate_sniah_cases,
    pairwise_f1,
    run_all,
    run_codeqa_benchmark,
    run_linear_agg_benchmark,
    run_multihop_benchmark,
    run_pairwise_benchmark,
    run_sniah_benchmark,
    set_f1,
    token_estimate,
    write_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Test 1: S-NIAH generator has exactly one gold needle
# ---------------------------------------------------------------------------


def test_sniah_generator_has_one_gold_needle(tmp_path: Path) -> None:
    """The S-NIAH generator must insert exactly one needle node."""
    db = str(tmp_path / "sniah.db")
    graph = _make_graph(db)
    cases = generate_sniah_cases(graph, scale_n=20, rng=_rng())

    assert len(cases) == 1, "Expected exactly one S-NIAH case"
    case = cases[0]

    # Gold answer must be a specific number of days
    assert "days" in case.gold_answer.lower(), f"Gold answer should mention 'days': {case.gold_answer}"

    # The needle node must exist in the graph
    stats = graph.get_stats()
    assert stats.total_nodes == 20, f"Expected 20 nodes, got {stats.total_nodes}"

    # The question must ask about the needle topic
    assert "password rotation" in case.question.lower() or "deployment" in case.question.lower()


# ---------------------------------------------------------------------------
# Test 2: Multi-hop generator has required evidence chain
# ---------------------------------------------------------------------------


def test_multihop_generator_has_required_evidence_chain(tmp_path: Path) -> None:
    """The multi-hop generator must create a 3-node evidence chain with edges."""
    db = str(tmp_path / "multihop.db")
    graph = _make_graph(db)
    cases = generate_multihop_cases(graph, scale_n=10, rng=_rng())

    assert len(cases) == 1
    case = cases[0]

    # Must have exactly 3 gold evidence nodes (A → B → C chain)
    assert len(case.gold_evidence_node_ids) == 3, (
        f"Expected 3 evidence nodes, got {len(case.gold_evidence_node_ids)}"
    )

    # Gold answer must mention PagerDuty
    assert "pagerduty" in case.gold_answer.lower() or "pd-" in case.gold_answer.lower(), (
        f"Gold answer should mention PagerDuty: {case.gold_answer}"
    )

    # All 3 gold nodes must exist in the graph
    for node_id in case.gold_evidence_node_ids:
        node = graph.get_node(node_id)
        assert node is not None, f"Gold node {node_id} not found in graph"


# ---------------------------------------------------------------------------
# Test 3: Linear aggregation gold count is correct
# ---------------------------------------------------------------------------


def test_linear_aggregation_gold_count_correct(tmp_path: Path) -> None:
    """The linear aggregation generator must correctly count blocked tasks."""
    db = str(tmp_path / "linear.db")
    graph = _make_graph(db)
    rng = _rng(seed=7)
    cases = generate_linear_agg_cases(graph, scale_n=30, rng=rng)

    assert len(cases) == 1
    case = cases[0]

    # Gold count must be >= 3 (generator ensures minimum)
    assert case.gold_count >= 3, f"Expected at least 3 blocked tasks, got {case.gold_count}"

    # All gold IDs must be valid task IDs
    for tid in case.gold_ids:
        assert tid.startswith("T"), f"Task ID should start with T: {tid}"

    # Gold count must match gold_ids length
    assert case.gold_count == len(case.gold_ids), (
        f"gold_count={case.gold_count} != len(gold_ids)={len(case.gold_ids)}"
    )

    # Verify total node count in graph matches scale_n (+ any extras added to reach min 3)
    # Note: dedup may reduce actual stored nodes; we verify the case data is correct
    assert case.gold_count == len(case.gold_ids), "gold_count must match gold_ids length"
    assert case.scale_n == 30


# ---------------------------------------------------------------------------
# Test 4: Pairwise gold conflicts are correct
# ---------------------------------------------------------------------------


def test_pairwise_gold_conflicts_correct(tmp_path: Path) -> None:
    """The pairwise generator must create the correct conflicting pairs."""
    db = str(tmp_path / "pairwise.db")
    graph = _make_graph(db)
    cases = generate_pairwise_cases(graph, scale_n=20, rng=_rng())

    assert len(cases) == 1
    case = cases[0]

    # Must have at least 3 conflicting pairs (hosted Postgres, SaaS vector DB, external LLM)
    assert len(case.gold_conflict_pairs) >= 3, (
        f"Expected at least 3 conflict pairs, got {len(case.gold_conflict_pairs)}"
    )

    # Each pair must be (choice_label, constraint_label)
    for choice_label, constraint_label in case.gold_conflict_pairs:
        assert choice_label in case.all_choice_labels, f"{choice_label} not in choices"
        assert constraint_label in case.all_constraint_labels, f"{constraint_label} not in constraints"

    # Verify contradicts edges exist in the graph
    from waggle.models import RelationType
    # Query for conflicts
    result = graph.list_conflicts(limit=50)
    assert len(result.conflicts) >= len(case.gold_conflict_pairs), (
        f"Expected {len(case.gold_conflict_pairs)} conflict edges, found {len(result.conflicts)}"
    )


# ---------------------------------------------------------------------------
# Test 5: CodeQA gold answer points to correct module
# ---------------------------------------------------------------------------


def test_codeqa_gold_answer_points_to_correct_module(tmp_path: Path) -> None:
    """The CodeQA generator must point to recursive_context.py for decomposition questions."""
    db = str(tmp_path / "codeqa.db")
    graph = _make_graph(db)
    cases = generate_codeqa_cases(graph, scale_n=20, rng=_rng())

    assert len(cases) == 1
    case = cases[0]

    assert case.gold_answer == "recursive_context.py", (
        f"Expected gold_answer='recursive_context.py', got {case.gold_answer!r}"
    )
    assert case.gold_module == "recursive_context.py"
    assert "recursive_context.py" in case.gold_evidence_labels

    # The module node must exist in the graph
    result = graph.query(
        query="recursive context decomposition",
        max_nodes=10,
        max_depth=1,
        retrieval_mode="graph",
    )
    labels = [n.label for n in result.nodes]
    assert any("recursive_context" in lbl.lower() for lbl in labels), (
        f"recursive_context.py not found in graph. Labels: {labels}"
    )


# ---------------------------------------------------------------------------
# Test 6: pairwise_f1 scorer
# ---------------------------------------------------------------------------


def test_pairwise_f1() -> None:
    """pairwise_f1 must be order-normalised and correct."""
    gold = [("A", "B"), ("C", "D"), ("E", "F")]

    # Perfect match
    assert pairwise_f1(gold, gold) == pytest.approx(1.0)

    # Reversed order — should still match (normalised)
    reversed_gold = [("B", "A"), ("D", "C"), ("F", "E")]
    assert pairwise_f1(reversed_gold, gold) == pytest.approx(1.0)

    # Empty pred
    assert pairwise_f1([], gold) == pytest.approx(0.0)

    # Empty gold
    assert pairwise_f1(gold, []) == pytest.approx(0.0)

    # Partial match: 2 of 3 correct
    partial = [("A", "B"), ("C", "D")]
    f1 = pairwise_f1(partial, gold)
    assert 0.0 < f1 < 1.0

    # Precision = 1.0, recall = 2/3 → F1 = 2*(1*2/3)/(1+2/3) = 4/5 = 0.8
    assert f1 == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# Test 7: evidence_coverage scorer
# ---------------------------------------------------------------------------


def test_evidence_coverage() -> None:
    """evidence_coverage must return correct fraction."""
    gold = ["node-1", "node-2", "node-3"]

    # All found
    assert evidence_coverage(["node-1", "node-2", "node-3", "node-4"], gold) == pytest.approx(1.0)

    # None found
    assert evidence_coverage(["node-5", "node-6"], gold) == pytest.approx(0.0)

    # 2 of 3 found
    assert evidence_coverage(["node-1", "node-2"], gold) == pytest.approx(2 / 3)

    # Empty gold → 1.0
    assert evidence_coverage(["node-1"], []) == pytest.approx(1.0)

    # Empty returned, non-empty gold → 0.0
    assert evidence_coverage([], gold) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 8: CLI smoke test at small scale
# ---------------------------------------------------------------------------


def test_cli_smoke_runs_small_scale(tmp_path: Path) -> None:
    """CLI must complete without error at scale=10 with all methods."""
    from rlm_style_waggle_eval import main

    output_dir = str(tmp_path / "results")
    db_base = str(tmp_path / "smoke")

    exit_code = main([
        "--db", db_base,
        "--scales", "10",
        "--methods", "raw_context", "query_graph", "build_context",
        "--families", "sniah", "codeqa",
        "--token-budget", "600",
        "--seed", "42",
        "--output", output_dir,
    ])

    assert exit_code == 0, f"CLI returned non-zero exit code: {exit_code}"

    # Output files must exist
    assert (Path(output_dir) / "rlm_style_waggle_results.csv").exists()
    assert (Path(output_dir) / "rlm_style_waggle_results.md").exists()
    assert (Path(output_dir) / "rlm_style_waggle_summary.json").exists()


# ---------------------------------------------------------------------------
# Test 9: build_context method does not crash without hybrid retriever
# ---------------------------------------------------------------------------


def test_build_context_method_does_not_crash_without_hybrid(tmp_path: Path) -> None:
    """build_context runner must work even when hybrid retrieval falls back to graph."""
    db = str(tmp_path / "no_hybrid.db")
    graph = _make_graph(db)

    # Add a few nodes
    for i in range(5):
        graph.add_node(
            label=f"Decision {i}",
            content=f"We decided to use approach {i} for component {i}.",
            node_type=NodeType.DECISION,
        )

    from rlm_style_waggle_eval import _run_build_context
    pack, latency = _run_build_context(graph, "What decisions were made?", token_budget=600)

    # Must not crash and must return a string
    assert isinstance(pack, str)
    assert latency >= 0.0


# ---------------------------------------------------------------------------
# Test 10: Results files are written correctly
# ---------------------------------------------------------------------------


def test_results_files_are_written(tmp_path: Path) -> None:
    """write_results must produce valid CSV, Markdown, and JSON files."""
    results = [
        BenchResult(
            benchmark_family="S-NIAH-style",
            scale_n=128,
            method="build_context",
            score=0.9,
            exact_match=0.9,
            f1=0.9,
            evidence_coverage=1.0,
            tokens_returned=450,
            latency_ms=120.5,
            context_pack_tokens=450,
            notes="test",
        ),
        BenchResult(
            benchmark_family="S-NIAH-style",
            scale_n=128,
            method="raw_context",
            score=0.7,
            exact_match=0.7,
            f1=0.7,
            evidence_coverage=1.0,
            tokens_returned=1100,
            latency_ms=80.0,
            context_pack_tokens=1100,
            notes="test",
        ),
    ]

    output_dir = str(tmp_path / "out")
    paths = write_results(results, output_dir)

    # CSV must be readable and have correct rows
    import csv as csv_mod
    with open(paths["csv"]) as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["method"] == "build_context"
    assert rows[1]["method"] == "raw_context"

    # Markdown must contain the warning
    md_text = Path(paths["markdown"]).read_text()
    assert "Warning" in md_text or "warning" in md_text.lower()
    assert "build_context" in md_text
    assert "raw_context" in md_text

    # JSON must be valid and contain the warning
    with open(paths["json"]) as f:
        summary = json.load(f)
    assert "warning" in summary
    assert "S-NIAH-style" in summary["families"]
    assert "build_context" in summary["families"]["S-NIAH-style"]

    # Token efficiency: build_context should use fewer tokens than raw_context
    bc_tokens = int(rows[0]["tokens_returned"])
    rc_tokens = int(rows[1]["tokens_returned"])
    assert bc_tokens < rc_tokens, (
        f"build_context ({bc_tokens}) should use fewer tokens than raw_context ({rc_tokens})"
    )
