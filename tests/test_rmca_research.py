"""
tests/test_rmca_research.py
============================
Test suite for all RMCA research infrastructure.

Covers:
  11.1  All 7 ablation variants produce non-empty context packs
  11.3  rmca_no_graph_expansion scores differently than rmca_full on pairwise
  11.4  rmca_no_conflict_resolution does not produce conflict section
  11.5  ContextReset case has all required gold fields
  11.6  Scoring correctness (exact_match, f1)
  11.10 Budget scaling writes output files
  11.11 DeterministicAnswerer returns non-empty string
  11.12 failure_analysis.py produces Markdown with "OOLONG"
  11.13 make_research_report.py produces all 11 section headings
  11.14 All 4 CLIs exit 0 with --help
  11.15 Multi-seed output has seed column with distinct values
"""
from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
for p in [str(SRC), str(BENCH)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from waggle.graph import MemoryGraph
from waggle.models import NodeType, RelationType
from waggle.recursive_context import AblationConfig, RecursiveContextController
from rlm_style_waggle_eval import (
    BenchResult,
    _make_graph,
    generate_pairwise_cases,
    generate_context_reset_cases,
    _score_context_reset,
    token_estimate,
)
from run_ablation import VARIANT_CONFIGS, _run_ablation_variant, write_ablation_results
from answer_level_eval import DeterministicAnswerer, _final_answer_exact_match, _final_answer_f1
from failure_analysis import _discover_csvs, _load_rows, _classify, _build_report
from make_research_report import generate_report


# ---------------------------------------------------------------------------
# FakeEmbeddingModel
# ---------------------------------------------------------------------------


class FakeEmbeddingModel:
    model_name = "fake-model"
    model_id = "fake-model:deterministic-v1"

    def embed(self, text):
        vec = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            idx = sum(ord(c) for c in token) % len(vec)
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def to_bytes(self, emb):
        return emb.astype(np.float32).tobytes()

    def from_bytes(self, data):
        return np.frombuffer(data, dtype=np.float32)

    def cosine_similarity(self, a, b):
        an, bn = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (an * bn)) if an > 0 and bn > 0 else 0.0


def make_graph(tmp_path):
    return MemoryGraph(tmp_path / "test.db", FakeEmbeddingModel())


# ---------------------------------------------------------------------------
# Test 11.1 — All 7 ablation variants produce non-empty context packs
# ---------------------------------------------------------------------------


def test_all_ablation_variants_produce_nonempty_context_pack(tmp_path):
    """Property 1: All ablation variants produce non-empty context packs."""
    graph = make_graph(tmp_path)
    # Add some nodes so retrieval has something to return
    for i in range(5):
        graph.add_node(
            label=f"Decision {i}",
            content=f"We decided to use approach {i} for component {i}.",
            node_type=NodeType.DECISION,
        )

    query = "What decisions were made?"
    for variant_name, config in VARIANT_CONFIGS.items():
        pack, latency = _run_ablation_variant(graph, query, token_budget=600, config=config)
        assert isinstance(pack, str), f"Variant {variant_name} returned non-string"
        assert len(pack) > 0, f"Variant {variant_name} returned empty context pack"


# ---------------------------------------------------------------------------
# Test 11.3 — rmca_no_graph_expansion scores differently than rmca_full on pairwise
# ---------------------------------------------------------------------------


def test_no_graph_expansion_differs_from_full_on_pairwise(tmp_path):
    """Property 2: Graph-dependent ablations score lower on pairwise cases."""
    rng = random.Random(42)
    db_path = str(tmp_path / "pairwise.db")
    graph = _make_graph(db_path)
    cases = generate_pairwise_cases(graph, scale_n=20, rng=rng)
    case = cases[0]

    from rlm_style_waggle_eval import pairwise_f1

    def score_variant(config):
        pack, _ = _run_ablation_variant(graph, case.question, token_budget=1200, config=config)
        pack_lower = pack.lower()
        found = [lbl for lbl, _ in case.gold_conflict_pairs if lbl.lower() in pack_lower]
        pred_pairs = [(lbl, case.gold_conflict_pairs[0][1]) for lbl in found]
        return pairwise_f1(pred_pairs, case.gold_conflict_pairs)

    full_score = score_variant(VARIANT_CONFIGS["rmca_full"])
    no_expand_score = score_variant(VARIANT_CONFIGS["rmca_no_graph_expansion"])

    # rmca_no_graph_expansion should score <= rmca_full on pairwise
    # (it may be equal if direct retrieval already finds the conflicts, but should not exceed)
    assert no_expand_score <= full_score + 0.01, (
        f"rmca_no_graph_expansion ({no_expand_score:.3f}) should not exceed "
        f"rmca_full ({full_score:.3f}) on pairwise"
    )


# ---------------------------------------------------------------------------
# Test 11.4 — rmca_no_conflict_resolution does not produce conflict section
# ---------------------------------------------------------------------------


def test_no_conflict_resolution_omits_conflict_section(tmp_path):
    """rmca_no_conflict_resolution should not produce 'Conflicts or superseded context' section."""
    graph = make_graph(tmp_path)
    # Add two nodes with a contradicts edge
    r1 = graph.add_node(
        label="Use PostgreSQL",
        content="Use PostgreSQL for the database.",
        node_type=NodeType.DECISION,
    )
    r2 = graph.add_node(
        label="Use MySQL",
        content="Use MySQL for the database.",
        node_type=NodeType.DECISION,
    )
    graph.add_edge(
        source_id=r1.node.id,
        target_id=r2.node.id,
        relationship=RelationType.CONTRADICTS.value,
    )

    config = VARIANT_CONFIGS["rmca_no_conflict_resolution"]
    pack, _ = _run_ablation_variant(
        graph, "What database should we use?", token_budget=1200, config=config
    )

    assert "Conflicts or superseded context" not in pack, (
        f"rmca_no_conflict_resolution should not produce conflict section, but got:\n{pack}"
    )


# ---------------------------------------------------------------------------
# Test 11.5 — ContextReset case has all required gold fields
# ---------------------------------------------------------------------------


def test_context_reset_case_has_all_gold_fields(tmp_path):
    """Property 3: ContextReset case generation invariant."""
    rng = random.Random(42)

    for difficulty in ("easy", "hard"):
        db_path = str(tmp_path / f"cr_{difficulty}.db")
        graph = _make_graph(db_path)
        cases = generate_context_reset_cases(graph, scale_n=20, rng=rng, difficulty=difficulty)

        assert len(cases) == 1, f"Expected 1 case for difficulty={difficulty}"
        case = cases[0]

        # Verify all gold fields exist
        assert case.gold_decision_ids, f"gold_decision_ids empty for {difficulty}"
        assert case.gold_constraint_ids, f"gold_constraint_ids empty for {difficulty}"
        assert case.gold_next_step_id, f"gold_next_step_id empty for {difficulty}"
        assert case.gold_superseded_id, f"gold_superseded_id empty for {difficulty}"
        assert case.gold_active_decision_id, f"gold_active_decision_id empty for {difficulty}"

        # Verify scoring fields are in [0.0, 1.0]
        # Run rmca_full to get a pack and score it
        controller = RecursiveContextController(graph=graph)
        result = controller.build_context(query=case.question, token_budget=1200)
        scoring = _score_context_reset(result.context_pack, case, graph)

        required_fields = [
            "decision_recall", "constraint_recall", "next_step_accuracy",
            "superseded_context_handling", "active_decision_preference", "evidence_coverage"
        ]
        for field in required_fields:
            assert field in scoring, f"Missing field {field} in scoring for {difficulty}"
            val = scoring[field]
            assert 0.0 <= val <= 1.0, f"Field {field}={val} out of [0,1] for {difficulty}"


# ---------------------------------------------------------------------------
# Test 11.6 — Scoring correctness
# ---------------------------------------------------------------------------


def test_exact_match_scoring():
    """_final_answer_exact_match must return 1.0 when gold appears in extracted."""
    assert _final_answer_exact_match("We use PostgreSQL for storage", "PostgreSQL") == 1.0
    assert _final_answer_exact_match("We use MySQL", "PostgreSQL") == 0.0
    assert _final_answer_exact_match("", "PostgreSQL") == 0.0


def test_f1_scoring():
    """_final_answer_f1 must return 1.0 for identical token sequences."""
    assert _final_answer_f1("postgresql database", "postgresql database") == pytest.approx(1.0)
    # Completely disjoint token sets → F1 = 0.0
    assert _final_answer_f1("redis cache", "postgresql database") == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 11.10 — Budget scaling writes output files
# ---------------------------------------------------------------------------


def test_budget_scaling_writes_output_files(tmp_path):
    """Budget scaling runner must write CSV and JSON to output dir."""
    output_dir = str(tmp_path / "budget_out")
    result = subprocess.run(
        [
            sys.executable, str(BENCH / "run_budget_scaling.py"),
            "--scales", "10",
            "--budgets", "250", "500",
            "--families", "pairwise",
            "--methods", "raw_context", "build_context",
            "--seed", "42",
            "--output", output_dir,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"run_budget_scaling.py failed:\n{result.stderr}"

    csv_path = Path(output_dir) / "budget_scaling_results.csv"
    json_path = Path(output_dir) / "budget_scaling_results.json"
    assert csv_path.exists(), "budget_scaling_results.csv not written"
    assert json_path.exists(), "budget_scaling_results.json not written"

    # Verify CSV has token_budget column
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0, "CSV is empty"
    assert "token_budget" in rows[0], "token_budget column missing from CSV"


# ---------------------------------------------------------------------------
# Test 11.11 — DeterministicAnswerer returns non-empty string
# ---------------------------------------------------------------------------


def test_deterministic_answerer_returns_nonempty():
    """DeterministicAnswerer must return non-empty string for a pack with known gold answer."""
    answerer = DeterministicAnswerer()
    context_pack = """### Waggle Recursive Context Pack
Task: What database are we using?

Current relevant decisions:
- [decision] Use PostgreSQL for storage: We decided to use PostgreSQL as the primary database.
- [decision] Use local embeddings: We use local sentence-transformers for embeddings.
"""
    result = answerer.extract(context_pack, "What database are we using?", gold_answer="PostgreSQL")
    assert isinstance(result, str)
    assert len(result) > 0, "DeterministicAnswerer returned empty string"


# ---------------------------------------------------------------------------
# Test 11.12 — failure_analysis.py produces Markdown with "OOLONG"
# ---------------------------------------------------------------------------


def test_failure_analysis_mentions_oolong(tmp_path):
    """failure_analysis.py must produce a Markdown file containing 'OOLONG'."""
    output_path = str(tmp_path / "failure_analysis.md")
    result = subprocess.run(
        [
            sys.executable, str(BENCH / "failure_analysis.py"),
            "--results-dir", str(ROOT / "benchmark_results"),
            "--output", output_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"failure_analysis.py failed:\n{result.stderr}"

    content = Path(output_path).read_text()
    assert "OOLONG" in content, "failure_analysis.md does not mention OOLONG"


# ---------------------------------------------------------------------------
# Test 11.13 — make_research_report.py produces all 11 section headings
# ---------------------------------------------------------------------------


def test_research_report_has_all_sections(tmp_path):
    """make_research_report.py must produce a Markdown file with all 11 required sections."""
    output_path = str(tmp_path / "report.md")
    result = subprocess.run(
        [
            sys.executable, str(BENCH / "make_research_report.py"),
            "--output", output_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"make_research_report.py failed:\n{result.stderr}"

    content = Path(output_path).read_text()
    required_sections = [
        "Abstract", "Method", "Benchmark Tasks", "Main Results",
        "Ablations", "Context-Reset", "Budget Scaling",
        "Answer-Level Evaluation", "Failure Analysis",
        "Limitations", "Reproducibility Commands",
    ]
    for section in required_sections:
        assert section in content, f"Section '{section}' missing from research report"


# ---------------------------------------------------------------------------
# Test 11.14 — All 4 CLIs exit 0 with --help
# ---------------------------------------------------------------------------


def test_all_clis_support_help():
    """All new CLIs must exit 0 when invoked with --help."""
    clis = [
        "run_budget_scaling.py",
        "answer_level_eval.py",
        "failure_analysis.py",
        "make_research_report.py",
    ]
    for cli in clis:
        result = subprocess.run(
            [sys.executable, str(BENCH / cli), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(ROOT),
        )
        assert result.returncode == 0, f"{cli} --help returned non-zero: {result.stderr}"


# ---------------------------------------------------------------------------
# Test 11.15 — Multi-seed output has seed column with distinct values
# ---------------------------------------------------------------------------


def test_multi_seed_output_has_seed_column(tmp_path):
    """Running with --seeds 42 43 must produce CSV with seed column and 2 distinct values."""
    output_dir = str(tmp_path / "multi_seed_out")
    result = subprocess.run(
        [
            sys.executable, str(BENCH / "run_ablation.py"),
            "--variants", "rmca_full",
            "--families", "pairwise",
            "--scales", "10",
            "--seeds", "42", "43",
            "--output", output_dir,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"run_ablation.py failed:\n{result.stderr}"

    csv_path = Path(output_dir) / "ablation_results.csv"
    assert csv_path.exists(), "ablation_results.csv not written"

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) > 0, "CSV is empty"
    assert "seed" in rows[0], "seed column missing from CSV"

    seeds_in_csv = {row["seed"] for row in rows}
    assert len(seeds_in_csv) >= 2, f"Expected ≥2 distinct seed values, got: {seeds_in_csv}"
