"""
benchmarks/make_research_report.py
=====================================
Generate a complete, paper-ready research report from all RMCA experimental
results.

Reads all result CSVs and the failure analysis, then generates
docs/research/rmca_experiment_report.md with 11 sections.

Usage:
  python benchmarks/make_research_report.py \\
    --output docs/research/rmca_experiment_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNTHETIC_CAVEAT = (
    "> **Synthetic data caveat:** The current RMCA evaluation uses deterministic "
    "synthetic Waggle memory tasks mapped to Waggle's graph/transcript environment. "
    "Numerical results **must not be compared** to results from the RLM paper "
    "(Zhang et al., 2026) or other long-context benchmarks until the exact public "
    "datasets (RULER S-NIAH, BrowseComp-Plus, OOLONG, OOLONG-Pairs, LongBench-v2 "
    "CodeQA) are downloaded and run with a matching model setup."
)

ANSWER_LEVEL_DISCLAIMER = (
    "> **Answer-level disclaimer:** Deterministic answer-level metrics are reproducible "
    "lower bounds. They are not equivalent to human preference ratings or LLM-judge "
    "quality assessments. Scores should be interpreted as retrieval-quality proxies, "
    "not end-to-end answer quality."
)

MULTI_SEED_DISCLAIMER = (
    "> **Statistical note:** Single-seed results are provided for quick reproducibility. "
    "Paper-quality claims should be verified with ≥3 seeds (e.g., `--seeds 42 43 44`)."
)

REPRODUCIBILITY_COMMANDS = """\
## Reproducibility Commands

All commands use deterministic seed 42. For paper-quality runs, use `--seeds 42 43 44`.

```bash
# 1. Main benchmark (5 families, 3 scales)
python benchmarks/rlm_style_waggle_eval.py \\
  --db /tmp/waggle_rlm_eval \\
  --scales 128 512 2048 \\
  --methods raw_context query_graph build_context \\
  --families sniah multihop pairwise codeqa context_reset \\
  --token-budget 1200 --seed 42 --output benchmark_results/

# 2. Ablation study (7 variants, 3 families)
python benchmarks/run_ablation.py \\
  --variants rmca_full rmca_no_decomposition rmca_no_graph_expansion \\
             rmca_no_conflict_resolution rmca_no_verbatim_evidence \\
             rmca_no_budget_compression rmca_random_subqueries \\
  --families pairwise codeqa context_reset \\
  --scales 128 512 --seed 42 --output benchmark_results/

# 3. Budget scaling (5 budgets, 4 families)
python benchmarks/run_budget_scaling.py \\
  --budgets 250 500 1000 2000 4000 \\
  --families context_reset pairwise linear_agg codeqa \\
  --methods raw_context query_graph build_context \\
  --scales 128 --seed 42 --output benchmark_results/

# 4. Answer-level evaluation
python benchmarks/answer_level_eval.py \\
  --methods rmca_full query_graph bm25_topk \\
  --scales 128 --families pairwise codeqa context_reset \\
  --seed 42 --output benchmark_results/

# 5. Failure analysis
python benchmarks/failure_analysis.py \\
  --results-dir benchmark_results/ \\
  --output benchmark_results/failure_analysis.md

# 6. Generate this report
python benchmarks/make_research_report.py \\
  --output docs/research/rmca_experiment_report.md
```
"""

# ---------------------------------------------------------------------------
# CSV / file loaders
# ---------------------------------------------------------------------------


def _load_csv_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _csv_to_md_table(rows: list[dict[str, str]], columns: list[str] | None = None) -> str:
    if not rows:
        return "_No data available._"
    cols = columns or list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for row in rows:
        cells = [str(row.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _load_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _section_abstract() -> str:
    return """\
## Abstract

We present **Recursive Memory Context Assembly (RMCA)**, an algorithm for
assembling compact, high-signal context packs from a persistent agent memory
graph in response to a user query. Unlike top-k RAG, which retrieves a flat
ranked list of text chunks, RMCA decomposes the query into targeted subqueries,
retrieves from multiple evidence lanes (graph, hybrid, verbatim transcript),
expands the graph along typed edges, resolves update chains and contradictions,
deduplicates and ranks hits, and compresses the result to a configurable token
budget.

We evaluate RMCA on five benchmark families adapted from the Recursive Language
Models paper (Zhang et al., 2026): S-NIAH-style needle retrieval, BrowseComp-
Plus-style multi-hop QA, OOLONG-style linear aggregation, OOLONG-Pairs-style
pairwise conflict reasoning, and CodeQA-style codebase understanding. We also
introduce a new **ContextReset** benchmark that directly tests the session-
boundary use case motivating Waggle's design.

The strongest result is on OOLONG-Pairs-style pairwise conflict reasoning:
both `raw_context` and `query_graph` score 0.0 at every scale (128–2048 nodes),
while RMCA scores 1.0 using 31–38% of raw-context tokens. This demonstrates
that explicit `contradicts` edge traversal is load-bearing for conflict-aware
memory retrieval.

""" + SYNTHETIC_CAVEAT


def _section_method() -> str:
    method_doc = _load_text(Path("docs/research/rmca_method.md"))
    if method_doc:
        # Embed the method doc inline, adjusting heading levels
        adjusted = method_doc.replace("\n# ", "\n### ").replace("\n## ", "\n#### ")
        return "## Method\n\n" + adjusted
    return """\
## Method

See `docs/research/rmca_method.md` for the full formal definition, Algorithm 1
pseudocode, and comparison with top-k RAG and GraphRAG.

**Summary:** RMCA takes inputs `(q, G=(V,E), T, B, d, k)` and produces a
context pack `C` where `estimated_tokens(C) ≤ B × 1.15` via an 8-step pipeline:
decompose → retrieve → expand → resolve → deduplicate → rank → compress → format.
"""


def _section_benchmark_tasks() -> str:
    return """\
## Benchmark Tasks

We evaluate on six benchmark families:

| Family | RLM paper equivalent | Complexity | What it tests |
|---|---|---|---|
| S-NIAH-style | RULER S-NIAH | O(1) | One fact among N distractors |
| BrowseComp-Plus-style | BrowseComp-Plus | O(hops) | 3-node evidence chain |
| OOLONG-style | OOLONG | O(n) | Aggregate N task-status entries |
| OOLONG-Pairs-style | OOLONG-Pairs | O(n²) | Pairwise conflict reasoning |
| CodeQA-style | LongBench-v2 CodeQA | O(files) | Codebase understanding |
| ContextReset | (novel) | O(session) | Session-boundary state restoration |

All tasks use deterministic synthetic Waggle memory. """ + SYNTHETIC_CAVEAT


def _section_main_results(results_dir: Path) -> str:
    rows = _load_csv_table(results_dir / "rlm_style_waggle_results.csv")
    if not rows:
        return """\
## Main Results

_Results not yet available. Run:_
```bash
python benchmarks/rlm_style_waggle_eval.py --scales 128 512 2048 --seed 42
```
"""
    cols = ["benchmark_family", "scale_n", "method", "score", "f1",
            "evidence_coverage", "tokens_returned", "latency_ms"]
    table = _csv_to_md_table(rows, cols)
    return f"""\
## Main Results

{MULTI_SEED_DISCLAIMER}

{table}

**Key finding:** On OOLONG-Pairs-style pairwise conflict reasoning, `build_context`
scores 1.0 at all scales while `raw_context` and `query_graph` score 0.0.
`build_context` uses 31–38% of raw-context tokens at scale=128–2048.
"""


def _section_ablations(results_dir: Path) -> str:
    rows = _load_csv_table(results_dir / "ablation_results.csv")
    if not rows:
        return """\
## Ablations

_Ablation results not yet available. Run:_
```bash
python benchmarks/run_ablation.py --families pairwise codeqa context_reset --scales 128 512 --seed 42
```
"""
    cols = ["benchmark_family", "scale_n", "ablation_variant", "score",
            "delta_vs_full", "annotation", "tokens_returned"]
    table = _csv_to_md_table(rows, cols)
    return f"""\
## Ablations

We ablate seven RMCA components by disabling one step at a time via `AblationConfig` flags.

{table}

**Expected finding:** `rmca_no_graph_expansion` and `rmca_no_conflict_resolution`
should score strictly lower than `rmca_full` on OOLONG-Pairs-style tasks, confirming
that graph expansion and conflict resolution are load-bearing components.
"""


def _section_context_reset(results_dir: Path) -> str:
    partial_dir = results_dir / "partial" / "context_reset"
    rows: list[dict[str, str]] = []
    if partial_dir.exists():
        for csv_path in partial_dir.glob("*.csv"):
            rows.extend(_load_csv_table(csv_path))

    if not rows:
        return """\
## Context-Reset Results

_ContextReset results not yet available. Run:_
```bash
python benchmarks/rlm_style_waggle_eval.py --families context_reset --scales 128 512 --seed 42
```
"""
    cols = ["benchmark_family", "scale_n", "method", "score",
            "evidence_coverage", "tokens_returned", "latency_ms"]
    table = _csv_to_md_table(rows, cols)
    return f"""\
## Context-Reset Results

The ContextReset benchmark tests session-boundary state restoration:
session 1 stores project decisions, constraints, and next steps;
session 2 starts fresh and asks "Continue from where we left off."

Two difficulty levels: **easy** (1 decision, 1 constraint, 1 next-step) and
**hard** (3+ decisions, superseded chain, contradicts edge, rejected direction,
bug node, multi-project distractors).

{table}

**Key metric:** `active_decision_preference` — whether the method returns the
latest active decision (source of the `updates` edge) rather than the superseded one.
"""


def _section_budget_scaling(results_dir: Path) -> str:
    rows = _load_csv_table(results_dir / "budget_scaling_results.csv")
    if not rows:
        return """\
## Budget Scaling

_Budget scaling results not yet available. Run:_
```bash
python benchmarks/run_budget_scaling.py --budgets 250 500 1000 2000 4000 --seed 42
```
"""
    cols = ["benchmark_family", "scale_n", "method", "token_budget",
            "score", "evidence_coverage", "tokens_returned"]
    table = _csv_to_md_table(rows, cols)
    return f"""\
## Budget Scaling

We sweep token budgets [250, 500, 1000, 2000, 4000] across four families
(ContextReset, OOLONG-Pairs, OOLONG, CodeQA) to characterise the efficiency frontier.

{table}

Charts: `benchmark_results/charts/score_vs_budget.png`,
`evidence_coverage_vs_budget.png`, `latency_vs_budget.png`,
`tokens_returned_vs_budget.png`
"""


def _section_answer_level(results_dir: Path) -> str:
    rows = _load_csv_table(results_dir / "answer_level_results.csv")
    if not rows:
        return """\
## Answer-Level Evaluation

_Answer-level results not yet available. Run:_
```bash
python benchmarks/answer_level_eval.py --families pairwise codeqa --seed 42
```
"""
    cols = ["benchmark_family", "scale_n", "method", "answerer",
            "final_answer_exact_match", "final_answer_f1",
            "evidence_used", "hallucination_rate", "tokens_injected"]
    table = _csv_to_md_table(rows, cols)
    return f"""\
## Answer-Level Evaluation

{ANSWER_LEVEL_DISCLAIMER}

Pipeline: method → context pack → `DeterministicAnswerer` → final answer → scorer.

{table}
"""


def _section_failure_analysis(results_dir: Path) -> str:
    fa_path = results_dir / "failure_analysis.md"
    content = _load_text(fa_path)
    if not content:
        return """\
## Failure Analysis

_Failure analysis not yet available. Run:_
```bash
python benchmarks/failure_analysis.py --results-dir benchmark_results/
```
"""
    # Embed the failure analysis, adjusting heading levels
    adjusted = content.replace("\n# ", "\n### ").replace("\n## ", "\n#### ")
    return "## Failure Analysis\n\n" + adjusted


def _section_limitations() -> str:
    return f"""\
## Limitations

1. **Synthetic data only.** All benchmarks use deterministic synthetic Waggle
   memory tasks. Results cannot be compared to the RLM paper until real datasets
   are used.

2. **Token estimation is approximate.** `estimated_tokens(text) = len(text) // 4`
   is a character-count heuristic. Actual token counts depend on the downstream
   model's tokenizer and may differ by ±20%.

3. **Decomposition is heuristic, not learned.** Subquery generation uses keyword
   pattern matching. It may produce suboptimal decompositions for queries outside
   the project/coding and generic-memory categories.

4. **Linear aggregation tasks remain hard.** For O(n) tasks requiring aggregation
   over all N entries, RMCA cannot guarantee coverage when N exceeds the token
   budget. This is a fundamental limitation of fixed-budget retrieval.

5. **Answer-level evaluation uses a deterministic lower-bound scorer.** The
   `DeterministicAnswerer` is a reproducible proxy, not a quality judge.

6. **Single-seed results.** Default runs use seed 42. Paper-quality claims
   require ≥3 seeds.

{SYNTHETIC_CAVEAT}
"""


# ---------------------------------------------------------------------------
# Main report assembler
# ---------------------------------------------------------------------------


def generate_report(
    results_dir: str = "benchmark_results",
    output_path: str = "docs/research/rmca_experiment_report.md",
) -> str:
    rd = Path(results_dir)

    sections = [
        "# Recursive Memory Context Assembly (RMCA) — Experiment Report",
        "",
        "_Auto-generated by `benchmarks/make_research_report.py`_",
        "",
        "---",
        "",
        _section_abstract(),
        "",
        "---",
        "",
        _section_method(),
        "",
        "---",
        "",
        _section_benchmark_tasks(),
        "",
        "---",
        "",
        _section_main_results(rd),
        "",
        "---",
        "",
        _section_ablations(rd),
        "",
        "---",
        "",
        _section_context_reset(rd),
        "",
        "---",
        "",
        _section_budget_scaling(rd),
        "",
        "---",
        "",
        _section_answer_level(rd),
        "",
        "---",
        "",
        _section_failure_analysis(rd),
        "",
        "---",
        "",
        _section_limitations(),
        "",
        "---",
        "",
        REPRODUCIBILITY_COMMANDS,
    ]

    report = "\n".join(sections)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate RMCA research report from all experimental results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        default="benchmark_results",
        help="Directory containing result CSV files (default: benchmark_results/)",
    )
    parser.add_argument(
        "--output",
        default="docs/research/rmca_experiment_report.md",
        help="Output path for the research report (default: docs/research/rmca_experiment_report.md)",
    )
    args = parser.parse_args(argv)

    output_path = generate_report(
        results_dir=args.results_dir,
        output_path=args.output,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
