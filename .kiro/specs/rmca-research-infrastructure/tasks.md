# Implementation Plan: RMCA Research Infrastructure

## Overview

Build the complete research infrastructure for Recursive Memory Context Assembly (RMCA): ablation study support, formal method documentation, stronger baselines, a new ContextReset benchmark family, answer-level evaluation, budget scaling analysis, failure analysis, and an automated research report generator. All components run offline, use deterministic seeds, and produce paper-ready outputs.

Implementation order follows the dependency graph: `AblationConfig` first (everything else depends on it), then static docs, then baselines (needed by all new runners), then the new benchmark family, then the analysis runners, then the report generator, then the test suite.

## Tasks

- [x] 1. Add `AblationConfig` to `src/waggle/recursive_context.py`
  - Add `AblationConfig` dataclass with six boolean flags: `decompose`, `graph_expand`, `conflict_resolve`, `verbatim_evidence`, `budget_compress`, `random_subqueries`, and `random_seed: int = 42`
  - Add `ablation: AblationConfig | None = None` parameter to `build_context` — default `None` preserves existing behaviour exactly
  - In `build_context` Step 1: when `ablation.decompose is False`, replace subquery list with a single subquery using the original query verbatim; when `ablation.random_subqueries is True`, replace decomposition with randomly sampled substrings of the query using `ablation.random_seed`
  - In `build_context` Step 2: when `ablation.verbatim_evidence is False`, remove `"verbatim"` from all subquery `retrieval_modes`
  - In `build_context` Step 3: when `ablation.graph_expand is False`, skip the `_expand_graph()` call entirely
  - In `build_context` Step 4: when `ablation.conflict_resolve is False`, skip `_resolve_updates_and_conflicts()` and return all hits unmodified with an empty conflict list
  - In `build_context` Step 7: when `ablation.budget_compress is False`, concatenate all ranked hits without enforcing the token budget limit
  - Export `AblationConfig` in the module's public surface (no `__all__` change needed — just ensure it is importable)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

- [x] 2. Write formal method document `docs/research/rmca_method.md`
  - Create `docs/research/` directory if it does not exist
  - Write the static Markdown file with these sections in order: Abstract, Notation Table (defining `q`, `G=(V,E)`, `T`, `B`, `d`, `k`, `C`, `estimated_tokens(·)`), Algorithm 1 pseudocode block covering all eight steps (decompose, retrieve, expand, resolve, deduplicate, rank, compress, format), Comparison with top-k RAG (three bullet differences), Comparison with GraphRAG (three bullet differences), Limitations and Scope (containing the Synthetic_Caveat verbatim)
  - All symbols must be defined in the notation table before first use
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. Add six stronger baselines to `benchmarks/rlm_style_waggle_eval.py`
  - [x] 3.1 Implement `_run_bm25_topk(graph, query, token_budget)`: load all nodes via `graph.aggregate()`, build a simple BM25 scorer over `label + " " + content` for each node, score the query, take top-k by BM25 score, concatenate `[node_type] label: content` lines until token budget exhausted; wrap body in `try/except` returning `("", latency)` on failure
    - _Requirements: 4.1, 4.7, 4.9_
  - [x] 3.2 Implement `_run_vector_topk(graph, query, token_budget)`: embed query with `_DeterministicEmbedding().embed(query)`, load all nodes, compute cosine similarity for each node using the embedding stored on the node or re-embedding its text, take top-k, concatenate until budget; same error handling
    - _Requirements: 4.2, 4.7, 4.9_
  - [x] 3.3 Implement `_run_hybrid_rrf(graph, query, token_budget)`: get BM25 ranking and vector ranking independently, fuse with RRF formula `score(d) = Σ 1/(60 + rank_i(d))`, take top-k by fused score, concatenate until budget; same error handling
    - _Requirements: 4.3, 4.7, 4.9_
  - [x] 3.4 Implement `_run_graph_expansion_no_recursion(graph, query, token_budget)`: single `graph.query(query, max_nodes=20, max_depth=1, retrieval_mode="hybrid")`, then for each returned node fetch one hop of neighbours via `graph.get_related(node_id, max_depth=1)`, deduplicate, concatenate until budget; no subquery decomposition; same error handling
    - _Requirements: 4.4, 4.7, 4.9_
  - [x] 3.5 Implement `_run_summary_memory(graph, query, token_budget)`: call `graph.prime_context()`, return `result.summary` truncated to budget; same error handling
    - _Requirements: 4.5, 4.7, 4.9_
  - [x] 3.6 Implement `_run_full_transcript_truncation(graph, query, token_budget)`: load transcript records in reverse-chronological order, concatenate `role: text` lines until budget exhausted; no graph traversal; same error handling
    - _Requirements: 4.6, 4.7, 4.9_
  - [x] 3.7 Register all six new runners in `_METHOD_RUNNERS` dict with keys `bm25_topk`, `vector_topk`, `hybrid_rrf`, `graph_expansion_no_recursion`, `summary_memory`, `full_transcript_truncation`; add them to the `--methods` choices list in the CLI `argparse` block
    - _Requirements: 4.8_
  - [x] 3.8 Extend `BenchResult` dataclass with new optional fields (all with defaults so existing CSV readers continue to work): `seed: int = 42`, `token_budget: int = 0`, `ablation_variant: str = ""`, `delta_vs_full: float = 0.0`, `annotation: str = ""`, `mean_score: float = 0.0`, `std_score: float = 0.0`; update `write_results()` to include these fields in the CSV fieldnames
    - _Requirements: 2.8, 6.3, 10.3_

- [x] 4. Add ContextReset benchmark family to `benchmarks/rlm_style_waggle_eval.py`
  - [x] 4.1 Add `ContextResetCase` dataclass with fields: `case_id`, `question`, `difficulty`, `gold_decision_ids`, `gold_constraint_ids`, `gold_next_step_id`, `gold_superseded_id`, `gold_active_decision_id`, `scale_n`
    - _Requirements: 3.1, 3.2, 3.3_
  - [x] 4.2 Implement `generate_context_reset_cases(graph, scale_n, rng, difficulty="easy")`:
    - **Easy path**: add 1 decision node, 1 constraint node, 1 next-step node (`NodeType.QUESTION`), 1 superseded decision node linked via `updates` edge (`source=active_decision, target=superseded`), fill remaining slots with distractor nodes from 1 unrelated project
    - **Hard path**: add ≥3 decision nodes, 1 superseded decision linked via `updates` to an active decision, 1 `contradicts` edge between two choices, 1 rejected-direction node (tagged `rejected`), 1 bug node (`NodeType.NOTE` tagged `bug`), 1 next-step node, fill remaining slots with distractors from ≥2 unrelated projects
    - Query is always `"Continue from where we left off"`
    - _Requirements: 3.1, 3.2, 3.4_
  - [x] 4.3 Implement `run_context_reset_benchmark(db_path, scale_n, methods, token_budget, rng, difficulty, include_latency, verbose)`:
    - Evaluate all eight methods: `no_memory`, `raw_context`, `bm25_topk`, `vector_topk`, `hybrid_rrf`, `query_graph`, `summary_memory`, `rmca_full` (map `rmca_full` to `build_context` runner)
    - `no_memory` returns `""` as context pack
    - Compute six scoring fields per result: `decision_recall`, `constraint_recall`, `next_step_accuracy`, `superseded_context_handling`, `active_decision_preference`, `evidence_coverage`; store as JSON string in `BenchResult.notes` and as separate columns in the context_reset CSV
    - `active_decision_preference`: `1.0` if only active decision present, `0.5` if both present, `0.0` if only superseded present
    - Write results to `benchmark_results/partial/context_reset/` using same CSV/MD/JSON format as existing families
    - Support `--scales` flag; support `--help`
    - _Requirements: 3.3, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_
  - [x] 4.4 Register `context_reset` in `_BENCHMARK_RUNNERS` and `_ALL_FAMILIES`; add `"context_reset"` to the `--families` choices in the CLI
    - _Requirements: 3.9_

- [x] 5. Implement ablation runner `benchmarks/run_ablation.py`
  - [x] 5.1 Define `VARIANT_CONFIGS: dict[str, AblationConfig]` mapping all seven variant names to their `AblationConfig` instances: `rmca_full`, `rmca_no_decomposition`, `rmca_no_graph_expansion`, `rmca_no_conflict_resolution`, `rmca_no_verbatim_evidence`, `rmca_no_budget_compression`, `rmca_random_subqueries`
    - _Requirements: 2.1_
  - [x] 5.2 Implement `_run_ablation_variant(graph, query, token_budget, config: AblationConfig)` wrapper that instantiates `RecursiveContextController` and calls `build_context(..., ablation=config)`; register it temporarily in `_METHOD_RUNNERS` for the duration of the ablation run
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_
  - [x] 5.3 Implement the main runner loop: for each `(variant, family, scale)` combination, run the benchmark using the ablation wrapper, collect `BenchResult` rows with `ablation_variant` field populated; use `atexit.register(_flush_partial, results_list, output_dir)` + `try/finally` for partial-run safety
    - _Requirements: 2.8, 2.13_
  - [x] 5.4 After all variants complete, compute `delta_vs_full = variant_score - rmca_full_score` per `(family, scale)` and populate the `delta_vs_full` column; populate `annotation` column with a human-readable string naming the responsible component (e.g., `"graph_expand responsible for +1.000 on OOLONG-Pairs"`)
    - _Requirements: 2.9, 2.10_
  - [x] 5.5 Write output to `benchmark_results/ablation_results.csv`, `benchmark_results/ablation_results.md`, `benchmark_results/ablation_results.json`; the Markdown table must include the `delta_vs_full` and `annotation` columns; add `--help` and `--variant`, `--families`, `--scales`, `--seed`, `--output` flags
    - _Requirements: 2.8, 2.12_
  - [x] 5.6 Add `--seeds` flag (space-separated list of integers); when multiple seeds provided, run full benchmark once per seed, aggregate with `mean_score` and `std_score`, emit per-seed rows plus aggregated summary row (`seed=-1`); Markdown tables display `mean ± std`; single-seed behaviour unchanged
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.7_

- [x] 6. Implement answer-level evaluation `benchmarks/answer_level_eval.py`
  - [x] 6.1 Define `AnswerLevelResult` dataclass with fields: `benchmark_family`, `scale_n`, `method`, `answerer`, `final_answer_exact_match`, `final_answer_f1`, `evidence_used`, `contradiction_correctness`, `hallucination_rate`, `tokens_injected`, `seed`, `notes`
    - _Requirements: 5.1_
  - [x] 6.2 Implement `DeterministicAnswerer.extract(context_pack, question)`: scan lines starting with `- [decision]`, `- [fact]`, `- [preference]`; extract content after the colon; if gold answer is ≤10 tokens, return first line containing all gold tokens; otherwise return first non-empty extracted line; label this as a lower-bound evaluator in all docstrings and output files
    - _Requirements: 5.1, 5.2_
  - [x] 6.3 Implement `OllamaAnswerer.extract(context_pack, question)`: POST to `http://localhost:11434/api/generate` with 30s timeout; fall back to `DeterministicAnswerer` with a `WARNING` log if Ollama is unavailable; activated only with `--answerer ollama`, disabled by default
    - _Requirements: 5.9_
  - [x] 6.4 Implement the five metric functions: `final_answer_exact_match` (gold in extracted, case-insensitive), `final_answer_f1` (token-level F1 using `tokenize_text`), `evidence_used` (fraction of gold node IDs whose `label: content` appears in pack), `contradiction_correctness` (linear interpolation over found conflict pairs), `hallucination_rate` (fraction of extracted-answer sentences with no token overlap with any node label+content)
    - _Requirements: 5.3, 5.4, 5.5, 5.6, 5.7_
  - [x] 6.5 Implement the main evaluation loop: for each `(method, scale, family)` combination, run the benchmark generator to get a graph and cases, run the method runner to get a context pack, run the answerer to extract an answer, compute all five metrics, collect `AnswerLevelResult` rows; include the disclaimer string in all output files
    - _Requirements: 5.8, 5.13_
  - [x] 6.6 Write output to `benchmark_results/answer_level_results.csv`, `benchmark_results/answer_level_results.md`, `benchmark_results/answer_level_results.json`; add `--methods`, `--scales`, `--families`, `--output`, `--seed`, `--answerer`, `--seeds`, `--help` flags
    - _Requirements: 5.10, 5.11, 5.12_

- [x] 7. Implement budget scaling runner `benchmarks/run_budget_scaling.py`
  - [x] 7.1 Implement the main runner loop: for each `(budget, family, method, scale)` combination, run the benchmark generator and method runner, record `BenchResult` with `token_budget` column populated; default budgets `[250, 500, 1000, 2000, 4000]`; default families `context_reset`, `pairwise`, `linear_agg`, `codeqa`; use `atexit.register` + `try/finally` for partial-run safety
    - _Requirements: 6.1, 6.2, 6.3, 6.9_
  - [x] 7.2 Write output to `benchmark_results/budget_scaling_results.csv`, `benchmark_results/budget_scaling_results.md`, `benchmark_results/budget_scaling_results.json`
    - _Requirements: 6.4_
  - [x] 7.3 Generate four charts using matplotlib (or ASCII fallback if matplotlib not installed): `score_vs_budget.png`, `evidence_coverage_vs_budget.png`, `latency_vs_budget.png`, `tokens_returned_vs_budget.png` in `benchmark_results/charts/`; reuse `METHOD_COLORS`, `METHOD_MARKERS`, `METHOD_LABELS` style constants from `plot_rlm_results.py`; exit 0 in ASCII fallback mode
    - _Requirements: 6.5, 6.6_
  - [x] 7.4 Add `--budgets`, `--families`, `--scales`, `--methods`, `--seed`, `--seeds`, `--output`, `--help` flags; default seed 42
    - _Requirements: 6.7, 6.8_

- [x] 8. Implement failure analysis `benchmarks/failure_analysis.py`
  - [x] 8.1 Implement CSV discovery: glob `benchmark_results/**/*_results.csv` and `benchmark_results/partial/**/*_results.csv`; if no files found, write stub `benchmark_results/failure_analysis.md` noting no results available and exit 0
    - _Requirements: 7.1, 7.7_
  - [x] 8.2 Implement win/loss/tie classification: for each `(family, scale)` group, compare `build_context` (or `rmca_full`) score against `max(score for method not in {"build_context", "rmca_full"})`; classify as Win/Tie/Loss
    - _Requirements: 7.2_
  - [x] 8.3 Write `benchmark_results/failure_analysis.md` with these sections: summary table of wins/losses/ties per family, "Why OOLONG linear aggregation remains hard" (token budget cannot cover all N entries at large scale — fundamental O(n) information need), "Why CodeQA does not prove RMCA beats query_graph" (both score 1.0 on synthetic tasks — task too easy), "Research Implications", Synthetic_Caveat
    - _Requirements: 7.3, 7.4_
  - [x] 8.4 Add `--results-dir`, `--output`, `--help` flags
    - _Requirements: 7.5, 7.6_

- [x] 9. Implement research report generator `benchmarks/make_research_report.py`
  - [x] 9.1 Implement section population logic for all eleven sections in order: Abstract (hardcoded template), Method (inline from design), Benchmark Tasks (hardcoded descriptions), Main Results (from `benchmark_results/rlm_style_waggle_results.csv` or placeholder), Ablations (from `benchmark_results/ablation_results.csv` or placeholder), Context-Reset (from `benchmark_results/partial/context_reset/*.csv` or placeholder), Budget Scaling (from `benchmark_results/budget_scaling_results.csv` or placeholder), Answer-Level Evaluation (from `benchmark_results/answer_level_results.csv` or placeholder), Failure Analysis (full text of `benchmark_results/failure_analysis.md` or placeholder), Limitations (hardcoded + Synthetic_Caveat), Reproducibility Commands (exact CLI commands with `--seed 42` for all experiments)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9_
  - [x] 9.2 Write output to `docs/research/rmca_experiment_report.md`; print output path to stdout; add `--output`, `--help` flags; include the multi-seed disclaimer in the report
    - _Requirements: 8.10, 8.11, 8.12, 10.6_

- [x] 10. Checkpoint — verify all runners are importable and `--help` exits 0
  - Run `python benchmarks/run_ablation.py --help`, `python benchmarks/run_budget_scaling.py --help`, `python benchmarks/answer_level_eval.py --help`, `python benchmarks/failure_analysis.py --help`, `python benchmarks/make_research_report.py --help` and confirm each exits with code 0
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Write test suite `tests/test_rmca_research.py`
  - [x] 11.1 Write test that runs all seven ablation variants against a fixed query and a small `MemoryGraph` (scale 10, `_DeterministicEmbedding`, `tempfile`) and asserts each produces a non-empty `context_pack` string
    - **Property 1: Ablation variants produce non-empty context packs**
    - **Validates: Requirements 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**
    - _Requirements: 2.2–2.7_
  - [ ]* 11.2 Write property test for ablation variant non-emptiness using `hypothesis`: for any non-empty query string and graph with ≥1 node, all seven variants produce non-empty context packs
    - **Property 1: Ablation variants produce non-empty context packs**
    - **Validates: Requirements 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**
  - [x] 11.3 Write test that verifies `rmca_no_graph_expansion` produces a different pairwise benchmark score than `rmca_full` on a graph with `contradicts` edges (use `generate_pairwise_cases` at scale 20)
    - **Property 2: Graph-dependent ablations score strictly lower than rmca_full on pairwise cases**
    - **Validates: Requirements 2.11**
    - _Requirements: 2.11_
  - [x] 11.4 Write test that verifies `rmca_no_conflict_resolution` produces a context pack that does NOT contain the string `"Conflicts or superseded context"` when conflict edges are present in the graph
    - _Requirements: 2.4_
  - [x] 11.5 Write test that generates a `ContextResetCase` at both easy and hard difficulty and asserts the result contains all required gold fields (`decision_recall`, `constraint_recall`, `next_step_accuracy`, `superseded_context_handling`, `active_decision_preference`, `evidence_coverage`) with values in `[0.0, 1.0]`
    - **Property 3: ContextReset case generation invariant**
    - **Validates: Requirements 3.1, 3.3**
    - _Requirements: 3.1, 3.3, 3.4_
  - [ ]* 11.6 Write property test for `active_decision_preference` scoring: for any context pack containing only the active decision node label, score equals `1.0`; for any pack containing only the superseded node label, score equals `0.0`
    - **Property 4: Active decision preference scoring is correct**
    - **Validates: Requirements 3.5**
  - [ ]* 11.7 Write property test for baseline runners: for any valid query string and `MemoryGraph` (including empty graphs), each of the six new baseline runners returns a `(str, float)` tuple without raising
    - **Property 5: Baseline runners handle any valid graph without raising**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.9**
  - [ ]* 11.8 Write property test for answer-level scoring: for any gold answer string `G` that appears verbatim in a context pack `P`, `final_answer_exact_match(P, G)` equals `1.0`; for identical token sequences, `token_f1` equals `1.0`; for disjoint token sequences, `token_f1` equals `0.0`
    - **Property 6: Answer-level scoring functions are correct**
    - **Validates: Requirements 5.3, 5.4**
  - [ ]* 11.9 Write property test for hallucination rate: for any context pack composed entirely of node label and content strings from the `MemoryGraph`, `hallucination_rate` equals `0.0`
    - **Property 7: Hallucination rate is zero for graph-sourced content**
    - **Validates: Requirements 5.7**
  - [x] 11.10 Write test that runs `run_budget_scaling.py` with `--scales 128 --budgets 250 500` via subprocess and verifies that output CSV and JSON files are written to the specified `tempdir` output directory
    - _Requirements: 6.1, 6.4_
  - [x] 11.11 Write test that verifies `DeterministicAnswerer.extract()` returns a non-empty string for a context pack containing a known gold answer string
    - _Requirements: 5.1_
  - [x] 11.12 Write test that verifies `failure_analysis.py` produces a Markdown file containing the substring `"OOLONG"` when run against the existing `benchmark_results/rlm_style_waggle_results.csv`
    - **Property 8: Failure analysis correctly classifies wins/losses/ties**
    - **Validates: Requirements 7.2**
    - _Requirements: 7.3_
  - [x] 11.13 Write test that verifies `make_research_report.py` produces a Markdown file containing all eleven required section headings: "Abstract", "Method", "Benchmark Tasks", "Main Results", "Ablations", "Context-Reset", "Budget Scaling", "Answer-Level Evaluation", "Failure Analysis", "Limitations", "Reproducibility Commands"
    - **Property 9: Research report always contains all required section headings**
    - **Validates: Requirements 8.1**
    - _Requirements: 8.1_
  - [x] 11.14 Write test that verifies each of the four CLIs exits with code 0 when invoked with `--help`: `benchmarks/run_budget_scaling.py`, `benchmarks/answer_level_eval.py`, `benchmarks/failure_analysis.py`, `benchmarks/make_research_report.py`
    - _Requirements: 6.8, 5.10, 7.5, 8.10_
  - [x] 11.15 Write test that runs any one benchmark runner with `--seeds 42 43` and verifies the output CSV contains a `seed` column and at least two distinct seed values
    - **Property 10: Multi-seed output contains seed column with distinct values**
    - **Validates: Requirements 10.3**
    - _Requirements: 10.3_

- [x] 12. Final checkpoint — full test suite passes
  - Run `pytest tests/test_rmca_research.py -v` and confirm all non-optional tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- `AblationConfig` (Task 1) is the critical dependency — all ablation and benchmark tasks depend on it
- The six new baselines (Task 3) must be registered before ContextReset (Task 4) and the ablation runner (Task 5), since both use the `_METHOD_RUNNERS` dispatch table
- Partial-run safety (`atexit` + `try/finally`) is required in Tasks 5, 6, and 7
- Property tests use `hypothesis` with `@settings(max_examples=100)`; add `hypothesis` to `requirements-dev.txt` if not present
- All tests use `_DeterministicEmbedding` and `tempfile`; no internet access or external model API required
- The `write_results()` function in `rlm_style_waggle_eval.py` must be updated (Task 3.8) before any runner that writes extended `BenchResult` fields
