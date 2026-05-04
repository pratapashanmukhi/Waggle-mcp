# Requirements Document

## Introduction

This feature turns Waggle MCP's existing Recursive Context Assembly (RMCA) implementation into defensible research evidence suitable for submission to ICLR/NeurIPS-level venues. The goal is not product improvement but research infrastructure: formal method documentation, ablation studies, stronger baselines, new benchmark families, answer-level evaluation, budget scaling analysis, failure analysis, and an automated research report generator. All components must run offline, require no external model API, use deterministic seeds, and produce outputs that can be used directly in a research paper.

The existing system has `RecursiveContextController.build_context` (the RMCA implementation), five RLM-style benchmark families (S-NIAH, BrowseComp-Plus, OOLONG, OOLONG-Pairs, CodeQA), and full results at scales 128/512/2048. The strongest existing result is OOLONG-Pairs, where `raw_context` and `query_graph` score 0.0 at every scale while `build_context` scores 1.0 using 31–38% of raw tokens.

## Glossary

- **RMCA**: Recursive Memory Context Assembly — the algorithm implemented in `RecursiveContextController.build_context` that decomposes a query into subqueries, retrieves from multiple lanes, expands the graph, resolves conflicts, deduplicates, ranks, and compresses to a token budget.
- **Memory_Graph**: The Waggle `MemoryGraph` instance (graph `G = (V, E)`) that stores nodes `V` and typed edges `E`.
- **Transcript_Store**: The verbatim transcript evidence store queried via the `verbatim` retrieval mode.
- **Context_Pack**: The assembled string output `C` produced by RMCA, where `estimated_tokens(C) ≤ B × 1.15`.
- **Token_Budget**: The integer `B` specifying the maximum desired token count for a Context_Pack.
- **Ablation_Variant**: A named configuration of RMCA that disables one component via a runtime flag, used to measure that component's contribution.
- **Benchmark_Family**: A named class of evaluation tasks sharing a common information-need structure (e.g., S-NIAH, OOLONG-Pairs, ContextReset).
- **Deterministic_Answerer**: A rule-based, model-free function that extracts a final answer from a Context_Pack using string matching and heuristics, enabling answer-level evaluation without an LLM.
- **RLM_Benchmark**: The five benchmark families adapted from the Recursive Language Models paper (Zhang et al., 2026), used as the primary evaluation suite.
- **Synthetic_Caveat**: The documented limitation that current benchmark tasks use deterministic synthetic Waggle memory rather than the exact public datasets from the RLM paper; numerical results must not be compared to RLM paper figures until real datasets are used.
- **Budget_Scaling**: An experiment that varies Token_Budget across a fixed set of values and measures score, evidence coverage, latency, and tokens returned.
- **Failure_Analysis**: A structured examination of cases where RMCA does not outperform baselines, with documented explanations and research implications.
- **Research_Report**: The auto-generated Markdown document `docs/research/rmca_experiment_report.md` that aggregates all experimental results into paper-ready sections.
- **BM25_Baseline**: A retrieval baseline using BM25 term-frequency scoring over the memory graph node contents.
- **Vector_Baseline**: A retrieval baseline using cosine similarity over deterministic embeddings.
- **Hybrid_RRF_Baseline**: A retrieval baseline combining BM25 and vector scores via Reciprocal Rank Fusion.
- **Graph_Expansion_Baseline**: A retrieval baseline that performs single-hop graph expansion without recursive subquery decomposition.
- **Summary_Memory_Baseline**: A retrieval baseline that returns only the `prime_context` summary without graph traversal.
- **Full_Transcript_Baseline**: A retrieval baseline that returns the most recent transcript entries truncated to the token budget.
- **ContextReset_Family**: A new benchmark family where session 1 stores project state and session 2 starts fresh and asks "Continue from where we left off."
- **Gold_Fields**: The set of required answer fields in a benchmark case: `decision_recall`, `constraint_recall`, `next_step_accuracy`, `superseded_context_handling`, `evidence_coverage`, `tokens_returned`, `latency_ms`.

---

## Requirements

### Requirement 1: Formal Method Document

**User Story:** As a researcher, I want a formal definition of the RMCA algorithm, so that reviewers can verify the method's novelty and reproducibility without reading source code.

#### Acceptance Criteria

1. THE Method_Document_Generator SHALL create the file `docs/research/rmca_method.md` containing a formal definition of RMCA with inputs `(q, G, T, B, d, k)` and output `C` where `estimated_tokens(C) ≤ B × 1.15`.
2. THE Method_Document_Generator SHALL include an Algorithm 1 pseudocode block covering all eight steps: decompose, retrieve, expand, resolve, deduplicate, rank, compress, format.
3. THE Method_Document_Generator SHALL include a section explaining how RMCA differs from top-k RAG (no subquery decomposition, no graph expansion, no conflict resolution) and from GraphRAG (no recursive decomposition, no token-budget compression, no verbatim evidence lane).
4. THE Method_Document_Generator SHALL define all symbols used in the formal definition in a notation table before first use.
5. THE Method_Document_Generator SHALL include the Synthetic_Caveat as a clearly labelled "Limitations and Scope" subsection.

---

### Requirement 2: Ablation Study Infrastructure

**User Story:** As a researcher, I want to run ablation variants of RMCA by toggling individual components off via flags, so that I can measure each component's contribution without duplicating code.

#### Acceptance Criteria

1. THE Ablation_Runner SHALL accept a `--variant` flag accepting values: `rmca_full`, `rmca_no_decomposition`, `rmca_no_graph_expansion`, `rmca_no_conflict_resolution`, `rmca_no_verbatim_evidence`, `rmca_no_budget_compression`, `rmca_random_subqueries`.
2. WHEN `--variant rmca_no_decomposition` is specified, THE Ablation_Runner SHALL replace subquery decomposition with a single pass using the original query verbatim.
3. WHEN `--variant rmca_no_graph_expansion` is specified, THE Ablation_Runner SHALL skip the graph expansion step (step 3 of RMCA) and use only direct retrieval results.
4. WHEN `--variant rmca_no_conflict_resolution` is specified, THE Ablation_Runner SHALL skip the update/conflict resolution step (step 4 of RMCA) and return all hits without superseded marking.
5. WHEN `--variant rmca_no_verbatim_evidence` is specified, THE Ablation_Runner SHALL exclude the verbatim retrieval mode from all subqueries and omit the Evidence section from the Context_Pack.
6. WHEN `--variant rmca_no_budget_compression` is specified, THE Ablation_Runner SHALL return all ranked hits concatenated without enforcing the Token_Budget limit.
7. WHEN `--variant rmca_random_subqueries` is specified, THE Ablation_Runner SHALL replace the deterministic subquery decomposition with randomly sampled substrings of the original query using a fixed seed.
8. THE Ablation_Runner SHALL write results to `benchmark_results/ablation_results.csv`, `benchmark_results/ablation_results.md`, and `benchmark_results/ablation_results.json`.
9. THE ablation report SHALL include a per-family delta column showing the score difference of each variant against `rmca_full` for the same family and scale.
10. THE ablation report SHALL include an annotation column explicitly naming which RMCA component is responsible for each observed gain or loss (e.g., "graph_expansion responsible for +1.000 on OOLONG-Pairs").
11. THE pairwise benchmark cases used in ablation SHALL be constructed so that `rmca_no_graph_expansion` and `rmca_no_conflict_resolution` each score strictly lower than `rmca_full`, verifying that graph expansion and conflict resolution are load-bearing components.
12. THE Ablation_Runner SHALL support `--help` and print usage information.
13. IF a partial ablation run is interrupted, THEN THE Ablation_Runner SHALL write completed rows to the output files before exiting, preserving partial results.

---

### Requirement 3: ContextReset Benchmark Family

**User Story:** As a researcher, I want a benchmark that tests whether RMCA can restore project state across a session boundary, so that I can evaluate the "continue from where we left off" use case that motivates Waggle's design.

#### Acceptance Criteria

1. THE ContextReset_Generator SHALL create benchmark cases where session 1 stores at least one decision node, one constraint node, one in-progress task node, one superseded node, and one next-step node in the Memory_Graph.
2. THE ContextReset_Generator SHALL create a session 2 query of the form "Continue from where we left off" with no additional context provided.
3. WHEN a ContextReset case is generated, THE ContextReset_Generator SHALL record gold fields: `decision_recall` (fraction of session-1 active decisions present in Context_Pack), `constraint_recall` (fraction of session-1 constraints present), `next_step_accuracy` (1.0 if the next-step node appears in Context_Pack), `superseded_context_handling` (1.0 if superseded node is marked superseded or absent from active sections), `evidence_coverage` (fraction of gold node IDs present), `tokens_returned`, `latency_ms`.
4. THE ContextReset_Generator SHALL support two difficulty levels:
   - **easy**: one decision, one constraint, one next-step node, and distractor memories from one unrelated project.
   - **hard**: multiple decisions (at least three), one superseded decision linked via an `updates` edge to an active decision, one `contradicts` edge between two choices, one rejected-direction node, one bug node, one next-step node, and distractor memories from at least two unrelated projects.
5. THE ContextReset_Benchmark SHALL score `active_decision_preference`: 1.0 if the method returns the latest active decision (the source of the `updates` edge) rather than the superseded one, 0.0 if it returns only the superseded decision, and 0.5 if it returns both without marking one as superseded.
6. THE ContextReset_Benchmark SHALL evaluate all eight methods: `no_memory`, `raw_context`, `bm25_topk`, `vector_topk`, `hybrid`, `query_graph`, `prime_context`, `rmca_full`.
7. WHEN `no_memory` method is evaluated, THE ContextReset_Benchmark SHALL return an empty string as the Context_Pack.
8. THE ContextReset_Benchmark SHALL support `--scales` to vary the number of session-1 nodes at scales 128, 512, and 2048.
9. THE ContextReset_Benchmark SHALL write results to `benchmark_results/partial/context_reset/` using the same CSV/MD/JSON format as existing benchmark families.
10. THE ContextReset_Benchmark SHALL support `--help` and print usage information.

---

### Requirement 4: Stronger Baselines

**User Story:** As a researcher, I want stronger retrieval baselines beyond `raw_context` and `query_graph`, so that RMCA's advantages are measured against competitive alternatives rather than weak baselines.

#### Acceptance Criteria

1. THE Baseline_Runner SHALL implement `bm25_topk`: retrieve the top-k nodes by BM25 score over node label and content, truncated to Token_Budget.
2. THE Baseline_Runner SHALL implement `vector_topk`: retrieve the top-k nodes by cosine similarity using the existing `_DeterministicEmbedding` model, truncated to Token_Budget.
3. THE Baseline_Runner SHALL implement `hybrid_rrf`: combine BM25 and vector rankings using Reciprocal Rank Fusion with `k=60`, truncated to Token_Budget.
4. THE Baseline_Runner SHALL implement `graph_expansion_no_recursion`: perform a single `query_graph` call followed by one hop of graph expansion, with no subquery decomposition, truncated to Token_Budget.
5. THE Baseline_Runner SHALL implement `summary_memory`: return only the `prime_context` summary string, with no graph traversal.
6. THE Baseline_Runner SHALL implement `full_transcript_truncation`: return the most recent transcript entries concatenated in reverse-chronological order, truncated to Token_Budget.
7. THE Baseline_Runner SHALL use only the existing `_DeterministicEmbedding` model and require no external ML library beyond numpy.
8. THE Baseline_Runner SHALL register all six new baselines in the `_METHOD_RUNNERS` dispatch table so they are available to all benchmark families via the `--methods` flag.
9. IF a baseline method raises an exception during evaluation, THEN THE Baseline_Runner SHALL log the exception at DEBUG level and return an empty string as the Context_Pack for that case.

---

### Requirement 5: Answer-Level Evaluation Pipeline

**User Story:** As a researcher, I want to measure final answer quality rather than only context retrieval quality, so that I can report metrics that directly reflect downstream task performance while remaining fully reproducible offline.

#### Acceptance Criteria

1. THE Answer_Level_Evaluator SHALL implement a `Deterministic_Answerer` that extracts a final answer from a Context_Pack using rule-based string matching with no external model API.
2. THE `Deterministic_Answerer` SHALL be explicitly described in all outputs and the research report as a **reproducible lower-bound evaluator**, not a substitute for human preference judgement or LLM-judge quality assessment.
3. THE Answer_Level_Evaluator SHALL compute `final_answer_exact_match`: 1.0 if the gold answer string appears verbatim (case-insensitive) in the extracted answer.
4. THE Answer_Level_Evaluator SHALL compute `final_answer_f1`: token-level F1 between the extracted answer tokens and the gold answer tokens.
5. THE Answer_Level_Evaluator SHALL compute `evidence_used`: fraction of gold evidence node IDs whose content appears in the Context_Pack.
6. THE Answer_Level_Evaluator SHALL compute `contradiction_correctness`: 1.0 if the Context_Pack correctly identifies all gold conflict pairs and 0.0 if it identifies none, with linear interpolation for partial identification.
7. THE Answer_Level_Evaluator SHALL compute `hallucination_rate`: fraction of sentences in the extracted answer that contain no token from any node in the Memory_Graph (approximated by checking against all node label and content tokens).
8. THE Answer_Level_Evaluator SHALL compute `tokens_injected`: the token count of the Context_Pack passed to the Deterministic_Answerer.
9. THE Answer_Level_Evaluator SHOULD support an optional `--answerer ollama` flag that, when specified, routes answer extraction through a local Ollama model instead of the Deterministic_Answerer. This flag SHALL be disabled by default and SHALL require a running Ollama instance; if Ollama is unavailable, the evaluator SHALL fall back to the Deterministic_Answerer and log a warning.
10. THE Answer_Level_Evaluator SHALL be implemented in `benchmarks/answer_level_eval.py` and support `--help`.
11. THE Answer_Level_Evaluator SHALL accept `--methods`, `--scales`, `--families`, `--output`, `--seed`, and `--answerer` flags.
12. THE Answer_Level_Evaluator SHALL write results to `benchmark_results/answer_level_results.csv`, `benchmark_results/answer_level_results.md`, and `benchmark_results/answer_level_results.json`.
13. THE research report and all answer-level output files SHALL include the following disclaimer: *"Deterministic answer-level metrics are reproducible lower bounds. They are not equivalent to human preference ratings or LLM-judge quality assessments. Scores should be interpreted as retrieval-quality proxies, not end-to-end answer quality."*

---

### Requirement 6: Budget Scaling Experiment

**User Story:** As a researcher, I want to measure how RMCA performance changes as the token budget varies, so that I can characterise the efficiency frontier and recommend practical budget settings.

#### Acceptance Criteria

1. THE Budget_Scaling_Runner SHALL evaluate all methods at token budgets 250, 500, 1000, 2000, and 4000.
2. THE Budget_Scaling_Runner SHALL evaluate the benchmark families: `context_reset`, `pairwise` (OOLONG-Pairs-style), `linear_agg` (OOLONG-style), and `codeqa` (CodeQA-style).
3. THE Budget_Scaling_Runner SHALL record per-row metrics: `score`, `evidence_coverage`, `latency_ms`, `tokens_returned`, `token_budget`.
4. THE Budget_Scaling_Runner SHALL write results to `benchmark_results/budget_scaling_results.csv`, `benchmark_results/budget_scaling_results.md`, and `benchmark_results/budget_scaling_results.json`.
5. THE Budget_Scaling_Runner SHALL generate four charts: `score_vs_budget.png`, `evidence_coverage_vs_budget.png`, `latency_vs_budget.png`, `tokens_returned_vs_budget.png` in `benchmark_results/charts/`.
6. IF matplotlib is not installed, THEN THE Budget_Scaling_Runner SHALL print ASCII tables instead of generating PNG charts and exit with code 0.
7. THE Budget_Scaling_Runner SHALL use a fixed random seed (default 42) and accept `--seed` to override it.
8. THE Budget_Scaling_Runner SHALL be implemented in `benchmarks/run_budget_scaling.py` and support `--help`.
9. IF a partial run is interrupted, THEN THE Budget_Scaling_Runner SHALL write completed rows to the output CSV before exiting.

---

### Requirement 7: Failure Analysis

**User Story:** As a researcher, I want an automated failure analysis that identifies where RMCA does not outperform baselines and explains why, so that the paper's limitations section is grounded in evidence rather than speculation.

#### Acceptance Criteria

1. THE Failure_Analyzer SHALL read all result CSV files from `benchmark_results/` and `benchmark_results/partial/` matching the pattern `*_results.csv`.
2. THE Failure_Analyzer SHALL identify cases where `build_context` (rmca_full) score is not strictly greater than the best baseline score for the same family, scale, and metric.
3. THE Failure_Analyzer SHALL write `benchmark_results/failure_analysis.md` containing: a summary table of wins/losses/ties per family, a section explaining why OOLONG linear aggregation remains hard (token budget cannot cover all N entries at large scale), a section explaining why CodeQA does not prove RMCA beats `query_graph` (both score 1.0 on synthetic tasks), and a "Research Implications" section.
4. THE Failure_Analyzer SHALL include the Synthetic_Caveat in the failure analysis document.
5. THE Failure_Analyzer SHALL be implemented in `benchmarks/failure_analysis.py` and support `--help`.
6. THE Failure_Analyzer SHALL accept `--results-dir` and `--output` flags.
7. IF no result CSV files are found, THEN THE Failure_Analyzer SHALL write a stub `failure_analysis.md` noting that no results are available and exit with code 0.

---

### Requirement 8: Research Report Generator

**User Story:** As a researcher, I want a single command that generates a complete, paper-ready research report from all experimental results, so that I can iterate on experiments and immediately see the updated narrative.

#### Acceptance Criteria

1. THE Report_Generator SHALL generate `docs/research/rmca_experiment_report.md` containing all of the following sections in order: Abstract, Method, Benchmark Tasks, Main Results, Ablations, Context-Reset, Budget Scaling, Answer-Level Evaluation, Failure Analysis, Limitations, Reproducibility Commands.
2. THE Report_Generator SHALL populate the Main Results section with a Markdown table of scores for all methods across all families and scales, sourced from `benchmark_results/rlm_style_waggle_results.csv` if present.
3. THE Report_Generator SHALL populate the Ablations section with a Markdown table sourced from `benchmark_results/ablation_results.csv` if present, or a placeholder noting the ablation has not been run.
4. THE Report_Generator SHALL populate the Context-Reset section with results sourced from `benchmark_results/partial/context_reset/` if present, or a placeholder.
5. THE Report_Generator SHALL populate the Budget Scaling section with results sourced from `benchmark_results/budget_scaling_results.csv` if present, or a placeholder.
6. THE Report_Generator SHALL populate the Answer-Level Evaluation section with results sourced from `benchmark_results/answer_level_results.csv` if present, or a placeholder.
7. THE Report_Generator SHALL include the full text of `benchmark_results/failure_analysis.md` in the Failure Analysis section if present, or a placeholder.
8. THE Report_Generator SHALL include the Synthetic_Caveat in the Limitations section.
9. THE Report_Generator SHALL include a Reproducibility Commands section listing the exact CLI commands needed to reproduce all experiments, with deterministic seeds.
10. THE Report_Generator SHALL be implemented in `benchmarks/make_research_report.py` and support `--help`.
11. THE Report_Generator SHALL accept `--output` to override the default output path.
12. WHEN the report is generated, THE Report_Generator SHALL print the output path to stdout.

---

### Requirement 10: Statistical Robustness

**User Story:** As a researcher, I want benchmark results reported with variance across multiple seeds, so that reviewers cannot dismiss findings as single-seed artefacts.

#### Acceptance Criteria

1. ALL benchmark runners (ablation, context_reset, budget_scaling, answer_level_eval, rlm_style_waggle_eval) SHOULD accept a `--seeds` flag that accepts a space-separated list of integer seeds (e.g., `--seeds 42 43 44`).
2. WHEN multiple seeds are provided, THEN each runner SHALL execute the full benchmark once per seed and aggregate results.
3. WHEN multiple seeds are provided, THEN all output CSV files SHALL include `seed`, `mean_score`, and `std_score` columns in addition to per-seed rows.
4. WHEN multiple seeds are provided, THEN all output Markdown tables SHALL display scores as `mean ± std` (e.g., `0.923 ± 0.041`).
5. THE default single-seed value SHALL remain 42 for quick runs; the recommended paper-quality run SHALL use at least three seeds (42, 43, 44).
6. THE research report SHALL note the number of seeds used for each experiment and include the disclaimer: *"Single-seed results are provided for quick reproducibility. Paper-quality claims should be verified with ≥3 seeds."*
7. IF only one seed is provided, THEN runners SHALL behave identically to the existing single-seed behaviour with no `std` column emitted.

---

**User Story:** As a researcher, I want a test suite that verifies all new research infrastructure components, so that I can trust the experimental results and reproduce them reliably.

#### Acceptance Criteria

1. THE Test_Suite SHALL include a test that runs all seven ablation variants and verifies each produces a non-empty Context_Pack for a fixed query and Memory_Graph.
2. THE Test_Suite SHALL include a test that verifies `rmca_no_graph_expansion` produces a different pairwise benchmark score than `rmca_full` when the gold conflict pairs require graph edge traversal to discover.
3. THE Test_Suite SHALL include a test that verifies `rmca_no_conflict_resolution` produces a Context_Pack that does not contain the string "Conflicts or superseded context" when conflict edges are present.
4. THE Test_Suite SHALL include a test that verifies a generated ContextReset case (both easy and hard difficulty) contains all required Gold_Fields: `decision_recall`, `constraint_recall`, `next_step_accuracy`, `superseded_context_handling`, `active_decision_preference`, `evidence_coverage`, `tokens_returned`, `latency_ms`.
5. THE Test_Suite SHALL include a test that runs `run_budget_scaling.py` with `--scales 128` and `--budgets 250 500` and verifies that output CSV and JSON files are written to the specified output directory.
6. THE Test_Suite SHALL include a test that verifies the Deterministic_Answerer returns a non-empty string for a Context_Pack containing a known gold answer.
7. THE Test_Suite SHALL include a test that verifies `failure_analysis.py` produces a Markdown file that contains the substring "OOLONG" when run against the existing `benchmark_results/rlm_style_waggle_results.csv`.
8. THE Test_Suite SHALL include a test that verifies `make_research_report.py` produces a Markdown file containing all required section headings: "Abstract", "Method", "Benchmark Tasks", "Main Results", "Ablations", "Context-Reset", "Budget Scaling", "Answer-Level Evaluation", "Failure Analysis", "Limitations", "Reproducibility Commands".
9. THE Test_Suite SHALL include a test that verifies each of the following CLIs exits with code 0 when invoked with `--help`: `benchmarks/run_budget_scaling.py`, `benchmarks/answer_level_eval.py`, `benchmarks/failure_analysis.py`, `benchmarks/make_research_report.py`.
10. THE Test_Suite SHALL include a test that runs any one benchmark runner with `--seeds 42 43` and verifies the output CSV contains a `seed` column and at least two distinct seed values.
11. THE Test_Suite SHALL use only deterministic seeds and require no internet access or external model API.
12. THE Test_Suite SHALL use the existing `_DeterministicEmbedding` model for all tests that require embeddings.
