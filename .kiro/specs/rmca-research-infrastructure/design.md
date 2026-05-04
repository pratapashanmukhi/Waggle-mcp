# Design Document: RMCA Research Infrastructure

## Overview

This document describes the technical design for turning Waggle MCP's existing Recursive Context Assembly (RMCA) implementation into defensible research evidence. The goal is a complete, offline-runnable research infrastructure: formal method documentation, ablation studies, stronger baselines, a new benchmark family, answer-level evaluation, budget scaling analysis, failure analysis, and an automated research report generator.

**Design constraints honoured throughout:**
- Minimal changes to `src/waggle/recursive_context.py` — only add `AblationConfig` dataclass and a thin flag-check layer inside `build_context`
- Variant flags instead of copy-paste controllers
- Shared `BenchResult` schema extended with new optional columns (backward-compatible)
- Deterministic offline execution; no external model API required
- Partial-run safe output writing via `atexit` + `try/finally`
- No product features — research evidence only

---

## Architecture

The nine components form three layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Core RMCA (minimal changes)                          │
│  src/waggle/recursive_context.py                                │
│    RecursiveContextController.build_context(ablation=...)       │
│    AblationConfig dataclass (new, 6 boolean flags)              │
└─────────────────────────────────────────────────────────────────┘
         │ used by
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — Benchmark infrastructure                             │
│  benchmarks/rlm_style_waggle_eval.py  (extended)                │
│    _METHOD_RUNNERS  ← 6 new baselines added                     │
│    generate_context_reset_cases()  (new)                        │
│    run_context_reset_benchmark()   (new)                        │
│                                                                 │
│  benchmarks/run_ablation.py         (new)                       │
│  benchmarks/run_budget_scaling.py   (new)                       │
│  benchmarks/answer_level_eval.py    (new)                       │
│  benchmarks/failure_analysis.py     (new)                       │
│  benchmarks/make_research_report.py (new)                       │
└─────────────────────────────────────────────────────────────────┘
         │ produces
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Outputs                                              │
│  benchmark_results/ablation_results.{csv,md,json}               │
│  benchmark_results/budget_scaling_results.{csv,md,json}         │
│  benchmark_results/answer_level_results.{csv,md,json}           │
│  benchmark_results/failure_analysis.md                          │
│  benchmark_results/partial/context_reset/                       │
│  docs/research/rmca_method.md                                   │
│  docs/research/rmca_experiment_report.md                        │
└─────────────────────────────────────────────────────────────────┘
```

The design deliberately avoids a new orchestration layer. Every new runner is a standalone CLI script that imports from `rlm_style_waggle_eval` and `recursive_context` — the same pattern used by `run_full_benchmark.py`.

---

## Components and Interfaces

### Component 1: Formal Method Document (`docs/research/rmca_method.md`)

A static Markdown file written once. No code generation needed. Structure:

1. **Abstract** — one-paragraph summary of RMCA
2. **Notation Table** — defines `q`, `G=(V,E)`, `T`, `B`, `d`, `k`, `C`, `estimated_tokens(·)`
3. **Algorithm 1** — pseudocode block with all eight steps labelled
4. **Comparison with top-k RAG** — three bullet differences
5. **Comparison with GraphRAG** — three bullet differences
6. **Limitations and Scope** — contains the Synthetic_Caveat verbatim

---

### Component 2: Ablation Infrastructure

#### 2a. `AblationConfig` dataclass (added to `src/waggle/recursive_context.py`)

```python
@dataclass
class AblationConfig:
    """Runtime flags for ablation study variants. All True = full RMCA."""
    decompose: bool = True           # Step 1: subquery decomposition
    graph_expand: bool = True        # Step 3: graph expansion
    conflict_resolve: bool = True    # Step 4: update/conflict resolution
    verbatim_evidence: bool = True   # Step 2: verbatim retrieval mode
    budget_compress: bool = True     # Step 7: token budget enforcement
    random_subqueries: bool = False  # Step 1 override: random substrings
    random_seed: int = 42            # seed for random_subqueries mode
```

`AblationConfig` is added as an optional parameter to `build_context`:

```python
def build_context(
    self,
    query: str,
    ...,
    ablation: AblationConfig | None = None,
) -> RecursiveContextResult:
```

When `ablation` is `None`, behaviour is identical to the current implementation (no regression). Each flag gates exactly one step:

| Flag | Step gated | Behaviour when False |
|---|---|---|
| `decompose=False` | Step 1 | Single subquery = original query verbatim |
| `graph_expand=False` | Step 3 | Skip `_expand_graph()` entirely |
| `conflict_resolve=False` | Step 4 | Skip `_resolve_updates_and_conflicts()`, return all hits unmodified |
| `verbatim_evidence=False` | Step 2 | Remove `"verbatim"` from all subquery retrieval modes |
| `budget_compress=False` | Step 7 | Concatenate all ranked hits without budget check |
| `random_subqueries=True` | Step 1 | Replace decomposition with random substrings of query |

The `random_subqueries` flag takes precedence over `decompose` when both are set.

#### 2b. `benchmarks/run_ablation.py`

CLI interface:

```
python benchmarks/run_ablation.py \
  --variant rmca_full \
  --families pairwise codeqa context_reset \
  --scales 128 512 2048 \
  --seed 42 \
  --output benchmark_results/
```

`--variant` maps to an `AblationConfig`:

```python
VARIANT_CONFIGS: dict[str, AblationConfig] = {
    "rmca_full":                  AblationConfig(),
    "rmca_no_decomposition":      AblationConfig(decompose=False),
    "rmca_no_graph_expansion":    AblationConfig(graph_expand=False),
    "rmca_no_conflict_resolution":AblationConfig(conflict_resolve=False),
    "rmca_no_verbatim_evidence":  AblationConfig(verbatim_evidence=False),
    "rmca_no_budget_compression": AblationConfig(budget_compress=False),
    "rmca_random_subqueries":     AblationConfig(random_subqueries=True),
}
```

The runner instantiates `RecursiveContextController` with the selected config and runs it through the existing `_BENCHMARK_RUNNERS` dispatch table. It adds a thin wrapper runner function `_run_ablation_variant(graph, query, token_budget, config)` that is registered in `_METHOD_RUNNERS` for the duration of the ablation run.

**Delta computation:** After all variants complete, the runner computes `delta = variant_score - rmca_full_score` per `(family, scale)` and writes it as an additional column. An `annotation` column maps each delta to the responsible component name (e.g., `"graph_expand responsible for +1.000 on OOLONG-Pairs"`).

**Output files:** `benchmark_results/ablation_results.csv`, `.md`, `.json`

**Partial-run safety:** Uses `atexit.register(_flush_partial, results_list, output_dir)` so that completed rows are written even on `KeyboardInterrupt` or timeout.

---

### Component 3: ContextReset Benchmark Family

Added to `benchmarks/rlm_style_waggle_eval.py` as new generator and runner functions.

#### `ContextResetCase` dataclass

```python
@dataclass
class ContextResetCase:
    case_id: str
    question: str                    # always "Continue from where we left off"
    difficulty: str                  # "easy" | "hard"
    gold_decision_ids: list[str]
    gold_constraint_ids: list[str]
    gold_next_step_id: str
    gold_superseded_id: str
    gold_active_decision_id: str     # source of the updates edge (hard only)
    scale_n: int
```

#### Difficulty parameterisation

**Easy:** 1 decision node, 1 constraint node, 1 next-step node (as a `question`-type node), 1 superseded node (linked via `updates` edge), distractors from 1 unrelated project to reach `scale_n`.

**Hard:** ≥3 decision nodes, 1 superseded decision linked via `updates` to an active decision, 1 `contradicts` edge between two choices, 1 rejected-direction node (tagged `rejected`), 1 bug node (`NodeType.NOTE` tagged `bug`), 1 next-step node, distractors from ≥2 unrelated projects.

The `updates` edge is always `source=active_decision, target=superseded_decision` (consistent with the existing RMCA convention: source updates target → target is superseded).

#### Scoring fields

The runner computes six scoring fields per result row:

| Field | Computation |
|---|---|
| `decision_recall` | `len(gold_decision_ids found in pack) / len(gold_decision_ids)` |
| `constraint_recall` | `len(gold_constraint_ids found in pack) / len(gold_constraint_ids)` |
| `next_step_accuracy` | `1.0` if gold next-step node label appears in pack, else `0.0` |
| `superseded_context_handling` | `1.0` if superseded node is absent from active sections OR marked `[superseded]`; `0.0` if it appears as active |
| `active_decision_preference` | `1.0` active only, `0.5` both present, `0.0` superseded only |
| `evidence_coverage` | fraction of all gold node IDs present in pack |

These are stored in the `notes` field of `BenchResult` as a JSON string, and also written as separate columns in the context_reset output CSV (extended schema).

#### Eight methods evaluated

`no_memory`, `raw_context`, `bm25_topk`, `vector_topk`, `hybrid_rrf`, `query_graph`, `prime_context`, `rmca_full`

`no_memory` returns `""` as the context pack.

**Output:** `benchmark_results/partial/context_reset/` using the same CSV/MD/JSON format.

---

### Component 4: Stronger Baselines

Six new runner functions added to `_METHOD_RUNNERS` in `benchmarks/rlm_style_waggle_eval.py`. All use only `_DeterministicEmbedding` and `SimpleBM25` (already in `retrieval/hybrid.py`).

#### `_run_bm25_topk(graph, query, token_budget)`

1. Load all nodes from the graph via `graph.get_stats()` + `graph.aggregate()`
2. Build `SimpleBM25({node.id: tokenize_text(node.label + " " + node.content) for node in nodes})`
3. Score query, take top-k by BM25 score
4. Concatenate `[node_type] label: content` lines until token budget exhausted

#### `_run_vector_topk(graph, query, token_budget)`

1. Embed query with `_DeterministicEmbedding().embed(query)`
2. Load all nodes with embeddings from the graph
3. Compute cosine similarity for each node
4. Take top-k, concatenate until budget

#### `_run_hybrid_rrf(graph, query, token_budget)`

1. Get BM25 ranking and vector ranking independently
2. Fuse with RRF: `score(d) = Σ 1/(k + rank_i(d))` where `k=60`
3. Take top-k by fused score, concatenate until budget

#### `_run_graph_expansion_no_recursion(graph, query, token_budget)`

1. Single `graph.query(query, max_nodes=20, max_depth=1, retrieval_mode="hybrid")`
2. For each returned node, fetch one hop of neighbours via `graph.get_related(node_id, max_depth=1)`
3. Deduplicate, concatenate until budget
4. No subquery decomposition

#### `_run_summary_memory(graph, query, token_budget)`

1. Call `graph.prime_context()`
2. Return `result.summary` (no graph traversal)
3. Truncate to budget if needed

#### `_run_full_transcript_truncation(graph, query, token_budget)`

1. Load transcript records in reverse-chronological order
2. Concatenate `role: text` lines until budget exhausted
3. No graph traversal

**Error handling:** All six runners wrap their body in `try/except Exception as exc: LOGGER.debug(...); return "", latency` — consistent with existing runners.

---

### Component 5: Answer-Level Evaluation (`benchmarks/answer_level_eval.py`)

#### `AnswerLevelResult` dataclass

```python
@dataclass
class AnswerLevelResult:
    benchmark_family: str
    scale_n: int
    method: str
    answerer: str                    # "deterministic" | "ollama"
    final_answer_exact_match: float
    final_answer_f1: float
    evidence_used: float
    contradiction_correctness: float
    hallucination_rate: float
    tokens_injected: int
    seed: int = 42
    notes: str = ""
```

#### `DeterministicAnswerer`

Rule-based string extractor. Labelled as lower-bound evaluator in all outputs.

**Extraction algorithm:**
1. Look for lines starting with `- [decision]`, `- [fact]`, `- [preference]` in the Context_Pack
2. Extract the content after the colon on each such line
3. If the gold answer is a short string (≤ 10 tokens), scan all extracted lines for the first line containing all gold tokens
4. Return that line as the extracted answer, or the first non-empty extracted line if no gold match

This is intentionally simple — it is a lower bound, not a quality evaluator.

#### `OllamaAnswerer` (optional, disabled by default)

```python
class OllamaAnswerer:
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        ...
    def extract(self, context_pack: str, question: str) -> str:
        # POST to /api/generate, timeout=30s
        # Falls back to DeterministicAnswerer on any exception
```

Activated only with `--answerer ollama`. Falls back to `DeterministicAnswerer` with a logged warning if Ollama is unavailable.

#### Metric implementations

| Metric | Implementation |
|---|---|
| `final_answer_exact_match` | `1.0 if gold.lower() in extracted.lower() else 0.0` |
| `final_answer_f1` | Token-level F1: `tokenize_text(extracted)` vs `tokenize_text(gold)` using existing `tokenize_text` from `waggle.intelligence` |
| `evidence_used` | Fraction of gold node IDs whose `label: content` appears in the context pack |
| `contradiction_correctness` | Linear interpolation: `len(found_conflict_pairs) / len(gold_conflict_pairs)` |
| `hallucination_rate` | Fraction of sentences in extracted answer with no token overlap with any node's label+content |
| `tokens_injected` | `token_estimate(context_pack)` using existing 4-chars-per-token heuristic |

**Disclaimer** (included in all output files and the research report):
> *"Deterministic answer-level metrics are reproducible lower bounds. They are not equivalent to human preference ratings or LLM-judge quality assessments. Scores should be interpreted as retrieval-quality proxies, not end-to-end answer quality."*

**CLI:**
```
python benchmarks/answer_level_eval.py \
  --methods rmca_full bm25_topk \
  --scales 128 512 \
  --families pairwise context_reset \
  --seed 42 \
  --answerer deterministic \
  --output benchmark_results/
```

---

### Component 6: Budget Scaling (`benchmarks/run_budget_scaling.py`)

Loops over `budgets × families × methods`, reusing existing `_METHOD_RUNNERS` and benchmark generators.

**Default budgets:** `[250, 500, 1000, 2000, 4000]`
**Default families:** `context_reset`, `pairwise`, `linear_agg`, `codeqa`

**Per-row output schema** (extends `BenchResult` with `token_budget` column):

```
benchmark_family, scale_n, method, token_budget, score, evidence_coverage,
latency_ms, tokens_returned, seed
```

**Partial-run safety:** `atexit.register(_flush_partial, rows, output_path)` — same pattern as ablation runner.

**Chart generation** (4 charts, matplotlib or ASCII fallback):
- `score_vs_budget.png` — score vs token_budget, one line per method, faceted by family
- `evidence_coverage_vs_budget.png`
- `latency_vs_budget.png`
- `tokens_returned_vs_budget.png`

Chart generation reuses the style constants from `plot_rlm_results.py` (`METHOD_COLORS`, `METHOD_MARKERS`, `METHOD_LABELS`).

---

### Component 7: Failure Analysis (`benchmarks/failure_analysis.py`)

**Input:** All `*_results.csv` files found under `benchmark_results/` and `benchmark_results/partial/` (recursive glob).

**Win/loss/tie classification:** For each `(family, scale)` group, compare `build_context` (or `rmca_full`) score against `max(score for method != build_context)`. Classify as:
- **Win:** `rmca_score > best_baseline_score`
- **Tie:** `rmca_score == best_baseline_score`
- **Loss:** `rmca_score < best_baseline_score`

**Output structure** (`benchmark_results/failure_analysis.md`):

1. Summary table: wins/losses/ties per family
2. Section: "Why OOLONG linear aggregation remains hard" — token budget cannot cover all N entries at large scale; this is a fundamental O(n) information need
3. Section: "Why CodeQA does not prove RMCA beats query_graph" — both score 1.0 on synthetic tasks; the task is too easy for the synthetic data
4. Section: "Research Implications" — what these failures mean for the paper's claims
5. Synthetic_Caveat

**Stub behaviour:** If no CSV files found, writes a stub noting no results are available and exits 0.

---

### Component 8: Research Report Generator (`benchmarks/make_research_report.py`)

Reads all result CSVs and the failure analysis, generates `docs/research/rmca_experiment_report.md`.

**Section population logic:**

| Section | Source | Fallback |
|---|---|---|
| Abstract | Hardcoded template | — |
| Method | Inline from design | — |
| Benchmark Tasks | Hardcoded descriptions | — |
| Main Results | `benchmark_results/rlm_style_waggle_results.csv` | Placeholder |
| Ablations | `benchmark_results/ablation_results.csv` | Placeholder |
| Context-Reset | `benchmark_results/partial/context_reset/*.csv` | Placeholder |
| Budget Scaling | `benchmark_results/budget_scaling_results.csv` | Placeholder |
| Answer-Level Evaluation | `benchmark_results/answer_level_results.csv` | Placeholder |
| Failure Analysis | `benchmark_results/failure_analysis.md` | Placeholder |
| Limitations | Hardcoded + Synthetic_Caveat | — |
| Reproducibility Commands | Generated from known CLI patterns | — |

The Reproducibility Commands section lists exact commands with `--seed 42` for all experiments.

---

### Component 9: Test Suite (`tests/test_rmca_research.py`)

12 test cases using `pytest`, `tempfile`, and `_DeterministicEmbedding`. No internet, no external model API.

---

## Data Models

### Extended `BenchResult` (backward-compatible)

The existing `BenchResult` dataclass gains optional fields with defaults so existing CSV readers continue to work:

```python
@dataclass
class BenchResult:
    # --- existing fields (unchanged) ---
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
    # --- new optional fields ---
    seed: int = 42
    token_budget: int = 0            # populated by budget_scaling runner
    ablation_variant: str = ""       # populated by ablation runner
    delta_vs_full: float = 0.0       # populated by ablation runner
    annotation: str = ""             # populated by ablation runner
    mean_score: float = 0.0          # populated when --seeds has >1 value
    std_score: float = 0.0           # populated when --seeds has >1 value
```

The `write_results()` function is updated to write only the fields that are non-default, keeping the CSV schema stable for existing consumers.

### `AnswerLevelResult` dataclass

Defined in `benchmarks/answer_level_eval.py` (see Component 5 above). Written to a separate CSV; not merged into `BenchResult`.

### `ContextResetCase` dataclass

Defined in `benchmarks/rlm_style_waggle_eval.py` (see Component 3 above). The scoring fields are stored in `BenchResult.notes` as a JSON string for the main CSV, and as separate columns in the context_reset-specific CSV.

---

## Multi-Seed Aggregation

When `--seeds 42 43 44` is provided:

1. Each runner executes the full benchmark once per seed, producing per-seed `BenchResult` rows with `seed=N`
2. After all seeds complete, an aggregation pass groups rows by `(benchmark_family, scale_n, method, token_budget, ablation_variant)` and computes `mean_score` and `std_score`
3. The output CSV contains both the per-seed rows (with `seed` column) and an aggregated summary row (with `seed=-1`, `mean_score`, `std_score`)
4. Markdown tables display `mean ± std` format for aggregated rows
5. Single-seed runs (`--seeds 42` or `--seed 42`) emit no `std_score` column — identical to current behaviour

---

## Partial-Run Safety Pattern

All new runners use the same two-layer safety pattern:

```python
import atexit

_partial_results: list[BenchResult] = []
_partial_output_dir: str = ""

def _flush_partial() -> None:
    if _partial_results and _partial_output_dir:
        _write_partial(_partial_results, _partial_output_dir)

atexit.register(_flush_partial)

def run_experiment(...):
    global _partial_results, _partial_output_dir
    _partial_output_dir = output_dir
    try:
        for config in experiment_configs:
            result = run_one(config)
            _partial_results.append(result)
    finally:
        _flush_partial()
```

`atexit` handles `KeyboardInterrupt` and normal exit. The `try/finally` handles exceptions within the loop. Partial results are written to `benchmark_results/partial/{runner_name}/` using the same CSV format as full results.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Ablation variants produce non-empty context packs

*For any* valid query string and non-empty `MemoryGraph`, all seven ablation variants (`rmca_full`, `rmca_no_decomposition`, `rmca_no_graph_expansion`, `rmca_no_conflict_resolution`, `rmca_no_verbatim_evidence`, `rmca_no_budget_compression`, `rmca_random_subqueries`) SHALL produce a non-empty `context_pack` string.

**Validates: Requirements 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**

---

### Property 2: Graph-dependent ablations score strictly lower than rmca_full on pairwise cases

*For any* pairwise benchmark case where the gold conflict pairs require traversal of `contradicts` edges to discover, the score of `rmca_no_graph_expansion` and the score of `rmca_no_conflict_resolution` SHALL each be strictly less than the score of `rmca_full`.

**Validates: Requirements 2.11**

---

### Property 3: ContextReset case generation invariant

*For any* generated `ContextResetCase` (at either easy or hard difficulty), the resulting `MemoryGraph` SHALL contain at least one node of each required type (decision, constraint, next-step, superseded), and the result dataclass SHALL contain all required gold fields (`decision_recall`, `constraint_recall`, `next_step_accuracy`, `superseded_context_handling`, `active_decision_preference`, `evidence_coverage`) with values in `[0.0, 1.0]`.

**Validates: Requirements 3.1, 3.3**

---

### Property 4: Active decision preference scoring is correct

*For any* context pack containing only the active decision node (the source of the `updates` edge), `active_decision_preference` SHALL equal `1.0`. *For any* context pack containing only the superseded decision node (the target of the `updates` edge), `active_decision_preference` SHALL equal `0.0`.

**Validates: Requirements 3.5**

---

### Property 5: Baseline runners handle any valid graph without raising

*For any* valid query string and `MemoryGraph` (including empty graphs), each of the six new baseline runners (`bm25_topk`, `vector_topk`, `hybrid_rrf`, `graph_expansion_no_recursion`, `summary_memory`, `full_transcript_truncation`) SHALL return a `(str, float)` tuple without raising an exception.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.9**

---

### Property 6: Answer-level scoring functions are correct

*For any* gold answer string `G` that appears verbatim in a context pack `P`, `final_answer_exact_match(P, G)` SHALL equal `1.0`. *For any* two identical token sequences `S`, `token_f1(S, S)` SHALL equal `1.0`. *For any* two token sequences with no common tokens, `token_f1` SHALL equal `0.0`.

**Validates: Requirements 5.3, 5.4**

---

### Property 7: Hallucination rate is zero for graph-sourced content

*For any* context pack composed entirely of node label and content strings from the `MemoryGraph`, `hallucination_rate` SHALL equal `0.0`.

**Validates: Requirements 5.7**

---

### Property 8: Failure analysis correctly classifies wins/losses/ties

*For any* result set where `build_context` score equals the best baseline score for a given `(family, scale)`, the failure analysis SHALL classify that row as a tie (not a win or loss).

**Validates: Requirements 7.2**

---

### Property 9: Research report always contains all required section headings

*For any* combination of present or absent result files (including the case where all result files are absent), the generated research report SHALL contain all eleven required section headings: "Abstract", "Method", "Benchmark Tasks", "Main Results", "Ablations", "Context-Reset", "Budget Scaling", "Answer-Level Evaluation", "Failure Analysis", "Limitations", "Reproducibility Commands".

**Validates: Requirements 8.1**

---

### Property 10: Multi-seed output contains seed column with distinct values

*For any* benchmark runner invoked with `--seeds S1 S2` where `S1 ≠ S2`, the output CSV SHALL contain a `seed` column and at least two rows with distinct seed values.

**Validates: Requirements 10.3**

---

## Error Handling

| Scenario | Handling |
|---|---|
| Baseline runner raises exception | `LOGGER.debug(exc)`, return `("", latency)` — consistent with existing runners |
| Ablation variant raises exception | Same as baseline; the row is written with `score=0.0` and `notes="error: ..."` |
| Ollama unavailable | Fall back to `DeterministicAnswerer`, log `WARNING` |
| No result CSVs found by failure_analysis | Write stub MD, exit 0 |
| matplotlib not installed | Print ASCII tables, exit 0 |
| Partial run interrupted | `atexit` + `try/finally` flush completed rows |
| Empty query to `build_context` | Existing guard returns `RecursiveContextResult(context_pack="No query provided.")` — unchanged |
| `AblationConfig.budget_compress=False` | No budget check; context pack may exceed budget — this is intentional and documented |

---

## Testing Strategy

### Unit tests (example-based)

- Verify all seven ablation variant names are accepted by `run_ablation.py --help`
- Verify ablation output CSV contains `delta_vs_full` and `annotation` columns
- Verify `DeterministicAnswerer` returns non-empty string for a pack containing a known gold answer
- Verify `failure_analysis.py` produces a Markdown file containing "OOLONG" when run against existing results CSV
- Verify `make_research_report.py` produces a file containing all 11 section headings
- Verify all four CLIs exit 0 with `--help`
- Verify budget scaling output CSV contains `token_budget` column

### Property-based tests

Property-based testing is appropriate here because:
- The ablation system has pure function behaviour: `AblationConfig` → `context_pack` string
- The scoring functions (`exact_match`, `token_f1`, `hallucination_rate`) are pure functions with clear universal properties
- The case generators have invariants that must hold for all generated cases
- The report generator has a structural invariant (all sections present) that holds regardless of input

**Library:** `hypothesis` (already available in the Python ecosystem; add to `requirements-dev.txt` if not present)

**Configuration:** Minimum 100 examples per property test. Each test tagged with:
```python
# Feature: rmca-research-infrastructure, Property N: <property_text>
@given(...)
@settings(max_examples=100)
def test_property_N_...:
```

### Integration tests

- Run `run_budget_scaling.py` with `--scales 128 --budgets 250 500` and verify CSV + JSON files are written
- Run any benchmark runner with `--seeds 42 43` and verify `seed` column with two distinct values
- Verify `rmca_no_graph_expansion` produces a different pairwise score than `rmca_full` on a graph with conflict edges

### Test file location

`tests/test_rmca_research.py` — 12 test cases covering all components, using `_DeterministicEmbedding` and `tempfile` throughout. No internet access, no external model API.
