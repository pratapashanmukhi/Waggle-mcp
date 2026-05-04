# Recursive Memory Context Assembly (RMCA)

## Abstract

Recursive Memory Context Assembly (RMCA) is an algorithm for assembling a
compact, high-signal context pack from a persistent agent memory graph in
response to a user query. Unlike top-k RAG, which retrieves a flat ranked
list of text chunks, RMCA decomposes the query into targeted subqueries,
retrieves from multiple evidence lanes (graph, hybrid, verbatim transcript),
expands the graph along typed edges, resolves update chains and
contradictions, deduplicates and ranks hits, and compresses the result to a
configurable token budget. The output is a structured context pack with
explicit provenance, conflict annotations, and superseded-memory handling.

RMCA is implemented in `waggle/recursive_context.py` as
`RecursiveContextController.build_context()` and exposed as the `build_context`
MCP tool in Waggle MCP.

---

## Notation

| Symbol | Type | Description |
|---|---|---|
| `q` | string | The user query or task description |
| `G = (V, E)` | graph | The persistent memory graph; `V` = nodes, `E` = typed edges |
| `v ∈ V` | node | A memory node with fields: `id`, `label`, `content`, `node_type`, `valid_from`, `valid_to`, `embedding` |
| `e ∈ E` | edge | A typed directed edge `(source_id, target_id, relationship)` where `relationship ∈ {updates, contradicts, depends_on, derived_from, part_of, relates_to, similar_to}` |
| `T` | transcript store | The verbatim conversation transcript store, queryable via the `verbatim` retrieval mode |
| `B` | integer | Token budget — the maximum desired token count for the output context pack |
| `d` | integer | Graph expansion depth — number of hops to traverse from seed nodes |
| `k` | integer | Maximum number of subqueries to generate in the decomposition step |
| `C` | string | The output context pack |
| `estimated_tokens(·)` | function | Token count approximation: `len(text) // 4` (1 token ≈ 4 characters) |
| `H` | list of hits | The working set of retrieved memory items during assembly |
| `sq_i` | subquery | The i-th decomposed subquery, with fields `query`, `purpose`, `priority`, `retrieval_modes` |

---

## Algorithm 1: Recursive Memory Context Assembly

```
Input:  q, G=(V,E), T, B, d, k
Output: C  where estimated_tokens(C) ≤ B × 1.15

1. DECOMPOSE
   ─────────
   If query intent is project/coding:
     subqueries ← [
       ("recent decisions about {topic}",       priority=1.00, modes=[graph, hybrid]),
       ("current unfinished tasks for {topic}",  priority=0.95, modes=[graph, hybrid]),
       ("constraints and rejected directions",   priority=0.90, modes=[graph, hybrid]),
       ("recent implementation details",         priority=0.85, modes=[graph, hybrid]),
       ("conflicts or updates in direction",     priority=0.80, modes=[graph]),
       (q,                                       priority=0.75, modes=[hybrid, verbatim]),
     ][:k]
   Else (generic memory query):
     subqueries ← [
       (q,                                       priority=1.00, modes=[hybrid, verbatim]),
       ("recent relevant facts about {topic}",   priority=0.90, modes=[graph, hybrid]),
       ("decisions related to {topic}",          priority=0.85, modes=[graph]),
       ("contradictions or conflicts",           priority=0.75, modes=[graph]),
       ("transcript evidence for {topic}",       priority=0.65, modes=[verbatim]),
     ][:k]

2. RETRIEVE
   ────────
   H ← []
   For each sq_i in subqueries:
     hits_i ← retrieve(sq_i.query, G, T,
                       modes=sq_i.retrieval_modes,
                       max_nodes=12, depth=d)
     H ← H ∪ hits_i

3. EXPAND
   ──────
   seeds ← top-5 nodes in H by score
   For each seed s in seeds[:3]:
     neighbours ← graph_hop(s, G, depth=min(d, 2))
     H ← H ∪ {n ∈ neighbours : n ∉ seeds}

4. RESOLVE
   ───────
   For each edge e = (src, tgt, rel) in edges(H):
     If rel = "updates":
       mark tgt as superseded; score(tgt) ×= 0.3
       score(src) = min(1.0, score(src) + 0.15)
     If rel = "contradicts":
       record conflict_entry(src, tgt)
   For each h ∈ H where h.valid_to < now:
     mark h as superseded; score(h) ×= 0.2

5. DEDUPLICATE
   ───────────
   H ← {argmax_{score} h : h.node_id} (keep highest-scored copy per node_id)

6. RANK
   ────
   H ← sort H by (is_superseded ASC, score DESC, label ASC)

7. COMPRESS
   ────────
   C ← ""
   budget_used ← 0
   max_tokens ← B × 1.15
   For section in [decisions, constraints, implementation, unfinished,
                   conflicts, superseded, evidence]:
     For each h in section:
       line ← format(h)
       If budget_used + estimated_tokens(line) > max_tokens: break
       C ← C + line
       budget_used += estimated_tokens(line)

8. FORMAT AND RETURN
   ──────────────────
   C ← prepend header "### Waggle Recursive Context Pack\nTask: {q}\n"
   Return C, provenance(H), conflict_entries
```

---

## Comparison with top-k RAG

Standard top-k RAG retrieves a flat ranked list of text chunks by semantic
similarity to the query and returns the top-k. RMCA differs in three
fundamental ways:

- **Subquery decomposition.** RAG issues a single query embedding. RMCA
  decomposes the query into up to `k` targeted subqueries covering different
  aspects of the information need (decisions, constraints, implementation
  details, conflicts, evidence). This allows retrieval of memory that is
  semantically distant from the original query but structurally relevant.

- **Graph expansion.** RAG operates on a flat chunk index. RMCA expands
  around retrieved nodes by traversing typed edges (`updates`, `contradicts`,
  `depends_on`, `derived_from`, `part_of`). This surfaces structurally linked
  memory that would not appear in a similarity ranking — in particular,
  conflict and update chains that are critical for agent continuity.

- **Conflict and update resolution.** RAG returns all retrieved chunks
  without distinguishing current from superseded information. RMCA explicitly
  detects `updates` and `contradicts` edges, marks superseded nodes, and
  surfaces conflict annotations in the context pack. This prevents agents
  from acting on stale or contradicted decisions.

---

## Comparison with GraphRAG-style expansion

GraphRAG-style systems (e.g., Microsoft GraphRAG) build a community-detection
graph over document chunks and retrieve community summaries. RMCA differs in
three ways:

- **Recursive decomposition.** GraphRAG issues a single query against
  pre-computed community summaries. RMCA recursively decomposes the query
  into subqueries and runs targeted retrieval for each, adapting the
  decomposition to the query's intent at runtime without pre-computation.

- **Token-budget compression.** GraphRAG returns community summaries of
  fixed size. RMCA enforces a configurable token budget `B` with a priority
  ordering (decisions > constraints > implementation > unfinished > conflicts
  > evidence), ensuring the context pack fits within the agent's context
  window regardless of graph size.

- **Verbatim evidence lane.** GraphRAG operates entirely on extracted graph
  facts. RMCA includes a verbatim transcript retrieval lane (`T`) that
  surfaces exact conversation snippets as evidence, enabling agents to cite
  the original source of a memory rather than only the extracted fact.

---

## Limitations and Scope

> **Synthetic data caveat:** The current RMCA evaluation uses deterministic
> synthetic Waggle memory tasks mapped to Waggle's graph/transcript
> environment. Numerical results from this evaluation **must not be compared**
> to results from the RLM paper (Zhang et al., 2026) or other long-context
> benchmarks until the exact public datasets (RULER S-NIAH, BrowseComp-Plus,
> OOLONG, OOLONG-Pairs, LongBench-v2 CodeQA) are downloaded and run with a
> matching model setup.

Additional limitations:

1. **Token estimation is approximate.** `estimated_tokens(text) = len(text) // 4`
   is a character-count heuristic. Actual token counts depend on the
   tokenizer used by the downstream model and may differ by ±20%.

2. **Decomposition is heuristic, not learned.** Subquery generation uses
   keyword pattern matching to detect query intent. It does not use an LLM
   and may produce suboptimal decompositions for queries outside the
   project/coding and generic-memory categories.

3. **Graph expansion is bounded.** Expansion is limited to `min(d, 2)` hops
   from the top-3 seed nodes. Deep reasoning chains requiring more than 2
   hops are not guaranteed to be surfaced.

4. **Linear aggregation tasks remain hard.** For tasks requiring aggregation
   over all N entries (e.g., "list all blocked tasks"), RMCA cannot guarantee
   coverage when N exceeds the token budget. This is a fundamental O(n)
   information need that no fixed-budget retrieval system can fully satisfy.

5. **No external model API required.** The first version of RMCA is fully
   local and offline. Answer-level evaluation uses a deterministic rule-based
   answerer, which is a reproducible lower bound and not equivalent to
   human preference or LLM-judge quality assessment.
