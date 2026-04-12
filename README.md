<p align="center">
  <img src="https://raw.githubusercontent.com/Abhigyan-Shekhar/graph-memory-mcp/main/assets/banner.png" alt="waggle-mcp" width="720"/>
</p>

<p align="center">
  <strong>Persistent, structured memory for AI agents — backed by a real knowledge graph.</strong><br/>
  Your LLM remembers facts, decisions, and context <em>across every conversation</em>.
</p>

<p align="center">
  <a href="https://pypi.org/project/waggle-mcp"><img src="https://img.shields.io/pypi/v/waggle-mcp?color=39d5cf&label=pypi" alt="PyPI"/></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/MCP-compatible-brightgreen" alt="MCP compatible"/>
  <img src="https://img.shields.io/badge/embeddings-local%2C%20no%20API%20key-orange" alt="Local embeddings"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT"/>
</p>

---

## Why waggle-mcp?

Most LLMs forget everything when the conversation ends.  
`waggle-mcp` fixes that by giving your AI a **persistent knowledge graph** it can read and write through any MCP-compatible client.

Waggle's key advantage is **token efficiency with structured context**:

| Without waggle-mcp | With waggle-mcp |
|--------------------------|----------------------|
| Context stuffed into a 200k-token prompt | **~4× fewer tokens** — compact subgraph, only relevant nodes retrieved |
| "What did we decide about the DB schema?" → ❌ Lost when the session ended | ✅ Recalls the decision node, when it was made, and what it contradicts |
| Flat bullet-list memory | Typed edges: `relates_to`, `contradicts`, `depends_on`, `updates`… |
| One session, one agent | Multi-tenant, multi-session, multi-agent |

> **Note on retrieval:** Waggle trades some raw recall coverage for dramatically lower token cost and richer relational context. See the [benchmark section](#performance--benchmarking) for honest numbers.

---

## Quick start — 30 seconds

```bash
pip install waggle-mcp
waggle-mcp init
```

The `init` wizard detects your MCP client, writes its config file, and creates
the database directory — no JSON editing required. Supports **Claude Desktop**,
**Cursor**, **Codex**, and a generic JSON fallback.

After init, restart your MCP client and your AI has persistent memory.  
No cloud service. No API key. Semantic search runs fully locally.

---

## See it in action

Here's a concrete before/after for a developer using the AI daily:

**Session 1** — April 10
```
User:  Let's use PostgreSQL. MySQL replication has been painful.
Agent: [calls observe_conversation()]
       → stores decision node: "Chose PostgreSQL over MySQL"
       → stores reason node:   "MySQL replication painful"
       → links them with a depends_on edge
```

**Session 2** — April 12 (fresh context window, no history)
```
User:  What did we decide about the database?
Agent: [calls query_graph("database decision")]
       → retrieves the decision node + linked reason from April 10

       "You decided on PostgreSQL on April 10. The reason recorded was
        that MySQL replication had been painful."
```

**Session 3** — April 14
```
User:  Actually, let's reconsider — the team is more familiar with MySQL.
Agent: [calls store_node() + store_edge(new_node → old_node, "contradicts")]
       → conflict is flagged automatically; both positions are preserved in the graph
```

> The agent never needed explicit instructions to remember or retrieve — it called
> the right tools based on the conversation, and the graph gave it the right context.

---

## How it works

Memory doesn't just get stored — it flows through a lifecycle:

```
You talk to your AI
        │
        ▼
  observe_conversation()          ← AI drops the turn in; facts extracted via structured LLM (regex fallback)
        │
        ▼
  Graph nodes are created         ← "Chose PostgreSQL" becomes a decision node
  Edges are inferred              ← linked to the "database" entity node
        │
        ▼
  Future conversation starts
        │
        ▼
  query_graph("DB schema")        ← semantic search finds the node from 3 sessions ago
        │
        ▼
  AI answers with full context    ← "You decided on PostgreSQL on Apr 10, here's why…"
```

Every node carries semantic embeddings computed **locally** using
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) —
a fast, lightweight model that runs entirely on-device with no API key or network
call required. This means semantic search works offline, costs nothing per query,
and keeps your data private.

---

## The magic tool: `observe_conversation`

> **This is the tool you'll use most.** You don't have to manually store facts — just
> tell the agent to observe each conversation turn and it handles the rest.

```
observe_conversation(user_message, assistant_response)
```

Under the hood, it:
1. Extracts atomic facts from both sides of the conversation
2. Deduplicates against existing nodes using semantic similarity
3. Creates typed edges between related concepts
4. Flags contradictions with existing stored beliefs

No instructions needed. No schema to define. Just observe.

Under the hood, every call runs a **Pydantic-validated LLM extraction pass** (with a regex fallback) to pull structured facts out of messy dialogue.

**Example:** `"Let's use PostgreSQL because MySQL replication is too painful."`

```json
{
  "facts": [
    {
      "label": "PostgreSQL for generic events",
      "content": "Chose PostgreSQL over MySQL because MySQL replication is too painful.",
      "node_type": "decision",
      "confidence": 0.95,
      "tags": ["llm-extracted", "confidence:0.95"]
    }
  ]
}
```

*Any extraction with `confidence < 0.5` or an invalid schema is silently dropped to prevent hallucination noise.*

---

## Memory model

**Node types** — what gets stored:

| Type | Example |
|------|---------| 
| `fact` | "The API uses JWT tokens" |
| `preference` | "User prefers dark mode" |
| `decision` | "Chose PostgreSQL over MySQL" |
| `entity` | "Project: waggle-mcp" |
| `concept` | "Rate limiting" |
| `question` | "Should we add GraphQL?" |
| `note` | "TODO: add integration tests" |

**Edge types** — how nodes connect:

`relates_to` · `contradicts` · `depends_on` · `part_of` · `updates` · `derived_from` · `similar_to`

---

## MCP tools

> Your AI calls these directly — you don't need to use them manually.

| Tool | What it does |
|------|-------------|
| `observe_conversation` | **Drop a conversation turn in — facts extracted, stored, and linked** |
| `query_graph` | Semantic + temporal search across the graph |
| `store_node` | Manually save a fact, preference, decision, or note |
| `store_edge` | Link two nodes with a typed relationship |
| `get_related` | Traverse edges from a specific node |
| `update_node` | Update content or tags on an existing node |
| `delete_node` | Remove a node and all its edges |
| `decompose_and_store` | Break long content into atomic nodes automatically |
| `graph_diff` | See what changed in the last N hours |
| `prime_context` | Generate a compact brief for a new conversation |
| `get_topics` | Detect topic clusters via community detection |
| `get_stats` | Node/edge counts and most-connected nodes |
| `export_graph_html` | Interactive browser visualization |
| `export_graph_backup` | Portable JSON backup |
| `import_graph_backup` | Restore from a JSON backup |

---

## Performance & Benchmarking

All numbers below are reproducible from the checked-in fixtures in `benchmarks/fixtures/` using the harness at [`scripts/benchmark_extraction.py`](./scripts/benchmark_extraction.py). Saved output artifacts live in [`tests/artifacts/`](./tests/artifacts/README.md).

**One command produces all the tables below** (extraction regex baseline, retrieval, dedup, and the comparative token-efficiency pilot):

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_extraction.py \
  --extraction-backend regex \
  --systems waggle rag_naive \
  --output tests/artifacts/benchmark_current.json
```

The LLM extraction row (75%) requires a separate run with a local Ollama instance — it is not included in `benchmark_current.json`:

```bash
# Requires Ollama running locally with qwen2.5:7b pulled
PYTHONPATH=src .venv/bin/python scripts/benchmark_extraction.py \
  --extraction-backend llm --ollama-model qwen2.5:7b --ollama-timeout-seconds 30
```


### Extraction accuracy

Corpus: 12 dialogue pairs covering simple recall, interruptions, reversals, vague statements, and conflicting signals (`benchmarks/fixtures/extraction_cases.json`).

| Backend | Cases | Accuracy |
|---------|-------|----------|
| Regex (fallback) | 12 | 33% |
| LLM (`qwen2.5:7b`, 30 s timeout) | 12 | 75% |

### Retrieval accuracy

Corpus: 18 nodes, 18 queries — 6 easy (direct paraphrase) and 12 hard (adversarial: semantic generalization, temporal disambiguation, indirect domain translation, privacy framing). Source: `benchmarks/fixtures/retrieval_cases.json`.

| Difficulty | Queries | Hit@k |
|------------|---------|-------|
| Easy | 6 | 6/6 = 100% |
| Hard (adversarial) | 12 | 9/12 = 75% |
| **Overall** | **18** | **15/18 = 83%** |

### Token efficiency vs. naive chunked-vector RAG

*The retrieval accuracy table above measures Waggle's standalone search quality. The comparison below uses a separate multi-session corpus designed to test token efficiency against a chunked-vector baseline.*

Corpus: 24 multi-session scenarios, 50 retrieval queries (`benchmarks/fixtures/comparative_eval.json`).

| System | Mean tokens | Median tokens | p95 tokens | Hit@k |
|--------|-------------|---------------|------------|-------|
| **Waggle** | **37.6** | **38.0** | **42.0** | 88% |
| Naive chunked-vector RAG | 152.1 | 154.0 | 163.0 | 100% |

**Waggle uses ~4× fewer tokens per retrieval** than the naive chunked baseline on this corpus.

The tradeoff is honest: the chunked baseline achieves 100% Hit@k on this corpus because the corpus is not yet hard enough to stress it. **The token efficiency advantage is real and large; the retrieval superiority claim is not yet supported at this corpus scale.** Corpus hardening is the next evaluation step.

### When extraction fails

> **User:** "Yeah, let's just do that thing we talked about."

The LLM assigns low confidence (`confidence < 0.5`) to ambiguous input; Waggle **drops the extraction silently** rather than storing a guess. The pipeline does **not** silently fall back to regex on timeout — backend failures surface as explicit errors that are logged.

<details>
<summary>Deduplication results (22-pair fixture — click to expand)</summary>

Corpus: 22 node pairs — 11 true duplicates (synonym, paraphrase, domain equivalence) and 11 false friends (same technology category, different technology). Source: `benchmarks/fixtures/dedup_cases.json`.

The pipeline runs five layers:
1. **Layer 0 — Entity-key hard block** — if both nodes name *different* technologies in the *same* category (e.g. `postgresql` vs `mysql`), merge is blocked unconditionally.
2. **Layer 0b — Numeric-conflict guard** — same entity but *different critical numbers* (e.g. `jwt` 15 min vs 1 hr) → block. Guards against merging distinct facts that share a technology but differ on a key value.
3. **Layer 1 — Exact string match** — normalized content or label equality.
4. **Layer 2 — Substring containment** — one sentence is a strict subset of the other.
5. **Layer 3 — Semantic similarity** — cosine via `all-MiniLM-L6-v2`:
   - Same-entity aggressive path: if both reference the **same** entity token, merge at cosine ≥ 0.60 (catches paraphrase true-dups like "fastapi was chosen" / "we chose fastapi because async")
   - Type-aware threshold: `decision`/`preference` → 0.82; `fact` → 0.92; `entity` → 0.97
   - Jaccard-boosted path: word overlap ≥ 0.35 AND cosine ≥ (type threshold − 0.05)
   - Conservative global fallback

Best measured: **18/22 = 82%** at threshold 0.82. **fp=0 across all thresholds** — no false-friend merges at any tested threshold.

The remaining 4 false-negatives are pure-paraphrase pairs with no recognisable entity anchor ("user prefers dark mode" / "user wants dark mode UI", "async non-negotiable" / "concurrent without blocking"). These require either semantic similarity fine-tuning or a learned paraphrase classifier to close.

Full threshold sweep and detailed methodology: [`tests/artifacts/README.md`](./tests/artifacts/README.md).

</details>

> Full artifacts, methodology, and rag_tuned comparison: [`tests/artifacts/README.md`](./tests/artifacts/README.md)


---

## Temporal queries — built-in, not bolted on

Most memory systems answer "what do you know about X?" — but can't answer
*when* you learned it or how knowledge changed over time.

`waggle-mcp` timestamps every node and understands temporal natural language:

| Query | What happens |
|-------|-------------|
| `query_graph("what did we decide recently")` | Filters nodes updated in the last 24–48h |
| `query_graph("what was the original plan")` | Retrieves the earliest version of relevant nodes |
| `query_graph("what changed last week")` | Returns a diff of nodes created/updated in that window |
| `graph_diff(since="48h")` | Explicit changelog: added nodes, updated nodes, new conflicts |

---

## Testing

Beyond empirical benchmarks, `waggle-mcp` ships with a comprehensive pytest suite covering both memory logic and server protocols. This guarantees core behaviours — multi-tenant isolation, conflict detection, semantic deduplication, MCP protocol handling, and explicit LLM backend failure — remain stable across updates.

<details>
<summary>View the 43 component and integration tests (click to expand)</summary>

```text
============================= test session starts ==============================
collected 43 items                                                             

tests/test_benchmark_harness.py::test_fixture_loading_is_auditable PASSED
tests/test_benchmark_harness.py::test_benchmark_report_includes_backend_labels_and_case_counts PASSED
tests/test_benchmark_harness.py::test_markdown_summary_includes_comparative_systems PASSED
tests/test_benchmark_harness.py::test_llm_benchmark_failure_is_explicit PASSED
tests/test_benchmark_harness.py::test_dedup_threshold_sweep_tracks_positive_and_negative_cases PASSED
tests/test_embeddings.py::test_embedding_bytes_round_trip PASSED
tests/test_embeddings.py::test_cosine_similarity_handles_orthogonal_vectors PASSED
tests/test_graph.py::test_add_query_and_related PASSED
tests/test_graph.py::test_update_delete_and_stats PASSED
tests/test_graph.py::test_exact_duplicate_nodes_are_reused_and_tags_are_merged PASSED
tests/test_graph.py::test_semantic_duplicate_nodes_reuse_existing_entry PASSED
tests/test_graph.py::test_entity_resolution_reuses_acronym_matches PASSED
tests/test_graph.py::test_query_ranking_uses_label_lexical_overlap PASSED
tests/test_graph.py::test_decompose_and_store_creates_nodes_and_edges PASSED
tests/test_graph.py::test_export_and_import_backup_round_trip PASSED
tests/test_graph.py::test_export_graph_html_creates_visualization_file PASSED
tests/test_graph.py::test_conflict_detection_creates_contradiction_edge PASSED
tests/test_graph.py::test_observe_conversation_extracts_nodes PASSED
tests/test_graph.py::test_query_supports_temporal_latest_and_oldest_bias PASSED
tests/test_graph.py::test_graph_diff_and_prime_context PASSED
tests/test_graph.py::test_get_topics_returns_clusters PASSED
tests/test_platform.py::test_api_key_hashing_round_trip PASSED
tests/test_platform.py::test_rate_limiter_enforces_request_and_concurrency_limits PASSED
tests/test_platform.py::test_tenant_scoping_isolated_within_same_sqlite_database PASSED
tests/test_platform.py::test_backup_round_trip_preserves_schema_and_tenant_metadata PASSED
tests/test_platform.py::test_http_app_health_auth_and_metrics PASSED
tests/test_platform.py::test_http_app_rate_limit_and_payload_limit PASSED
tests/test_server.py::test_store_node_and_stats_tool PASSED
tests/test_server.py::test_export_graph_html_tool PASSED
tests/test_server.py::test_decompose_and_store_tool_persists_subgraph PASSED
tests/test_server.py::test_export_and_import_backup_tools PASSED
tests/test_server.py::test_store_node_reports_deduplication PASSED
tests/test_server.py::test_store_node_reports_conflicts PASSED
tests/test_server.py::test_observe_conversation_tool PASSED
tests/test_server.py::test_graph_diff_prime_context_and_topics_tools PASSED
tests/test_server.py::test_recent_resource_serialization PASSED
tests/test_server.py::test_unknown_tool_raises PASSED
tests/test_server.py::test_invalid_tool_inputs_return_structured_errors PASSED
tests/test_server.py::test_tool_payload_limit_is_enforced PASSED
tests/test_server.py::test_default_graph_uses_sqlite_backend_by_default PASSED
tests/test_server.py::test_default_graph_can_build_neo4j_backend PASSED
tests/test_server.py::test_default_graph_requires_neo4j_connection_settings PASSED
tests/test_stdio_integration.py::test_server_stdio_initialize_and_basic_calls PASSED

============================== 43 passed in 4.92s ==============================
```
</details>

---

## Installation

<details>
<summary>Local / development (SQLite, no extra services)</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
waggle-mcp init        # ← writes your client config automatically
```

Key variables for local mode:

| Variable | What it does |
|----------|-------------|
| `WAGGLE_BACKEND=sqlite` | Local file DB, zero setup |
| `WAGGLE_TRANSPORT=stdio` | Connects to desktop MCP clients |
| `WAGGLE_DB_PATH` | Where the graph is stored (default: `memory.db`) |

</details>

<details>
<summary>Production (Neo4j backend)</summary>

```bash
pip install -e ".[dev,neo4j]"

WAGGLE_TRANSPORT=http \
WAGGLE_BACKEND=neo4j \
WAGGLE_DEFAULT_TENANT_ID=workspace-default \
WAGGLE_NEO4J_URI=bolt://localhost:7687 \
WAGGLE_NEO4J_USERNAME=neo4j \
WAGGLE_NEO4J_PASSWORD=change-me \
waggle-mcp
```

</details>

<details>
<summary>Docker</summary>

```bash
docker build -t waggle-mcp:latest .

docker run --rm -p 8080:8080 \
  -e WAGGLE_TRANSPORT=http \
  -e WAGGLE_BACKEND=neo4j \
  -e WAGGLE_DEFAULT_TENANT_ID=workspace-default \
  -e WAGGLE_NEO4J_URI=bolt://host.docker.internal:7687 \
  -e WAGGLE_NEO4J_USERNAME=neo4j \
  -e WAGGLE_NEO4J_PASSWORD=change-me \
  waggle-mcp:latest
```

</details>

<details>
<summary>Manual client configuration</summary>

**Claude Desktop — `claude_desktop_config.json`**

```json
{
  "mcpServers": {
    "waggle": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "waggle.server"],
      "env": {
        "PYTHONPATH": "/path/to/waggle-mcp/src",
        "WAGGLE_TRANSPORT": "stdio",
        "WAGGLE_BACKEND": "sqlite",
        "WAGGLE_DB_PATH": "~/.waggle/memory.db",
        "WAGGLE_DEFAULT_TENANT_ID": "local-default",
        "WAGGLE_MODEL": "all-MiniLM-L6-v2"
      }
    }
  }
}
```

**Codex — `codex_config.toml`**

```toml
[mcp_servers.waggle]
command = "/path/to/.venv/bin/python"
args    = ["-m", "waggle.server"]
cwd     = "/path/to/waggle-mcp"
env     = {
  PYTHONPATH                     = "/path/to/waggle-mcp/src",
  WAGGLE_TRANSPORT         = "stdio",
  WAGGLE_BACKEND           = "sqlite",
  WAGGLE_DB_PATH           = "~/.waggle/memory.db",
  WAGGLE_DEFAULT_TENANT_ID = "local-default",
  WAGGLE_MODEL             = "all-MiniLM-L6-v2"
}
```

A pre-filled example is in [`codex_config.example.toml`](./codex_config.example.toml).

</details>

---

## Environment variables

<details>
<summary>Click to expand full reference</summary>

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `WAGGLE_BACKEND` | `sqlite` | `sqlite` or `neo4j` |
| `WAGGLE_TRANSPORT` | `stdio` | `stdio` or `http` |
| `WAGGLE_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model (local inference) |
| `WAGGLE_DEFAULT_TENANT_ID` | `local-default` | default tenant |
| `WAGGLE_EXPORT_DIR` | — | optional export directory |

### SQLite

| Variable | Default | Description |
|----------|---------|-------------|
| `WAGGLE_DB_PATH` | `memory.db` | path to the SQLite file |

### HTTP service

| Variable | Default | Description |
|----------|---------|-------------|
| `WAGGLE_HTTP_HOST` | `0.0.0.0` | bind host |
| `WAGGLE_HTTP_PORT` | `8080` | bind port |
| `WAGGLE_LOG_LEVEL` | `INFO` | log level |
| `WAGGLE_RATE_LIMIT_RPM` | `120` | global rate limit (req/min) |
| `WAGGLE_WRITE_RATE_LIMIT_RPM` | `60` | write-tool rate limit |
| `WAGGLE_MAX_CONCURRENT_REQUESTS` | `8` | concurrency cap |
| `WAGGLE_MAX_PAYLOAD_BYTES` | `1048576` | max request size |
| `WAGGLE_REQUEST_TIMEOUT_SECONDS` | `30` | per-request timeout |

### Neo4j

| Variable | Description |
|----------|-------------|
| `WAGGLE_NEO4J_URI` | Bolt URI, e.g. `bolt://localhost:7687` |
| `WAGGLE_NEO4J_USERNAME` | Neo4j username |
| `WAGGLE_NEO4J_PASSWORD` | Neo4j password |
| `WAGGLE_NEO4J_DATABASE` | Neo4j database name |

### LLM Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `WAGGLE_EXTRACT_BACKEND` | `auto` | `auto` \| `llm` \| `regex` |
| `WAGGLE_EXTRACT_MODEL` | `mistral` | Ollama model name |
| `WAGGLE_EXTRACT_MIN_CONFIDENCE` | `0.5` | float 0–1, facts below this are dropped |
| `WAGGLE_OLLAMA_URL` | `http://localhost:11434` | Base URL for local Ollama |

</details>

---

<details>
<summary>Admin commands</summary>

```bash
# Create a tenant
waggle-mcp create-tenant --tenant-id workspace-a --name "Workspace A"

# Issue an API key (raw key returned once — store it securely)
waggle-mcp create-api-key --tenant-id workspace-a --name "ci-agent"

# List keys for a tenant
waggle-mcp list-api-keys --tenant-id workspace-a

# Revoke a key
waggle-mcp revoke-api-key --api-key-id <id>

# Migrate SQLite data → Neo4j
WAGGLE_BACKEND=neo4j WAGGLE_NEO4J_URI=bolt://localhost:7687 \
WAGGLE_NEO4J_USERNAME=neo4j WAGGLE_NEO4J_PASSWORD=change-me \
  waggle-mcp migrate-sqlite --db-path ./memory.db --tenant-id workspace-a
```

</details>

<details>
<summary>Kubernetes & observability</summary>

Full production deployment assets are in [`deploy/`](./deploy/):

| Path | What's inside |
|------|--------------|
| `deploy/kubernetes/` | Deployment, Service, Ingress (TLS), NetworkPolicy, HPA, PDB, cert-manager, ExternalSecrets — see [`deploy/kubernetes/README.md`](./deploy/kubernetes/README.md) |
| `deploy/observability/` | Prometheus scrape config, Grafana dashboard, one-command Docker Compose observability stack |

Operational runbooks are in [`docs/runbooks/`](./docs/runbooks/):

- [API key rotation](./docs/runbooks/api-key-rotation.md) — zero-downtime create-then-revoke
- [Incident response](./docs/runbooks/incident-response.md) — Neo4j down, OOM, rate storm, auth failures
- [Backup & restore](./docs/runbooks/backup-restore.md) — manual and automated drill
- [Tenant onboarding](./docs/runbooks/onboarding.md) — new tenant checklist
- [Secret management](./docs/runbooks/secret-management.md) — External Secrets + cert-manager

</details>

<details>
<summary>Architecture & project layout</summary>

```
waggle-mcp
├── Core domain    graph CRUD · dedup · local embeddings · conflict detection · export/import
├── Transport      stdio MCP (Codex/Desktop) · streamable HTTP MCP (Kubernetes)
└── Platform       config · auth · tenant isolation · rate limiting · logging · metrics
```

**Backend:**
- Local/dev → SQLite (zero config, instant start)
- Production → Neo4j (`WAGGLE_TRANSPORT=http` requires `WAGGLE_BACKEND=neo4j`)

```
waggle-mcp/
├── assets/                   ← banner + demo SVG
├── benchmarks/fixtures/      ← checked-in eval datasets
├── deploy/
│   ├── kubernetes/           ← full K8s manifests + guide
│   └── observability/        ← Prometheus + Grafana stack
├── docs/runbooks/            ← operational runbooks
├── scripts/
│   ├── benchmark_extraction.py
│   ├── load_test.py / .sh
│   └── backup_restore_drill.py / .sh
├── src/waggle/         ← server, graph, neo4j_graph, auth, config …
├── tests/artifacts/    ← saved benchmark runs
├── Dockerfile
├── pyproject.toml
└── README.md
```

</details>

---

## Testing

```bash
.venv/bin/pytest -q
```

Coverage: graph CRUD, deduplication, conflict detection, tenant isolation,
backup/import, stdio MCP, HTTP auth/health/metrics, payload limits.

---

## License

MIT — see [LICENSE](./LICENSE).
