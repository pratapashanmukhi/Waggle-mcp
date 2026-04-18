<p align="center">
  <img src="https://raw.githubusercontent.com/Abhigyan-Shekhar/graph-memory-mcp/main/assets/banner.png" alt="waggle-mcp" width="720"/>
</p>

<p align="center">
  <strong>Persistent, structured memory for AI agents — typically lower-token than chunk-based retrieval, often 2-4× on factual lookups.</strong><br/>
  Your LLM remembers facts, decisions, and context <em>across every conversation</em>, backed by a real knowledge graph.
</p>

<p align="center">
  <a href="https://pypi.org/project/waggle-mcp"><img src="https://img.shields.io/pypi/v/waggle-mcp?color=39d5cf&label=pypi" alt="PyPI"/></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/MCP-compatible-brightgreen" alt="MCP compatible"/>
  <img src="https://img.shields.io/badge/embeddings-local%2C%20no%20API%20key-orange" alt="Local embeddings"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT"/>
</p>

<p align="center">
  <a href="https://glama.ai/mcp/servers/Abhigyan-Shekhar/Waggle-mcp"><img src="https://glama.ai/mcp/servers/Abhigyan-Shekhar/Waggle-mcp/badges/card.svg" alt="Waggle-mcp MCP server"/></a>
  <a href="https://glama.ai/mcp/servers/Abhigyan-Shekhar/Waggle-mcp"><img src="https://glama.ai/mcp/servers/Abhigyan-Shekhar/Waggle-mcp/badges/score.svg" alt="Waggle-mcp MCP server score"/></a>
</p>

---

## What's New — v0.1.7

- **Benchmark harness**: end-to-end `WaggleAdapter` connecting the graph engine to ConvoMem / MemBench runners with automated exact-match scoring and latency logging.
- **LongMemEval integration**: CLI-driven ingestion and retrieval evaluation against the official LongMemEval split (held-out `81.6% R@5`).
- **Logging utilities**: structured log helpers (`logging_utils`) for consistent, level-aware output across all subsystems.
- **Evidence tracking**: new `evidence.py` module records source provenance on stored nodes so reasoning chains are fully traceable.
- **Observability stack**: Grafana dashboard, Prometheus config, and Docker Compose overlay in `deploy/observability/`.
- **Kubernetes manifests**: production-grade `deployment.yaml`, network policy, external-secret, and certificate templates under `deploy/kubernetes/`.
- **Operational runbooks**: incident response, secret management, API-key rotation, and onboarding guides added to `docs/runbooks/`.

---

## Why waggle-mcp?

`waggle-mcp` is a local-first memory layer for MCP-compatible AI clients, built on a persistent knowledge graph. It gives your AI a persistent knowledge graph it can read and write through any MCP-compatible client (Claude Desktop, Cursor, Codex, Antigravity, etc.).

| Stuffed context | Structured retrieval |
|-----------------|----------------------|
| Huge prompts every session | Compact subgraph retrieved at query time |
| Session-local memory | Persistent multi-session memory |
| Flat notes and chunks | Typed nodes and edges: decisions, reasons, contradictions |
| "What changed?" requires replaying logs | Temporal queries and diffs are first-class |

Waggle often uses materially fewer tokens than naive chunked retrieval on factual lookups, while graph-traversal queries intentionally spend more context to include reasoning chains such as updates, contradictions, and dependencies.

---

## Quick start

```bash
pip install waggle-mcp
waggle-mcp init
# Restart your MCP client. Done.
```

`init` detects your MCP client, writes its config, and creates the local database directory. Default mode is local SQLite with on-device embeddings. Antigravity and manual configuration details are in [docs/reference.md](./docs/reference.md).

Manual MCP setup examples for **Codex**, **Claude Code**, **Cursor**, and **Antigravity** are in [docs/reference.md](./docs/reference.md#manual-client-configuration).

---

## Using It In MCP Clients

Once Waggle is installed in an MCP client, people normally do not run `waggle-mcp` commands by hand during everyday use. They talk to the agent normally, and the agent uses Waggle's MCP tools to store and retrieve memory.

### Codex

Typical pattern:
- You work in a normal Codex thread.
- Codex calls `observe_conversation`, `store_node`, `store_edge`, `query_graph`, or `prime_context` when memory is useful.
- On a later task, Codex can pull back the connected subgraph instead of relying on the current chat window alone.

Example:
- You say: `Remember that we chose PostgreSQL because MySQL replication was painful.`
- Codex stores that as structured memory.
- Days later you ask: `What did we decide about the database?`
- Codex can call `query_graph` and recover the earlier decision plus its reason.

### Claude Code

Typical pattern:
- You configure Waggle as an MCP server in Claude Code.
- Claude Code uses Waggle tools to persist decisions, preferences, architecture notes, and project state across sessions.
- `prime_context` and `export_context_bundle` are useful when starting a fresh task or handing context to another model.

### Cursor

Typical pattern:
- Cursor uses Waggle over MCP while you work in the editor.
- Facts and decisions can be saved as graph memory instead of getting lost in past chats.
- Later questions like `why did we change this?` or `what superseded this decision?` can be answered from connected nodes and edges.

### Antigravity

Typical pattern:
- Antigravity can use Waggle as its persistent memory backend through MCP.
- Conversation turns can be extracted with `observe_conversation`.
- Linked context can be exported with `export_context_bundle` or edited through the Markdown vault workflow.

For a built-in CLI explanation of the feature surface, run:

```bash
waggle-mcp features
```

---

## See it in action

**Session 1** — April 10
```text
User:  Let's use PostgreSQL. MySQL replication has been painful.
Agent: [calls observe_conversation()]
       → stores decision node: "Chose PostgreSQL over MySQL"
       → stores reason node:   "MySQL replication painful"
       → links them with a depends_on edge
```

**Session 2** — April 12 (fresh context window, no history)
```text
User:  What did we decide about the database?
Agent: [calls query_graph("database decision")]
       → retrieves the decision node + linked reason from April 10

       "You decided on PostgreSQL on April 10. The reason recorded was
        that MySQL replication had been painful."
```

**Session 3** — April 14
```text
User:  Actually, let's reconsider — the team is more familiar with MySQL.
Agent: [calls store_node() + store_edge(new_node → old_node, "contradicts")]
       → both positions are preserved, and the contradiction is explicit
```

---

## Key Features

- **Automatic Extraction**: `observe_conversation` ingests facts into the graph without manual schema work.
- **Portable Context**: `export_context_bundle` generates Markdown/JSON context packs for another AI.
- **Vault Round-trip**: `export_markdown_vault` / `import_markdown_vault` for Obsidian-style node editing.
- **Conflict Resolution**: `list_conflicts` / `resolve_conflict` to manage contradictions without losing history.
- **Deterministic Fallback**: Stable SHA-256 hashing for reliable, reproducible offline operation when transformer models are unavailable.

---

## Benchmarks & Verification

Waggle performance is verified against checked-in fixtures and automated regression tests.

### Project Fixtures
| Area | Corpus | Result |
|------|--------|--------|
| Extraction | 25-case deterministic fixture | `100.0%` |
| Retrieval | 18-query retrieval fixture | `83.3% Hit@k` |
| Query stress | 40 adversarial retrieval-only cases | `97.5% Hit@k`, `97.5% exact support` |
| Deduplication | 22 cases (semi-semantic) | `0` false merges at the selected threshold; `77.3%` overall due to conservative false negatives |
| Automated tests | Infrastructure & logic | `91 passed` |

### External Benchmarks
| Benchmark | Coverage | Metric | Status |
|-----------|----------|--------|--------|
| **LongMemEval** | 500 questions | `81.6% R@5` held-out deterministic | Verified |

- **LongMemEval note**: The checked-in full-split `97.4% R@5` result is useful as a retrieval ceiling on the saved benchmark setup, but the held-out `81.6%` split is the more honest number for generalization.
- **Deduplication**: Zero false-positive merges across the threshold sweep. Accuracy limited by conservative similarity bounds.
- **Comparative benchmark note**: The comparative Waggle-vs-RAG corpus is still evolving. For current per-family/token numbers, use the checked-in artifact index in [tests/artifacts/README.md](./tests/artifacts/README.md) rather than this top-level summary.

Detailed benchmark artifacts and the new **[Benchmark Methodology](./docs/benchmark-methodology.md)** guide provide full traceability.

---

## Known Limitations

- **Best on structured recall, weaker on benchmark-style answer synthesis**: Waggle is strongest when the problem is "retrieve the right facts and relationships" rather than "emit one benchmark-formatted final answer from memory."
- **Edges matter**: Isolated `store_node` writes do not create graph context by themselves. Connected context comes from `store_edge`, `observe_conversation`, `decompose_and_store`, and automatic contradiction/update detection.
- **Graph retrieval is not always the cheapest mode**: factual lookups are often much cheaper than chunked RAG, but graph-expansion queries intentionally use more tokens to carry reasoning context.
- **Deduplication is conservative by design**: the system prefers missed merges over unsafe merges, which protects correctness but leaves some semantically similar duplicates unmerged.
- **README numbers are intentionally narrow**: only the most stable benchmark claims are summarized here; per-family and evolving comparative numbers live in the artifact docs instead.

For operational details, scaling considerations, tool-level behavior, and the full MCP feature surface, see [docs/reference.md](./docs/reference.md).

---

## Reference & Docs

Detailed reference material lives in external documentation:

- **[docs/reference.md](./docs/reference.md)**: Environment variables, admin commands, Docker setup, and full tool surface.
- **[deploy/kubernetes/README.md](./deploy/kubernetes/README.md)**: Production deployment.
- **[docs/runbooks/](./docs/runbooks/)**: Operations and troubleshooting.
- **[tests/artifacts/README.md](./tests/artifacts/README.md)**: Benchmark artifacts and traceability.

---

## License

MIT — see [LICENSE](./LICENSE).
