# Waggle Briefing

This is the short-form "what to say about Waggle" document.

It is intentionally a media-kit style summary:
- use it for pitches, intros, and product overviews
- keep it separate from the README so setup, usage, and verification details stay in one canonical place

## One-line summary

Waggle is a local-first MCP memory layer that stores conversation as a persistent knowledge graph and exports or retrieves connected context across sessions and across clients.

## What it does

- Persists durable conversation memory as typed graph nodes and edges
- Extracts facts, decisions, preferences, questions, and rationale from conversation
- Retrieves scoped memory with `query_graph`
- Builds a compact session starter with `prime_context`
- Detects and preserves contradictions instead of overwriting history
- Exports portable context bundles for other sessions or other IDEs
- Imports graph backups and markdown vaults back into memory
- Runs locally by default with SQLite, or on Neo4j for service deployments

## Best user-facing commands

- `waggle-mcp init`
- `waggle-mcp features`
- `waggle-mcp ingest-transcript-handoff`
- `waggle-mcp export-context-bundle`

## Proof points

- Checked-in comparison snapshot: about `2.6x` fewer tokens on factual lookups versus naive RAG baseline
- LongMemEval saved artifacts:
  - `graph_raw`: `97.4% R@5`, `88.4% Exact@5`
  - `graph_hybrid`: `96.4% R@5`, `85.6% Exact@5`
- Comprehensive feature demo: `46` calls, `0` errors, `25` unique actions covered

## How it is verified

- MCP stdio integration tests cover startup, tool registration, prompts, and resources
- Transcript handoff tests cover ingest, deduplication, extraction, and export behavior
- Smoke test exercises live `store_node`, `query_graph`, and `graph://stats`
- Checked-in benchmark artifacts document retrieval quality and token use

