# Runbook: Backup and Restore

**Audience:** Platform team / on-call engineer  
**Applies to:** graph-memory-mcp (both SQLite and Neo4j backends)  
**Last reviewed:** 2026-04-12

---

## Overview

The graph is exported as a portable JSON file that includes all nodes, edges,
`tenant_id`, and a `schema_version` field.  The same file can be imported into
any backend.

Use the automated drill script for routine validation:
`scripts/backup_restore_drill.sh`

---

## Manual backup procedure

### Via MCP tool (HTTP mode)

```bash
BASE=https://graph-memory.example.com
API_KEY=<your-raw-key>
BACKUP_FILE="backup-$(date +%Y%m%d-%H%M%S).json"

curl -s -X POST "$BASE/mcp" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"export_graph_backup\",\"arguments\":{\"output_path\":\"/app/tmp/$BACKUP_FILE\"}}}" \
  | jq .

# Copy the backup out of the pod
kubectl cp graph-memory-<pod>:/app/tmp/$BACKUP_FILE ./$BACKUP_FILE
```

### Via Python script (direct, for SQLite)

```bash
GRAPH_MEMORY_BACKEND=sqlite \
GRAPH_MEMORY_DB_PATH=./memory.db \
GRAPH_MEMORY_DEFAULT_TENANT_ID=local-default \
  python -c "
from graph_memory.graph import MemoryGraph
from graph_memory.embeddings import EmbeddingModel
g = MemoryGraph('memory.db', EmbeddingModel('all-MiniLM-L6-v2'), tenant_id='local-default')
r = g.export_graph_backup('backup.json')
print(f'Exported {r.node_count} nodes and {r.edge_count} edges')
"
```

---

## Manual restore procedure

> [!CAUTION]
> Importing merges data into the target tenant.  If you want a clean restore,
> purge the tenant's data in Neo4j first:
> ```cypher
> MATCH (n {tenant_id: 'workspace-a'}) DETACH DELETE n
> ```

```bash
BASE=https://graph-memory.example.com
API_KEY=<your-raw-key>
BACKUP_FILE=backup-20260412-120000.json

# Upload the backup file into the pod
kubectl cp $BACKUP_FILE graph-memory-<pod>:/app/tmp/$BACKUP_FILE

# Trigger import via MCP tool
curl -s -X POST "$BASE/mcp" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"import_graph_backup\",\"arguments\":{\"input_path\":\"/app/tmp/$BACKUP_FILE\"}}}" \
  | jq '{nodes_created:.structuredContent.nodes_created, nodes_updated:.structuredContent.nodes_updated, edges_created:.structuredContent.edges_created}'
```

---

## Verifying a restore

After import, run a spot-check via `query_graph` or `get_stats`:

```bash
curl -s -X POST "$BASE/mcp" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_stats","arguments":{}}}' \
  | jq '.structuredContent | {total_nodes, total_edges}'
```

Compare `total_nodes` and `total_edges` against the `node_count` and
`edge_count` values from the export step.

---

## Automated drill

Run the automated end-to-end drill (requires a running HTTP server):

```bash
# Minimal
GRAPH_MEMORY_HOST=http://localhost:8080 \
GRAPH_MEMORY_API_KEY=<key> \
  ./scripts/backup_restore_drill.sh

# With JSON output for CI
.venv/bin/python scripts/backup_restore_drill.py \
  --host http://localhost:8080 \
  --api-key <key> \
  --json
```

See [backup_restore_drill.sh](../../scripts/backup_restore_drill.sh) and
[backup_restore_drill.py](../../scripts/backup_restore_drill.py) for details.

---

## Backup retention policy (recommended)

| Frequency | Retain for |
|-----------|-----------|
| Daily | 7 days |
| Weekly | 4 weeks |
| Monthly | 12 months |

Store backups in object storage (S3 / GCS) outside the cluster.
