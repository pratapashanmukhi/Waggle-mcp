# Runbook: New Tenant Onboarding

**Audience:** Platform team  
**Applies to:** graph-memory-mcp in hosted HTTP mode  
**Last reviewed:** 2026-04-12

---

## Overview

Each tenant gets an isolated namespace in the graph.  All nodes and edges are
stamped with `tenant_id`; cross-tenant queries are not possible.

---

## Pre-checklist

- [ ] Obtain the requested `tenant_id` name (e.g., `team-alpha`).
      Must be DNS-safe: lowercase letters, digits, hyphens only.
- [ ] Confirm the tenant's primary use-case (number of agents, expected RPM).
- [ ] Confirm the initial API key owner and rotation schedule.

---

## Step 1 — Create the tenant

```bash
graph-memory-mcp create-tenant \
  --tenant-id <tenant-id> \
  --name "<Human Readable Name>"
```

This initialises the tenant's namespace in the backend (SQLite or Neo4j).

---

## Step 2 — Create the first API key

```bash
graph-memory-mcp create-api-key \
  --tenant-id <tenant-id> \
  --name "primary-agent"
```

> [!IMPORTANT]
> The raw key is returned only once. Send it to the tenant via a secure
> channel (e.g., 1Password share link, Vault, or encrypted email).

Record the `api_key_id` in the key inventory.

---

## Step 3 — Verify tenant isolation

From your own shell with the new key:

```bash
BASE=https://graph-memory.example.com

# Health check
curl -sf -H "X-API-Key: <raw-key>" "$BASE/health/ready"

# Write a node
curl -s -X POST "$BASE/mcp" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <raw-key>" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"store_node","arguments":{"label":"onboarding-test","content":"Tenant isolation check","node_type":"fact"}}}'

# Read it back
curl -s -X POST "$BASE/mcp" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <raw-key>" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"query_graph","arguments":{"query":"onboarding-test"}}}'

# Confirm no cross-tenant leakage by querying with a different tenant key
# and verifying the node does NOT appear.
```

---

## Step 4 — Document the tenant

Add an entry to your internal tenant registry:

| Field | Value |
|-------|-------|
| `tenant_id` | `team-alpha` |
| `display_name` | Team Alpha |
| `created_at` | 2026-04-12 |
| `primary_owner` | alice@example.com |
| `key_rotate_by` | 2026-07-12 |
| `expected_rpm` | 60 |

---

## Step 5 — Hand over to tenant

Share with the tenant:

- The raw API key (via secure channel).
- The endpoint: `https://graph-memory.example.com/mcp`
- Headers required: `Content-Type: application/json`, `X-API-Key: <key>`
- The list of available MCP tools (from `README.md`).
- Link to the API key rotation runbook: [api-key-rotation.md](./api-key-rotation.md).

---

## Off-boarding a tenant

```bash
# 1. Revoke all API keys for the tenant
graph-memory-mcp list-api-keys --tenant-id <tenant-id>
graph-memory-mcp revoke-api-key --api-key-id <id1>
graph-memory-mcp revoke-api-key --api-key-id <id2>
# ... repeat for all keys

# 2. Export a final backup of their data before deletion (optional)
#    (via the MCP tool using a remaining admin key before revocation)

# 3. Remove the tenant entry from the key inventory.
# 4. Data retention: per your data-retention policy, either purge or archive
#    the tenant's nodes/edges in Neo4j.
```
