# Runbook: API Key Rotation

**Audience:** Platform / ops team  
**Applies to:** graph-memory-mcp HTTP (hosted) mode  
**Last reviewed:** 2026-04-12

---

## Why rotate API keys?

- Suspected or confirmed credential leak
- Periodic security policy requirement (e.g., every 90 days)
- Employee / service off-boarding
- Audit finding

---

## Zero-downtime rotation procedure

The strategy is **create-then-revoke**: issue the new key first, update all
consumers, then revoke the old key.  No downtime is required.

### Step 1 — Create the replacement key

```bash
# Replace <tenant-id> and <name> with real values
graph-memory-mcp create-api-key \
  --tenant-id <tenant-id> \
  --name "ci-agent-rotated-$(date +%Y%m%d)"
```

> [!IMPORTANT]
> The raw key is returned **once**. Copy it to a secure store immediately.
> Only a SHA-256 hash is persisted in the database.

Record the new `api_key_id` and raw key.

### Step 2 — Distribute the new key to consumers

For each consumer of the old key (CI pipelines, agent configs, Codex TOML,
monitoring scripts):

1. Update the secret or environment variable to the new raw key value.
2. Verify the consumer can authenticate:
   ```bash
   curl -sf -H "X-API-Key: <NEW_RAW_KEY>" \
     https://graph-memory.example.com/health/ready
   ```

### Step 3 — Verify no consumer is still using the old key

Check the `/metrics` endpoint for `graph_memory_auth_failures_total`.
A spike during this window may indicate a consumer still using the old key.

```bash
curl -s https://graph-memory.example.com/metrics \
  | grep auth_failures
```

Allow a soak period (≥ 5 minutes) before proceeding.

### Step 4 — Revoke the old key

```bash
# Get the old key ID from the list command
graph-memory-mcp list-api-keys --tenant-id <tenant-id>

# Revoke by ID
graph-memory-mcp revoke-api-key --api-key-id <old-api-key-id>
```

The old key is immediately rejected by the server; no restart required.

### Step 5 — Confirm revocation

```bash
curl -v -H "X-API-Key: <OLD_RAW_KEY>" \
  https://graph-memory.example.com/health/ready
# Expect: HTTP 401
```

---

## Emergency key revocation (suspected leak)

If a key must be revoked immediately without a ready replacement:

```bash
graph-memory-mcp revoke-api-key --api-key-id <compromised-key-id>
```

Then follow steps 1–4 above to issue and distribute a new key.
Monitor `graph_memory_auth_failures_total` for the next 30 minutes.

---

## Key inventory

Track all active keys in your team's secret manager. Each entry should record:

| Field | Example |
|-------|---------|
| `api_key_id` | `abc123...` |
| `tenant_id` | `workspace-a` |
| `name` | `ci-agent` |
| `issued_at` | `2026-01-10` |
| `rotate_by` | `2026-04-10` |
| `owned_by` | `platform-team` |

---

## Related runbooks

- [Incident response](./incident-response.md)
- [Secret management](./secret-management.md)
- [Onboarding a new tenant](./onboarding.md)
