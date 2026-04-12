# Runbook: Incident Response

**Audience:** On-call engineer  
**Applies to:** graph-memory-mcp in hosted HTTP mode  
**Last reviewed:** 2026-04-12

---

## Severity levels

| Level | Definition | Response time |
|-------|-----------|---------------|
| P1 | Service completely unavailable / all requests failing | Immediate |
| P2 | Degraded performance (high latency / error rate > 5 %) | < 15 min |
| P3 | Partial failure (one tool broken, overall service healthy) | < 1 h |
| P4 | Cosmetic / logging / metric anomaly | Next business day |

---

## Scenario 1 — Neo4j unreachable

**Symptoms:**
- `/health/ready` returns HTTP 503
- `graph_memory_tool_requests_total{status="error"}` spikes
- Logs contain `ServiceUnavailableError` or `neo4j.exceptions`

**Steps:**
1. Confirm Neo4j connectivity from within the pod:
   ```bash
   kubectl exec -it deploy/graph-memory -- \
     python -c "from neo4j import GraphDatabase; \
     GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j','change-me')).verify_connectivity()"
   ```
2. Check Neo4j pod / service health:
   ```bash
   kubectl get pod -l app=neo4j
   kubectl logs -l app=neo4j --tail=50
   ```
3. If Neo4j is down, restart it:
   ```bash
   kubectl rollout restart deployment/neo4j
   ```
4. Once Neo4j is back, graph-memory readiness probe will recover automatically
   (no restart needed).
5. If Neo4j is not in-cluster, verify the `GRAPH_MEMORY_NEO4J_URI` points to the
   correct host and the network / firewall allows port 7687.

---

## Scenario 2 — Embedding model failure

**Symptoms:**
- `store_node`, `query_graph`, `decompose_and_store` return errors
- Startup log contains `embedding model` error
- `graph_memory_startup_validation_seconds` metric absent

**Steps:**
1. Check pod logs for the model error:
   ```bash
   kubectl logs deploy/graph-memory --tail=100 | grep -i embed
   ```
2. Common causes:
   - **Model not downloaded**: The HuggingFace model downloads on first boot.
     Increase `startupProbe.failureThreshold` or pull the model into the
     Docker image at build time.
   - **Wrong model name**: Verify `GRAPH_MEMORY_MODEL` in the ConfigMap matches
     a valid sentence-transformers model name.
3. Force a pod restart after fixing config:
   ```bash
   kubectl rollout restart deployment/graph-memory
   ```

---

## Scenario 3 — Rate limit spike (429 storm)

**Symptoms:**
- `graph_memory_rate_limit_rejections_total` rises sharply
- Clients receive HTTP 429 responses

**Steps:**
1. Identify the offending API key from logs:
   ```bash
   kubectl logs deploy/graph-memory --tail=200 \
     | grep rate_lim | grep -oP '"api_key_id":"[^"]+"' | sort | uniq -c | sort -rn
   ```
2. If a runaway agent is hammering the API, revoke its key:
   ```bash
   graph-memory-mcp revoke-api-key --api-key-id <id>
   ```
3. If legitimate traffic is hitting limits, temporarily raise the cap:
   ```bash
   kubectl edit configmap graph-memory-config
   # Increase GRAPH_MEMORY_RATE_LIMIT_RPM
   kubectl rollout restart deployment/graph-memory
   ```
4. Long-term: tune rate limit values in `configmap.yaml` and re-apply.

---

## Scenario 4 — Sustained auth failures

**Symptoms:**
- `graph_memory_auth_failures_total` > baseline
- Logs show `AuthenticationError: Invalid API key`

**Steps:**
1. Check if a recently rotated key was not fully distributed:
   - See [api-key-rotation.md](./api-key-rotation.md) Step 2.
2. If this looks like a brute-force attempt, check source IPs at the ingress
   and consider adding an IP allow-list annotation to the Ingress.
3. If a key is compromised:
   ```bash
   graph-memory-mcp revoke-api-key --api-key-id <compromised-id>
   ```

---

## Scenario 5 — High memory / OOM kill

**Symptoms:**
- Pod restarts with `OOMKilled` in `kubectl describe pod`
- Large graph or many concurrent decompose_and_store / observe_conversation calls

**Steps:**
1. Check current memory usage:
   ```bash
   kubectl top pod -l app=graph-memory
   ```
2. Raise the memory limit in `deployment.yaml` and re-apply.
3. If community detection (`get_topics`) is running on a very large graph, it
   loads the entire adjacency matrix in memory. Consider adding a node count
   guard before calling it in production.

---

## General diagnostics

```bash
# Tail live logs
kubectl logs -f deploy/graph-memory

# Check recent events
kubectl get events --sort-by=.lastTimestamp | tail -20

# Port-forward for direct access
kubectl port-forward svc/graph-memory 8080:80
curl http://localhost:8080/health/ready
curl http://localhost:8080/metrics

# Quick metric snapshot
curl -s http://localhost:8080/metrics | grep graph_memory
```

---

## Escalation

| Condition | Escalate to |
|-----------|-------------|
| Neo4j data corruption suspected | DB team + backup restore (see [backup-restore.md](./backup-restore.md)) |
| Security incident | Security team + revoke all keys for affected tenant |
| Persistent OOM after limit increase | Platform team for capacity planning |
