# Runbook: Secret Management

**Audience:** Platform team  
**Applies to:** graph-memory-mcp Kubernetes deployment  
**Last reviewed:** 2026-04-12

---

## Principles

1. **Never commit real credentials to Git.**  
   `secret.example.yaml` is a template with placeholder values only.
2. **Use external secret management in production.**  
   `external-secret.example.yaml` integrates with AWS Secrets Manager,
   HashiCorp Vault, or GCP Secret Manager via the
   [External Secrets Operator](https://external-secrets.io/).
3. **TLS certificates are managed by cert-manager.**  
   See `certificate.yaml` — do not hand-manage TLS secrets.

---

## Kubernetes Secret (manual / dev path)

Use this path only for local or staging environments without an external
secrets backend.

```bash
# Copy the example and fill real values
cp deploy/kubernetes/secret.example.yaml deploy/kubernetes/secret.yaml

# Edit secret.yaml — set GRAPH_MEMORY_NEO4J_USERNAME and
# GRAPH_MEMORY_NEO4J_PASSWORD to real values.
# IMPORTANT: add secret.yaml to .gitignore immediately.

echo "deploy/kubernetes/secret.yaml" >> .gitignore

kubectl apply -f deploy/kubernetes/secret.yaml
```

---

## External Secrets Operator (production path)

### Install the operator

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets --create-namespace
```

### Configure a SecretStore

Choose your backend.  Example for AWS Secrets Manager:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: my-secret-store
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        # Uses IRSA (IAM Roles for Service Accounts) — no static credentials needed
        jwt:
          serviceAccountRef:
            name: external-secrets
            namespace: external-secrets
```

### Create the secret in AWS

```bash
aws secretsmanager create-secret \
  --name prod/graph-memory/neo4j \
  --secret-string '{"username":"neo4j","password":"<real-password>"}'
```

### Apply the ExternalSecret

```bash
cp deploy/kubernetes/external-secret.example.yaml \
   deploy/kubernetes/external-secret.yaml
# Edit: set the correct key path and SecretStore name.
kubectl apply -f deploy/kubernetes/external-secret.yaml
```

The operator will create and keep `graph-memory-secret` in sync.
The deployment picks up changes on pod restart or via a rolling update.

### Verify sync

```bash
kubectl get externalsecret graph-memory-secret
# READY column should show True or Synced
```

---

## TLS Certificate management (cert-manager)

### Install cert-manager

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml
```

### Apply issuers and certificate

```bash
# Edit certificate.yaml: replace ops@example.com and graph-memory.example.com
kubectl apply -f deploy/kubernetes/certificate.yaml
```

### Verify certificate issuance

```bash
kubectl describe certificate graph-memory-tls
# Look for: "Certificate is up to date and has not expired"

kubectl get secret graph-memory-tls -o jsonpath='{.data.tls\.crt}' \
  | base64 -d | openssl x509 -noout -dates
```

### Promote from staging to production

1. Verify the staging cert is issued and the Ingress serves HTTPS correctly.
2. Edit `certificate.yaml`: change `issuerRef.name` from `letsencrypt-staging`
   to `letsencrypt-prod`.
3. Edit `ingress.yaml`: change the annotation to `cert-manager.io/cluster-issuer: letsencrypt-prod`.
4. Apply both:
   ```bash
   kubectl apply -f deploy/kubernetes/certificate.yaml
   kubectl apply -f deploy/kubernetes/ingress.yaml
   ```
5. cert-manager will issue a new, browser-trusted certificate within minutes.

---

## Rotating Neo4j credentials

1. Update the secret in your external store (AWS Secrets Manager / Vault):
   ```bash
   aws secretsmanager put-secret-value \
     --secret-id prod/graph-memory/neo4j \
     --secret-string '{"username":"neo4j","password":"<new-password>"}'
   ```
2. The External Secrets Operator syncs within `refreshInterval` (default 2 min).
3. Trigger a rolling restart to pick up the new secret:
   ```bash
   kubectl rollout restart deployment/graph-memory
   ```
4. Also update the Neo4j server to accept the new password before restarting
   graph-memory to avoid a gap.

---

## Related runbooks

- [API key rotation](./api-key-rotation.md)
- [Incident response](./incident-response.md)
