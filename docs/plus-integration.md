# Waggle Plus Integration Contract

This document defines the public contract that `Waggle Core` expects from a private `waggle_plus` package.

## Distribution model

- Core package: `waggle-mcp`
- Private extension module: `waggle_plus`

Core never bundles paid features. Instead, it detects an optional private module at runtime and enables reserved Plus-only surfaces when the module is present and advertises the required capabilities.

## Module detection

Core resolves the Plus module name from:

1. `WAGGLE_PLUS_MODULE`
2. fallback default: `waggle_plus`

Detection can be disabled with:

```env
WAGGLE_PLUS_DISABLED=true
```

## Capability advertisement

The private module should expose:

```python
WAGGLE_PLUS_CAPABILITIES = ("oidc_sso", "rbac")
```

Current reserved capability names in Core:

- `oidc_sso`
- `rbac`

## Identity provider contract

To back the reserved admin identity routes, the private module must provide one of:

```python
def build_identity_provider():
    ...
```

or:

```python
WAGGLE_PLUS_IDENTITY_PROVIDER = ...
```

The resulting provider object must implement:

```python
provider.status() -> dict[str, object]
provider.authorize_url(*, redirect_uri: str, state: str) -> str
provider.exchange_code(*, code: str, redirect_uri: str, state: str) -> dict[str, object]
provider.map_roles(*, session: dict[str, object]) -> dict[str, object]
provider.check_permission(*, resolved: dict[str, object], permission: str, resource_type: str = "", resource_id: str = "") -> dict[str, object]
```

## Reserved Core routes

When Plus is installed and advertises `oidc_sso`, Core exposes these routes as the public integration surface:

- `GET /api/admin/identity/provider`
- `POST /api/admin/identity/authorize`
- `POST /api/admin/identity/callback`
- `POST /api/admin/identity/roles/resolve`
- `POST /api/admin/identity/permissions/check`

Both routes require `admin:read` when an API key is presented.

Without Plus, Core intentionally returns `501` with a machine-readable contract payload.

## Normalized callback session payload

`exchange_code()` should return a JSON-serializable object with:

- required: `subject`
- optional: `issuer`, `email`, `display_name`, `roles`, `expires_at`

Core treats this as an identity-session payload contract only. It does not persist users or sessions yet.

## Role-mapping payload

`map_roles()` should return a JSON-serializable object with:

- required: `primary_role`
- optional: `roles`, `permissions`, `source_claims`

Supported Waggle role names in the public contract:

- `Owner`
- `Admin`
- `Developer`
- `Viewer`
- `Auditor`

## Permission decision payload

`check_permission()` should return a JSON-serializable object with:

- required: `allowed`
- optional: `reason`, `matched_roles`, `matched_permissions`

## Example private module shape

```python
WAGGLE_PLUS_CAPABILITIES = ("oidc_sso", "rbac")


class OIDCProvider:
    def status(self) -> dict[str, object]:
        return {
            "issuer": "https://id.example.com",
            "client_id": "waggle-plus-client",
            "supported_roles": ["Owner", "Admin", "Developer", "Viewer", "Auditor"],
        }

    def authorize_url(self, *, redirect_uri: str, state: str) -> str:
        return f"https://id.example.com/authorize?redirect_uri={redirect_uri}&state={state}"

    def exchange_code(self, *, code: str, redirect_uri: str, state: str) -> dict[str, object]:
        return {
            "subject": "user_123",
            "issuer": "https://id.example.com",
            "email": "admin@example.com",
            "display_name": "Admin User",
            "roles": ["Owner", "Admin"],
            "expires_at": "2026-05-06T12:30:00Z",
        }

    def map_roles(self, *, session: dict[str, object]) -> dict[str, object]:
        source_roles = list(session.get("roles", []))
        primary_role = "Admin" if "Admin" in source_roles else "Viewer"
        return {
            "primary_role": primary_role,
            "roles": source_roles,
            "permissions": ["read:data", "write:data"] if primary_role == "Admin" else ["read:data"],
            "source_claims": {"roles": source_roles},
        }

    def check_permission(
        self,
        *,
        resolved: dict[str, object],
        permission: str,
        resource_type: str = "",
        resource_id: str = "",
    ) -> dict[str, object]:
        permissions = list(resolved.get("permissions", []))
        allowed = permission in permissions
        return {
            "allowed": allowed,
            "reason": "matched_permission" if allowed else "permission_missing",
            "matched_roles": list(resolved.get("roles", [])),
            "matched_permissions": [permission] if allowed else [],
        }


def build_identity_provider() -> OIDCProvider:
    return OIDCProvider()
```
