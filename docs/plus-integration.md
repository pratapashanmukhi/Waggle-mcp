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
```

## Reserved Core routes

When Plus is installed and advertises `oidc_sso`, Core exposes these routes as the public integration surface:

- `GET /api/admin/identity/provider`
- `POST /api/admin/identity/authorize`

Both routes require `admin:read` when an API key is presented.

Without Plus, Core intentionally returns `501` with a machine-readable contract payload.

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


def build_identity_provider() -> OIDCProvider:
    return OIDCProvider()
```
