from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any, Protocol

_DEFAULT_PLUS_MODULE = "waggle_plus"
_TRUE_VALUES = {"1", "true", "yes", "on"}
PLUS_CAPABILITY_OIDC_SSO = "oidc_sso"
PLUS_CAPABILITY_RBAC = "rbac"


class IdentityProviderProtocol(Protocol):
    def status(self) -> dict[str, Any]: ...

    def authorize_url(self, *, redirect_uri: str, state: str) -> str: ...

    def exchange_code(self, *, code: str, redirect_uri: str, state: str) -> dict[str, Any]: ...

    def map_roles(self, *, session: dict[str, Any]) -> dict[str, Any]: ...

    def check_permission(
        self,
        *,
        resolved: dict[str, Any],
        permission: str,
        resource_type: str = "",
        resource_id: str = "",
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class PlusStatus:
    module_name: str
    enabled: bool
    installed: bool
    version: str | None
    capabilities: tuple[str, ...]
    message: str
    import_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "enabled": self.enabled,
            "installed": self.installed,
            "version": self.version,
            "capabilities": list(self.capabilities),
            "message": self.message,
            "import_error": self.import_error,
        }


def _resolve_module_name(module_name: str | None = None) -> str:
    resolved = (module_name or os.environ.get("WAGGLE_PLUS_MODULE", _DEFAULT_PLUS_MODULE)).strip()
    return resolved or _DEFAULT_PLUS_MODULE


def detect_plus(module_name: str | None = None) -> PlusStatus:
    resolved_name = _resolve_module_name(module_name)
    if os.environ.get("WAGGLE_PLUS_DISABLED", "").strip().lower() in _TRUE_VALUES:
        return PlusStatus(
            module_name=resolved_name,
            enabled=False,
            installed=False,
            version=None,
            capabilities=(),
            message="Waggle Plus detection is disabled by WAGGLE_PLUS_DISABLED.",
        )

    try:
        module = import_module(resolved_name)
    except ModuleNotFoundError as exc:
        if exc.name == resolved_name:
            return PlusStatus(
                module_name=resolved_name,
                enabled=True,
                installed=False,
                version=None,
                capabilities=(),
                message="Waggle Plus is coming soon. Core is fully usable without it.",
            )
        return PlusStatus(
            module_name=resolved_name,
            enabled=True,
            installed=False,
            version=None,
            capabilities=(),
            message="Waggle Plus could not be imported because one of its dependencies is missing.",
            import_error=str(exc),
        )
    except Exception as exc:
        return PlusStatus(
            module_name=resolved_name,
            enabled=True,
            installed=False,
            version=None,
            capabilities=(),
            message="Waggle Plus was detected but failed to import cleanly.",
            import_error=str(exc),
        )

    capabilities = _read_capabilities(module)
    detected_version = _read_version(module, resolved_name)
    return PlusStatus(
        module_name=resolved_name,
        enabled=True,
        installed=True,
        version=detected_version,
        capabilities=capabilities,
        message="Waggle Plus module detected.",
    )


def load_plus_module(module_name: str | None = None) -> tuple[PlusStatus, Any | None]:
    status = detect_plus(module_name)
    if not status.installed:
        return status, None
    try:
        return status, import_module(status.module_name)
    except Exception:
        return detect_plus(module_name), None


def has_plus_capability(capability: str, module_name: str | None = None) -> bool:
    status = detect_plus(module_name)
    return status.installed and capability.strip() in status.capabilities


def load_identity_provider(module_name: str | None = None) -> tuple[PlusStatus, Any | None]:
    status, module = load_plus_module(module_name)
    if module is None:
        return status, None
    provider_factory = getattr(module, "build_identity_provider", None)
    if callable(provider_factory):
        return status, provider_factory()
    provider = getattr(module, "WAGGLE_PLUS_IDENTITY_PROVIDER", None)
    return status, provider


def validate_identity_provider(provider: Any) -> list[str]:
    if provider is None:
        return ["status", "authorize_url", "exchange_code", "map_roles", "check_permission"]
    missing: list[str] = []
    for method_name in ("status", "authorize_url", "exchange_code", "map_roles", "check_permission"):
        if not callable(getattr(provider, method_name, None)):
            missing.append(method_name)
    return missing


def describe_plus_contract(module_name: str | None = None) -> dict[str, Any]:
    resolved_name = _resolve_module_name(module_name)
    return {
        "module_name": resolved_name,
        "distribution": "private-package",
        "capabilities": {
            "identity": [PLUS_CAPABILITY_OIDC_SSO],
            "access_control": [PLUS_CAPABILITY_RBAC],
        },
        "identity_provider": {
            "factory": "build_identity_provider",
            "fallback_attribute": "WAGGLE_PLUS_IDENTITY_PROVIDER",
            "required_methods": ["status", "authorize_url", "exchange_code", "map_roles", "check_permission"],
            "reserved_routes": [
                {"path": "/api/admin/identity/provider", "method": "GET", "required_scope": "admin:read"},
                {"path": "/api/admin/identity/authorize", "method": "POST", "required_scope": "admin:read"},
                {"path": "/api/admin/identity/callback", "method": "POST", "required_scope": "admin:read"},
                {"path": "/api/admin/identity/roles/resolve", "method": "POST", "required_scope": "admin:read"},
                {"path": "/api/admin/identity/permissions/check", "method": "POST", "required_scope": "admin:read"},
            ],
            "session_payload": {
                "required_fields": ["subject"],
                "optional_fields": ["issuer", "email", "display_name", "roles", "expires_at"],
            },
            "role_mapping": {
                "required_fields": ["primary_role"],
                "optional_fields": ["roles", "permissions", "source_claims"],
                "supported_roles": ["Owner", "Admin", "Developer", "Viewer", "Auditor"],
            },
            "permission_check": {
                "required_fields": ["allowed"],
                "optional_fields": ["reason", "matched_roles", "matched_permissions"],
            },
        },
    }


def _read_capabilities(module: Any) -> tuple[str, ...]:
    raw_capabilities = getattr(module, "WAGGLE_PLUS_CAPABILITIES", ())
    if raw_capabilities is None:
        return ()
    values: list[str] = []
    for item in raw_capabilities:
        value = str(item).strip()
        if value:
            values.append(value)
    return tuple(values)


def _read_version(module: Any, module_name: str) -> str | None:
    explicit_version = getattr(module, "__version__", None)
    if explicit_version:
        return str(explicit_version)
    candidates = [module_name, module_name.replace("_", "-")]
    for candidate in candidates:
        try:
            return package_version(candidate)
        except PackageNotFoundError:
            continue
    return None
