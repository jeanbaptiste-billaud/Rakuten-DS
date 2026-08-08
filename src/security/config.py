"""Validated configuration for the SPIRE/Infisical security bootstrap."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse


class SecurityConfigurationError(ValueError):
    """Raised when the security bootstrap is configured incorrectly."""


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise SecurityConfigurationError(f"Missing required environment variable: {name}")
    return value


def _positive_float(environment: Mapping[str, str], name: str, default: float) -> float:
    raw_value = environment.get(name, str(default))
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise SecurityConfigurationError(f"{name} must be a number") from exc
    if value <= 0:
        raise SecurityConfigurationError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class SecuritySettings:
    """Public metadata needed to authenticate a workload and fetch its secrets."""

    spiffe_endpoint_socket: str
    spiffe_audience: str
    expected_spiffe_id: str | None
    infisical_domain: str
    infisical_identity_id: str
    infisical_project_id: str
    infisical_environment: str
    infisical_secret_path: str
    spire_timeout_seconds: float = 5.0
    http_timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "SecuritySettings":
        env = os.environ if environment is None else environment
        endpoint = env.get(
            "SPIFFE_ENDPOINT_SOCKET", "unix:///run/spire/sockets/agent.sock"
        ).strip()
        if not endpoint.startswith("unix://"):
            raise SecurityConfigurationError("SPIFFE_ENDPOINT_SOCKET must use unix://")

        domain = env.get("INFISICAL_DOMAIN", "http://infisical:8080").strip().rstrip("/")
        parsed_domain = urlparse(domain)
        if parsed_domain.scheme not in {"http", "https"} or not parsed_domain.netloc:
            raise SecurityConfigurationError("INFISICAL_DOMAIN must be an HTTP(S) URL")

        secret_path = _required(env, "INFISICAL_SECRET_PATH")
        if not secret_path.startswith("/"):
            raise SecurityConfigurationError("INFISICAL_SECRET_PATH must start with '/'")

        expected_id = env.get("SPIFFE_EXPECTED_ID", "").strip() or None
        if expected_id is not None and not expected_id.startswith("spiffe://"):
            raise SecurityConfigurationError("SPIFFE_EXPECTED_ID must be a SPIFFE ID")

        return cls(
            spiffe_endpoint_socket=endpoint,
            spiffe_audience=env.get("SPIFFE_AUDIENCE", "infisical").strip() or "infisical",
            expected_spiffe_id=expected_id,
            infisical_domain=domain,
            infisical_identity_id=_required(env, "INFISICAL_IDENTITY_ID"),
            infisical_project_id=_required(env, "INFISICAL_PROJECT_ID"),
            infisical_environment=env.get("INFISICAL_ENV", "dev").strip() or "dev",
            infisical_secret_path=secret_path,
            spire_timeout_seconds=_positive_float(env, "SPIRE_TIMEOUT_SECONDS", 5.0),
            http_timeout_seconds=_positive_float(env, "SECURITY_HTTP_TIMEOUT_SECONDS", 10.0),
        )
