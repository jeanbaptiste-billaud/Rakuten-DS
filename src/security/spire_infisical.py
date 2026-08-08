"""Exchange a SPIRE JWT-SVID for an Infisical access token and retrieve secrets."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import SecuritySettings

ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
JsonRequest = Callable[[str, str, dict[str, Any] | None, str | None, float], dict[str, Any]]


class SecurityBootstrapError(RuntimeError):
    """A fail-closed error that is safe to display without leaking credentials."""


def _request_json(
    url: str,
    method: str,
    payload: dict[str, Any] | None,
    bearer_token: str | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    request = Request(url=url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is validated config
            raw_response = response.read()
    except HTTPError as exc:
        raise SecurityBootstrapError(
            f"Infisical request failed with HTTP status {exc.code}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise SecurityBootstrapError("Infisical is unreachable") from exc

    try:
        decoded = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecurityBootstrapError("Infisical returned an invalid JSON response") from exc
    if not isinstance(decoded, dict):
        raise SecurityBootstrapError("Infisical returned an unexpected response")
    return decoded


def fetch_jwt_svid(
    settings: SecuritySettings,
    workload_client_factory: Callable[..., Any] | None = None,
) -> str:
    """Fetch a short-lived JWT-SVID from the local SPIRE Workload API."""
    if os.environ.get("SPIFFE_ENDPOINT_SOCKET") != settings.spiffe_endpoint_socket:
        os.environ["SPIFFE_ENDPOINT_SOCKET"] = settings.spiffe_endpoint_socket

    if workload_client_factory is None:
        try:
            from spiffe import WorkloadApiClient
        except ImportError as exc:
            raise SecurityBootstrapError("The 'spiffe' Python package is not installed") from exc
        workload_client_factory = WorkloadApiClient

    try:
        with workload_client_factory(
            default_timeout=settings.spire_timeout_seconds
        ) as workload_client:
            jwt_svid = workload_client.fetch_jwt_svid(
                audience={settings.spiffe_audience},
                timeout=settings.spire_timeout_seconds,
            )
    except SecurityBootstrapError:
        raise
    except Exception as exc:
        raise SecurityBootstrapError("Unable to obtain a JWT-SVID from SPIRE") from exc

    spiffe_id = str(jwt_svid.spiffe_id)
    if settings.expected_spiffe_id and spiffe_id != settings.expected_spiffe_id:
        raise SecurityBootstrapError("SPIRE returned an unexpected workload identity")
    token = getattr(jwt_svid, "token", "")
    if not token:
        raise SecurityBootstrapError("SPIRE returned an empty JWT-SVID")
    return token


def login_infisical(
    settings: SecuritySettings,
    jwt_svid: str,
    request_json: JsonRequest = _request_json,
) -> str:
    """Exchange the JWT-SVID for a short-lived Infisical access token."""
    response = request_json(
        f"{settings.infisical_domain}/api/v1/auth/spiffe-auth/login",
        "POST",
        {"identityId": settings.infisical_identity_id, "jwt": jwt_svid},
        None,
        settings.http_timeout_seconds,
    )
    access_token = response.get("accessToken")
    if not isinstance(access_token, str) or not access_token:
        raise SecurityBootstrapError("Infisical did not return an access token")
    return access_token


def fetch_infisical_secrets(
    settings: SecuritySettings,
    access_token: str,
    request_json: JsonRequest = _request_json,
) -> dict[str, str]:
    """Retrieve the configured secret path and normalize it as an environment map."""
    query = urlencode(
        {
            "projectId": settings.infisical_project_id,
            "environment": settings.infisical_environment,
            "secretPath": settings.infisical_secret_path,
            "viewSecretValue": "true",
            "expandSecretReferences": "true",
            "recursive": "false",
        }
    )
    response = request_json(
        f"{settings.infisical_domain}/api/v4/secrets?{query}",
        "GET",
        None,
        access_token,
        settings.http_timeout_seconds,
    )

    secret_entries = response.get("secrets")
    if not isinstance(secret_entries, list):
        raise SecurityBootstrapError("Infisical returned an invalid secrets payload")

    secrets: dict[str, str] = {}
    for entry in secret_entries:
        if not isinstance(entry, dict):
            raise SecurityBootstrapError("Infisical returned an invalid secret entry")
        key = entry.get("secretKey")
        value = entry.get("secretValue")
        if not isinstance(key, str) or not ENVIRONMENT_KEY.fullmatch(key):
            raise SecurityBootstrapError("A secret key is not a valid environment variable")
        if not isinstance(value, str):
            raise SecurityBootstrapError(f"Secret {key!r} has no string value")
        if key in secrets:
            raise SecurityBootstrapError(f"Duplicate secret key: {key}")
        secrets[key] = value
    return secrets


def load_workload_secrets(
    settings: SecuritySettings,
    *,
    workload_client_factory: Callable[..., Any] | None = None,
    request_json: JsonRequest = _request_json,
) -> dict[str, str]:
    """Run the complete bootstrap while keeping both short-lived tokens in memory."""
    jwt_svid = fetch_jwt_svid(settings, workload_client_factory)
    access_token = login_infisical(settings, jwt_svid, request_json)
    return fetch_infisical_secrets(settings, access_token, request_json)
