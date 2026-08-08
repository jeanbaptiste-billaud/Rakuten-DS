from __future__ import annotations

from dataclasses import replace
from urllib.parse import parse_qs, urlparse

import pytest

from src.security.config import SecurityConfigurationError, SecuritySettings
from src.security.spire_infisical import (
    SecurityBootstrapError,
    fetch_infisical_secrets,
    fetch_jwt_svid,
    load_workload_secrets,
)


def settings() -> SecuritySettings:
    return SecuritySettings(
        spiffe_endpoint_socket="unix:///run/spire/sockets/agent.sock",
        spiffe_audience="infisical",
        expected_spiffe_id="spiffe://rakuten.local/workload/training",
        infisical_domain="https://infisical.example.test",
        infisical_identity_id="identity-id",
        infisical_project_id="project-id",
        infisical_environment="dev",
        infisical_secret_path="/workers/training",
    )


class JwtSvid:
    token = "jwt-svid-value"
    spiffe_id = "spiffe://rakuten.local/workload/training"


class WorkloadClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def fetch_jwt_svid(self, **kwargs):
        assert kwargs["audience"] == {"infisical"}
        return JwtSvid()


def test_settings_are_validated():
    with pytest.raises(SecurityConfigurationError, match="INFISICAL_IDENTITY_ID"):
        SecuritySettings.from_env(
            {
                "INFISICAL_PROJECT_ID": "project",
                "INFISICAL_SECRET_PATH": "/path",
            }
        )


def test_unexpected_spiffe_identity_is_rejected():
    with pytest.raises(SecurityBootstrapError, match="unexpected workload identity"):
        fetch_jwt_svid(
            replace(settings(), expected_spiffe_id="spiffe://rakuten.local/workload/other"),
            WorkloadClient,
        )


def test_complete_exchange_does_not_return_tokens():
    calls = []

    def request_json(url, method, payload, bearer, timeout):
        calls.append((url, method, payload, bearer, timeout))
        if url.endswith("/api/v1/auth/spiffe-auth/login"):
            assert payload == {"identityId": "identity-id", "jwt": "jwt-svid-value"}
            assert bearer is None
            return {"accessToken": "infisical-access-token"}
        assert bearer == "infisical-access-token"
        query = parse_qs(urlparse(url).query)
        assert query["projectId"] == ["project-id"]
        assert query["secretPath"] == ["/workers/training"]
        return {"secrets": [{"secretKey": "MLFLOW_TOKEN", "secretValue": "value"}]}

    result = load_workload_secrets(
        settings(), workload_client_factory=WorkloadClient, request_json=request_json
    )

    assert result == {"MLFLOW_TOKEN": "value"}
    assert len(calls) == 2
    assert "jwt-svid-value" not in repr(result)
    assert "infisical-access-token" not in repr(result)


def test_invalid_or_duplicate_secret_keys_fail_closed():
    def invalid_key(*_args):
        return {"secrets": [{"secretKey": "BAD-KEY", "secretValue": "value"}]}

    with pytest.raises(SecurityBootstrapError, match="valid environment variable"):
        fetch_infisical_secrets(settings(), "access-token", invalid_key)

    def duplicate_key(*_args):
        return {
            "secrets": [
                {"secretKey": "VALID_KEY", "secretValue": "first"},
                {"secretKey": "VALID_KEY", "secretValue": "second"},
            ]
        }

    with pytest.raises(SecurityBootstrapError, match="Duplicate secret key"):
        fetch_infisical_secrets(settings(), "access-token", duplicate_key)
