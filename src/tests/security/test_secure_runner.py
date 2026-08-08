import pytest

from src.security.secure_runner import build_child_environment, parse_command
from src.security.spire_infisical import SecurityBootstrapError


def test_build_child_environment_scrubs_bootstrap_credentials():
    child = build_child_environment(
        {
            "PATH": "/usr/bin",
            "INFISICAL_TOKEN": "legacy-token",
            "INFISICAL_TRAINING_CLIENT_ID": "legacy-id",
            "INFISICAL_TRAINING_CLIENT_SECRET": "legacy-secret",
            "INFISICAL_PROJECT_ID": "public-project-metadata",
        },
        {"DATABASE_URL": "postgres://secret"},
    )

    assert child["PATH"] == "/usr/bin"
    assert child["DATABASE_URL"] == "postgres://secret"
    assert "INFISICAL_TOKEN" not in child
    assert "INFISICAL_TRAINING_CLIENT_ID" not in child
    assert "INFISICAL_TRAINING_CLIENT_SECRET" not in child
    assert "INFISICAL_PROJECT_ID" not in child


def test_secrets_cannot_override_security_configuration():
    with pytest.raises(SecurityBootstrapError, match="may not override"):
        build_child_environment({}, {"SPIFFE_ENDPOINT_SOCKET": "malicious"})


def test_parse_command_preserves_arguments_without_shell_joining():
    assert parse_command(["--", "python", "worker.py", "value with spaces"]) == [
        "python",
        "worker.py",
        "value with spaces",
    ]


def test_parse_command_requires_an_executable():
    with pytest.raises(SystemExit):
        parse_command([])
