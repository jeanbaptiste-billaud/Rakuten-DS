"""Authenticate, inject secrets into one child process, then replace this process."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence

from .config import SecurityConfigurationError, SecuritySettings
from .spire_infisical import SecurityBootstrapError, load_workload_secrets

LOGGER = logging.getLogger("security-bootstrap")
CONTROL_VARIABLES = {
    "SPIFFE_ENDPOINT_SOCKET",
    "SPIFFE_AUDIENCE",
    "SPIFFE_EXPECTED_ID",
    "SPIRE_TIMEOUT_SECONDS",
    "INFISICAL_DOMAIN",
    "INFISICAL_IDENTITY_ID",
    "INFISICAL_PROJECT_ID",
    "INFISICAL_ENV",
    "INFISICAL_SECRET_PATH",
    "SECURITY_HTTP_TIMEOUT_SECONDS",
}
LEGACY_CREDENTIAL_SUFFIXES = ("_CLIENT_ID", "_CLIENT_SECRET")


def build_child_environment(
    base_environment: Mapping[str, str], secrets: Mapping[str, str]
) -> dict[str, str]:
    """Build the worker environment without forwarding bootstrap credentials."""
    forbidden = CONTROL_VARIABLES.intersection(secrets)
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise SecurityBootstrapError(f"Secrets may not override security controls: {names}")

    child_environment = dict(base_environment)
    for key in list(child_environment):
        if key == "INFISICAL_TOKEN" or key.endswith(LEGACY_CREDENTIAL_SUFFIXES):
            child_environment.pop(key, None)
    for key in CONTROL_VARIABLES:
        child_environment.pop(key, None)
    child_environment.update(secrets)
    return child_environment


def parse_command(arguments: Sequence[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(
        description="Authenticate a SPIFFE workload and run a command with Infisical secrets."
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(arguments)
    command = list(parsed.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after '--'")
    return command


def main(arguments: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("SECURITY_LOG_LEVEL", "INFO"))
    command = parse_command(arguments)
    try:
        settings = SecuritySettings.from_env()
        secrets = load_workload_secrets(settings)
        child_environment = build_child_environment(os.environ, secrets)
    except (SecurityConfigurationError, SecurityBootstrapError) as exc:
        LOGGER.error("Security bootstrap refused to start the worker: %s", exc)
        return 78

    LOGGER.info(
        "Security bootstrap completed; launching %s with %d injected secrets",
        command[0],
        len(secrets),
    )
    try:
        os.execvpe(command[0], command, child_environment)
    except FileNotFoundError:
        LOGGER.error("Worker executable was not found: %s", command[0])
        return 127
    return 0


if __name__ == "__main__":
    sys.exit(main())
