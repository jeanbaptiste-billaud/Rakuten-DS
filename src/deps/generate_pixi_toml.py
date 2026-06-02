#!/usr/bin/env python3
"""
Generate a pixi.toml manifest from a central dependency catalog YAML.

Expected YAML shape, compatible with docker_dependencies_catalog_uniformized.yaml:

catalog:
  pypi:
    fastapi: ==0.128.0
    uvicorn:
      version: ==0.40.0
      extras: [standard]
features:
  api_gateway:
    pypi: [fastapi, uvicorn]
environments:
  api-gateway: [api_gateway]

Optional YAML shape:

workspace:
  channels: [conda-forge]
  platforms: [linux-64]
catalog:
  conda:
    python: 3.12.*
  pypi: {...}
features:
  common:
    conda: [python]
    pypi: [pydantic]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def toml_string(value: str) -> str:
    """Small TOML string serializer for simple strings."""
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def toml_list(values: list[str]) -> str:
    return "[" + ", ".join(toml_string(str(v)) for v in values) + "]"


def toml_value(value: Any) -> str:
    """Serialize the limited value types used in this manifest."""
    if isinstance(value, str):
        return toml_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return toml_list([str(v) for v in value])
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            parts.append(f"{key} = {toml_value(val)}")
        return "{ " + ", ".join(parts) + " }"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def normalize_requirement(spec: Any) -> Any:
    """
    Convert catalog entries to Pixi-compatible dependency specs.

    YAML string:
      fastapi: ==0.128.0
    becomes:
      fastapi = "==0.128.0"

    YAML dict:
      uvicorn:
        version: ==0.40.0
        extras: [standard]
    becomes:
      uvicorn = { version = "==0.40.0", extras = ["standard"] }
    """
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        allowed = {"version", "extras", "markers", "index", "git", "branch", "tag", "rev", "path", "editable"}
        unknown = set(spec) - allowed
        if unknown:
            raise ValueError(f"Unsupported dependency keys {sorted(unknown)} in spec {spec!r}")
        return spec
    raise ValueError(f"Dependency spec must be a string or dict, got: {spec!r}")


def get_catalog(data: dict[str, Any], kind: str) -> dict[str, Any]:
    return dict(data.get("catalog", {}).get(kind, {}) or {})


def feature_dependencies(
    feature_name: str,
    feature_def: dict[str, Any],
    catalog: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    names = feature_def.get(kind, []) or []
    deps = {}
    missing = []
    for package_name in names:
        if package_name not in catalog:
            missing.append(package_name)
            continue
        deps[package_name] = normalize_requirement(catalog[package_name])
    if missing:
        raise KeyError(
            f"Feature {feature_name!r} references missing {kind} dependencies: {', '.join(missing)}"
        )
    return deps


def write_table(lines: list[str], header: str, values: dict[str, Any]) -> None:
    if not values:
        return
    lines.append(f"[{header}]")
    for key in sorted(values):
        lines.append(f"{key} = {toml_value(values[key])}")
    lines.append("")


def generate_pixi_toml(
    data: dict[str, Any],
    python_version: str | None = None,
    include_empty_environments: bool = False,
) -> str:
    workspace = data.get("workspace", {}) or {}
    channels = workspace.get("channels", ["conda-forge"])
    platforms = workspace.get("platforms", ["linux-64"])

    conda_catalog = get_catalog(data, "conda")
    pypi_catalog = get_catalog(data, "pypi")
    features = data.get("features", {}) or {}
    environments = data.get("environments", {}) or {}

    if not features:
        raise ValueError("No features found in YAML.")
    if not environments:
        # Fallback: one environment per feature.
        environments = {name: [name] for name in features}

    lines: list[str] = []
    lines.append("# Generated file. Do not edit manually.")
    lines.append("# Source of truth: dependencies YAML catalog.")
    lines.append("")

    lines.append("[workspace]")
    lines.append(f"channels = {toml_list(channels)}")
    lines.append(f"platforms = {toml_list(platforms)}")
    lines.append("")

    # Optional global Python dependency for all generated environments.
    # If you need per-feature Python versions, put python in catalog.conda and feature.<name>.conda.
    # if python_version:
    #     lines.append("[dependencies]")
    #     lines.append(f"python = {toml_string(python_version)}")
    #     lines.append("")

    non_empty_feature_names: set[str] = set()

    for feature_name in sorted(features):
        feature_def = features[feature_name] or {}
        conda_deps = feature_dependencies(feature_name, feature_def, conda_catalog, "conda")
        pypi_deps = feature_dependencies(feature_name, feature_def, pypi_catalog, "pypi")

        # Pixi needs a Python interpreter inside each environment that resolves PyPI deps.
        if python_version and pypi_deps and "python" not in conda_deps:
            conda_deps["python"] = python_version

        if conda_deps or pypi_deps:
            non_empty_feature_names.add(feature_name)

        write_table(lines, f"feature.{feature_name}.dependencies", conda_deps)
        write_table(lines, f"feature.{feature_name}.pypi-dependencies", pypi_deps)

    filtered_envs: dict[str, list[str]] = {}
    for env_name, env_features in environments.items():
        env_features = list(env_features or [])
        missing_features = [f for f in env_features if f not in features]
        if missing_features:
            raise KeyError(
                f"Environment {env_name!r} references unknown features: {', '.join(missing_features)}"
            )
        has_deps = any(f in non_empty_feature_names for f in env_features)
        if include_empty_environments or has_deps:
            filtered_envs[env_name] = env_features

    if filtered_envs:
        lines.append("[environments]")
        for env_name in sorted(filtered_envs):
            lines.append(f"{env_name} = {toml_list(filtered_envs[env_name])}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate pixi.toml from a central dependencies YAML file.")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Path to dependencies YAML catalog.")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Path to generated pixi.toml.")
    parser.add_argument(
        "--python",
        dest="python_version",
        default=None,
        help="Optional global Python version for [dependencies], e.g. '3.12.*'.",
    )
    parser.add_argument(
        "--include-empty-environments",
        action="store_true",
        help="Also emit environments whose features contain no Python/PyPI dependencies.",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    toml = generate_pixi_toml(
        data,
        python_version=args.python_version,
        include_empty_environments=args.include_empty_environments,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(toml, encoding="utf-8")
    print(f"Generated {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
