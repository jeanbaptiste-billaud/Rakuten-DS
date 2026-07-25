# AGENTS.md

## Project overview

Rakuten-DS is a Python MLOps project for product-text classification. It combines data pipelines, model training and serving, Airflow orchestration, and supporting services such as MinIO, MLflow, Prometheus/Grafana, and Evidently.

This branch is the V2 refactor. The target infrastructure is provisioned with Terraform and configured with Ansible. Docker Compose files are V1 legacy material and must not be extended for new V2 work. Podman remains a V2 runtime implementation detail; Kubernetes is the planned V3 target and is out of scope unless explicitly requested.

The roadmap is staged: V2 adds Traefik and centralized logs with Grafana Alloy and Loki; V3 evolves the text-classification model, deploys Kubernetes through Helm and Terraform, evaluates Kubeflow alongside or in place of selected Airflow duties, and improves drift monitoring beyond training-time checks. V4 integrates the image model and its image-quality-control workload, which needs substantially more development resources. V5 is expected to add an AI agent that drives quality-control logic, drift analysis, and alerts. These are future directions, not authorization to introduce later-version technologies into V2 work.

The branch flow is intentional: development happens on `dev`; a completed version of `dev` is promoted to `main`; the `deploy_branch/` subtree and its deployment dependencies are promoted to the `deploy` branch. The `dvc` branch stores versioned data and is currently updated by Airflow DAGs. The V2 target is to move CI/CD responsibility out of Airflow to a dedicated orchestrator (Jenkins, GitHub Actions, or GitLab CI/CD).

## Repository layout

- `src/`: application, data, model, pipeline, and shared utility code.
- `src/tests/`: unit tests, mirroring the `src/` packages.
- `docker/`: service images and runtime-specific code; preserve compatibility only when explicitly required.
- `deploy_branch/deploy/`: the deployment subtree intentionally promoted to the `deploy` branch. It contains deployment assets and dependencies used by the V2 Terraform stack; the Compose runtime within it remains V1 legacy.
- `configs/`: training and preprocessing configuration.
- `infra/ansible/`: V2 bootstrap, configuration, secret seeding, and deployment automation.
- `infra/terraform/`: V2 resource provisioning, including the Infisical project and machine identities.
- `infra/security/`: credential-flow documentation and security guidance.
- `models/`: tracked model artifacts; do not replace them unless the task explicitly requires it.
- `data/` and `.dvc/`: versioned datasets and DVC configuration; treat as data artifacts.

## Python changes

- Target Python 3.12, matching CI.
- Keep imports and package names consistent with the existing `src/` structure.
- Add or update focused pytest coverage under the matching `src/tests/` package when behavior changes.
- Prefer small, typed, side-effect-aware functions. Keep I/O, network calls, and service configuration at module boundaries.
- Do not introduce a formatter or linter configuration unless requested; CI currently runs pytest and has Flake8 checks commented out.

## Validation

Run the relevant focused tests first. For a broad local check, use:

```powershell
$env:PYTHONPATH = "."
pytest src/tests/ --ignore=src/tests/services/mlflow_writing_test.py --doctest-modules
```

Some tests and services require optional dependencies or running infrastructure. Report those prerequisites or failures rather than weakening tests.

## Infrastructure and deployment

- For V2 infrastructure changes, make Terraform the source of truth for resources and Ansible the source of truth for host and service configuration.
- Treat `deploy_branch/deploy/compose/` as V1 reference material. The current Terraform stack intentionally mounts selected assets from `deploy_branch/deploy/`; preserve this deployment-subtree contract, but do not add new Compose services or workflows.
- Keep Terraform modules, root resources, variables, outputs, and Ansible inventories/playbooks consistent when changing infrastructure contracts.
- Podman is used by V2 bootstrap playbooks for containers, networks, and volumes. Dockerfiles may remain build inputs, but do not make Docker Compose a runtime dependency for new work.
- For Airflow work, preserve DAG IDs and task contracts unless the change explicitly includes a migration.
- Airflow currently orchestrates data updates to the `dvc` branch. Do not move CI/CD behavior out of Airflow until the dedicated CI/CD orchestrator and its equivalent pipeline contracts are explicitly in scope.
- Do not introduce Kubernetes manifests, Helm charts, or a migration to Kubernetes unless the task explicitly targets V3.
- Avoid provisioning, destroying, or rebuilding infrastructure unless the task calls for it.

## Roadmap guardrails

- In V2, treat Traefik, Grafana Alloy, and Loki as operational infrastructure. Define ingress boundaries, log labels, retention, access controls, and dashboards as code; do not put credentials or high-cardinality request data into logs.
- Treat V3 model evolution, Helm, Terraform-managed Kubernetes deployment, Kubeflow integration, and continuous drift monitoring as one explicit migration. Preserve current data, model, DAG, and service contracts until its scope and acceptance criteria are defined.
- Kubeflow may replace selected Airflow responsibilities or coexist with Airflow; do not assume a full replacement. Define ownership boundaries for scheduling, training, serving, metadata, and CI/CD before implementation.
- V3 drift monitoring must extend beyond training-time evaluation: define production signals, reference-data policy, thresholds, delayed-label handling, alert routing, and retraining decision ownership.
- V4 image-model work must include image-quality-control acceptance criteria, annotation/data-quality governance, resource sizing, and separate model/data lineage before integration into the MLOps platform.
- V5 AI quality-control automation must begin with bounded, auditable recommendations and human approval for impactful actions. It must not be granted implicit authority to retrain, promote, roll back, or change infrastructure.
- When later-version work begins, keep model artifacts, DVC data lineage, MLflow tracking, Infisical/SPIRE workload identity, logs, metrics, and dashboards traceable through the transition.

## Images and container runtime (V2)

- The canonical image/dependency catalogue is `docker/python_library_catalogs/stable.yaml`. It defines Pixi features/environments and image names; update it before changing an image dependency or adding an image.
- Generate `pixi.toml` from that catalogue with `src/deps/generate_pixi_toml.py`, then generate the matching lock file. Keep `pixi.toml` and `pixi.lock` synchronized with the catalogue.
- The V2 image build entry point is `infra/ansible/playbooks/dev/build_docker_images.yaml`, which uses Podman. The legacy `docker/docker-images-build.sh` is Docker Buildx-based, has stale paths/tags, and is reference material only.
- Most application images use a multi-stage build: the shared `pixi` builder resolves a locked environment, then the runtime copies only Pixi, its environment, and the service code. Retain this pattern for comparable services.
- Build arguments `PIXI_ENV`, `PIXI_CATALOG_DIR`, and `INFISICAL_CLI_IMAGE` are part of the build contract. Do not hard-code an environment-specific dependency set in individual Dockerfiles.
- Preserve the least-privilege runtime pattern: data, training, DVC, and BentoML images create `trainusr` (UID/GID 1000) and run as it. New Python HTTP services should also explicitly use a non-root user unless a documented runtime constraint prevents it.
- Image names and exact tags used by Terraform are a deployment contract. Update the catalogue, image build automation, Terraform image map, and consuming Airflow task together when changing them.
- Do not use mutable tags such as `latest` for new production-facing base images, CLI images, or service images. Pin a reviewed version or immutable digest.
- Define health checks in the image when the service can self-report health; Terraform health checks must target a real endpoint exposed by that image.

## Infisical lifecycle

Infisical spans both IaC tools; preserve this order and separation of responsibilities:

1. Ansible runs the bootstrap playbooks in `infra/ansible/playbooks/bootstrap/` to install, configure, bootstrap, and seed the Infisical service.
2. Terraform in `infra/terraform/infisical/` provisions the Infisical project, secret-folder structure, machine identities, Universal Auth methods, memberships, and path-scoped privileges.
3. Ansible transfers application secret values from Ansible Vault into the provisioned Infisical paths.

- Secret values are owned by Ansible Vault and Ansible seeding tasks, never Terraform configuration.
- Terraform state includes Universal Auth client secrets; treat state files and generated `tfvars` as sensitive and keep them out of Git.
- When introducing a secret-consuming service, update its Vault source, Infisical target path, Terraform identity privilege, and the consuming runtime configuration together.

## Workload identity trajectory: SPIRE

The current V2 implementation includes the Infisical CLI in most application images and uses Infisical Universal Auth credentials for deployment and selected workloads. This is transitional: static machine-identity client secrets must not be baked into images, committed, or broadly injected into container environments.

SPIRE is the intended workload-identity layer for the next V2 security increment. It will attest workloads and issue SPIFFE identities that Infisical can trust, so a workload can obtain its required secrets without first carrying a long-lived Infisical machine-identity credential.

- Treat SPIRE deployment, trust domain, registration entries, selectors, and the Infisical SPIFFE integration as IaC owned by `infra/`; do not add ad-hoc identity bootstrap logic to application images.
- Keep every workload identity narrow: one identity per service or Airflow task role, bound to its runtime attributes and granted only the Infisical paths it needs.
- Keep bootstrap/admin credentials separate from workload credentials. SPIRE removes the bootstrap problem for runtime workloads, not the need to bootstrap and protect Infisical itself.
- Until SPIRE is implemented and validated end to end, retain the existing Infisical path/identity controls and do not remove the current secure bootstrap flow.
- When SPIRE is introduced, document the delivery mechanism (agent/socket, file, or supported integration), rotation behavior, failure mode, and local-development equivalent before migrating a workload.

## Data, models, and secrets

- Never commit credentials, tokens, `.env` files, vault material, or real connection strings. Use documented secret-management flows in `infra/security/` and Ansible vaults.
- Do not run destructive DVC, MinIO, database, Docker volume, or Terraform operations without explicit user approval.
- Avoid modifying large datasets, model binaries, generated reports, or spreadsheet outputs unless they are explicitly in scope.

## Change hygiene

- Preserve unrelated working-tree changes.
- Update documentation when a user-facing command, service interface, configuration option, or deployment procedure changes.
- Keep commits scoped; use imperative, concise commit messages when asked to prepare one.
