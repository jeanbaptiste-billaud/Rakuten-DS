# Terraform Docker stack

Ce dossier reproduit `deploy_branch/deploy/compose/docker-compose.yml` avec le
provider Terraform Docker, branché par défaut sur le socket rootless Podman:

```bash
cd infra/terraform/docker
terraform init
terraform apply
```

Les chemins montés pointent par défaut vers `deploy_branch/deploy` du dépôt
courant. Pour déployer depuis un autre clone:

```bash
terraform apply -var 'project_dir=/chemin/vers/deploy_branch'
```

## Structure

Le découpage suit le pattern provider/images/infrastructure/modules/variables
utilisé dans l'article OneUptime sur Podman avec Terraform:

- `main.tf`: version Terraform, provider Docker et socket Podman.
- `variables.tf`: paramètres exposés, dont les credentials marqués `sensitive`.
- `locals.tf`: chemins dérivés, catalogue d'images, volumes et environnement Airflow commun.
- `images.tf`: pull et gestion des images.
- `infrastructure.tf`: réseau Docker/Podman et volumes nommés.
- `containers.tf`: instancie les modules locaux par domaine.
- `modules/core`: Postgres, MinIO, MLflow, drift detector, model serving.
- `modules/monitoring`: Prometheus, StatsD exporter, Grafana.
- `modules/frontend`: API gateway, auth service, reverse proxy, Streamlit.
- `modules/airflow`: init, DAG processor, API server et scheduler Airflow.
- `outputs.tf`: informations utiles sur le déploiement.

Les noms du réseau et des volumes viennent de `infra/config/docker.yaml`, lue
par Terraform et Ansible. Le runtime ne les redéclare donc qu'une seule fois.

## Infisical

La stack Docker lit ses credentials applicatifs via le provider
`infisical/infisical`.

Le provider doit être authentifié par l'orchestrateur de déploiement via
`TF_VAR_infisical_provider_client_id` et
`TF_VAR_infisical_provider_client_secret` ou leurs équivalents d'environnement.
Le bootstrap Ansible/Vault reste une source valable pour injecter ces valeurs,
mais Terraform ne dépend pas de `init.sh`.

Par défaut, le module lit les secrets dans:

- `workspace_id = TF_VAR_infisical_project_id`
- `env_slug = TF_VAR_infisical_env`
- `folder_path = /`

Clés attendues dans Infisical:

- `POSTGRES_AIRFLOW_USER`
- `POSTGRES_AIRFLOW_PASSWORD`
- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `/services/mlflow`: `MLFLOW_S3_ACCESS_KEY`, `MLFLOW_S3_SECRET_KEY`
- optionnellement à la racine: `AIRFLOW_WWW_USER_USERNAME`, `AIRFLOW_WWW_USER_PASSWORD`
  sinon la stack retombe sur `airflow/airflow`

## Secrets

Oui, on peut utiliser Infisical ou Ansible Vault pour fournir les credentials,
mais il faut distinguer deux choses:

- Ansible Vault reste la meilleure source de bootstrap/seed local.
- Infisical est la meilleure source runtime pour les workloads et les identities.

Terraform n'est pas un bon coffre-fort: les variables `sensitive` masquent
l'affichage CLI, mais les valeurs injectées dans `docker_container.env` peuvent
quand même être stockées dans le state Terraform. Il faut donc protéger le
backend state et éviter de committer des `*.tfvars` réels.

Pattern Infisical CLI recommandé:

```bash
infisical run --env dev --path /services/terraform-docker -- \
  terraform apply
```

Le wrapper fournit uniquement les identifiants du provider Infisical:
`TF_VAR_infisical_provider_client_id` et
`TF_VAR_infisical_provider_client_secret`.
La stack elle-même lit ensuite les secrets applicatifs depuis Infisical.

Pattern Ansible Vault possible:

```bash
ansible-playbook ... --vault-password-file ~/.config/rakuten/ansible/.vault_pass
```

Le playbook peut exporter les mêmes variables `TF_VAR_*` avant d'appeler
`terraform apply`.

Pour les credentials Universal Auth Infisical injectés dans Airflow, passer une
map Terraform:

```hcl
infisical_identity_credentials = {
  data_ingestion = {
    client_id     = "..."
    client_secret = "..."
  }
  model_build = {
    client_id     = "..."
    client_secret = "..."
  }
}
```

Les noms de clé sont transformés en variables d'environnement Airflow comme
`INFISICAL_DATA_INGESTION_ID_CLIENT_ID`.
