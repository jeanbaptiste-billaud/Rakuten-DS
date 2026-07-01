# V2 Credentials Flows Inventory

Etat des lieux des processus qui consomment des credentials et des acces
externes/internes associes. Ce document sert de base pour definir les machine
identities Infisical, les scopes de secrets et les futurs droits MinIO/Postgres.

## Contexte V1 / V2

La V1 du projet est une infrastructure de demonstration: elle valide les flux
MLOps, l'orchestration Airflow, le stockage MinIO, le tracking MLflow, le
serving et les premiers reflexes d'automatisation. Elle contient deja une
intuition de scalabilite et de securisation, mais son modele de secrets,
d'authentification et de composition reste trop fragile pour servir de base
robuste.

La V2 est le chantier de durcissement de l'infrastructure. L'objectif est de
passer d'une stack demonstrative et partiellement manuelle a une infrastructure
securisee, reproductible et gouvernee par l'IaC:

- Ansible pour le bootstrap local et l'initialisation des briques de confiance;
- Infisical pour la distribution runtime des secrets;
- des identities separees par perimetre de droits;
- des comptes MinIO non-root par usage;
- une suppression progressive des `.env` et des secrets en clair;
- une trajectoire vers Terraform puis Kubernetes, sans traiter ce dernier point
  dans la presente migration.

Les problemes listes dans ce document sont donc les problemes connus de la V1.
Les changements en cours constituent les premieres briques de la V2.

## Principes

- Une identity Infisical doit representer un perimetre de permissions, pas un
  conteneur precis.
- Les comptes root/admin ne doivent servir qu'a l'initialisation ou a
  l'administration, pas aux workloads applicatifs.
- Les jobs Airflow lances par `DockerOperator` doivent recevoir seulement les
  secrets necessaires a leur tache.
- Les secrets de bootstrap restent dans Ansible Vault. Infisical devient la
  source d'injection runtime.
- Le suffixe des identities est `-id` pour marquer leur nature d'identity
  Infisical. L'environnement reste porte par le scope Infisical (`dev`) ou par
  le chemin de secrets, pas par le nom de l'identity.

## Sources V1 Et V2

| Source | Fichier | Role |
| --- | --- | --- |
| Ansible Vault | `infra/ansible/inventory/group_vars/all/secrets.vault.yaml` | Source seed chiffree des secrets V2 |
| Seed Infisical | `infra/ansible/playbooks/bootstrap/tasks/infisical/bootstrap_infisical.yaml` | Charge les secrets du vault dans Infisical |
| Template seed | `infra/ansible/playbooks/bootstrap/tasks/infisical/templates/seed_secrets.yaml.j2` | Normalise les noms de secrets |
| Runtime seed | `infra/ansible/playbooks/bootstrap/tasks/infisical/templates/runtime_secrets.yaml.j2` | Prepare les chemins `/workloads/*` consommes par les jobs V2 |
| Compose runtime | `deploy_branch/deploy/compose/docker-compose.yml` | V1: injecte encore des variables secretes via `.env` |
| Airflow DAG common | `deploy_branch/deploy/airflow/dags/common_task.py` | V2: centralise les profils Infisical des `DockerOperator` |

## Secrets Connus

| Famille | Variables / donnees | Producteur actuel | Consommateurs actuels |
| --- | --- | --- | --- |
| MinIO root | `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` | Vault puis `.env`/compose | MinIO et bootstrap/admin uniquement |
| MinIO S3 compat | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | V2: comptes `minio_service_accounts.*` | training, preprocessing, sync bucket, MLflow |
| Postgres root/admin | `postgres.root.*`, `postgres.admin.*` | Vault | bootstrap DB, restauration, maintenance |
| Postgres Airflow | `postgres.airflow.*`, `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` | Vault puis `.env`/compose | Airflow scheduler/apiserver/worker |
| Postgres MLflow | `postgres.mlflow.*`, `MLFLOW_BACKEND_STORE_URI` | Vault puis `.env`/compose | MLflow tracking server |
| Airflow API | `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | `.env`/defaut code actuellement | API gateway |
| Airflow web admin | `_AIRFLOW_WWW_USER_USERNAME`, `_AIRFLOW_WWW_USER_PASSWORD` | `.env`/compose | Initialisation Airflow |
| DagsHub | `dagshub.user`, `dagshub.password` | Vault | DVC pull/push |
| GitHub App | `github_app.*` | Vault | Git clone/push ou automatisation future |
| Docker Hub | `docker_hub.*` | Vault | Build/push images |
| Auth service tokens | `docker/auth-service/auth.json` hashes | Fichier dans image | Auth service |
| MinIO client config | `docker/minio-client/mc-config/config.json*` | Supprime en V2 | Ne doit plus etre versionne |
| Infisical bootstrap | `infisical.*` | Vault | Bootstrap Infisical uniquement |

## Workloads Runtime

| Workload | Declencheur | Connexions | Secrets requis | Droits metier | Identity candidate |
| --- | --- | --- | --- | --- | --- |
| `postgres` | Compose | Stockage Postgres local | `POSTGRES_USER`, `POSTGRES_PASSWORD` | Init DB et roles | `postgres-admin-id` ou bootstrap Ansible direct |
| `minio` | Compose | Stockage S3 local | `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` | Admin serveur MinIO | `minio-admin-id` uniquement bootstrap/admin |
| `mlflow-server` | Compose | Postgres MLflow, MinIO artifacts | `MLFLOW_BACKEND_STORE_URI`, credentials S3, auth MLflow | Lire/ecrire runs, metrics, artifacts | `mlflow-server-id` |
| `airflow-*` | Compose | Postgres Airflow, Docker socket | URI Postgres Airflow, fernet/API secrets | Orchestration DAGs | `airflow-id` |
| `api-gateway` | Compose | Airflow API, model-serving HTTP | `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | Declencher DAGs, lire modele via HTTP sans secret | `api-gateway-id` |
| `auth-service` | Compose | Fichier local `auth.json` | Hashes de tokens applicatifs | Service temporaire V1, remplace par Traefik en V2 | Pas d'identity cible |
| `model-serving` | Compose | Aucun secret runtime identifie | Aucun actuellement | Servir le modele BentoML embarque | Pas d'identity au depart |
| `drift-detector` | Compose/DAG | `model-serving` HTTP | Aucun actuellement | Lire prediction/metadata, ecrire rapports locaux | Pas d'identity au depart |
| `streamlit-dashboard` | Compose | MLflow/Prometheus/Grafana via URLs | Aucun secret vu dans le compose actuel | Dashboard lecture | A definir si appels proteges plus tard |
| `prometheus` | Compose | Scrape HTTP | Aucun secret actuellement | Scraper metrics | Pas d'identity |
| `grafana` | Compose | Prometheus | Aucun, anonymous admin actuellement | Dashboards | A securiser plus tard |

## DAGs Airflow

### Common DockerOperator Environment

En V1, `deploy_branch/deploy/airflow/dags/common_task.py` injectait a tous les
containers:

- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_S3_ENDPOINT_URL`

En V2, `docker_common_args()` n'injecte plus ces credentials. Les taches qui
ont besoin de secrets utilisent un profil explicite et executent leur commande
via `infisical run --path /workloads/...`.

### `rakuten_create_dataset`

| Tache | Image | Connexions | Secrets requis | Droits minimaux | Identity candidate |
| --- | --- | --- | --- | --- | --- |
| `create_dataset` | `jbbillaud/rakuten:spacy-v3.8.11` | MinIO `raw`, MinIO `dataset` | S3 access key/secret | Read `raw`, write `dataset` | `data-ingestion-id` |
| `preprocessing` | `jbbillaud/rakuten:spacy-v3.8.11` | MinIO `dataset`, MinIO `preprocessed` | S3 access key/secret | Read `dataset`, write `preprocessed` | `data-preprocessing-id` |

### `rakuten_enrich_dataset`

| Tache | Image | Connexions | Secrets requis | Droits minimaux | Identity candidate |
| --- | --- | --- | --- | --- | --- |
| `enrich_dataset` | `jbbillaud/rakuten:spacy-v3.8.11` | MinIO `raw`, MinIO `dataset` | S3 access key/secret | Read `raw`, read/write `dataset` | `data-enrichment-id` |
| `preprocessing` | `jbbillaud/rakuten:spacy-v3.8.11` | MinIO `dataset`, MinIO `preprocessed` | S3 access key/secret | Read `dataset`, write `preprocessed` | `data-preprocessing-id` |

### `rakuten_train_model`

| Tache | Image | Connexions | Secrets requis | Droits minimaux | Identity candidate |
| --- | --- | --- | --- | --- | --- |
| `generate_data_lineage_json_file` | `jbbillaud/rakuten:dvc-v3.66.1` | DVC repo volume | Aucun secret reseau direct dans la tache | Lire repo DVC local | Pas d'identity au depart |
| `train_model` | `jbbillaud/rakuten:sklearn-v1.8.0` | MinIO `preprocessed`, MLflow server, MinIO artifacts via MLflow | S3 access key/secret, `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL` | Read `preprocessed`, write MLflow runs/artifacts | `training-id` |
| `get_model_run_id_task` | Airflow process | `model-serving` HTTP | Aucun | Lire metadata modele courant | Pas d'identity |
| `model_comparison` | `jbbillaud/rakuten:sklearn-v1.8.0` | MLflow server | `MLFLOW_TRACKING_URI`; S3 credentials si MLflow telecharge des artifacts | Read runs/metrics/artifacts | `model-evaluation-id` |
| `build_model` | `jbbillaud/rakuten:bentoml-v1.4.33` | MLflow server, MinIO artifacts, Docker socket | MLflow + S3 credentials | Read model/artifacts, build image locale | `model-build-id` |
| `stop_model_serving` / `start_model_serving` | Airflow process | Docker Compose / Docker socket | Aucun secret applicatif, mais socket Docker | Stop/start service | `airflow-id`; securite hors Infisical |

### `rakuten_backup_pipeline`

| Tache | Image | Connexions | Secrets requis | Droits minimaux | Identity candidate |
| --- | --- | --- | --- | --- | --- |
| `backup_minio_volume` | `jbbillaud/rakuten:minio-client-*` | MinIO buckets | MinIO credentials | Read `dataset`, `preprocessed`, MLflow bucket | `backup-id` |
| `backup_logs_and_reports` | `jbbillaud/rakuten:dvc-v3.66.1` | Volumes locaux | Aucun secret reseau direct | Lire volume | Pas d'identity |
| `backup_mlflow_db` | `postgres:17` | Postgres `mlflow_db` | Postgres MLflow read/dump credentials | Dump DB | `backup-id` ou `postgres-backup-id` |
| `backup_airflow_db` | `postgres:17` | Postgres `airflow_db` | Postgres Airflow read/dump credentials | Dump DB | `backup-id` ou `postgres-backup-id` |
| `git_dvc_commit` | `jbbillaud/rakuten:dvc-v3.66.1` | Git remote, DVC remote | GitHub/DagsHub credentials | Commit/push Git, DVC push | `dvc-publisher-id` |

## Scripts Et Modules

| Script/module | Connexions | Secrets lus | Remarque |
| --- | --- | --- | --- |
| `src/utils/minio_utils.py` | MinIO S3 API | `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Point d'entree commun S3 pour les jobs data |
| `src/utils/sync_bucket.py` | MinIO S3 API via `minio_utils` | Idem | Utilise `folder` comme nom de bucket |
| `src/pipeline_steps_df/training/train_text_model.py` | MLflow + MinIO artifacts | `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`, AWS credentials | V2: echoue si Infisical n'injecte pas les variables requises |
| `src/pipeline_steps_df/training/train_text_model_with_drift.py` | MLflow + MinIO + drift-detector HTTP | Idem | V2: echoue si Infisical n'injecte pas les variables requises |
| `src/model_serving/build_step.py` | MLflow + artifacts | `MLFLOW_TRACKING_URI`, probablement S3 credentials selon artifact store | Echoue si `MLFLOW_TRACKING_URI` absent |
| `src/pipeline_steps_df/model_promotion_decision.py` | MLflow | `MLFLOW_TRACKING_URI`; S3 si artifacts necessaires | Compare runs |
| `docker/minio-client/minio_backup.sh` | MinIO via `mc` | `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` | V2: utilise le compte non-root `backup-id` |
| `docker/minio-client/minio_init.sh` | MinIO via `mc` | `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` | Bootstrap/admin uniquement |
| `docker/minio-client/mc-config/config.json*` | MinIO via `mc` | Alias MinIO avec credential en clair | Supprime en V2; la config doit etre regeneree au runtime |
| `deploy_branch/boostrap/init.sh` | GitHub, DagsHub, Postgres | `GITHUB_TOKEN`, `DAGSHUB_USER`, `DAGSHUB_PASSWORD`, `POSTGRES_PASSWORD` | Legacy bash a remplacer par Ansible/Terraform |

## Problemes V1 Connus

1. Les credentials MinIO root sont reutilises comme credentials applicatifs.
   Mitigation V2: comptes `minio_service_accounts.*`, MLflow et jobs Airflow
   commencent a utiliser des credentials non-root.
2. Les DAGs injectent le meme bloc de secrets a tous les containers.
   Mitigation V2: `docker_common_args()` utilise des profils Infisical par
   tache.
3. Les scripts training contiennent des valeurs de secours en clair
   (`minio`, `minio123`) qui masquent les erreurs d'injection.
   Mitigation V2: les fallbacks ont ete remplaces par des erreurs explicites.
4. `AIRFLOW_CONN_POSTGRES_DEFAULT` contient encore un placeholder dans le
   compose.
5. `dag_backup.py` dump `airflow_db` avec les identifiants `mlflow/mlflow`,
   probablement incorrect ou temporaire.
6. `auth.json` est copie dans l'image `auth-service`; les hashes sont
   versionnes avec le code.
7. `docker/minio-client/mc-config/config.json*` contient une configuration
   client MinIO avec credential en clair.
   Mitigation V2: les fichiers ont ete supprimes du workspace; l'historique Git
   devra etre traite separement si necessaire.
8. `grafana` est en anonymous admin. Ce n'est pas prioritaire pour Infisical,
   mais c'est a traiter avant une approche prod.
9. Le socket Docker donne des privileges eleves a Airflow et au job
   `build_model`; Infisical ne reduit pas ce risque.

### Limites Du Service D'authentification V1

`auth-service` est un service transitoire de demonstration. Ses limites sont
structurelles:

- les hashes de tokens sont embarques dans `docker/auth-service/auth.json`;
- le fichier est copie dans l'image au build;
- la gestion des roles est statique;
- il n'y a pas de rotation ou de cycle de vie des tokens;
- il n'y a pas de federation d'identite ni d'integration propre avec un
  reverse proxy moderne;
- le service n'est pas un composant cible de la V2.

La V2 prevoit de remplacer ce bloc par Traefik et une strategie
d'authentification plus propre. Les identities Infisical ne doivent donc pas
etre construites autour de `auth-service`.

### Limites Du Reverse Proxy V1

Le reverse proxy V1 sert principalement de point d'entree demonstratif:

- il ne porte pas encore une politique d'authentification robuste;
- il depend du service d'auth V1;
- la separation entre routes publiques, internes et admin reste limitee;
- la gestion TLS, middlewares, headers de securite et providers dynamiques n'est
  pas encore traitee comme de l'IaC.

En V2, le reverse proxy doit devenir une brique d'infrastructure explicite,
probablement Traefik, configuree comme code et separee du mecanisme de secrets
applicatifs.

## Initialisation Des Services

Les services lances par Compose/Terraform ont deux categories de secrets:

- bootstrap/admin: credentials root, creation de DB, creation de buckets,
  creation d'utilisateurs techniques;
- runtime: secrets utilises par le process long-lived apres initialisation.

Le pattern cible est:

1. Ansible demarre Infisical et seed les secrets depuis le vault.
2. Ansible execute les taches de bootstrap des services avec les secrets lus
   depuis Infisical ou encore depuis le vault si Infisical n'est pas disponible.
3. Les services long-lived ne recoivent plus les secrets root/admin.
4. Quand le service supporte les secrets par fichier (`*_FILE`), le secret est
   monte en fichier temporaire plutot qu'en variable d'environnement.
5. Quand le service exige des variables d'environnement, le conteneur est lance
   via un wrapper `infisical run -- <commande>` ou via l'agent Infisical, avec
   une identity dediee au service.

Ce point reste critique: certains logiciels imposent les variables
d'environnement au demarrage. Infisical reduit la diffusion et la persistance
des secrets, mais ne rend pas magiquement invisible un secret deja charge dans
l'environnement du process. Pour ces cas, on limite le blast radius par des
credentials non-root, des identities separees, des TTL courts et des droits
minimaux.

## Proposition D'identities Infisical

| Identity | Secrets Infisical cibles | Workloads | Scope attendu |
| --- | --- | --- | --- |
| `airflow-id` | Airflow DB URI, fernet/API secret, eventuellement Docker runtime config | Airflow services | Orchestration uniquement |
| `api-gateway-id` | Airflow API credentials | API gateway | Declencher DAGs |
| `mlflow-server-id` | MLflow DB URI, S3 artifacts credentials, auth MLflow | MLflow server | Read/write MLflow DB et artifacts |
| `data-ingestion-id` | S3 credentials ingestion | `create_dataset` | Read `raw`, write `dataset` |
| `data-enrichment-id` | S3 credentials enrichment | `enrich_dataset` | Read `raw`, read/write `dataset` |
| `data-preprocessing-id` | S3 credentials preprocessing | `preprocessing` | Read `dataset`, write `preprocessed` |
| `training-id` | MLflow URI, S3 training credentials | `train_model` | Read `preprocessed`, write MLflow artifacts |
| `model-evaluation-id` | MLflow read credentials | `model_comparison` | Read MLflow runs/metrics |
| `model-build-id` | MLflow/S3 model read credentials | `build_model` | Read model artifacts |
| `backup-id` | MinIO backup credentials, Postgres dump credentials | backup DAG | Read backups sources, write DVC data volume |
| `dvc-publisher-id` | GitHub/DagsHub credentials | `git_dvc_commit` | Git push + DVC push |

## Decisions A Prendre

- Definir les comptes MinIO non-root par usage.
- Definir le mecanisme de protection MLflow.
- Definir le mode exact d'utilisation de l'agent Infisical dans les taches
  `DockerOperator`.
- Sortir `auth-service` du plan cible V2, puisqu'il sera remplace par Traefik.
- Garder la creation initiale des identities Infisical dans le bootstrap
  Ansible. Terraform pourra consommer Infisical ensuite, mais ne doit pas etre
  necessaire pour creer les identities de bootstrap.

## Migration Cible

1. Supprimer les secrets communs de `docker_common_args()`. Fait en V2.
2. Ajouter un mapping par tache Airflow vers une identity/profil de secrets.
   Fait en V2.
3. Remplacer les fallbacks en clair des scripts Python par des erreurs
   explicites si les variables obligatoires sont absentes. Fait en V2.
4. Creer des credentials MinIO separes par usage. Generation vault ajoutee en
   V2; creation effective des users MinIO a brancher au bootstrap MinIO.
5. Creer les machine identities Infisical et leurs droits projet/env/path via
   Ansible bootstrap.
6. Injecter les secrets via Infisical CLI/agent en dev. Premiere integration
   `infisical run` ajoutee dans les `DockerOperator`.
7. Remplacer l'injection Docker Compose par Terraform puis Kubernetes auth.
   Hors scope de cette migration.
