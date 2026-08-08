# Infrastructure Rakuten DS

## Objet du document

Ce document décrit les deux générations d'infrastructure présentes dans le
dépôt :

- la **V1**, actuellement matérialisée par les fichiers de `deploy_branch` et
  orchestrée avec Docker Compose ;
- la **V2**, en cours de construction dans `infra`, fondée sur Podman rootless,
  Ansible, Terraform, Infisical et SPIRE.

La V2 reprend les services fonctionnels de la V1, mais sépare davantage le
bootstrap, la gestion déclarative des ressources, les secrets et les identités
des workloads. Tant que la migration n'est pas terminée, les deux arborescences
coexistent et ne doivent pas être considérées comme deux stacks à lancer
simultanément sur la même machine : elles emploient les mêmes ports et noms de
volumes.

## Vue d'ensemble fonctionnelle

Le système couvre le cycle MLOps suivant :

1. DVC récupère les données et sauvegardes versionnées depuis DagsHub.
2. MinIO stocke les données brutes, enrichies, prétraitées et les artefacts.
3. Airflow orchestre l'ingestion, la préparation, l'entraînement et
   l'évaluation.
4. MLflow conserve les expériences et les métadonnées dans PostgreSQL, avec
   les artefacts dans MinIO.
5. BentoML sert le modèle ; l'API Gateway et le reverse proxy exposent les
   parcours applicatifs.
6. Evidently détecte le drift.
7. Prometheus, StatsD Exporter et Grafana assurent l'observabilité.
8. Streamlit fournit un tableau de bord utilisateur.

## V1 — Docker Compose

### Statut et sources de vérité

La V1 correspond au déploiement historique. Ses principaux fichiers sont :

- `start.sh` : point d'entrée du démarrage ;
- `deploy_branch/boostrap/docker-compose.yml` : services nécessaires au
  bootstrap ;
- `deploy_branch/deploy/compose/docker-compose.yml` : stack applicative ;
- `deploy_branch/deploy/compose/.env` : paramètres locaux du déploiement ;
- `deploy_branch/deploy/monitoring` : configuration Prometheus/Grafana ;
- `deploy_branch/deploy/airflow` : DAGs, plugins, journaux et configuration
  Airflow.

La configuration Compose et le fichier `.env` constituent la source de vérité
du runtime V1. La procédure historique restaure les données avant de démarrer
l'ensemble des services.

### Architecture

```text
Navigateur / client
        |
        +--> reverse-proxy:8085 --> auth-service
        |                       +--> API / drift detector
        +--> api-gateway:8000 ------> model-serving
        +--> streamlit:8501
        +--> airflow:8080
        +--> mlflow:5000
        +--> grafana:3000 --> prometheus:9090

Airflow --> tâches de données et d'entraînement
   |              |                 |
   +----------> MinIO <--------- MLflow
   |                                |
   +--------------------------> PostgreSQL
```

Tous les conteneurs rejoignent le réseau Compose nommé `mlflow-network`.
PostgreSQL, MinIO et les répertoires de travail utilisent des volumes externes,
ce qui permet de recréer les conteneurs sans effacer les données.

### Services et exposition

| Domaine | Service V1 | Rôle | Port hôte |
|---|---|---|---:|
| Données | PostgreSQL 17 | Métadonnées Airflow et MLflow | non exposé |
| Données | MinIO | API S3 et console | 9000, 9001 |
| MLOps | MLflow Server | Tracking et registre d'expériences | 5000 |
| MLOps | Model Serving | Inférence BentoML | `${MODEL_PORT}` |
| MLOps | Drift Detector | Analyse Evidently | 8003 |
| Orchestration | Airflow API Server | API et interface Airflow | 8080 |
| Orchestration | Airflow Scheduler | Planification des DAGs | interne |
| Orchestration | Airflow DAG Processor | Analyse des DAGs | interne |
| Application | API Gateway | Point d'entrée FastAPI | 8000 |
| Application | Auth Service | Authentification interne | interne |
| Application | Reverse Proxy | Entrée HTTP | 8085 |
| Application | Streamlit | Tableau de bord | 8501 |
| Monitoring | Prometheus | Collecte de métriques | 9090 |
| Monitoring | StatsD Exporter | Relais Airflow vers Prometheus | 9102, 8125/UDP |
| Monitoring | Grafana | Visualisation | 3000 |

### Persistance

| Volume V1 | Contenu |
|---|---|
| `pgdata` | cluster PostgreSQL |
| `minio_data` | objets MinIO |
| `dvc_data` | clone et cache de données DVC |
| `logs_and_reports` | journaux, rapports et résultats Evidently |
| `airflow_vol` | données persistantes Airflow historiques |

### Secrets en V1

Les credentials sont principalement injectés par variables d'environnement et
fichiers `.env`. Cette méthode est simple, mais présente plusieurs limites :

- le cycle de vie et la rotation des secrets ne sont pas centralisés ;
- un secret peut être visible dans l'environnement ou dans l'inspection d'un
  conteneur ;
- les droits sont souvent partagés par plusieurs services ;
- l'identité technique d'un workload dépend d'un couple identifiant/mot de
  passe statique.

### Démarrage V1

La procédure historique est :

```bash
./start.sh
```

Le script prépare les données DVC, initialise MinIO et PostgreSQL, puis lance
la stack Compose. Les détails restent dépendants des credentials présents dans
les `.env` et de la disponibilité des images référencées par Compose.

## V2 — Podman, Ansible, Terraform, Infisical et SPIRE

### Statut

La V2 est **en cours de construction**. Les briques de bootstrap, de secrets,
de déploiement Terraform et d'identité sont présentes, mais la migration
complète des workloads et leur consommation effective de SPIFFE ne sont pas
encore terminées.

Ses sources principales sont :

- `infra/config/docker.yaml` : noms communs du réseau et des volumes ;
- `infra/ansible` : installation, génération des secrets et bootstrap ;
- `infra/terraform/infisical` : projet, dossiers, identités et permissions
  Infisical ;
- `infra/terraform/docker` : réseau, volumes, images et conteneurs applicatifs ;
- `infra/security/credentials-flows.md` : modèle de circulation des secrets ;
- `infra/ansible/playbooks/bootstrap/templates/spire` : configuration SPIRE.

### Principes d'architecture

```text
                    Ansible Vault
                         |
                         | bootstrap / seed
                         v
   Ansible ---------> Infisical <--------- Terraform Infisical
      |                  |                       |
      |                  | secrets runtime       +-- identités
      |                  v                       +-- permissions
      |          Terraform Docker
      |                  |
      v                  v
  bootstrap       Podman rootless
  DVC/DB/S3       réseau rakuten_network
  SPIRE                  |
      |                  +-- services MLOps V2
      v
 Workload API <-- SPIRE Agent <-- socket API Podman + PID hôte
      |
      +-- X.509-SVID / JWT-SVID pour les workloads enregistrés
```

La séparation des responsabilités cible le modèle suivant :

| Outil | Responsabilité V2 |
|---|---|
| Ansible | Préparer l'hôte, générer le Vault, créer les volumes, restaurer les données, initialiser MinIO/PostgreSQL, Infisical et SPIRE |
| Terraform Infisical | Déclarer le projet, les dossiers, les identités machine et leurs permissions |
| Infisical | Distribuer les secrets runtime avec des droits par workload |
| Terraform Docker | Déclarer le réseau, les volumes, les images et les conteneurs via l'API compatible Docker de Podman |
| Podman rootless | Exécuter les conteneurs sans daemon root |
| SPIRE | Attester les workloads et leur délivrer des identités SPIFFE à durée courte |

### Runtime et ressources communes

Le runtime V2 utilise le réseau `rakuten_network`. Les noms sont centralisés
dans `infra/config/docker.yaml` afin qu'Ansible et Terraform emploient les
mêmes ressources.

| Volume V2 | Usage |
|---|---|
| `dvc_data` | données DVC |
| `pgdata` | cluster PostgreSQL |
| `logs_and_reports` | rapports et journaux partagés |
| `redis_data` | persistance Redis pour Infisical |
| `minio_data` | stockage objet |
| `airflow_vol` | persistance Airflow |
| `spire_server_data` | base SQLite et clés du SPIRE Server |
| `spire_agent_data` | clés et cache du SPIRE Agent |
| `spire_agent_socket` | socket de la SPIFFE Workload API |

### Images durcies

La migration vers Docker Hardened Images est progressive :

| Composant | Image V2 |
|---|---|
| PostgreSQL | `dhi.io/postgres:17` |
| Airflow | `dhi.io/airflow:3-compat` dans la stack Terraform |
| SPIRE Server | `dhi.io/spire-server:1.15.1` |
| SPIRE Agent | `dhi.io/spire-agent:1.15.1` |

Les autres services utilisent encore des images historiques, officielles ou
spécifiques au projet. La V2 ne doit donc pas encore être décrite comme une
stack intégralement DHI. Une authentification préalable à `dhi.io` est requise
pour tirer les images hardened.

### Bootstrap Ansible

Le bootstrap suit cet ordre logique :

1. créer les volumes et le réseau Podman ;
2. démarrer et enrôler SPIRE Server/Agent ;
3. récupérer les données DVC ;
4. extraire les journaux et rapports initiaux ;
5. démarrer PostgreSQL et restaurer les dumps Airflow/MLflow ;
6. démarrer MinIO, créer les buckets et charger les données ;
7. créer les comptes MinIO dédiés aux workloads ;
8. démarrer et configurer Infisical via son playbook dédié.

Les tâches sensibles utilisent `no_log: true`. Elles sont conçues pour
conserver les secrets existants et ne les régénérer qu'en cas de demande
explicite.

### Gestion des secrets V2

Le flux prévu est :

```text
génération locale -> Ansible Vault chiffré -> seed Infisical
                                             |
                                             +-> identité data-ingestion
                                             +-> identité preprocessing
                                             +-> identité training
                                             +-> identité mlflow
                                             +-> identités backup/build/evaluation
```

Le playbook `create_ansible_vault.yaml` génère notamment les rôles PostgreSQL
`root`, `admin`, `mlflow`, `airflow` et `infisical`. L'identifiant
root est demandé à l'utilisateur et les mots de passe sont générés avec
OpenSSL. Jenkins n'est pas déployé : l'intégration et le déploiement continus
sont assurés par GitHub Actions. Le Vault local sert de source de bootstrap ; Infisical devient la
source runtime.

Terraform marque les variables sensibles, mais cela ne chiffre pas le state.
Les states de `infra/terraform/docker` et `infra/terraform/infisical` doivent
donc être considérés comme sensibles, rester hors de Git et être placés à
terme dans un backend protégé.

### Identité SPIFFE/SPIRE

Le domaine de confiance est `rakuten.local`. Le nœud Podman est enrôlé sous :

```text
spiffe://rakuten.local/agent/rakuten-host
```

Le SPIRE Agent :

- fonctionne avec `pid: host` pour identifier le processus appelant ;
- monte le socket rootless Podman afin de relier ce PID à un conteneur ;
- combine les attesteurs `unix` et `docker` ;
- expose la Workload API dans le volume `spire_agent_socket` ;
- utilise un trust bundle explicite et un jeton d'enrôlement à usage unique.

Un workload compatible doit monter le volume du socket et définir :

```text
SPIFFE_ENDPOINT_SOCKET=unix:///run/spire/sockets/agent.sock
```

Il doit également disposer d'une entrée d'enregistrement SPIRE associant son
SPIFFE ID à des sélecteurs suffisamment restrictifs, par exemple UID, label et
digest d'image. Cette création d'entrées reste actuellement une opération
explicite ; elle n'est pas encore pilotée par Terraform.

Le montage `:ro` du socket Podman empêche sa modification comme fichier, mais
ne réduit pas les méthodes disponibles dans l'API Podman. Le SPIRE Agent reste
donc un composant privilégié qui doit être traité comme faisant partie de la
base de confiance de la machine.

### Déploiement V2 prévu

```bash
# 1. Dépendances Ansible
ansible-galaxy collection install -r infra/ansible/requirements.yaml

# 2. Secrets de bootstrap
ansible-playbook infra/ansible/playbooks/dev/create_ansible_vault.yaml

# 3. Runtime rootless et registre DHI
systemctl --user enable --now podman.socket
docker login dhi.io

# 4. Données, services de bootstrap et SPIRE
ansible-playbook infra/ansible/playbooks/bootstrap/bootstrap.yaml

# 5. Infisical et ses identités
ansible-playbook infra/ansible/playbooks/bootstrap/setup_infisical.yaml

# 6. Stack applicative déclarative
terraform -chdir=infra/terraform/docker init
terraform -chdir=infra/terraform/docker apply
```

Les variables `TF_VAR_infisical_provider_client_id`,
`TF_VAR_infisical_provider_client_secret` et `TF_VAR_infisical_project_id`
doivent être disponibles avant l'application de la stack Terraform Docker.

## Correspondance V1 vers V2

| Sujet | V1 | V2 cible |
|---|---|---|
| Runtime | Docker Engine | Podman rootless |
| Orchestration infra | scripts + Compose | Ansible + Terraform |
| Secrets | `.env` | Vault pour bootstrap, Infisical au runtime |
| Identité workload | credentials statiques | Universal Auth puis SPIFFE/SPIRE |
| Réseau | `mlflow-network` | `rakuten_network` |
| Déclaration des conteneurs | un fichier Compose | modules Terraform par domaine |
| Initialisation des données | scripts/Compose | tâches Ansible idempotentes |
| Images durcies | Airflow seulement selon le Compose courant | PostgreSQL, Airflow et SPIRE, migration progressive |
| Moindre privilège | comptes souvent partagés | comptes et permissions par workload |

## Travaux restant pour terminer la V2

Les principaux points encore ouverts sont :

1. migrer ou reconstruire les autres images applicatives sur des bases DHI ;
2. monter la Workload API dans les conteneurs qui consommeront des SVID ;
3. automatiser les entrées SPIRE et ajouter des labels d'attestation stables ;
4. remplacer progressivement les secrets statiques par une authentification
   fondée sur l'identité du workload lorsque les applications le permettent ;
5. protéger les states Terraform avec un backend distant chiffré et verrouillé ;
6. supprimer les valeurs de repli non sûres, notamment les credentials Airflow
   par défaut ;
7. ajouter des tests de déploiement, de restauration et de rotation des
   secrets/SVID ;
8. formaliser la sauvegarde de PostgreSQL, MinIO, Infisical et des données
   persistantes SPIRE ;
9. retirer la V1 uniquement après validation fonctionnelle et plan de retour
   arrière de la V2.

## Règles d'exploitation pendant la transition

- Ne pas démarrer V1 et V2 simultanément sur le même hôte.
- Ne jamais committer les `.env`, Vault déchiffrés, tfvars réels, tokens
  Infisical ou states Terraform.
- Épingler les versions d'images de production ; éviter `latest`.
- Sauvegarder les volumes avant toute migration destructive.
- Tester une rotation de secret sur un environnement non productif.
- Vérifier les sélecteurs SPIRE avant d'autoriser un workload : un sélecteur
  trop large peut délivrer la même identité à plusieurs conteneurs.
- Considérer l'accès au socket Podman, au socket SPIRE et aux states Terraform
  comme des accès privilégiés.

## Sécurité des workers V2 et trajectoire des prochaines versions

### Exécution SPIFFE avant la tâche fonctionnelle

Les workers Airflow exécutent `src.security.secure_runner` avant leur commande
métier. Le `DockerOperator` ajoute le label `org.rakuten.workload`, monte
`spire_agent_socket` en lecture seule et transmet uniquement l'UUID public de
l'identité Infisical. Ansible associe ce label à un SPIFFE ID dans SPIRE.

```text
DockerOperator -> container labellisé -> SPIRE Agent -> JWT-SVID
                                             |
                                             v
                                      Infisical SPIFFE Auth
                                             |
                                             v
                              secrets injectés au processus métier
```

Terraform Infisical produit la map publique `identity_ids` et ne crée plus de
client secrets Universal Auth. Le provider Infisical 0.16 ne fournit cependant
pas encore de ressource SPIFFE Auth. Il faut donc attacher cette méthode via
l'API ou l'interface Infisical avec le trust domain `rakuten.local`, l'audience
`infisical` et le SPIFFE ID exact du workload avant le démarrage des DAG.

SPIRE publie son bundle de clés publiques sur l'endpoint privé
`https://spire-server:8443`. Le profil Infisical à utiliser est **HTTPS Web
Bundle**, avec la CA `~/.config/rakuten/spire/tls/ca.crt` et un rafraîchissement
d'une heure. Infisical récupère ainsi automatiquement les nouvelles clés JWT
lors des rotations de l'autorité SPIRE ; aucun JWKS statique ne doit être
recopié à chaque déploiement.

Le certificat TLS de l'endpoint possède le SAN `spire-server` et l'endpoint
n'est joignable que depuis `rakuten_network`. Il ne doit être publié ni sur un
port hôte, ni par le futur reverse proxy, ni par Tailscale Funnel.

Les images `trainer`, `spacy`, `bentoml`, `dvc`, `minio-client` et
`postgres-backup` embarquent le bootstrap. Le JWT-SVID et l'access token
Infisical restent en mémoire et ne sont jamais passés à la commande métier.

### Reverse proxy, API du portfolio et Tailscale Funnel

La V2 prévoit le remplacement du reverse proxy historique. La solution cible
devra assurer le routage par service, TLS, les en-têtes de sécurité,
l'authentification et les limites de débit de l'API d'inférence.

À terme, le projet sera associé à un **Tailscale Funnel** afin d'exposer de
manière contrôlée l'inférence et certains dashboards via une API consommable
depuis le portfolio, en complément de Streamlit. Seules les routes
explicitement publiques devront traverser le Funnel. Airflow, MLflow,
Infisical, SPIRE et les interfaces d'administration resteront privés.

### Grafana Loki, Alloy et agent IA d'analyse des logs

Les prochaines versions ajouteront **Grafana Loki** pour centraliser et
rechercher les logs, **Grafana Alloy** pour les collecter, les transformer et
les acheminer, et Grafana pour leur exploration. Les logs d'Airflow, Podman,
SPIRE, Infisical, du reverse proxy et des applications devront être structurés,
horodatés et corrélés avec des labels à cardinalité maîtrisée.

Cette centralisation est une étape indispensable avant l'implémentation d'un
agent IA fondé sur un LLM pour analyser les incidents et anomalies. Cet agent
devra utiliser une identité en lecture seule, limiter ses requêtes Loki à une
fenêtre temporelle, masquer les secrets et données personnelles, citer les
logs sources et ne jamais lancer automatiquement une remédiation destructive.

Ordre recommandé :

1. remplacer et durcir le reverse proxy ;
2. définir les routes publiques et privées ;
3. intégrer Tailscale Funnel pour le portfolio ;
4. déployer Alloy, Loki et les règles de rétention ;

### Construction des images V2

La construction des images ne passe plus par
`docker/docker-images-build.sh`. La source de vérité est le catalogue
`docker/python_library_catalogs/stable.yaml`, consommé par le playbook Ansible
`infra/ansible/playbooks/dev/build_docker_images.yaml`. Celui-ci régénère les
fichiers Pixi puis construit et pousse les images avec Podman.

```bash
# Toutes les images déclarées dans le catalogue
ansible-playbook -i infra/ansible/inventory/hosts.ini \
  infra/ansible/playbooks/dev/build_docker_images.yaml \
  --tags build-all

# Une seule image
ansible-playbook -i infra/ansible/inventory/hosts.ini \
  infra/ansible/playbooks/dev/build_docker_images.yaml \
  --tags build-one \
  -e target_image=postgres-backup
```

Les tags produits suivent la convention `<nom>-latest`, par exemple
`minio-client-latest` et `postgres-backup-latest`.
