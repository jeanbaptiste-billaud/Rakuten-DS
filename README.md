# 🏭 Rakuten MLOps - Pipeline de Classification Produits

## 📋 Vue d'ensemble

Pipeline MLOps de production pour la classification automatique de produits Rakuten basée sur leurs descriptions textuelles. Le projet implémente une architecture microservices complète avec orchestration de données, entraînement automatisé, détection de drift et monitoring en temps réel.

L'infrastructure déployée couvre l'ensemble du cycle de vie MLOps : acquisition et versioning des données (MinIO + DVC), orchestration des pipelines (Airflow), expérimentation et tracking (MLflow), déploiement de modèles (BentoML), monitoring des performances (Prometheus/Grafana) et détection de dégradation (Evidently).

**Architecture des branches** :
- `dev` : Branche de développement
- `main` : Branche principale stable utilisée comme backup et source d'origine des branches 'deploy' et 'data'
- `deploy` : Branche de déploiement, contient les scripts et configurations pour l'exécution
- `data` : Branche de données versionnées avec DVC (clonée automatiquement par `init.sh`)

## 🏗️ Architecture Logicielle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          COUCHE PRÉSENTATION                            │
│         Grafana:3000  │  MLflow UI:5000  │  Airflow:8080                │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
┌──────────────────────────────────────────────────────────────────────────┐
│                         COUCHE APPLICATION                               │
│                                                                          │
│  ┌──────────────────┐                        ┌─────────────────────┐     │
│  │  API Gateway     │ ─────────────────────▶ │  Model Serving      │     │
│  │  (FastAPI:8000)  │                        │  (BentoML:3001)     │     │
│  └──────────────────┘                        └─────────────────────┘     │
│                                                         ▲                │
│                                                         │                │
│  ┌──────────────────┐       ┌──────────────────┐        │                │
│  │  MLflow Server   │ ◀───▶ │  Model Builder   │────────┘                │
│  │  (Tracking)      │       │  (BentoML)       │                         │
│  └──────────────────┘       └──────────────────┘                         │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  Drift Detector (Evidently:8003)                                 │    │
│  │  - Analyse post-training (run_id, y_true, y_pred, metrics)       │    │
│  │  - Comparaison baseline vs run courant                           │    │
│  │  - Rapports HTML + UI dédiée                                     │    │
│  │  - Data lineage des prédictions                                  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│           ▲                                     │                        │
│           │ (predict proxy)                     │ (evaluation)           │
│           └────────────── Model Serving ────────┘                        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
┌──────────────────────────────────────────────────────────────────────────┐
│                      COUCHE ORCHESTRATION                                │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  Airflow (Scheduler + API Server + DAG Processor)                │    │
│  │  - Orchestration des pipelines :                                 │    │
│  │    • Data processing (fetch, enrich, preprocess)                 │    │
│  │    • Training (entraînement, validation, tracking)               │    │
│  │    • CI/CD (build, test, deploy)                                 │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────┐                                                    │
│  │  DVC Container   │  (Pull/Push données versionnées)                   │
│  └──────────────────┘                                                    │
└──────────────────────────────────────────────────────────────────────────┘
                                      │
┌──────────────────────────────────────────────────────────────────────────┐
│                    COUCHE DONNÉES & MONITORING                           │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐    │
│  │  MinIO S3    │  │  PostgreSQL  │  │  MLflow      │  │ Prometheus │    │
│  │  (Agrégation │  │  (Metadata)  │  │  (Artifacts) │  │ (Metrics)  │    │
│  │   données)   │  │  :5432       │  │  S3 backend  │  │ :9090      │    │
│  │  :9000       │  │              │  │              │  │            │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘    │
│        ▲                                      ▲                          │
│        │                                      │                          │
│        └──────────────────┬───────────────────┘                          │
│                           │                                              │
│              Buckets : raw, dataset, preprocessed, mlflow                │
└──────────────────────────────────────────────────────────────────────────┘
```

### Flux de données

**Pipeline de données (Airflow)** : 
1. Les DAGs Airflow orchestrent les DockerOperators qui exécutent les tâches de traitement
2. Fetch des données depuis MinIO → Enrichissement → Preprocessing
3. Upload des résultats vers MinIO
4. Déclenchement de l'entraînement si nécessaire

**Pipeline d'entraînement** : 
1. Les données prétraitées sont récupérées depuis MinIO
2. Entraînement du modèle (SVM) avec logging MLflow
3. Le Model Builder (BentoML) package le modèle en service conteneurisé
4. Build de l'image Docker du modèle
5. Déploiement du service Model Serving

**Pipeline d'inférence** : 
1. Requête utilisateur → API Gateway (FastAPI)
2. API Gateway → Model Serving (BentoML)
3. Le modèle effectue preprocessing + prédiction
4. Retour de la classe prédite avec probabilités

**Monitoring** : 
1. Prometheus scrape les métriques exposées par tous les services
2. Grafana visualise via dashboards personnalisés
3. Drift Detector analyse les distributions de prédictions
4. Alertes en cas de dégradation détectée

## 🚀 Démarrage Rapide

### Prérequis système

- **Docker** (version 20.10+ avec support `docker compose`)
- **Git**
- **Credentials** :
  - GitHub Personal Access Token (PAT) pour cloner le repository privé
  - DagsHub user + token pour l'authentification DVC

Vérification :
```bash
docker --version
docker compose version
git --version
```

### Installation et déploiement

```bash
# Cloner la branche de déploiement
git clone -b deploy https://github.com/jeanbaptiste-billaud/Rakuten-DS.git
cd Rakuten-DS

# Lancer le script de démarrage
./start.sh
```

Le script `start.sh` effectue automatiquement :

1. **Phase DVC** (`init.sh`) :
   - Clone de la branche `data` contenant les datasets versionnés
   - Configuration de l'authentification DVC avec DagsHub
   - Pull des données via DVC
   - Restauration de la base PostgreSQL (métadonnées MLflow)

2. **Phase MinIO** :
   - Démarrage du stockage S3 MinIO
   - Création et population des buckets (raw, dataset, preprocessed, mlflow)

3. **Phase Services** :
   - Démarrage de tous les services via `docker compose up -d`

**Credentials demandés** :
- GitHub token (PAT) : Pour cloner le repository privé
- DagsHub user : Votre nom d'utilisateur DagsHub
- DagsHub token : Votre token d'API DagsHub

### Pour le développement

Si vous souhaitez contribuer au projet, clonez plutôt la branche `dev` :

```bash
git clone -b dev https://github.com/jeanbaptiste-billaud/Rakuten-DS.git
cd Rakuten-DS
# Suivre les instructions de développement dans CONTRIBUTING.md
```

### Accès aux interfaces

Une fois tous les services démarrés (attendre ~2-3 minutes), accéder aux différentes interfaces :

| Service | URL | Credentials |
|---------|-----|-------------|
| **MLflow UI** | http://localhost:5000 | - |
| **Grafana Dashboards** | http://localhost:3000 | Auto-login (anonymous admin) |
| **Prometheus Metrics** | http://localhost:9090 | - |
| **Airflow API Server** | http://localhost:8080 | airflow / airflow |
| **API Gateway Swagger** | http://localhost:8000/docs | - |
| **MinIO Console** | http://localhost:9001 | Voir .env (MINIO_ROOT_USER/PASSWORD) |
| **Drift Detector API** | http://localhost:8003/docs | - |
| **Model Serving** | http://localhost:3001 | - |

Vérifier l'état des services :
```bash
docker compose ps
```

### Test de l'API de prédiction

```bash
# Health check
curl http://localhost:8000/health

# Prédiction simple
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Console PlayStation 5 avec manette sans fil DualSense"}'

# Résultat attendu:
# {
#   "predicted_class": 50,
#   "confidence": 0.92,
#   "probabilities": {...}
# }
```

## 🛠️ Commandes Essentielles

### Gestion de l'infrastructure

```bash
# Voir l'état de tous les services
docker compose ps

# Voir les logs de tous les services
docker compose logs -f

# Logs d'un service spécifique
docker compose logs -f mlflow-server
docker compose logs -f airflow-scheduler
docker compose logs --tail=100 drift-detector

# Arrêter tous les services
docker compose down

# Redémarrer un service spécifique
docker compose restart api-gateway
docker compose restart model-serving

# Arrêter et supprimer les volumes (⚠️ perte de données)
docker compose down -v
```

### Gestion des données et modèles

```bash
# Accéder au conteneur DVC pour manipuler les données
docker compose run --rm dvc bash

# Vérifier le contenu d'un bucket MinIO
docker compose exec minio-client mc ls myminio/dataset
docker compose exec minio-client mc ls myminio/mlflow

# Backup des buckets MinIO vers local
docker compose exec minio-client python -m src.minio.minio_backup

# Lister les expériences MLflow (via conteneur)
docker compose exec mlflow-server mlflow experiments list
```

### Monitoring et debugging

```bash
# Vérifier les métriques Prometheus
curl http://localhost:9090/api/v1/query?query=model_requests_total

# Inspecter les métriques d'un service
curl http://localhost:8002/metrics

# Accéder à un conteneur pour debug
docker compose exec trainer bash
docker compose exec data python

# Vérifier la connectivité réseau entre services
docker compose exec api-gateway ping preprocessing
docker compose exec mlflow-server curl http://minio:9000/minio/health/live
```

### Airflow DAG management

```bash
# Lister les DAGs
docker compose exec airflow-webserver airflow dags list

# Déclencher un DAG
docker compose exec airflow-webserver airflow dags trigger rakuten_ml_pipeline

# Voir l'état des tâches
docker compose exec airflow-webserver airflow tasks list rakuten_ml_pipeline

# Tester une tâche spécifique
docker compose exec airflow-webserver airflow tasks test rakuten_ml_pipeline fetch_raw_data 2025-01-09
```

## 📊 Captures d'Écran des Interfaces

### MLflow Tracking UI

![*[Insérer capture d'écran de l'interface MLflow montrant les expérimentations et métriques]*](docs/screenshots/mlflow_experiments.png)
**Emplacement**: `docs/screenshots/mlflow_experiments.png`

Points d'intérêt à capturer : liste des runs, comparaison de métriques (accuracy, F1-score), visualisation des paramètres, artefacts stockés.

---

### Grafana Dashboard - Model Monitoring

![*[Insérer capture d'écran du dashboard Grafana avec métriques temps réel]*](docs/screenshots/grafana_dashboard.png)

**Emplacement**: `docs/screenshots/grafana_dashboard.png`

Points d'intérêt à capturer : total des prédictions, prédictions par minute, confiance moyenne, entropie (drift indicator), histogrammes de confiance et entropie, CPU/Memory usage.

---

### Prometheus Metrics


![*[Insérer capture d'écran de l'interface Prometheus avec targets et queries]*](docs/screenshots/prometheus_metrics.png)

**Emplacement**: `docs/screenshots/prometheus_metrics.png`

Points d'intérêt à capturer : status des targets (api-gateway, preprocessing, model-serving), exemple de requête PromQL, graphiques de métriques brutes.

---

### Airflow Pipeline DAG


![*[Insérer capture d'écran du DAG Airflow en mode Graph]*](docs/screenshots/airflow_dag_graph.png)

**Emplacement**: `docs/screenshots/airflow_dag_graph.png`

Points d'intérêt à capturer : graphe complet du DAG `rakuten_ml_pipeline`, état des tâches (success/failed), durée d'exécution, logs d'une tâche.

---

### MinIO Console - Buckets


![*[Insérer capture d'écran de l'interface MinIO avec les buckets]*](docs/screenshots/minio_buckets.png)

**Emplacement**: `docs/screenshots/minio_buckets.png`

Points d'intérêt à capturer : liste des buckets (raw, dataset, preprocessed, mlflow), contenu d'un bucket, statistiques de stockage.

---

### API Gateway Swagger UI


![*[Insérer capture d'écran de l'interface Swagger avec endpoints]*](docs/screenshots/api_swagger.png)

**Emplacement**: `docs/screenshots/api_swagger.png`

Points d'intérêt à capturer : documentation des endpoints /predict et /health, exemple de requête/réponse, modèles Pydantic.

---

### Drift Detection Report


![*[Insérer capture d'écran d'un rapport Evidently de détection de drift]*](docs/screenshots/drift_report.png)

**Emplacement**: `docs/screenshots/drift_report.png`

Points d'intérêt à capturer : comparaison des distributions (référence vs production), métriques de drift, alertes de dégradation.

---

## 📁 Organisation des Screenshots

Créer l'arborescence suivante pour organiser les captures d'écran :

```
docs/
├── screenshots/
│   ├── README.md                    # Index des captures avec descriptions
│   ├── mlflow_experiments.png
│   ├── mlflow_artifacts.png
│   ├── grafana_dashboard.png
│   ├── grafana_alerts.png
│   ├── prometheus_metrics.png
│   ├── prometheus_targets.png
│   ├── airflow_dag_graph.png
│   ├── airflow_task_logs.png
│   ├── minio_buckets.png
│   ├── minio_browser.png
│   ├── api_swagger.png
│   ├── api_prediction_test.png
│   └── drift_report.png
```

## 🔧 Configuration Avancée

### Variables d'environnement

Le fichier `.env` contient toutes les configurations modifiables. Les variables principales sont les credentials MinIO (`MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`), les ports exposés, les paramètres PostgreSQL et MLflow, ainsi que les UIDs pour les permissions Docker.

### Ajout d'un nouveau service

Pour ajouter un service à l'architecture, créer un nouveau dossier dans `docker/`, définir le Dockerfile, ajouter le service dans `docker-compose.yml` avec les bonnes dépendances et variables d'environnement, puis exposer les métriques Prometheus via un endpoint `/metrics`.

### Personnalisation des dashboards Grafana

Les dashboards sont provisionnés automatiquement depuis `monitoring/grafana/provisioning/dashboards/`. Modifier les fichiers JSON existants ou ajouter de nouveaux dashboards en respectant le format Grafana.

## 🐛 Troubleshooting

**Services ne démarrent pas** : 
- Vérifier que les ports ne sont pas déjà utilisés : `netstat -tulpn | grep <port>` (Linux) ou `lsof -i :<port>` (macOS)
- Consulter les logs : `docker compose logs <service>`
- S'assurer que Docker dispose d'assez de ressources (RAM ≥ 4GB, CPU ≥ 2 cores)
- Vérifier que les volumes externes existent : `docker volume ls`

**Volumes manquants** :
Si vous rencontrez l'erreur "volume not found", créer manuellement les volumes :
```bash
docker volume create pgdata
docker volume create minio_data
docker volume create dvc_data
docker volume create logs_and_reports
```

**Erreurs de permissions** : 
- Vérifier que les UIDs dans `.env` correspondent à votre utilisateur : `id -u`
- Pour diagnostic temporaire : `chmod -R 777 logs_and_reports/`

**MinIO inaccessible** : 
- Vérifier le healthcheck : `docker compose ps minio`
- Tester la connexion : `curl http://localhost:9000/minio/health/live`
- Vérifier les credentials dans `.env`

**MLflow ne peut pas écrire les artefacts** : 
- Vérifier que `MLFLOW_S3_ENDPOINT_URL` pointe vers `http://minio:9000` (nom du service Docker)
- Vérifier les credentials AWS dans les variables d'environnement
- Tester l'accès MinIO : `docker compose exec minio-client mc ls myminio/mlflow`

**Airflow DAG ne s'affiche pas** : 
- Vérifier la syntaxe Python du DAG
- Consulter les logs du dag-processor : `docker compose logs airflow-dag-processor`
- Le DAG doit être dans le dossier monté `/opt/airflow/dags`

**Problème avec DVC** :
- Vérifier les credentials DagsHub dans `.env`
- Re-exécuter `init.sh` pour reconfigurer DVC
- Vérifier manuellement : `docker compose run --rm dvc dvc status`

## 📚 Documentation Technique

### Endpoints API disponibles

**API Gateway** (`http://localhost:8000`) :
- `GET /` - Informations sur le service
- `GET /health` - État de santé de tous les services
- `POST /predict` - Prédiction de classification
- `GET /metrics` - Métriques Prometheus

**Preprocessing Service** (`http://localhost:8001`) :
- `GET /` - Informations sur le service
- `GET /health` - État de santé
- `POST /preprocess` - Nettoyage et lemmatisation de texte
- `GET /metrics` - Métriques Prometheus

**Model Serving** (`http://localhost:8002`) :
- `POST /predict` - Prédiction SVM
- `GET /metrics` - Métriques Prometheus incluant confiance et entropie

### Métriques Prometheus exposées

Chaque service expose des métriques standardisées : compteurs de requêtes par endpoint et méthode, histogrammes de latence, métriques métier (confiance moyenne, entropie des prédictions, distribution des classes prédites).

### Structure du DAG Airflow

Le pipeline exécute séquentiellement : initialisation des permissions du volume, fetch des données depuis MinIO, enrichissement du dataset, preprocessing avec spaCy, upload des résultats vers MinIO. Chaque tâche est un DockerOperator utilisant l'image `spacy:3.7.5`.

## 🤝 Contribution

Pour contribuer au projet, forker le repository, créer une branche de feature, suivre les conventions PEP8, ajouter des tests si nécessaire, et soumettre une pull request avec une description détaillée des changements.

## 📄 Licence

Projet développé dans le cadre du challenge Rakuten France, adapté pour démonstration MLOps.
