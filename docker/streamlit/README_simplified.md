# 🏭 Rakuten MLOps - Classification Produits

## 📋 À Propos

Pipeline MLOps de production pour la classification automatique de produits Rakuten basée sur leurs descriptions textuelles. Architecture microservices complète avec orchestration, monitoring temps réel et détection de drift.

## 🏗️ Architecture

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

## 🎯 Composants Principaux

**Data & Training**
- **MinIO** : Stockage S3-compatible (données + artefacts)
- **Airflow** : Orchestration des pipelines de données
- **MLflow** : Tracking expérimentations et versioning modèles
- **PostgreSQL** : Métadonnées MLflow

**Inférence**
- **API Gateway** (FastAPI) : Point d'entrée unique
- **Preprocessing** : Nettoyage texte avec spaCy
- **Model Serving** (BentoML) : Service de prédiction SVM

**Monitoring**
- **Prometheus** : Collecte de métriques
- **Grafana** : Dashboards de visualisation
- **Drift Detector** (Evidently) : Détection dégradation modèle

## 🚀 Démarrage Rapide

```bash
# Cloner le projet
git clone -b deploy https://github.com/jeanbaptiste-billaud/Rakuten-DS.git
cd Rakuten-DS

# Lancer l'infrastructure
./start.sh
```

**Le script effectue** :
1. Clone de la branche `data` avec DVC
2. Pull des datasets versionnés
3. Démarrage MinIO + création buckets
4. Démarrage de tous les services

## 🌐 Accès aux Services

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow UI | http://localhost:5000 | - |
| Grafana | http://localhost:3000 | Auto-login |
| Prometheus | http://localhost:9090 | - |
| Airflow | http://localhost:8080 | airflow / airflow |
| API Gateway | http://localhost:8000/docs | - |
| MinIO Console | http://localhost:9001 | Voir `.env` |
| Drift Detector | http://localhost:8003/docs | - |
| Model Serving | http://localhost:3001 | - |

## 🧪 Test de l'API

```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Console PlayStation 5 avec manette DualSense"}'

# Résultat attendu
{
  "predicted_class": "Jeux Video",
  "confidence": 0.94,
  "all_probabilities": {...}
}
```

## 📊 Commandes Utiles

**Docker**
```bash
# Voir l'état des services
docker compose ps

# Logs d'un service
docker compose logs -f <service>

# Redémarrer un service
docker compose restart <service>

# Arrêter l'infrastructure
docker compose down
```

**Airflow**
```bash
# Déclencher le pipeline
docker compose exec airflow-apiserver airflow dags trigger rakuten_ml_pipeline

# Voir les DAGs
docker compose exec airflow-apiserver airflow dags list

# Logs d'une tâche
docker compose exec airflow-apiserver airflow tasks logs rakuten_ml_pipeline <task_id> <date>
```

**MinIO**
```bash
# Lister les buckets
docker compose exec minio-client mc ls myminio

# Vérifier un bucket
docker compose exec minio-client mc ls myminio/dataset
```

## 🔧 Flux de Données

**Pipeline de données** :
1. Airflow orchestre les tâches Docker
2. Fetch données MinIO → Enrichissement → Preprocessing
3. Upload résultats vers MinIO

**Pipeline d'entraînement** :
1. Récupération données prétraitées (MinIO)
2. Entraînement SVM + logging MLflow
3. Packaging modèle avec BentoML
4. Déploiement du service

**Pipeline d'inférence** :
1. Requête → API Gateway
2. API Gateway → Model Serving
3. Preprocessing + Prédiction
4. Retour classe + probabilités

## 🐛 Troubleshooting Rapide

**Services ne démarrent pas**
```bash
# Vérifier les logs
docker compose logs <service>

# Vérifier les ports
netstat -tulpn | grep <port>  # Linux
lsof -i :<port>                # macOS
```

**Volumes manquants**
```bash
docker volume create pgdata
docker volume create minio_data
docker volume create dvc_data
docker volume create logs_and_reports
```

**MLflow artefacts**
```bash
# Vérifier accès MinIO
docker compose exec minio-client mc ls myminio/mlflow
```

**Airflow DAG invisible**
```bash
# Vérifier logs dag-processor
docker compose logs airflow-dag-processor
```

## 📚 Endpoints API

**API Gateway** (`:8000`)
- `GET /` - Infos service
- `GET /health` - État de santé
- `POST /predict` - Classification
- `GET /metrics` - Métriques Prometheus

**Model Serving** (`:3001`)
- `POST /predict` - Prédiction SVM
- `GET /metrics` - Métriques (confiance, entropie)

## 🤝 Architecture des Branches

- `deploy` : Déploiement production
- `data` : Données versionnées DVC
- `main` : Branche stable
- `dev` : Développement

## 📄 Licence

Projet Rakuten France - Démonstration MLOps
