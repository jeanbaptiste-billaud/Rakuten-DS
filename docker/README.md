# Architecture Microservices - Rakuten Text Classification

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│    API Gateway (FastAPI:8000)       │
│    - POST /predict                  │
│    - GET /health                    │
└───────────┬─────────────────────────┘
            │
    ┌───────┴────────┐
    │                │
┌───▼──────────┐  ┌──▼───────────────┐
│ Preprocessing│  │  Model Service   │
│ Service      │  │  (SVM)           │
│ (spaCy:8001) │  │  (8002)          │
│              │  │                  │
│ - Lemmatiser │  │ - TF-IDF         │
│ - Stopwords  │  │ - SVM RBF        │
│ - Nettoyage  │  │ - Prédiction     │
└──────────────┘  └──────────────────┘
```

## 📦 Services

### 1. API Gateway (port 8000)
- **Rôle** : Point d'entrée unique, orchestration
- **Endpoints** :
  - `POST /predict` : Classification de texte
  - `GET /health` : État des services

### 2. Preprocessing Service (port 8001)
- **Rôle** : Nettoyage et lemmatisation du texte
- **Technologie** : spaCy (fr_core_news_sm)
- **Traitement** :
  - Suppression stopwords
  - Lemmatisation
  - Normalisation

### 3. Model Service (port 8002)
- **Rôle** : Inférence avec le modèle SVM
- **Modèle** : TF-IDF + SVM (RBF kernel)
- **Input** : Texte prétraité
- **Output** : Classe + probabilités

## 🚀 Démarrage rapide

### Prérequis
- Docker et Docker Compose installés
- Modèle SVM dans `models/SVM/model.pkl`

### Lancer les services

```bash
# À la racine du projet Rakuten-DS
docker-compose -f docker-compose.yml up --build
```

**Temps de démarrage** : 2-3 minutes (téléchargement modèle spaCy)

### Vérifier l'état

```bash
# Health check
curl http://localhost:8000/health

# Devrait retourner :
# {
#   "status": "healthy",
#   "services": {
#     "api_gateway": "healthy",
#     "preprocessing": "healthy",
#     "model": "healthy"
#   }
# }
```

## 📡 Utilisation de l'API

### Prédiction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Console de jeux vidéo PlayStation 5 avec manette sans fil"
  }'
```

**Réponse** :
```json
{
  "predicted_class": 50,
  "confidence": 0.92,
  "probabilities": {
    "10": 0.01,
    "40": 0.03,
    "50": 0.92,
    "60": 0.04
  }
}
```

### Test avec Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Ordinateur portable Dell 15 pouces"}
)

result = response.json()
print(f"Classe: {result['predicted_class']}")
print(f"Confiance: {result['confidence']:.2%}")
```

## 🛠️ Développement

### Structure des dossiers

```
docker/
├── api-gateway/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── preprocessing/
│   ├── Dockerfile
│   ├── service.py
│   └── requirements.txt
├── model/
│   ├── Dockerfile
│   ├── service.py
│   └── requirements.txt
└── README.md

docker-compose.yml
```

### Logs

```bash
# Tous les services
docker-compose logs -f

# Service spécifique
docker-compose logs -f api-gateway
docker-compose logs -f preprocessing
docker-compose logs -f model
```

### Rebuild d'un service

```bash
# Rebuild complet
docker-compose up --build

# Rebuild d'un service spécifique
docker-compose up --build api-gateway
```

### Arrêter les services

```bash
# Arrêt simple
docker-compose down

# Arrêt + suppression volumes
docker-compose down -v
```

## 🧪 Tests

### Test unitaire d'un service

```bash
# Preprocessing
curl -X POST "http://localhost:8001/preprocess" \
  -H "Content-Type: application/json" \
  -d '{"text": "Ceci est un test"}'

# Model (nécessite texte prétraité)
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"text_cleaned": "test texte lemmatise"}'
```

### Test de charge (optionnel)

```bash
# Installer apache-bench
sudo apt install apache2-utils

# 100 requêtes, 10 concurrentes
ab -n 100 -c 10 -T 'application/json' \
   -p test_payload.json \
   http://localhost:8000/predict
```

## 🔧 Configuration

### Variables d'environnement

Dans `docker-compose.yml`, vous pouvez ajuster :

```yaml
environment:
  - PREPROCESSING_SERVICE_URL=http://preprocessing:8001
  - MODEL_SERVICE_URL=http://model:8002
  - MODEL_PATH=/app/models/SVM/model.pkl  # Chemin du modèle
```

### Volumes

- `./models:/app/models:ro` → Modèle SVM en lecture seule
- `./logs:/app/logs` → Logs partagés

## 📊 Monitoring

### Métriques disponibles

- Temps de réponse par service
- Nombre de prédictions
- Taux d'erreur
- Confiance moyenne

### Ajout de Prometheus (Phase 4)

Décommenter dans `docker-compose.yml` :

```yaml
# prometheus:
#   image: prom/prometheus
#   volumes:
#     - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
#   ports:
#     - "9090:9090"
```

## 🐛 Troubleshooting

### Modèle non trouvé

```bash
# Vérifier le chemin
ls -lh models/SVM/model.pkl

# Rebuild avec logs
docker-compose up --build model
```

### Service preprocessing lent au démarrage

Normal : téléchargement du modèle spaCy (~40 MB). Attendre 1-2 minutes.

### Erreur de communication entre services

```bash
# Vérifier le réseau Docker
docker network inspect rakuten-ds_rakuten-network

# Ping entre services
docker exec rakuten-api-gateway ping preprocessing
```

## 📈 Prochaines étapes (Phase 3)

- [ ] Ajouter le Training Service
- [ ] Ajouter le Data Acquisition Service
- [ ] Intégrer Kubernetes
- [ ] CI/CD avec GitHub Actions
- [ ] Monitoring (Prometheus + Grafana)

## 📝 Notes

- **PEP8** : Code conforme aux conventions Python
- **Logs** : Format structuré pour faciliter le débogage
- **Health checks** : Surveillance automatique de l'état des services
- **Timeouts** : 30s pour les requêtes longues