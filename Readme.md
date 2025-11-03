# 📦 Rakuten-DS - MLOps Pipeline

Pipeline MLOps pour la classification de produits Rakuten. Le projet a évolué d'une approche multimodale (image + texte) vers une **architecture microservices focalisée sur le texte** pour démontrer les bonnes pratiques MLOps.

---

## 🎯 Contexte du projet

**Projet initial** : Classification multimodale (image + texte) des produits Rakuten  
**Évolution MLOps** : Architecture microservices conteneurisée avec focus sur la classification textuelle

Ce repository documente la transition vers une infrastructure MLOps de production, en suivant une feuille de route par phases :
- ✅ **Phase 1** : Fondations & Conteneurisation (Dev Container)
- ✅ **Phase 2** : Microservices, Docker Compose, API d'inférence
- 🔄 **Phase 3** : Orchestration Kubernetes, CI/CD
- 📅 **Phase 4** : Monitoring, détection de dérive, maintenance

---

## 🚀 Démarrage rapide

### Prérequis

- Docker et Docker Compose
- VS Code (recommandé) ou IDE compatible devcontainer
- Git
- Accès au repository DagsHub (pour les données)

### 1️⃣ Cloner le repository

```bash
git clone https://github.com/jeanbaptiste-billaud/Rakuten-DS.git
cd Rakuten-DS
```

### 2️⃣ Configurer DVC (accès aux données)

```bash
# Dans le DevContainer ou votre terminal
dvc remote modify origin --local auth basic 
dvc remote modify origin --local user <Votre_Nom_Utilisateur>
dvc remote modify origin --local password <Votre_Token_DagsHub>

# Télécharger les données
dvc pull
```

> **Note** : Les credentials DVC sont disponibles dans l'onglet "Data" du projet DagsHub

### 3️⃣ Lancer l'architecture microservices

```bash
# Depuis votre terminal WSL2 ou Linux
docker-compose up --build
```

**Accès à l'API** :
- 🌐 Interface Swagger : http://localhost:8000/docs
- 🔍 Health check : http://localhost:8000/health
- 📡 Endpoint de prédiction : `POST http://localhost:8000/predict`

---

## 🏗️ Architecture Phase 2 - Microservices

```
┌─────────────────────────────────┐
│  API Gateway (FastAPI:8000)     │
│  - POST /predict                │
│  - GET /health                  │
└────────────┬────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼──────────┐  ┌──▼──────────────┐
│ Preprocessing│  │  Model Service  │
│ Service      │  │  (SVM)          │
│ (spaCy:8001) │  │  (8002)         │
│              │  │                 │
│ - Lemmatiser │  │ - TF-IDF        │
│ - Stopwords  │  │ - SVM RBF       │
│ - Nettoyage  │  │ - Prédiction    │
└──────────────┘  └─────────────────┘
```

### Services déployés

| Service | Port | Rôle | Technologie |
|---------|------|------|-------------|
| **API Gateway** | 8000 | Orchestration & point d'entrée | FastAPI |
| **Preprocessing** | 8001 | Nettoyage et lemmatisation texte | spaCy (fr_core_news_sm) |
| **Model Service** | 8002 | Inférence classification | TF-IDF + SVM (RBF) |

---

## 📡 Utilisation de l'API

### Test rapide avec curl

```bash
# Health check
curl http://localhost:8000/health

# Prédiction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Console PlayStation 5 avec manette sans fil DualSense"}'
```

### Réponse type

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

### Interface interactive

Accédez à l'interface Swagger pour tester l'API de manière interactive :
```
http://localhost:8000/docs
```

---

## 📁 Structure du projet

```bash
Rakuten-DS/
│
├── docker/                        # 🆕 Architecture microservices (Phase 2)
│   ├── api-gateway/               # Service API Gateway
│   ├── preprocessing/             # Service de prétraitement texte
│   ├── model/                     # Service d'inférence SVM
│   ├── README.md                  # Documentation détaillée microservices
│   └── test_api.py                # Script de test automatisé
│
├── docker-compose_lionel.yml             # 🆕 Orchestration des services
│
├── .devcontainer/                 # Configuration DevContainer
│   ├── devcontainer.json
│   └── requirements.txt
│
├── src/                           # Code source du projet initial
│   ├── data_module_df/            # Chargement, preprocessing, balancing
│   ├── models_module_df/          # Modèles (texte, image, fusion)
│   ├── pipeline_steps_df/         # Étapes du pipeline
│   └── visualization_module_df/   # Visualisations et SHAP
│
├── models/                        # Modèles entraînés
│   └── SVM/
│       └── model.pkl              # Modèle SVM de classification texte
│
├── data/                          # Données (gérées par DVC)
│   ├── dataset/
│   ├── preprocessed/
│   └── raw/
│
├── configs/                       # Fichiers de configuration
│   ├── config.yaml
│   ├── preprocess.yaml
│   └── train_image.yaml
│
└── README.md                      # Ce fichier
```

---

## 🧠 Workflow MLOps

### Phase 1 : Dev Container ✅

Le projet utilise un **DevContainer** pour assurer un environnement de développement reproductible :

```bash
# Ouvrir le projet dans VS Code
code Rakuten-DS

# VS Code détecte automatiquement le devcontainer
# Cliquer sur "Reopen in Container"
```

### Phase 2 : Microservices & API ✅

Architecture microservices avec Docker Compose :
- ✅ 3 services découplés (Gateway, Preprocessing, Model)
- ✅ API REST pour l'inférence
- ✅ Health checks automatiques
- ✅ Documentation Swagger intégrée

### Phase 3 : Orchestration & CI/CD 🔄 (À venir)

- Kubernetes pour la scalabilité
- GitHub Actions pour le CI/CD
- Tests automatisés
- Déploiement continu

### Phase 4 : Monitoring & Maintenance 📅 (À venir)

- Prometheus pour les métriques
- Grafana pour les dashboards
- Evidently pour la détection de dérive
- Réentraînement automatisé

---

## 🔧 Développement

### DevContainer (environnement de développement)

Le projet est configuré pour fonctionner dans un DevContainer (compatible VS Code et PyCharm) :

```bash
# Depuis le DevContainer
pip install -r .devcontainer/requirements.txt
pip install dvc[all]

# Configurer DVC
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <password>

# Récupérer les données
dvc pull
```

### Docker Compose (services de production)

```bash
# Build et lancement
docker-compose up --build

# Logs en temps réel
docker-compose logs -f

# Arrêt des services
docker-compose down
```

**Documentation détaillée** : Consultez `docker/README.md` pour plus d'informations sur l'architecture microservices.

---

## 📊 Versioning des données et modèles

Le projet utilise **DVC** (Data Version Control) pour gérer :
- Les datasets (raw, preprocessed)
- Les modèles entraînés
- Intégration avec DagsHub pour le stockage

```bash
# Ajouter de nouvelles données
dvc add data/dataset/new_data.csv
git add data/dataset/new_data.csv.dvc
git commit -m "Add new dataset"
dvc push

# Récupérer les données versionnées
dvc pull
```

---

## 🧪 Tests

### Test manuel via Swagger

```
http://localhost:8000/docs
```

### Test via curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Smartphone Samsung Galaxy S23 256 GB"}'
```

### Script de test automatisé

```bash
# Depuis WSL2 (avec Python installé)
python3 docker/test_api.py
```

---

## 🛠️ Technologies utilisées

### Phase 1-2 (Actuel)
- **Containerisation** : Docker, Docker Compose
- **Dev Environment** : DevContainer (VS Code, PyCharm)
- **Backend API** : FastAPI
- **ML Framework** : scikit-learn (SVM)
- **NLP** : spaCy (lemmatisation, stopwords)
- **Versioning** : DVC + DagsHub

### Phase 3-4 (À venir)
- **Orchestration** : Kubernetes
- **CI/CD** : GitHub Actions
- **Monitoring** : Prometheus, Grafana
- **ML Monitoring** : Evidently

---

## 📝 Projet initial (Historique)

Le projet initial incluait une approche **multimodale** (texte + image) avec :
- Extraction de features image via ResNet
- Classification textuelle via SVM
- Fusion des modalités
- Pipeline orchestré via `main_orchestrator.py`

Pour le **POC MLOps**, le focus est mis sur la **classification textuelle uniquement**, avec un dataset réduit (~10 000 échantillons) et réentraînements hebdomadaires incrémentaux.

### Ancien pipeline (référence)

```bash
# Lancement du pipeline complet (multimodal)
python app/main_orchestrator.py --configs configs.yaml --stages stage01 stage02 stage03 stage04
```

---

## 👥 Équipe & Contribution

Projet développé dans le cadre du challenge Rakuten France 2020, adapté pour démontrer les bonnes pratiques MLOps.

**Phase 2 (Microservices)** : Architecture conteneurisée avec Docker Compose  
**Deadline** : 24 octobre

---

## 📚 Documentation

- **Architecture microservices** : `docker/README.md`
- **Configuration DevContainer** : `.devcontainer/devcontainer.json`
- **API Documentation** : http://localhost:8000/docs (après lancement)

---

## 🐛 Troubleshooting

### Problème : Services ne démarrent pas

```bash
# Vérifier Docker
docker ps
docker-compose ps

# Rebuild complet
docker-compose down -v
docker-compose up --build
```

### Problème : Modèle SVM introuvable

```bash
# Vérifier la présence du modèle
ls -lh models/SVM/model.pkl

# Télécharger avec DVC si absent
dvc pull
```

### Problème : Erreur de credentials DVC

```bash
# Reconfigurer les credentials
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <token>
```

---

## 📈 Roadmap

- [x] Phase 1 : DevContainer & environnement reproductible
- [x] Phase 2 : Architecture microservices avec Docker Compose
- [ ] Phase 3 : CI/CD pipeline avec GitHub Actions
- [ ] Phase 3 : Migration vers Kubernetes
- [ ] Phase 4 : Monitoring (Prometheus/Grafana)
- [ ] Phase 4 : Détection de dérive (Evidently)
- [ ] Phase 4 : Réentraînement automatisé

---

## 📧 Contact

Pour toute question sur le projet, consultez la documentation dans `docker/README.md` ou les issues GitHub.