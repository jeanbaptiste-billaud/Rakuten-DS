# 📦 Rakuten-DS

Pipeline MLOps modulaire pour la classification de produits Rakuten à partir d'images et de descriptions textuelles. Le projet combine du traitement d'image, de texte et une fusion multimodale, avec une architecture modulaire pensée pour la production.

---

## 🚀 Prérequis

* Python 3.8+
* `pip install -r requirements.txt`
* Données prétraitées disponibles via Google Drive (ou générées via le stage01)

---

## 📁 Arborescence du projet

```bash
Rakuten-DS/
│
├── app/
│   └── main_orchestrator.py        # Script principal de lancement pipeline
│
├── config.yaml                     # Fichier de configuration central
│
├── pipeline_steps/                # Étapes de traitement modulaire
│   ├── stage01_initialisation.py
│   ├── stage02_eval_image.py
│   ├── stage03_eval_text.py
│   └── stage04_fusion_multimodale.py
│
├── src/
│   ├── data_module_def/           # Scripts de chargement, preprocessing, balancing...
│   ├── models_module_def/         # Gestion des modèles image/texte + fusion
│   └── visualization_module_def/  # Visualisation + SHAP
│
├── data/                          # Données (brutes, prétraitées, modèles, rapports)
└── README.md                      # (ce fichier)
```

---

## 🧠 Étapes du pipeline

| Stage | Nom                  | Rôle principal                             |
| ----- | -------------------- | ------------------------------------------ |
| 01    | `Initialisation`     | Prétraitement, splits, extraction features |
| 02    | `Évaluation image`   | Modèle image (XGBoost/MLP)                 |
| 03    | `Évaluation texte`   | Modèle texte (SVM)                         |
| 04    | `Fusion multimodale` | Fusion texte/image + évaluation finale     |

---

## ▶️ Lancement du pipeline

```bash
python app/main_orchestrator.py --config config.yaml --stages stage01 stage02 stage03 stage04
```

Tu peux également exécuter un sous-ensemble des étapes, par exemple :

```bash
python app/main_orchestrator.py --stages stage03 stage04
```

---

## 📦 Modules techniques (`src/`)

* `data_module_def/` : chargement, splits, balancing, feature extraction (ResNet, TF-IDF)
* `models_module_def/` : classifieurs image, texte, fusion, sauvegarde/rechargement
* `visualization_module_def/` : courbes, matrices de confusion, SHAP, etc.

---

## 📌 Auteur

Projet développé dans le cadre du challenge Rakuten 2020. Architecture adaptée pour la démonstration Streamlit et l'automatisation MLOps.
