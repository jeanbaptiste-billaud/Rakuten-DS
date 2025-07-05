# 📁 pipeline\_steps

Ce dossier contient les différentes étapes de la pipeline MLOps, organisées en scripts modulaires. Chaque stage peut être exécuté individuellement ou appelé depuis `main_orchestrator.py` (dans `app/`).

---

## 🔹 Stage 01 — Initialisation

**Fichier :** `stage01_initialisation.py`

* Charge le fichier de configuration `config.yaml`
* Vérifie ou télécharge les données prétraitées
* Exécute le prétraitement si `force_preprocessing: true`
* Génère les splits d'entraînement / validation / test
* Applique le balancing et l'extraction de features texte et image

---

## 🔹 Stage 02 — Entraînement et évaluation du modèle image

**Fichier :** `stage02_eval_image.py` **Classe :** `StageImagePipeline`

* Charge les données image (`X_train`, `X_test`, ...)
* Initialise, entraîne, sauvegarde un classifieur image (XGBoost ou MLP)
* Évalue les performances et trace la matrice de confusion

---

## 🔹 Stage 03 — Entraînement et évaluation du modèle texte

**Fichier :** `stage03_eval_text.py` **Classe :** `StageTextPipeline`

* Charge les données texte (`text_train`, `text_test`, ...)
* Initialise, entraîne, sauvegarde un classifieur texte (SVM)
* Évalue les performances et trace la matrice de confusion

---

## 🔹 Stage 04 — Fusion multimodale

**Fichier :** `stage04_fusion_multimodale.py` **Classe :** `StageFusionPipeline`

* Charge les prédictions des modèles texte et image
* Applique une stratégie de fusion (`mean`, `product`, etc.)
* Évalue le modèle multimodal fusionné
* Génère la matrice de confusion fusionnée

---

## ▶️ Exécution orchestrée

Pour enchaîner les étapes automatiquement, utiliser :

```bash
python app/main_orchestrator.py --configs configs.yaml --stages stage01 stage02 stage03 stage04
```

Chaque étape peut être lancée indépendamment ou combinée.
