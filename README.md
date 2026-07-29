# Rakuten — Classification textuelle

Ce dépôt contient uniquement le noyau de classification textuelle des produits
Rakuten et les pointeurs DVC vers les données nécessaires.

## Contenu

- `data/raw/all_raw_data.csv.dvc` : données textuelles brutes ;
- `data/dataset.dvc` : jeu de données texte constitué ;
- `data/preprocessed.dvc` : données texte prétraitées ;
- `src/data_module_df/` : chargement, équilibrage, indexation et préparation ;
- `src/models_module_df/` : classifieur texte SVM/TF-IDF et utilitaires ;
- `src/pipeline_steps_df/data/` : étapes de constitution et de prétraitement ;
- `src/visualization_module_df/` : visualisations et explicabilité du modèle ;
- `src/tests/` : tests unitaires du périmètre conservé.

Les composants d'infrastructure et d'exploitation (Docker, Airflow, MLflow,
MinIO, BentoML, API, monitoring et déploiement) ne font pas partie de cette
branche.

## Installation

Le projet cible Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r src/tests/requirements.txt
```

Pour récupérer les données versionnées :

```bash
dvc pull
```

## Préparation des données

Depuis la racine du dépôt :

```bash
PYTHONPATH=. python -m src.pipeline_steps_df.data.make_original_dataset
PYTHONPATH=. python -m src.pipeline_steps_df.data.preprocessing
```

## Tests

```bash
PYTHONPATH=. pytest src/tests
```
