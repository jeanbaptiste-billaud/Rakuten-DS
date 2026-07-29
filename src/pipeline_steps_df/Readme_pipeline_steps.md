# Étapes de préparation des données textuelles

Ce dossier contient les scripts autonomes conservés pour construire et
prétraiter le jeu de données texte :

- `data/make_original_dataset.py` équilibre et constitue le jeu de données ;
- `data/preprocessing.py` nettoie et prépare les champs textuels ;
- `data/enrich_raw_dataset.py` ajoute de nouveaux échantillons textuels.

Les scripts doivent être lancés depuis la racine du dépôt avec `PYTHONPATH=.`,
après récupération des données avec DVC.
