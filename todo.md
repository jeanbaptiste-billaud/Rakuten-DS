- transformer script boostrap/init.sh en tâches ansible + playbook

- créer/télécharger helm-cart pour chaque image docker:
  - airflow
  - mlflow
  - minio
  - postgres
  - grafana
  - prometheus
  - bentoml
  - evidently


- airflow:
  - supprimer tache bashoperator dans pipeline entrainement
  - remplacer par Trigger Jenkins job:
    - bento build & container
    - push registry
    - deploy
    - Healthcheck


- ZenML
  - découper entraînement en steps propres
  - gérer train / evaluate / compare
  - logger vers MLflow
  - produire un modèle candidat