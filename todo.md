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
  - remplacer par un workflow GitHub Actions réutilisable :
    - bento build & container
    - push registry
    - deploy
    - Healthcheck

