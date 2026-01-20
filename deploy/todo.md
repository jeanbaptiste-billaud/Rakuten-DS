- initialisation du projet : → ok, à finaliser et tester
  - branche deploy cloné sur host
  - script pour builder image dvc, créer volume et télécharger le repo
    - branche data cloné dans volume dvc_data via image docker dvc
    - récupération des données via dvc
    - décompression des archives dans les volumes associés
    - création et population des bucket par minio

- Airflow:
  - training: comparaison nouveau vs ancien → si mieux bento (branchOperator + empty)
    - httpoperator: récupération du run_id du model qui tourne
    - implémenter script de comparaison → import des data du modèle actuel via mlflow (run_id), data nouveau modèle via volume airflow
    - branchoperator: build nouveau modèle ou non.
  - implémenter httptriger pour chaque dag ; en attente maj api-gateway par flo

- script start.sh
- impression ecran + check prise en compte readme 