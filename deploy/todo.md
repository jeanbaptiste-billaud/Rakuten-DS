- modifier les scripts services pour qu'ils utilisent les différents scripts de src montés dans les dockers :
  - preprocessing → utilities : sera en charge de la manipulation des dataset et de la création du model dockérisé par bentoml
  - model → à remplacer par le docker créé par bentoml

- entrainement: récupérer hash du dernier commit des données et le passer en variable d'env au docker d'entrainement

- initialisation du projet : → ok, à finaliser et tester
  - branche deploy cloné sur host
  - script pour builder image dvc, créer volume et télécharger le repo
    - branche data cloné dans volume dvc_data via image docker dvc
    - récupération des données via dvc
    - décompression des archives dans les volumes associés
    - création et population des bucket par minio

- remplacer bind mount dossier airflow par un volume dédié → ok

- implémenter la logique de backup similaire à bootstrap mais via un dag airflow:
  - 1 backup/volume <-- → 1 docker operator task 

- Airflow:
  - dag bento: redémarrage du docker de service après le push de l'image
  - training: comparaison nouveau vs ancien → si mieux bento (branchOperator + empty)
    - Si R2 score (drift) < 5% par rapport précédent, on garde modèle

- script start.sh
- impression ecran + check prise en compte readme 