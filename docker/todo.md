- modifier les scripts services pour qu'ils utilisent les différents scripts de src montés dans les dockers:
  - preprocessing -> utilities: sera en charge de la manipulation des dataset et de la création du model dockérisé par bentoml
  - model -> à remplacer par le docker créé par bentoml

- entrainement: récupérer hash du dernier commit des données et le passer en variable d'env au docker d'entrainement

- initialisation du projet:
  - branche github dédiée
  - docker file dvc
  - script pour builder image dvc, créer volume et télécharger le repo