#!/bin/sh

# Désactive l'arrêt sur erreur
set +e

echo $(pwd)

#docker volume create mlflow_db
#
docker run --rm \
  -v mlflow_db:/db_data \
  -v ../data/mlflow_db.tar.gz:/backup/mlflow_db.tar.gz \
  alpine \
  tar xzvf /backup/mlfow_db.tar.gz -C /db_data