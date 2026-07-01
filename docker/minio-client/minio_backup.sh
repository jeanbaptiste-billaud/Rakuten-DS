#!/bin/sh
set -euo pipefail

export MC_CONFIG_DIR=/tmp/.mc
rm -rf "$MC_CONFIG_DIR"
mkdir -p "$MC_CONFIG_DIR"

set +e

echo "🔗 Configuration de l'alias MinIO..."
mc alias set myminio http://${MINIO_HOST}:${MINIO_PORT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}

BUCKETS_LIST=("dataset" "preprocessed", ${MINIO_MLFLOW_BUCKET})

DATA_PATH='/dvc_data/Rakuten-DS/data'

echo "Début de la synchronisation des buckets MinIO..."
echo "--------------------------------------------------"

for BUCKET in "${BUCKETS_LIST[@]}"; do

    SOURCE="myminio/${BUCKET}/"
    DESTINATION="${DATA_PATH}/${BUCKET}/"

    echo "▶️ Traitement du bucket : ${BUCKET}"

    # 1. Condition IF : Vérifier si le bucket n'est PAS vide
    # La commande mc ls liste le contenu.
    # grep -q '.' renvoie 0 (succès) si au moins une ligne (un objet) est trouvée.
    if [ -n "$(mc ls ${SOURCE})" ] ; then

        echo "   ✅ Le bucket n'est pas vide. Préparation de la synchronisation..."

        # S'assurer que le répertoire de destination local existe
        mkdir -p "${DESTINATION}"

        # 2. Synchronisation du distant vers le local
        # --overwrite : Assure que les fichiers modifiés sont mis à jour.
        # ATTENTION : L'option --remove est OMISE pour ne pas supprimer les fichiers/dossiers locaux supplémentaires.
        mc mirror --overwrite --md5 "${SOURCE}" "${DESTINATION}"

        # Vérifier le code de retour de mc mirror
        if [ $? -eq 0 ]; then
            echo "   👍 Synchronisation de '${BUCKET}' terminée avec succès dans ${DESTINATION}"
        else
            echo "   ❌ ERREUR : La synchronisation de '${BUCKET}' a échoué. Code de retour : $?"
        fi

    else
        echo "   ⚠️ Le bucket est vide ou inaccessible. Synchronisation ignorée."
    fi

    echo "---"

done

echo "✅ Toutes les vérifications et synchronisations sont terminées."
