#!/bin/sh

BUCKETS_LIST=("raw" "dataset" "preprocessed")

DATA_PATH="/mnt/data"

echo "Début de la synchronisation des buckets MinIO..."
echo "--------------------------------------------------"

for BUCKET in "${BUCKETS_LIST[@]}"; do

    SOURCE="myminio/${BUCKET}/"
    DESTINATION="${CHEMIN_LOCAL_BASE}/${BUCKET}/"

    echo "▶️ Traitement du bucket : ${BUCKET}"

    # 1. Condition IF : Vérifier si le bucket n'est PAS vide
    # La commande mc ls liste le contenu.
    # grep -q '.' renvoie 0 (succès) si au moins une ligne (un objet) est trouvée.
    if mc ls "${SOURCE}" | grep -q '.'; then

        echo "   ✅ Le bucket n'est pas vide. Préparation de la synchronisation..."

        # S'assurer que le répertoire de destination local existe
        mkdir -p "${DESTINATION}"

        # 2. Synchronisation du distant vers le local
        # --overwrite : Assure que les fichiers modifiés sont mis à jour.
        # ATTENTION : L'option --remove est OMISE pour ne pas supprimer les fichiers/dossiers locaux supplémentaires.
        mc mirror --overwrite "${SOURCE}" "${DESTINATION}"

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
