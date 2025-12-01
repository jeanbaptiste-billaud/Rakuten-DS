#!/bin/bash
set -e

TOKEN_FILE="/app/auth.json"

if [ ! -f "$TOKEN_FILE" ]; then
  echo "❌ Fichier d'authentification introuvable : $TOKEN_FILE"
  exit 1
fi

# Extraire tous les tokens
ADMIN_HASH=$(jq -r '.users.admin.token_hash' $TOKEN_FILE)
USER_HASH=$(jq -r '.users.user.token_hash' $TOKEN_FILE)

# Vérifier que la variable AUTH_TOKEN est présente
if [ -z "$AUTH_TOKEN" ]; then
  echo "❌ Variable AUTH_TOKEN manquante."
  exit 1
fi

# Calcul du hash du token fourni
TOKEN_HASH=$(python3 -c "import hashlib; print(hashlib.sha256('$AUTH_TOKEN'.encode()).hexdigest())")

ROLE=""

if [ "$TOKEN_HASH" == "$ADMIN_HASH" ]; then
  ROLE="admin"
elif [ "$TOKEN_HASH" == "$USER_HASH" ]; then
  ROLE="user"
else
  echo "🚫 Token invalide : accès refusé."
  exit 1
fi

echo "✅ Authentification réussie. Rôle : <$ROLE>"
exec "$@"
