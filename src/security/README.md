# Bootstrap de sécurité des workers

Ce package exécute la phase de sécurité avant la phase fonctionnelle d'un
worker éphémère :

1. connexion à la Workload API locale de SPIRE par socket Unix ;
2. demande d'un JWT-SVID pour l'audience `infisical` ;
3. contrôle optionnel du SPIFFE ID attendu ;
4. échange du JWT-SVID contre un access token Infisical de courte durée ;
5. lecture du chemin de secrets autorisé ;
6. remplacement du processus par la commande métier, avec les secrets dans
   son environnement uniquement.

Les JWT-SVID et access tokens Infisical ne sont ni journalisés, ni écrits sur
disque, ni transmis dans la ligne de commande du worker. Le bootstrap échoue
fermé : la commande métier ne démarre pas si l'identité ou les secrets ne
peuvent pas être validés.

## Utilisation

Le container doit recevoir le socket de l'agent SPIRE en lecture seule :

```text
/run/spire/sockets/agent.sock
```

Puis Airflow peut lancer le worker ainsi :

```python
command = [
    "python",
    "-m",
    "src.security.secure_runner",
    "--",
    "python",
    "/src/train_text_model.py",
]
```

Pour conserver temporairement une commande métier shell existante :

```python
command = [
    "python",
    "-m",
    "src.security.secure_runner",
    "--",
    "sh",
    "-lc",
    script,
]
```

La première forme est préférable car elle n'ajoute pas d'interprétation shell.

## Variables publiques requises

| Variable | Description |
|---|---|
| `INFISICAL_IDENTITY_ID` | UUID de l'identité Infisical configurée en SPIFFE Auth |
| `INFISICAL_PROJECT_ID` | Projet Infisical autorisé pour l'identité |
| `INFISICAL_SECRET_PATH` | Chemin de secrets du worker, commençant par `/` |

Variables facultatives :

| Variable | Défaut | Description |
|---|---|---|
| `SPIFFE_ENDPOINT_SOCKET` | `unix:///run/spire/sockets/agent.sock` | Workload API SPIRE |
| `SPIFFE_AUDIENCE` | `infisical` | Audience du JWT-SVID |
| `SPIFFE_EXPECTED_ID` | vide | SPIFFE ID exact attendu |
| `INFISICAL_DOMAIN` | `http://infisical:8080` | URL Infisical |
| `INFISICAL_ENV` | `dev` | Environnement Infisical |
| `SPIRE_TIMEOUT_SECONDS` | `5` | Timeout Workload API |
| `SECURITY_HTTP_TIMEOUT_SECONDS` | `10` | Timeout HTTP Infisical |

Aucune de ces variables ne doit contenir un mot de passe. Les anciennes
variables `INFISICAL_TOKEN`, `*_CLIENT_ID` et `*_CLIENT_SECRET` sont supprimées
avant le démarrage de la commande métier.

## Images couvertes

Le package et sa dépendance `spiffe` sont intégrés aux environnements `trainer`,
`spacy`, `bentoml` et `dvc`. L'image PostgreSQL utilisée directement pour
`pg_dump` doit être remplacée par une petite image worker dédiée avant de
migrer cette tâche vers ce bootstrap Python.
