# Bootstrap de l'infrastructure

## Secrets PostgreSQL

Le playbook de développement crée le Vault local et génère un mot de passe
aléatoire de 256 bits pour chaque rôle PostgreSQL. L'identifiant root est
demandé à l'exécution (valeur par défaut : `postgres`). Les valeurs existantes
sont conservées ; utiliser `-e vault_force_recreate=true` uniquement pour une
rotation volontaire.

```bash
ansible-playbook infra/ansible/playbooks/dev/create_ansible_vault.yaml
```

Le fichier de secrets est chiffré par Ansible Vault. Son mot de passe reste
dans `~/.config/rakuten/ansible/.vault_pass` avec le mode `0600`. Les tâches qui
manipulent les credentials sont protégées par `no_log: true`.

## SPIRE

Le bootstrap déploie un SPIRE Server et un SPIRE Agent avec les images Docker
Hardened Images 1.15.1 épinglées. Cette version est la dernière version stable
présente dans le catalogue DHI et prend en charge Podman rootless. Il crée :

- un domaine de confiance `rakuten.local` ;
- un stockage persistant séparé pour le serveur et l'agent ;
- un volume `spire_agent_socket` contenant la Workload API ;
- une identité de nœud `spiffe://rakuten.local/agent/rakuten-host` ;
- un jeton d'enrôlement à usage unique, masqué dans les logs Ansible ;
- une attestation combinée Unix + conteneur via l'API Podman rootless.
- un bundle endpoint HTTPS privé sur `https://spire-server:8443` ;
- une CA interne persistante pour authentifier cet endpoint.

Le certificat racine à fournir au profil **HTTPS Web Bundle** d'Infisical se
trouve dans `~/.config/rakuten/spire/tls/ca.crt`. L'endpoint n'est pas publié
sur l'hôte et reste accessible uniquement depuis `rakuten_network`.

Le serveur DHI s'exécute avec l'UID/GID `65532`. Ansible conserve la clé TLS
comme propriété de l'utilisateur de déploiement, mappe son groupe vers le GID
`65532` avec `podman unshare` et applique le mode `0640`. La clé privée de la
CA reste en `0600` et n'est jamais montée dans le conteneur SPIRE.

L'agent partage le namespace PID hôte et monte le socket Podman en lecture
seule. Ces accès sont nécessaires pour rattacher le PID appelant au conteneur.

Le bundle transmis à l'agent est exporté au format PEM. Le jeton d'enrôlement
est demandé au CLI SPIRE au format JSON et sa propriété `value` est extraite
sous `no_log`, ce qui évite de dépendre du format d'affichage humain. Avant de
générer un token, Ansible teste la santé d'un éventuel agent existant. Un agent
sain est réutilisé ; un agent absent, arrêté ou malsain est automatiquement
ré-enrôlé et recréé avec un token à usage unique.

```bash
podman login dhi.io
ansible-playbook infra/ansible/playbooks/bootstrap/bootstrap.yaml
```

Un workload qui doit obtenir un X.509-SVID monte le volume
`spire_agent_socket` sur `/run/spire/sockets` et utilise :

```text
SPIFFE_ENDPOINT_SOCKET=unix:///run/spire/sockets/agent.sock
```

Le playbook crée les entrées workload déclarées dans `spire.workloads`. Chaque
entrée associe le label Docker/Podman à un SPIFFE ID :

```text
docker:label:org.rakuten.workload:training
  -> spiffe://rakuten.local/workload/training
```

Le socket Podman rootless est monté en lecture seule dans `spire-agent` à
`/run/podman/podman.sock`. Ce montage est nécessaire pour que l'attesteur
Docker retrouve les labels du conteneur appelant à partir de son PID.

## Liste de suivi DVC

Le bootstrap recherche `.dvc/dvc_tracking_list.yaml` en priorité dans le dépôt
cloné depuis la branche `dvc`, stocké dans le volume `dvc_data`. Cette branche
reste donc la source normale de la liste des artefacts à restaurer.

Si le fichier n'est pas encore présent dans cette branche, Ansible utilise
temporairement la copie du dépôt contrôleur et affiche un message de fallback.
Le bootstrap ne reste ainsi pas bloqué pendant la synchronisation des branches.
Il échoue explicitement uniquement si le fichier est absent des deux sources.
Dès que la branche `dvc` contient sa copie, celle-ci est automatiquement
sélectionnée sans modification supplémentaire du playbook.

## Initialisation PostgreSQL

`create_ansible_vault.yaml` génère les identifiants et mots de passe des
comptes `root`, `admin`, `airflow`, `mlflow` et `infisical`. Le
bootstrap consomme ensuite directement ces valeurs chiffrées pour :

1. démarrer PostgreSQL avec le compte root lors d'une première initialisation ;
2. resynchroniser le mot de passe root sur un volume déjà existant ;
3. créer les rôles manquants ou mettre à jour leurs mots de passe ;
4. restaurer `airflow_db.dump` et `mlflow_db.dump` si les bases sont absentes ;
5. créer les autres bases manquantes et imposer leur propriétaire attendu.

Les secrets sont protégés par `no_log` et ne sont pas écrits dans une commande
Terraform ou un fichier d'environnement. Les restaurations sont idempotentes :
une base existante n'est pas écrasée lors d'un bootstrap normal.

Une restauration destructive volontaire doit être demandée explicitement :

```bash
ansible-playbook infra/ansible/playbooks/bootstrap/bootstrap.yaml \
  -e postgres_force_restore=true
```

Cette option exécute `pg_restore --clean` pour Airflow et MLflow.

L'image DHI PostgreSQL 17 utilise `PGDATA=/var/lib/postgresql/17/data` et le
volume V2 dédié `pgdata_dhi`. L'ancien volume `pgdata`, créé avec l'image
upstream et son chemin `/var/lib/postgresql/data`, n'est ni modifié ni supprimé
automatiquement. Cette séparation évite de forcer la récupération d'un ancien
cluster qui n'aurait pas été arrêté proprement. Les dumps DVC servent à
initialiser le nouveau volume DHI.

## Initialisation MinIO

Le bootstrap reprend la logique du script historique : il démarre MinIO,
attend que l'endpoint de santé réponde, puis lance l'image `minio-client` pour
créer et alimenter les buckets `raw`, `dataset`, `preprocessed` et
`mlflow-artifacts` depuis les données restaurées dans le volume `dvc_data`.
L'initialisation utilise `minio_init.py` et le SDK Python MinIO : création
idempotente avec `bucket_exists`/`make_bucket`, puis upload avec `fput_object`.
Le backup shellless utilise `list_objects(..., recursive=True)` et
`fget_object`. L'image reste basée sur le runtime Python DHI et n'embarque plus
de shell ni le binaire `mc`. Le volume DVC est monté en lecture seule pendant
l'initialisation et l'endpoint interne reste `minio:9000`.

Les comptes et politiques granulaires sont créés lorsque la section
`minio_service_accounts` existe dans le Vault. Un ancien Vault qui ne la
contient pas n'empêche pas la création et la population des buckets ; relancer
`create_ansible_vault.yaml` permet d'ajouter ces comptes en conservant les
secrets existants.

```bash
ansible-playbook infra/ansible/playbooks/bootstrap/bootstrap.yaml --tags minio
```
