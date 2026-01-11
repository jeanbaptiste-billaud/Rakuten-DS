import streamlit as st
import os
import requests

st.set_page_config(
    page_title="Airflow - Rakuten MLOps",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Airflow - Orchestration des Pipelines")

st.markdown("""
Apache Airflow orchestre les workflows de données et d'entraînement ML 
à travers des DAGs (Directed Acyclic Graphs).
""")

# Récupération de l'URL depuis les variables d'environnement
airflow_url = os.getenv("AIRFLOW_URL", "http://airflow-apiserver:8080")

# Affichage de l'interface Airflow en iframe
st.markdown("### Interface Airflow")

# Option 1 : Iframe (si CORS autorisé)
st.markdown(f"""
<iframe src="{airflow_url}" width="100%" height="800" frameborder="0"></iframe>
""", unsafe_allow_html=True)

# Option 2 : Lien direct (alternative)
st.markdown("---")
st.markdown(f"**Accès direct :** [Ouvrir Airflow dans un nouvel onglet]({airflow_url.replace('airflow-apiserver', 'localhost')})")

st.info("""
**Credentials Airflow :**
- **Username :** airflow
- **Password :** airflow
""")

# Informations sur les DAGs via l'API
st.markdown("---")
st.markdown("### 📊 Statut des DAGs")

try:
    # Récupération de la liste des DAGs via l'API Airflow
    response = requests.get(
        f"{airflow_url}/api/v1/dags",
        auth=('airflow', 'airflow'),
        timeout=5
    )
    
    if response.status_code == 200:
        dags_data = response.json()
        dags = dags_data.get('dags', [])
        
        if dags:
            col1, col2, col3 = st.columns(3)
            
            total_dags = len(dags)
            active_dags = sum(1 for dag in dags if not dag.get('is_paused', True))
            paused_dags = total_dags - active_dags
            
            col1.metric("Total DAGs", total_dags)
            col2.metric("Actifs", active_dags)
            col3.metric("En Pause", paused_dags)
            
            # Liste détaillée des DAGs
            st.markdown("#### Liste des DAGs")
            
            import pandas as pd
            
            dags_info = []
            for dag in dags:
                dags_info.append({
                    'DAG ID': dag.get('dag_id', 'N/A'),
                    'Description': dag.get('description', 'N/A')[:50] + '...' if dag.get('description') and len(dag.get('description', '')) > 50 else dag.get('description', 'N/A'),
                    'Statut': '▶️ Actif' if not dag.get('is_paused', True) else '⏸️ Pause',
                    'Schedule': dag.get('schedule_interval', 'N/A'),
                    'Tags': ', '.join(dag.get('tags', []))
                })
            
            df_dags = pd.DataFrame(dags_info)
            st.dataframe(df_dags, use_container_width=True)
        else:
            st.info("Aucun DAG configuré")
    else:
        st.warning(f"⚠️ Impossible de récupérer les DAGs (Status: {response.status_code})")
        
except Exception as e:
    st.warning(f"⚠️ Impossible de se connecter à l'API Airflow: {str(e)}")
    st.info("L'API Airflow peut ne pas être accessible depuis le conteneur Streamlit.")

# Informations utiles
with st.expander("ℹ️ Aide - Airflow"):
    st.markdown("""
    ### Concepts clés :
    
    **DAG (Directed Acyclic Graph) :**
    - Collection de tâches organisées avec leurs dépendances
    - Définit l'ordre d'exécution des tâches
    - Peut être schedulé ou déclenché manuellement
    
    **Tasks :**
    - Unité de travail dans un DAG
    - Peut être un script Python, un job Spark, une requête SQL, etc.
    - Reliées par des dépendances (upstream/downstream)
    
    **Operators :**
    - `PythonOperator` : Exécute du code Python
    - `BashOperator` : Exécute des commandes shell
    - `DockerOperator` : Lance un conteneur Docker
    - `EmailOperator` : Envoie des emails
    
    ### DAGs typiques dans ce projet :
    
    1. **data_ingestion_dag** : 
       - Récupération des données depuis MinIO
       - Nettoyage et préprocessing
       - Stockage dans le data warehouse
    
    2. **model_training_dag** :
       - Chargement des données d'entraînement
       - Entraînement du modèle SVM
       - Logging dans MLflow
       - Déploiement du meilleur modèle
    
    3. **drift_detection_dag** :
       - Analyse périodique du drift
       - Génération de rapports Evidently
       - Alerting si drift détecté
    
    ### Navigation Airflow :
    
    1. **DAGs** : Liste de tous les DAGs
    2. **Grid View** : Vue chronologique des runs
    3. **Graph View** : Visualisation du DAG
    4. **Calendar View** : Vue calendrier des exécutions
    5. **Task Duration** : Analyse de performance
    6. **Logs** : Logs détaillés de chaque task
    
    ### Commandes utiles :
    
    ```bash
    # Déclencher un DAG manuellement
    airflow dags trigger <dag_id>
    
    # Lister les DAGs
    airflow dags list
    
    # Tester une task
    airflow tasks test <dag_id> <task_id> <execution_date>
    
    # Voir les logs
    airflow tasks logs <dag_id> <task_id> <execution_date>
    ```
    
    ### API Airflow :
    
    ```python
    import requests
    
    # Authentification
    auth = ('airflow', 'airflow')
    
    # Lister les DAGs
    response = requests.get(
        'http://localhost:8080/api/v1/dags',
        auth=auth
    )
    
    # Déclencher un DAG
    response = requests.post(
        'http://localhost:8080/api/v1/dags/<dag_id>/dagRuns',
        auth=auth,
        json={}
    )
    ```
    """)

# Section statistiques
st.markdown("---")
st.markdown("### 📈 Statistiques d'Exécution")

try:
    # Récupération des dernières exécutions
    response = requests.get(
        f"{airflow_url}/api/v1/dags/~/dagRuns",
        auth=('airflow', 'airflow'),
        params={'limit': 10},
        timeout=5
    )
    
    if response.status_code == 200:
        runs_data = response.json()
        runs = runs_data.get('dag_runs', [])
        
        if runs:
            import pandas as pd
            
            runs_info = []
            for run in runs:
                runs_info.append({
                    'DAG ID': run.get('dag_id', 'N/A'),
                    'Run ID': run.get('dag_run_id', 'N/A')[:30] + '...' if len(run.get('dag_run_id', '')) > 30 else run.get('dag_run_id', 'N/A'),
                    'Statut': run.get('state', 'N/A'),
                    'Début': run.get('start_date', 'N/A')[:19] if run.get('start_date') else 'N/A',
                    'Fin': run.get('end_date', 'N/A')[:19] if run.get('end_date') else 'En cours'
                })
            
            df_runs = pd.DataFrame(runs_info)
            st.dataframe(df_runs, use_container_width=True)
        else:
            st.info("Aucune exécution récente")
    
except Exception as e:
    st.warning(f"⚠️ Impossible de récupérer les statistiques d'exécution")
