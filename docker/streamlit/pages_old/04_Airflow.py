import streamlit as st

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

airflow_url = "http://localhost:8080"

st.markdown("### Interface Airflow")

st.markdown(f"""
<iframe src="{airflow_url}" width="100%" height="800" frameborder="0"></iframe>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"**[🔗 Ouvrir Airflow dans un nouvel onglet]({airflow_url})**")

st.info("""
**Credentials Airflow :**
- **Username :** airflow
- **Password :** airflow
""")

with st.expander("ℹ️ Aide - Airflow"):
    st.markdown("""
    ### Concepts clés :
    
    **DAG (Directed Acyclic Graph) :**
    - Collection de tâches organisées avec leurs dépendances
    - Définit l'ordre d'exécution des tâches
    - Peut être schedulé ou déclenché manuellement
    
    **Operators :**
    - `PythonOperator` : Exécute du code Python
    - `BashOperator` : Exécute des commandes shell
    - `DockerOperator` : Lance un conteneur Docker
    - `EmailOperator` : Envoie des emails
    
    ### Commandes utiles :
    
    ```bash
    # Déclencher un DAG manuellement
    airflow dags trigger <dag_id>
    
    # Lister les DAGs
    airflow dags list
    
    # Tester une task
    airflow tasks test <dag_id> <task_id> <execution_date>
    ```
    """)