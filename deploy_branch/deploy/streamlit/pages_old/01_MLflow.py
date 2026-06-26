import streamlit as st

st.set_page_config(
    page_title="MLflow - Rakuten MLOps",
    page_icon="📊",
    layout="wide"
)

st.title("📊 MLflow - Tracking d'Expérimentations")

st.markdown("""
MLflow permet de suivre vos expérimentations de machine learning, 
de versionner vos modèles et de gérer le cycle de vie des modèles.
""")

# URL accessible depuis le navigateur
mlflow_url = "http://localhost:5000"

# Affichage de l'interface MLflow en iframe
st.markdown("### Interface MLflow")

st.markdown(f"""
<iframe src="{mlflow_url}" width="100%" height="800" frameborder="0"></iframe>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"**[🔗 Ouvrir MLflow dans un nouvel onglet]({mlflow_url})**")

# Informations utiles
with st.expander("ℹ️ Aide - MLflow"):
    st.markdown("""
    ### Fonctionnalités principales :
    
    - **Experiments** : Organisez vos expérimentations par projets
    - **Runs** : Chaque run enregistre les paramètres, métriques et artefacts
    - **Models** : Registre centralisé de vos modèles ML
    - **Artifacts** : Stockage des modèles et fichiers associés
    
    ### Commandes API utiles :
    
    ```python
    import mlflow
    
    mlflow.set_tracking_uri("http://localhost:5000")
    
    with mlflow.start_run():
        mlflow.log_param("param1", 5)
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_artifact("model.pkl")
    ```
    """)