import streamlit as st
import os

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

# Récupération de l'URL depuis les variables d'environnement
mlflow_url = os.getenv("MLFLOW_URL", "http://mlflow-server:5000")

# Affichage de l'interface MLflow en iframe
st.markdown("### Interface MLflow")

# Option 1 : Iframe (si CORS autorisé)
st.markdown(f"""
<iframe src="{mlflow_url}" width="100%" height="800" frameborder="0"></iframe>
""", unsafe_allow_html=True)

# Option 2 : Lien direct (alternative)
st.markdown("---")
st.markdown(f"**Accès direct :** [Ouvrir MLflow dans un nouvel onglet]({mlflow_url.replace('mlflow-server', 'localhost')})")

# Informations utiles
with st.expander("ℹ️ Aide - MLflow"):
    st.markdown("""
    ### Fonctionnalités principales :
    
    - **Experiments** : Organisez vos expérimentations par projets
    - **Runs** : Chaque run enregistre les paramètres, métriques et artefacts
    - **Models** : Registre centralisé de vos modèles ML
    - **Artifacts** : Stockage des modèles et fichiers associés
    
    ### Navigation :
    
    1. **Experiments** : Voir la liste des expérimentations
    2. **Runs** : Comparer les performances des différents runs
    3. **Models** : Gérer les versions de vos modèles
    4. **Artifacts** : Télécharger modèles et fichiers
    
    ### Commandes API utiles :
    
    ```python
    import mlflow
    
    # Définir le serveur MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Créer une expérimentation
    mlflow.create_experiment("mon_experience")
    
    # Logger des métriques
    with mlflow.start_run():
        mlflow.log_param("param1", 5)
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_artifact("model.pkl")
    ```
    """)
