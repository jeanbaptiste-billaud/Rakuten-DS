import streamlit as st

st.set_page_config(
    page_title="Drift Detection - Rakuten MLOps",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Drift Detection - Surveillance du Modèle")

st.markdown("""
Le système de détection de drift surveille la qualité et la stabilité du modèle 
en analysant les distributions de données et les prédictions.
""")

drift_url = "http://localhost:8003"

st.markdown("### 📡 Service Drift Detector")

st.markdown(f"**[🔗 Ouvrir l'API Drift Detector]({drift_url}/docs)**")

st.markdown("---")

# Informations sur le drift
st.markdown("### 📊 Qu'est-ce que le Drift ?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📉 Data Drift")
    st.markdown("""
    Les features changent mais 
    la relation feature-target 
    reste stable.
    
    **Exemple :** Nouvelles 
    catégories de produits
    """)

with col2:
    st.markdown("#### 🔄 Concept Drift")
    st.markdown("""
    La relation entre features 
    et target change.
    
    **Exemple :** Changement dans 
    les préférences clients
    """)

with col3:
    st.markdown("#### 🏷️ Label Drift")
    st.markdown("""
    La distribution des 
    classes change.
    
    **Exemple :** Certaines 
    catégories plus fréquentes
    """)

# Informations utiles
with st.expander("ℹ️ Aide - Drift Detection"):
    st.markdown("""
    ### Métriques de surveillance :
    
    - **Entropie** : Mesure l'incertitude des prédictions
      - Valeur basse (< 2.0) : Modèle confiant
      - Valeur élevée (> 2.5) : Modèle incertain → Drift possible
    
    - **Confiance Moyenne** : Probabilité moyenne de la classe prédite
      - Haute (> 0.8) : Prédictions fiables
      - Basse (< 0.7) : Prédictions incertaines
    
    ### Rapports Evidently :
    
    Les rapports générés incluent :
    - **Data Drift Report** : Détection du drift sur les features
    - **Target Drift Report** : Analyse de la dérive des prédictions
    - **Data Quality Report** : Validation de la qualité des données
    - **Classification Performance** : Métriques de performance
    
    ### Actions en cas de drift détecté :
    
    1. **Analyser** : Identifier les features qui dérivent
    2. **Investiguer** : Comprendre la cause
    3. **Corriger** : Ré-entraîner le modèle avec nouvelles données
    4. **Monitorer** : Suivre l'évolution après correction
    
    ### API Drift Detector :
    
    ```python
    import requests
    
    # Déclencher une analyse de drift
    response = requests.post(
        'http://localhost:8003/analyze_drift',
        json={
            'reference_data': [...],
            'current_data': [...]
        }
    )
    ```
    """)

st.markdown("---")
st.markdown("Pour voir les rapports de drift détaillés, accédez à l'API via le lien ci-dessus.")