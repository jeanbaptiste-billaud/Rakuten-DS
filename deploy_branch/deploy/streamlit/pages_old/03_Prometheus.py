import streamlit as st

st.set_page_config(
    page_title="Prometheus - Rakuten MLOps",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Prometheus - Métriques Système")

st.markdown("""
Prometheus collecte et stocke les métriques de tous les services 
pour le monitoring et l'alerting.
""")

prometheus_url = "http://localhost:9090"

st.markdown("### Interface Prometheus")

st.markdown(f"""
<iframe src="{prometheus_url}" width="100%" height="800" frameborder="0"></iframe>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"**[🔗 Ouvrir Prometheus dans un nouvel onglet]({prometheus_url})**")

with st.expander("ℹ️ Aide - Prometheus & PromQL"):
    st.markdown("""
    ### Métriques disponibles :
    
    **Modèle ML :**
    - `model_requests_total` : Nombre total de requêtes
    - `model_prediction_confidence` : Confiance des prédictions
    - `model_entropy_last_value` : Entropie (indicateur de drift)
    - `model_request_duration_seconds` : Latence des requêtes
    
    ### Exemples de requêtes PromQL :
    
    ```promql
    # Taux de requêtes par seconde
    rate(model_requests_total[5m])
    
    # Latence moyenne
    rate(model_request_duration_seconds_sum[5m]) / 
    rate(model_request_duration_seconds_count[5m])
    
    # Top 5 des catégories prédites
    topk(5, sum by (predicted_class) (model_requests_total))
    ```
    """)