import streamlit as st

st.set_page_config(
    page_title="Grafana - Rakuten MLOps",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Grafana - Monitoring Visuel")

st.markdown("""
Grafana affiche les dashboards de monitoring pour visualiser les métriques 
en temps réel de vos modèles et services.
""")

grafana_url = "http://localhost:3000"

st.markdown("### Dashboards Grafana")

st.markdown(f"""
<iframe src="{grafana_url}" width="100%" height="800" frameborder="0"></iframe>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"**[🔗 Ouvrir Grafana dans un nouvel onglet]({grafana_url})**")

with st.expander("ℹ️ Aide - Grafana"):
    st.markdown("""
    ### Dashboards disponibles :
    
    - **Model Performance** : Métriques de performance du modèle
    - **System Metrics** : CPU, RAM, Disk des services
    - **Request Metrics** : Latence, throughput des API
    - **Drift Detection** : Surveillance de la dérive du modèle
    
    ### Configuration :
    
    - **Authentification** : Auto-login activé
    - **Source de données** : Prometheus configuré par défaut
    - **Refresh** : Rafraîchissement automatique toutes les 30s
    """)