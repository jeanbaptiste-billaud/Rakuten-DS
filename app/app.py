import streamlit as st
import pandas as pd
import numpy as np
from preprocess import ProductClassificationPipeline, PipelineConfig
import shap
import matplotlib.pyplot as plt
import os
from PIL import Image

# Initialisation de l'application
st.title("Classification Multimodale Rakuten - Démonstration")

@st.cache_resource
def load_pipeline():
    """Charge le pipeline et les modèles (mis en cache)"""
    config = PipelineConfig.from_yaml('config.yaml')
    pipeline = ProductClassificationPipeline(config)
    pipeline.prepare_data(force_preprocess=False)
    
    # Charger les modèles
    pipeline.load_model('xgboost')
    # pipeline.load_model('neural_net')
    
    try:
        import joblib
        pipeline.text_model = joblib.load('data/models/SVM/model.pkl')
        print("Modèle SVM texte chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle texte: {e}")
    
    return pipeline

# Charger le pipeline
pipeline = load_pipeline()

# Interface utilisateur
st.sidebar.header("Options")
model_type = st.sidebar.selectbox(
    "Modèle Image",
    ["xgboost", "neural_net"]
)
fusion_strategy = st.sidebar.selectbox(
    "Stratégie de fusion",
    ["mean", "product", "weighted"]
)
show_explanations = st.sidebar.checkbox("Afficher les explications", value=True)

# Formulaire d'entrée
st.header("Entrée de données")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Texte")
    text_input = st.text_area("Description du produit", "Saisir la description du produit ici...")

with col2:
    st.subheader("Image")
    uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_column_width=True)
        
        # Sauvegarder l'image temporairement
        temp_image_path = os.path.join("temp_uploads", uploaded_file.name)
        os.makedirs("temp_uploads", exist_ok=True)
        image.save(temp_image_path)
    else:
        temp_image_path = None
        
        
# Prédiction
if st.button("Classifier") and temp_image_path is not None:
    try:
        # Charger le modèle spécifié
        pipeline.load_model(model_type)
        
        # Effectuer la prédiction
        results = pipeline.predict_multimodal(text_input, temp_image_path, fusion_strategy)
        
        # Afficher les résultats
        st.header("Résultats")
        st.subheader(f"Classe prédite: {results['predicted_class_name']} (code {results['predicted_class']})")
        
        # Afficher les probabilités des top 5 classes
        top_indices = np.argsort(results['probabilities'])[-5:][::-1]
        top_probs = results['probabilities'][top_indices]
        top_classes = [pipeline.category_names[pipeline.idx_to_category[idx]] for idx in top_indices]
        
        prob_df = pd.DataFrame({
            "Classe": top_classes,
            "Probabilité": top_probs
        })
        
        st.bar_chart(prob_df.set_index("Classe"))
        
        # Comparaison des prédictions individuelles
        st.subheader("Prédictions par modalité")
        st.write(f"Texte uniquement: {pipeline.category_names[results['text_prediction']]}")
        st.write(f"Image uniquement: {pipeline.category_names[results['image_prediction']]}")
        
        # Explications SHAP
        if show_explanations:
            st.header("Explications du modèle")
            
            with st.spinner("Génération des explications..."):
                explanations = pipeline.get_model_explanations(text_input, temp_image_path, fusion_strategy)
                
                # Afficher les figures SHAP générées par le pipeline
                if 'figures' in explanations:
                    st.subheader("Valeurs SHAP - Modèle texte")
                    st.pyplot(explanations['figures']['text'])
                    
                    st.subheader("Valeurs SHAP - Modèle image")
                    st.pyplot(explanations['figures']['image'])
                    
                    st.subheader("Valeurs SHAP - Modèle fusionné")
                    st.pyplot(explanations['figures']['fusion'])
                
                # Afficher la synthèse des caractéristiques importantes
                if 'feature_importance' in explanations:
                    st.subheader("Synthèse des caractéristiques importantes")
                    
                    # Créer une figure pour les explications simplifiées
                    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Top features texte avec des étiquettes plus descriptives
                    text_indices = explanations['feature_importance'].get('text_indices', range(len(explanations['feature_importance']['text'])))
                    text_labels = [f"Feature {idx}" for idx in text_indices]
                    
                    ax[0].barh(range(len(explanations['feature_importance']['text'])),
                             explanations['feature_importance']['text'])
                    ax[0].set_yticks(range(len(text_labels)))
                    ax[0].set_yticklabels(text_labels)
                    ax[0].set_title("Top caractéristiques texte")
                    
                    # Top features image
                    image_indices = explanations['feature_importance'].get('image_indices', range(len(explanations['feature_importance']['image'])))
                    image_labels = [f"Feature {idx}" for idx in image_indices]
                    
                    ax[1].barh(range(len(explanations['feature_importance']['image'])),
                             explanations['feature_importance']['image'])
                    ax[1].set_yticks(range(len(image_labels)))
                    ax[1].set_yticklabels(image_labels)
                    ax[1].set_title("Top caractéristiques image")
                    
                    st.pyplot(fig)
                
                # Afficher l'importance relative des modalités
                if 'text_shap' in explanations and 'image_shap' in explanations:
                    # Calculer l'importance relative
                    text_importance = np.mean(np.abs(explanations['text_shap']))
                    image_importance = np.mean(np.abs(explanations['image_shap']))
                    total = text_importance + image_importance
                    
                    # Ajouter un petit graphique en camembert pour une visualisation plus claire
                    st.subheader("Contribution relative des modalités")
                    fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                    ax_pie.pie([text_importance/total*100, image_importance/total*100], 
                           labels=['Texte', 'Image'], 
                           autopct='%1.1f%%',
                           colors=['#ff9999','#66b3ff'])
                    ax_pie.set_title(f"Impact relatif des modalités ({fusion_strategy})")
                    st.pyplot(fig_pie)
                    
                    # Afficher les pourcentages explicitement
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Contribution Texte", f"{text_importance/total*100:.1f}%")
                    with col2:
                        st.metric("Contribution Image", f"{image_importance/total*100:.1f}%")
                
    except Exception as e:
        st.error(f"Erreur lors de la classification: {str(e)}")
        import traceback
        st.error(traceback.format_exc())  # Afficher la stack trace pour le debug

# Afficher les informations sur le projet
st.sidebar.markdown("---")
st.sidebar.header("À propos")
st.sidebar.info(
    "Ce démonstrateur permet de classifier des produits Rakuten "
    "en utilisant à la fois le texte et l'image du produit. "
    "Il utilise des modèles XGBoost et MLP pré-entraînés "
    "et fournit des explications via SHAP."
)