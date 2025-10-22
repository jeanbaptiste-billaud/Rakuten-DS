# inference_text_model_bento.py
import bentoml
import pandas as pd
from src.data_module_df.data_text import preprocess_dataframe

# --- 1️⃣ Chargement du modèle BentoML ---
MODEL_NAME = "text_classifier_svm:latest"
print(f"📦 Chargement du modèle BentoML '{MODEL_NAME}'...")
model_ref = bentoml.sklearn.get(MODEL_NAME)
model_runner = model_ref.to_runner()
model_runner.init_local()  # utilisation locale sans serveur REST

# --- 2️⃣ Exemple d'inférence sur un batch de textes ---
texts = [
    "appareil photo reflex professionnel canon",
    "chaussures de course running homme",
    "roman historique sur la première guerre mondiale",
    "console de jeux vidéo portable nintendo switch"
]

df = pd.DataFrame({"designation_description": texts})
df = preprocess_dataframe(df, "designation_description")

# --- 3️⃣ Prédictions ---
print("🔮 Prédiction des classes...")
preds = model_runner.predict.run(df["text_cleaned"])
df["predicted_label"] = preds

# (Optionnel) probabilités associées
try:
    probs = model_runner.predict_proba.run(df["text_cleaned"])
except Exception:
    probs = None

# --- 4️⃣ Affichage ---
for i, text in enumerate(texts):
    print(f"\nTexte : {text}")
    print(f"→ Classe prédite : {preds[i]}")
    if probs is not None:
        print(f"→ Confiance max : {max(probs[i]):.4f}")

# todo: remplacer texts par un text importé ou un csv
# todo: sauvegarder le résultat de la pred ?