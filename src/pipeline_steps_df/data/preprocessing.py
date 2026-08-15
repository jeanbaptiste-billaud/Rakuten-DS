# preprocessing.py
import os

import pandas as pd
from src.data_module_df.data_text import preprocess_dataframe, save_class_distribution

WORKDIR = os.getenv("WORKDIR", "/workspace")
def main():
    data_dir = os.path.join(WORKDIR, "data")

    raw_path = os.path.join(data_dir, "dataset/raw_dataset.csv")
    output_dir = os.path.join(data_dir, "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    print("📂 Chargement du dataset brut...")
    df = pd.read_csv(raw_path)

    # Fusion des colonnes texte
    df["designation_description"] = (
            df["designation"].fillna("") + " " + df["description"].fillna("")
    ).str.strip()

    print("🔄 Prétraitement du texte...")
    df = preprocess_dataframe(df, "designation_description")

    print("💾 Sauvegarde des données prétraitées...")
    processed_path = os.path.join(output_dir, "preprocessed_text.csv")
    df.to_csv(processed_path, index=False)
    print(f"✅ Données sauvegardées dans {processed_path}")

    # Sauvegarde de la répartition des classes
    save_class_distribution(df, "prdtypecode", os.path.join(output_dir, "class_distribution.json"))

    print("🎉 Prétraitement terminé avec succès.")

if __name__ == "__main__":
    main()
