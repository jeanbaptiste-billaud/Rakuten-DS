# enrich_raw_dataset.py
import pandas as pd
import numpy as np

def main(seed=42):
    dataset_path = "data/dataset/raw_dataset.csv"
    all_path = "data/raw/all_raw_data.csv"

    np.random.seed(seed)
    n_new_samples = 1000

    print("📂 Lecture des fichiers...")
    df_raw = pd.read_csv(dataset_path)
    df_all = pd.read_csv(all_path)

    # Identifier les nouveaux samples (via productid)
    used_ids = set(df_raw['productid'])
    df_candidates = df_all[~df_all['productid'].isin(used_ids)]

    if len(df_candidates) < n_new_samples:
        raise ValueError(f"Pas assez de nouveaux échantillons disponibles ({len(df_candidates)} restants).")

    print(f"🎯 Sélection de {n_new_samples} nouveaux échantillons...")
    df_new = df_candidates.sample(n=n_new_samples, random_state=seed)

    # Concaténation et sauvegarde
    df_updated = pd.concat([df_raw, df_new], ignore_index=True)
    df_updated.to_csv(dataset_path, index=False)

    print(f"✅ {n_new_samples} nouveaux échantillons ajoutés à {dataset_path} (total: {len(df_updated)})")

if __name__ == "__main__":
    main()
