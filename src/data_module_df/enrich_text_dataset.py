# pipeline_steps/enrich_raw_dataset.py
import os
import pandas as pd
import numpy as np
from src.common_utils import get_project_root

def main():
    root = get_project_root()
    data_dir = os.path.join(root, "data", "raw")

    raw_path = os.path.join(data_dir, "raw_data.csv")
    all_path = os.path.join(data_dir, "all_raw_data.csv")
    n_new_samples = 1000

    np.random.seed(42)

    print(f"📂 Lecture des fichiers : {raw_path} et {all_path}")
    df_raw = pd.read_csv(raw_path)
    df_all = pd.read_csv(all_path)

    used_ids = set(df_raw["productid"])
    df_candidates = df_all[~df_all["productid"].isin(used_ids)]

    if len(df_candidates) < n_new_samples:
        raise ValueError(f"Pas assez de nouveaux échantillons disponibles ({len(df_candidates)} restants).")

    print(f"🎯 Sélection de {n_new_samples} nouveaux échantillons...")
    df_new = df_candidates.sample(n=n_new_samples, random_state=42)

    df_updated = pd.concat([df_raw, df_new], ignore_index=True)
    df_updated.to_csv(raw_path, index=False)

    print(f"✅ {n_new_samples} nouveaux échantillons ajoutés ({len(df_updated)} total).")

if __name__ == "__main__":
    main()
