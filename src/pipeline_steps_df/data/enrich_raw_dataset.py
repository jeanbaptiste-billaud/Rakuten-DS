# enrich_raw_dataset.py
import os
import pandas as pd

from src.data_module_df.enrich_logic import select_new_samples, enrich_dataset
from src.utils.common_utils import get_project_root


def main(n_new_samples=1000, seed=42):
    # Détermination du root du projet
    root = os.getenv("WORKDIR", get_project_root())
    data_dir = os.path.join(root, "data")

    all_data_path = os.path.join(data_dir, "raw/all_raw_data.csv")
    dataset_path = os.path.join(data_dir, "dataset/raw_dataset.csv")

    print(f"📂 Lecture des fichiers : {dataset_path} et {all_data_path}")

    df_all = pd.read_csv(all_data_path)
    df_dataset = pd.read_csv(dataset_path)

    print(f"🎯 Sélection de {n_new_samples} nouveaux échantillons...")
    df_new = select_new_samples(df_dataset=df_dataset,
                                df_all=df_all,
                                n_new_samples=n_new_samples,
                                seed=seed)

    # Fusion datasets
    df_updated = enrich_dataset(df_dataset, df_new)

    # Sauvegarde
    df_updated.to_csv(dataset_path, index=False)

    print(f"✅ {n_new_samples} nouveaux échantillons ajoutés ({len(df_updated)} au total).")


if __name__ == "__main__":
    main()
