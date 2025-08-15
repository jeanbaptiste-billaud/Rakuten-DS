import os

import pandas as pd

from src.common_utils import create_symlink_folder
from src.data_module_df.data_balancing import create_balanced_dataset
from src.data_module_df.data_config import PipelineConfig
from src.data_module_df.data_image import create_image_dataset, extract_resnet_features
from src.data_module_df.data_indexing import generate_splits, save_indices, load_indices
from src.data_module_df.data_text import prepare_text_column

if __name__ == "__main__":
    cfg = PipelineConfig()
    csv_file = cfg.raw_dir / "raw_data.csv"
    raw_csv_df = pd.read_csv(csv_file)

    # Equilibrage des données
    # indices_balanced = create_balanced_dataset(cfg, raw_csv_df)
    # balanced_csv_df = raw_csv_df.iloc[indices_balanced, :].reset_index(drop=True)
    # balanced_csv_df["designation_description"] = (balanced_csv_df["designation"] + " " + balanced_csv_df["description"].fillna("")).str.strip()
    # balanced_csv_df = balanced_csv_df.drop(columns=["designation", "description"])
    # indices_split = generate_splits(balanced_csv_df.prdtypecode, random_state=cfg.training.random_state)
    # save_indices(indices_split, cfg.processed_dir / "indices_split.npz")
    # balanced_csv_df.to_csv(cfg.processed_dir / "balanced_data.csv", index=False)
    indices_split = load_indices(cfg.processed_dir / "indices_split.npz")
    balanced_csv_df = pd.read_csv(cfg.processed_dir / "balanced_data.csv")

    # prétraitement des données images
    # for key, index in indices_split.items():
    #     df = balanced_csv_df.iloc[index, :]
    #     dataset = create_image_dataset(df, cfg.raw_dir)
    #     prep_data = extract_resnet_features(dataset, device="cuda", batch_size=cfg.training.batch_size, num_workers=cfg.num_workers)
    #     file_name = "image_" + key.replace("_idx", ".npz")
    #     save_indices(indices_split, cfg.processed_dir / file_name)

    # prétraitement des données textes
    for key, index in indices_split.items():
        print(key)
        df = balanced_csv_df.iloc[index, :]
        prep_df = prepare_text_column(df, cfg, "designation_description")
        file_name = "texte_" + key.replace("_idx", ".npz")
        save_indices(prep_df, cfg.processed_dir / file_name)

    create_symlink_folder(cfg.version, os.path.dirname(cfg.processed_dir))

