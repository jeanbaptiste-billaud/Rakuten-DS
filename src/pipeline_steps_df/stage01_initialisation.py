# stage01_initialisation.py
import argparse
from src.data_module_df.data_config import PipelineConfig
from src.data_module_df.data_indexing import generate_splits, save_indices
from src.data_module_df.data_loader import load_training_data
from src.data_module_df.data_image import prepare_image_data
from src.data_module_df.data_text import prepare_text_data


def main(config_path):
    print("🔧 Chargement de la configuration...")
    config = PipelineConfig.from_yaml(config_path)
    config.validate_paths()
    config.ensure_preprocessed_data()

    if config.force_preprocessing:
        print("📊 Chargement des données d'entraînement...")
        X_df, y_df = load_training_data()
        y = y_df['prdtypecode'].values

        print("✂️ Génération des splits stratifiés...")
        indices = generate_splits(y, test_size=0.2, val_size=0.2, random_state=config.random_state)
        save_indices(indices)

        print("🧼 Prétraitement des images...")
        prepare_image_data(config)

        print("🧼 Prétraitement du texte...")
        prepare_text_data(config)

    print("✅ Initialisation terminée.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialisation de la pipeline MLOps")
    parser.add_argument('--config', type=str, default="config.yaml", help="Chemin vers le fichier de configuration YAML")
    args = parser.parse_args()
    main(args.config)
