# pipeline_steps/make_original_dataset.py
import logging
import os
import pandas as pd
from src.data_module_df.data_balancing import generate_text_dataset
from src.utils.common_utils import get_project_root


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    root = os.getenv("WORKDIR", get_project_root())
    data_dir = os.path.join(root, "data")

    input_csv = os.path.join(data_dir, "raw/all_raw_data.csv")

    output_dir = os.path.join(data_dir, "dataset")
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "raw_dataset.csv")

    logger.info(f"📂 Lecture du dataset source : {input_csv}")
    df = pd.read_csv(input_csv)

    logger.info("⚖️ Génération d’un dataset équilibré (10 000 échantillons)...")
    balanced_df = generate_text_dataset(df,
                                        target_size=10_000,
                                        min_per_class=300,
                                        random_state=42,
                                        save_path=output_csv)

    logger.info(f"✅ Dataset d’origine créé ({len(balanced_df)} échantillons)")
    logger.info(f"💾 Fichier sauvegardé dans {output_csv}")


if __name__ == "__main__":
    main()
