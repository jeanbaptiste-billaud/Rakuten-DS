# main_pipeline_runner.py
import argparse
from src.data_module_df.data_config import PipelineConfig
from src.pipeline_steps_df.initialisation import main as run_stage01
from src.pipeline_steps_df.image_preprocessing import StageImagePipeline
from src.pipeline_steps_df.text_preprocessing import StageTextPipeline
from src.pipeline_steps_df.fusion_multimodale import StageFusionPipeline

import sys
import os

def main(config_path, stages):
    config = None

    if "stage01" in stages:
        print("\n🔁 Lancement stage 01 - Initialisation")
        config = run_stage01(config_path)

    if "stage02" in stages:
        print("\n🧠 Lancement stage 02 - Image")
        stage2 = StageImagePipeline(config) # todo: charger tous les fichiers de processed
        stage2.init_stage()
        stage2.train()
        stage2.infer()

    if "stage03" in stages:
        print("\n🧠 Lancement stage 03 - Texte")
        stage3 = StageTextPipeline(config)
        stage3.init_stage()
        stage3.train()
        stage3.infer()

    if "stage04" in stages:
        print("\n🤝 Lancement stage 04 - Fusion multimodale")
        stage4 = StageFusionPipeline(config)
        stage4.init_stage()
        stage4.infer()


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    parser = argparse.ArgumentParser(description="Lanceur global de la pipeline MLOps")
    parser.add_argument("--configs", type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs.yaml")),
                        help="Chemin du fichier configs YAML")
    parser.add_argument("--stages", nargs="+", default=["stage01", "stage02", "stage03", "stage04"],
                        help="Stages à exécuter parmi : stage01, stage02, stage03, stage04")
    args = parser.parse_args()

    main(args.config, args.stages)
