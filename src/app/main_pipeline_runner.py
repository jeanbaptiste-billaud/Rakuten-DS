# main_pipeline_runner.py
import argparse
from src.data_module_df.data_config import PipelineConfig
from src.pipeline_steps_df.stage01_initialisation import main as run_stage01
from src.pipeline_steps_df.stage02_eval_image import StageImagePipeline
from src.pipeline_steps_df.stage03_eval_text import StageTextPipeline
from src.pipeline_steps_df.stage04_fusion_multimodale import StageFusionPipeline


def main(config_path, stages):
    config = PipelineConfig.from_yaml(config_path)

    if "stage01" in stages:
        print("\n🔁 Lancement stage 01 - Initialisation")
        run_stage01(config_path)

    if "stage02" in stages:
        print("\n🧠 Lancement stage 02 - Image")
        stage2 = StageImagePipeline(config)
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
    parser = argparse.ArgumentParser(description="Lanceur global de la pipeline MLOps")
    parser.add_argument("--config", type=str, default="config.yaml", help="Chemin du fichier config YAML")
    parser.add_argument("--stages", nargs="+", default=["stage01", "stage02", "stage03", "stage04"],
                        help="Stages à exécuter parmi : stage01, stage02, stage03, stage04")
    args = parser.parse_args()

    main(args.config, args.stages)
