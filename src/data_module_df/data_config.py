# data_config.py
import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


class PipelineConfig:
    def __init__(self):
        # 1. Localisation du projet et du YAML
        this_file = Path(__file__).resolve()
        self.project_root = this_file.parents[2]
        self.config_path = self.project_root / "configs" / "config.yaml"

        # 2. Chargement YAML
        self.yaml_cfg = OmegaConf.load(self.config_path)

        # 3. Attributs simples (hors blocs imbriqués)
        for key, value in self.yaml_cfg.items():
            if key not in ["data", "image_model", "text_model", "output", "multithreading"]:
                setattr(self, key, value)

        # 4. Flags d'entraînement
        self.train_image = self.yaml_cfg["image_model"].get('image_force_training', False)
        self.train_text = self.yaml_cfg["text_model"].get('text_force_training', False)

        # 5. Process tous les chemins
        self.process_paths()
        self.image_dir = self.raw_dir / "images"

        # 6. Multithrading setup
        self.num_workers = 1
        self.multithreading()


    def process_paths(self):
        """Construit tous les chemins utiles et crée les dossiers si nécessaire."""

        # -- DATA PATHS --
        data_cfg = self.yaml_cfg.get('data', {})

        for key, rel_path in data_cfg.items():
            path = self.project_root / rel_path
            setattr(self, key, path)

        # versionnée ou latest
        if self.force_preprocessing:
            self.processed_dir = self.processed_dir / f"v{self.version}"
        else:
            self.processed_dir = self.processed_dir / "latest"

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # -- MODEL PATHS --
        image_model_cfg = self.yaml_cfg.get("image_model", {})
        text_model_cfg = self.yaml_cfg.get("text_model", {})

        if self.train_image:
            self.image_model_path = self.project_root / image_model_cfg["image_model_path"]
        else:
            self.image_model_path = self.project_root / image_model_cfg["image_model_path"] / f"v{self.version}"

        if self.train_text:
            self.text_model_path = self.project_root / text_model_cfg["text_model_path"]
        else:
            self.text_model_path = self.project_root / text_model_cfg["text_model_path"] / f"v{self.version}"

        self.image_model_path.mkdir(parents=True, exist_ok=True)
        self.text_model_path.mkdir(parents=True, exist_ok=True)

    def multithreading(self):
        config = self.yaml_cfg.get("multithreading", {})

        if config["activate"]:
            if config["auto"]:
                self.num_workers = min(16, os.cpu_count()//2)
            else:
                self.num_workers = config["num_workers"]


    def to_dict(self):
        return self.__dict__

    def save_yaml(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as f:
            self.yaml.dump(self.to_dict(), f)

    def update_and_save_config(self):
        self.yaml_cfg["version"] += 1
        self.save_yaml(self.config_path)

    def validate_paths(self):
        for path in [self.data_path, self.image_dir, self.model_path]:# Où sont définis ces attributs ?
            if not path.exists():
                raise FileNotFoundError(f"Chemin introuvable : {path}")

    def ensure_preprocessed_data(self):
        expected = [
            "X_test.npz", "X_test_split.npz", "X_train.npz",
            "test_split_indices.npz", "train_indices.npz",
            "y_test_split.npz", "y_train.npz"
        ]
        data_dir = self.processed_dir
        missing = [f for f in expected if not (data_dir / f).exists()]

        if missing:
            print(f"⚠️ Données prétraitées manquantes : {missing}")
            print("⬇️ Récupération via DVC…")

            try:
                subprocess.run(["dvc", "pull"], cwd=self.project_root, check=True, capture_output=True, text=True)
                print("✅ DVC pull terminé avec succès.")
            except subprocess.CalledProcessError as e:
                print("❌ Échec de DVC pull.")
                print("📄 Stdout:", e.stdout)
                print("📄 Stderr:", e.stderr)
                raise RuntimeError("Erreur lors de l'exécution de 'dvc pull'")

            # Re-vérification
            still_missing = [f for f in expected if not (data_dir / f).exists()]
            if still_missing:
                raise FileNotFoundError(f"❌ Les fichiers suivants sont toujours manquants après 'dvc pull' : {still_missing}")
            print("✅ Tous les fichiers nécessaires sont bien présents.")
