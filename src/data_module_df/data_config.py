# data_config.py
import os
import yaml
from src.common_utils import download_and_extract_from_drive

class PipelineConfig:
    def __init__(self, config_dict):
        # Chemins
        self.data_path = config_dict.get('data_path', '../data')
        self.image_dir = config_dict.get('image_dir', '../data/images')
        self.model_path = config_dict.get('model_path', '../data/models')
        self.output_dir = config_dict.get('output_dir', '../data/reports')
        self.plot_dir = config_dict.get('plot_dir', '../data/plots')

        # Paramètres entraînement
        self.batch_size = config_dict.get('batch_size', 32)
        self.target_size = config_dict.get('target_size', 2000)
        self.random_state = config_dict.get('random_state', 42)
        self.num_workers = config_dict.get('num_workers', -1)
        self.early_stopping_patience = config_dict.get('early_stopping_patience', 5)

        # Choix des modèles
        self.image_model_type = config_dict.get('image_model_type', 'xgboost')
        self.text_model_type = config_dict.get('text_model_type', 'svm')

        # Contrôle du pipeline
        self.force_preprocessing = config_dict.get('force_preprocessing', False)
        self.save_outputs = config_dict.get('save_outputs', True)

        # Fusion multimodale
        self.fusion_strategy = config_dict.get('fusion_strategy', 'mean')

        # Évaluation avancée
        self.crossval_folds = config_dict.get('crossval_folds', 5)
        self.optuna_trials = config_dict.get('optuna_trials', 50)

        # Source de données prétraitées (Google Drive ZIP)
        self.preprocessed_drive_url = config_dict.get(
            'preprocessed_drive_url',
            'https://drive.google.com/file/d/1guhuHp0dVRPWCtZ7570jEsTub6m2RrRF/view'
            # 'https://drive.google.com/uc?id=1D7R4EpSc3NYpVsk_4UHQD3CoBLWyYoJe'
        )

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)

        config_dir = os.path.dirname(os.path.abspath(path))
        for key in cfg:
            if "path" in key or "dir" in key:
                cfg[key] = os.path.abspath(os.path.join(config_dir, cfg[key]))

        return cls(cfg)

    def to_dict(self):
        return self.__dict__

    def save_yaml(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    def validate_paths(self):
        for path in [self.data_path, self.image_dir, self.model_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Chemin introuvable : {path}")

    def ensure_preprocessed_data(self):
        expected = ["X_test.npz", "X_test_split.npz", "X_train.npz", "test_split_indices.npz", "train_indices.npz", "y_test_split.npz",
                    "y_train.npz"]
        data_dir = os.path.join(self.data_path, 'processed')
        missing = [f for f in expected if not os.path.exists(os.path.join(data_dir, f))]
        if missing:
            print(f"⚠️ Données prétraitées manquantes : {missing}")
            print("⬇️ Téléchargement depuis Google Drive en cours...")
            download_and_extract_from_drive(self.preprocessed_drive_url, data_dir)
            print("✅ Données prétraitées téléchargées.")

            # Vérification post-téléchargement
            still_missing = [f for f in expected if not os.path.exists(os.path.join(data_dir, f))]
            if still_missing:
                raise FileNotFoundError(f"❌ Les fichiers suivants sont toujours manquants après téléchargement : {still_missing}")
            print("✅ Tous les fichiers nécessaires sont bien présents.")
