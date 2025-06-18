# stage02_eval_image.py
from src.data_module_df.data_loader import load_processed_npz
from src.models_module_df.model_image_classifier import ImageClassifierManager
from src.visualization_module_df.viz_utils import plot_confusion_matrix


class StageImagePipeline:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None

    def init_stage(self):
        print("📥 Chargement des données image prétraitées...")
        self.X_train = load_processed_npz("X_train")
        self.X_val = load_processed_npz("X_val")
        self.X_test = load_processed_npz("X_test")

        print("🧠 Initialisation du modèle image...")
        self.model = ImageClassifierManager(
            model_type=self.config.image_model_type,
            model_dir=self.config.model_path,
            input_dim=self.X_train['features'].shape[1],
            num_classes=len(set(self.X_train['labels']))
        )

    def train(self):
        print("🚀 Entraînement du modèle image...")
        self.model.train(self.X_train['features'], self.X_train['labels'])
        self.model.save()
        print("✅ Modèle sauvegardé.")

    def infer(self):
        print("🔎 Inférence sur le jeu de test image...")
        preds, probs = self.model.predict(self.X_test['features'])
        scores = self.model.evaluate(self.X_test['labels'], preds, probs)

        print("\n📊 Évaluation du modèle image :")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

        plot_confusion_matrix(
            y_true=self.X_test['labels'],
            y_pred=preds,
            title=f"Matrice de confusion - modèle image ({self.model.model_type})",
            save_path=f"{self.model.model_dir}/confusion_matrix_{self.model.model_type}.png"
        )
        return scores
