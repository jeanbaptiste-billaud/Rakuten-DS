# fusion_multimodale.py
from src.data_module_df.data_loader import load_processed_npz
from src.data_module_df.data_fusion_manager import FusionManager
from src.models_module_df.model_text_classifier import TextClassifier
from src.models_module_df.model_image_classifier import ImageClassifierManager
from src.visualization_module_df.viz_utils import plot_confusion_matrix


class StageFusionPipeline:
    def __init__(self, config):
        self.config = config
        self.image_model = None
        self.text_model = None
        self.X_test_image = None
        self.X_test_text = None
        self.y_test = None
        self.fusion_manager = FusionManager()

    def init_stage(self):
        print("📥 Chargement des données multimodales prétraitées...")
        image = load_processed_npz("X_test")
        text = load_processed_npz("text_test")

        self.X_test_image = image['features']
        self.X_test_text = text['text']
        self.y_test = image['labels']  # doit être égal à text['labels']

        print("🔄 Chargement des modèles texte et image...")
        self.image_model = ImageClassifierManager(
            model_type=self.config.image_model_type,
            model_dir=self.config.model_path
        )
        self.image_model.load()

        self.text_model = TextClassifier(model_path=f"{self.config.model_path}/SVM/model.pkl")
        self.text_model.load()

    def infer(self):
        print("🔎 Inférence des deux modèles...")
        img_preds, img_probs = self.image_model.predict(self.X_test_image)
        txt_preds, txt_probs = self.text_model.predict(self.X_test_text)

        print(f"🔗 Fusion des prédictions via stratégie '{self.config.fusion_strategy}'...")
        fused_probs = self.fusion_manager.fuse_predictions(txt_probs, img_probs, strategy=self.config.fusion_strategy)
        fused_preds = fused_probs.argmax(axis=1)

        scores = self.image_model.evaluate(self.y_test, fused_preds, fused_probs)

        print("\n📊 Évaluation du modèle fusionné :")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

        plot_confusion_matrix(
            y_true=self.y_test,
            y_pred=fused_preds,
            title="Matrice de confusion - fusion multimodale",
            save_path=f"{self.config.output_dir}/confusion_matrix_fusion.png"
        )
        return scores
