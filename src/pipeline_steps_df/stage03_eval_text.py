# stage03_eval_text.py
from src.data_module_df.data_loader import load_processed_npz
from src.models_module_df.model_text_classifier import TextClassifier
from src.visualization_module_df.viz_utils import plot_confusion_matrix


class StageTextPipeline:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None

    def init_stage(self):
        print("📥 Chargement des données texte prétraitées...")
        self.X_train = load_processed_npz("text_train")
        self.X_val = load_processed_npz("text_val")
        self.X_test = load_processed_npz("text_test")

        print("🧠 Initialisation du modèle texte...")
        self.model = TextClassifier(model_path=f"{self.config.model_path}/SVM/model.pkl")

    def train(self):
        print("🚀 Entraînement du modèle texte...")
        self.model.train(self.X_train['text'], self.X_train['labels'])
        self.model.save()
        print("✅ Modèle sauvegardé.")

    def infer(self):
        print("🔎 Inférence sur le jeu de test texte...")
        preds, probs = self.model.predict(self.X_test['text'])
        scores = self.model.evaluate(self.X_test['labels'], preds, probs)

        print("\n📊 Évaluation du modèle texte :")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

        plot_confusion_matrix(
            y_true=self.X_test['labels'],
            y_pred=preds,
            title="Matrice de confusion - modèle texte (SVM)",
            save_path=f"{self.model.model_path.replace('model.pkl', 'confusion_matrix_text.png')}"
        )
        return scores
