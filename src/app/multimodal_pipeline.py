# multimodal_pipeline.py
import os

from src.data_module_df.data_fusion_manager import FusionManager
from src.data_module_df.data_image import prepare_image_data
from src.data_module_df.data_text import prepare_text_data
from src.models_module_df.model_image_classifier import ImageClassifierManager
from src.models_module_df.model_text_classifier import TextClassifier


class MultimodalPipeline:
    def __init__(self, config):
        self.config = config
        self.fusion_manager = FusionManager()

        # Ces attributs seront remplis dans prepare_all()
        self.X_train_img = self.X_val_img = self.X_test_img = None
        self.y_train = self.y_val = self.y_test = None
        self.X_train_text = self.X_val_text = self.X_test_text = None

        # Modèles instanciés
        self.image_model = ImageClassifierManager('xgboost', config.model_path)
        self.text_model = TextClassifier(os.path.join(config.model_path, 'SVM', 'model.pkl'))

    def prepare_all(self):
        self.X_train_img, self.X_val_img, self.X_test_img, y_train_img, y_val_img, y_test_img = prepare_image_data(self.config)
        self.X_train_text, self.X_val_text, self.X_test_text, y_train_txt, y_val_txt, y_test_txt = prepare_text_data(self.config)

        assert (y_train_img == y_train_txt).all()
        self.y_train, self.y_val, self.y_test = y_train_img, y_val_img, y_test_img

    def train_all(self):
        self.image_model.train(self.X_train_img, self.y_train)
        self.image_model.save()

        self.text_model.train(self.X_train_text, self.y_train)
        self.text_model.save()

    def predict_all(self):
        img_preds, img_probs = self.image_model.predict(self.X_test_img)
        txt_preds, txt_probs = self.text_model.predict(self.X_test_text)
        fused_probs = self.fusion_manager.fuse_predictions(txt_probs, img_probs, strategy='mean')
        fused_preds = fused_probs.argmax(axis=1)
        return txt_preds, txt_probs, img_preds, img_probs, fused_preds, fused_probs

    def evaluate_all(self):
        txt_preds, txt_probs, img_preds, img_probs, fused_preds, fused_probs = self.predict_all()

        scores = {
            'text': self.text_model.evaluate(self.y_test, txt_preds, txt_probs),
            'image': self.image_model.evaluate(self.y_test, img_preds, img_probs),
            'fusion': self.image_model.evaluate(self.y_test, fused_preds, fused_probs)
        }
        return scores
