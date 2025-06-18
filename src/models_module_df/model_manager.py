# model_manager.py
import os
import pickle

import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score


class ModelManager:
    def __init__(self, model_type, config):
        self.model_type = model_type
        self.model = None
        self.config = config

    def train(self, X, y):
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**self.config)
            self.model.fit(X, y)
        else:
            raise NotImplementedError("Only XGBoost is currently supported.")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        pred = self.model.predict(X)
        proba = self.model.predict_proba(X)
        return pred, proba

    def evaluate(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        return {'accuracy': acc, 'weighted_f1': f1}

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(os.path.join(path, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
